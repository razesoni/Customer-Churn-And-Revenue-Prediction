"""Streamlit dashboard for the customer churn model. Talks to the FastAPI service."""
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("CHURN_API_URL", "http://localhost:8000").rstrip("/")
ASSETS = Path(__file__).parent / "assets"

RISK_COLORS = {"Low": "#2e9e6b", "Medium": "#e3a008", "High": "#d64545"}

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


# ----------------------------------------------------------------------------- helpers
def stretch(fn, *args, **kwargs):
    """Full-width rendering that works on both old and new Streamlit APIs."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def show_plot(fig, key):
    stretch(st.plotly_chart, fig, key=key)


def show_image(name, caption):
    path = ASSETS / name
    if path.exists():
        stretch(st.image, str(path), caption=caption)
    else:
        st.info(f"{name} not found in frontend/assets.")


def api_get(path, **params):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def api_error(e: Exception) -> str:
    if isinstance(e, requests.HTTPError) and e.response is not None:
        try:
            return str(e.response.json().get("detail", e.response.text))
        except Exception:
            return e.response.text
    return str(e)


@st.cache_data(ttl=600, show_spinner=False)
def get_schema():
    return api_get("/schema")


@st.cache_data(ttl=600, show_spinner=False)
def get_metrics():
    return api_get("/metrics")


@st.cache_data(ttl=600, show_spinner=False)
def get_importance(importance_type, top_n):
    return api_get("/feature-importance", importance_type=importance_type, top_n=top_n)


# ----------------------------------------------------------------------------- API check
try:
    api_get("/health")
    schema = get_schema()
    metrics = get_metrics()
except Exception as e:
    st.title("📉 Customer Churn Predictor")
    st.error(f"Can't reach the FastAPI backend at **{API_URL}**.")
    st.code("cd backend && uvicorn main:app --port 8000", language="bash")
    st.caption(f"Details: {e}")
    st.stop()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Decision threshold",
        0.05, 0.95, 0.50, 0.01,
        help="A customer is flagged as a churner when their probability is at or above this value. "
             "Lower = catch more churners (higher recall) but more false alarms. "
             "The scores in the notebook use 0.50.",
    )
    st.caption("🟢 API connected")
    st.caption(f"`{API_URL}`")
    st.divider()
    st.caption("Model: XGBoost pipeline · AUC-ROC (test): "
               f"**{metrics['test']['roc_auc']:.4f}**")

st.title("📉 Customer Churn Predictor")
st.caption("Video-streaming subscribers · XGBoost model served through FastAPI")

tab_predict, tab_batch, tab_perf, tab_insights, tab_about = st.tabs(
    ["🔮 Predict", "📁 Batch & live evaluation", "📊 Model performance", "🧠 Feature insights", "ℹ️ About"]
)

cats = schema["categories"]

# Starting values for the single-customer form
DEFAULTS = {
    "AccountAge": 30, "MonthlyCharges": 12.0, "TotalCharges": 360.0,
    "ViewingHoursPerWeek": 20.0, "AverageViewingDuration": 90.0,
    "ContentDownloadsPerMonth": 20, "UserRating": 3.0,
    "SupportTicketsPerMonth": 3, "WatchlistSize": 10,
}


# ----------------------------------------------------------------------------- evaluation plots
def render_evaluation(ev: dict, prefix: str):
    """Live versions of the notebook's evaluation graphs, computed from an uploaded labelled CSV."""
    st.subheader("Live evaluation (from the 'Churn' column in your file)")
    c = st.columns(6)
    c[0].metric("ROC-AUC", f"{ev['roc_auc']:.4f}")
    c[1].metric("Accuracy", f"{ev['accuracy']:.4f}")
    c[2].metric("Churn precision", f"{ev['precision']:.2f}")
    c[3].metric("Churn recall", f"{ev['recall']:.2f}")
    c[4].metric("Churn F1", f"{ev['f1']:.2f}")
    c[5].metric("Avg precision", f"{ev['average_precision']:.4f}")

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_scatter(x=ev["roc"]["fpr"], y=ev["roc"]["tpr"], mode="lines",
                        name=f"ROC (AUC = {ev['roc_auc']:.4f})", line=dict(color="darkorange", width=3))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                        line=dict(color="navy", dash="dash"))
        fig.update_layout(title="ROC curve", xaxis_title="False positive rate",
                          yaxis_title="True positive rate", legend=dict(x=0.5, y=0.1), height=400)
        show_plot(fig, f"{prefix}_roc")
    with right:
        fig = go.Figure()
        fig.add_scatter(x=ev["pr"]["recall"], y=ev["pr"]["precision"], mode="lines",
                        name=f"PR (AP = {ev['average_precision']:.4f})", line=dict(color="royalblue", width=3))
        fig.add_hline(y=ev["churn_rate"], line_dash="dot", line_color="gray",
                      annotation_text="Baseline (churn rate)")
        fig.update_layout(title="Precision-Recall curve", xaxis_title="Recall",
                          yaxis_title="Precision", height=400)
        show_plot(fig, f"{prefix}_pr")

    left, right = st.columns(2)
    with left:
        (tn, fp), (fn, tp) = ev["confusion_matrix"]
        fig = px.imshow(
            [[tn, fp], [fn, tp]], text_auto=",d", color_continuous_scale="Blues",
            x=["Predicted retained", "Predicted churn"], y=["Actual retained", "Actual churn"],
        )
        fig.update_layout(title=f"Confusion matrix (threshold {ev['threshold']:.2f})",
                          coloraxis_showscale=False, height=400)
        show_plot(fig, f"{prefix}_cm")
    with right:
        fig = go.Figure()
        fig.add_scatter(x=ev["gains"]["population"], y=ev["gains"]["gain"], mode="lines",
                        name="Model", line=dict(color="seagreen", width=3))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random targeting",
                        line=dict(color="gray", dash="dash"))
        fig.update_layout(title="Cumulative gains", xaxis_title="Share of customers contacted (riskiest first)",
                          yaxis_title="Share of churners captured", height=400)
        show_plot(fig, f"{prefix}_gains")

    dec = pd.DataFrame(ev["deciles"])
    fig = px.bar(dec, x="decile", y="churn_rate", text=dec["lift"].map(lambda v: f"{v:.2f}x"),
                 labels={"decile": "Risk decile (1 = riskiest)", "churn_rate": "Actual churn rate"})
    fig.add_hline(y=ev["churn_rate"], line_dash="dot", annotation_text="Overall churn rate")
    fig.update_layout(title="Churn rate by risk decile (labels show lift)", height=380)
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(tickformat=".0%")
    show_plot(fig, f"{prefix}_deciles")

    with st.expander("Classification report"):
        rep = pd.DataFrame(ev["report"]).T
        st.dataframe(rep.style.format({"precision": "{:.2f}", "recall": "{:.2f}",
                                       "f1-score": "{:.2f}", "support": "{:,.0f}"}))


# ----------------------------------------------------------------------------- TAB 1: predict
with tab_predict:
    st.subheader("Score a single customer")

    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Account & billing**")
            account_age = st.number_input("Account age (months)", 0, 600, DEFAULTS["AccountAge"], 1)
            monthly = st.number_input("Monthly charges", 0.0, 1000.0, DEFAULTS["MonthlyCharges"], 0.5)
            total = st.number_input("Total charges", 0.0, 100000.0, DEFAULTS["TotalCharges"], 10.0)
            subscription = st.selectbox("Subscription type", cats["SubscriptionType"])
            payment = st.selectbox("Payment method", cats["PaymentMethod"])
            paperless = st.selectbox("Paperless billing", cats["PaperlessBilling"])
        with c2:
            st.markdown("**Viewing behaviour**")
            hours = st.number_input("Viewing hours / week", 0.0, 168.0, DEFAULTS["ViewingHoursPerWeek"], 0.5)
            duration = st.number_input("Average viewing duration", 0.0, 1000.0, DEFAULTS["AverageViewingDuration"], 1.0)
            downloads = st.number_input("Content downloads / month", 0, 1000, DEFAULTS["ContentDownloadsPerMonth"], 1)
            content_type = st.selectbox("Content type", cats["ContentType"])
            genre = st.selectbox("Genre preference", cats["GenrePreference"])
            device = st.selectbox("Device registered", cats["DeviceRegistered"])
        with c3:
            st.markdown("**Engagement & profile**")
            rating = st.slider("User rating", 0.0, 5.0, DEFAULTS["UserRating"], 0.1)
            tickets = st.number_input("Support tickets / month", 0, 100, DEFAULTS["SupportTicketsPerMonth"], 1)
            watchlist = st.number_input("Watchlist size", 0, 1000, DEFAULTS["WatchlistSize"], 1)
            multi = st.selectbox("Multi-device access", cats["MultiDeviceAccess"])
            parental = st.selectbox("Parental control", cats["ParentalControl"])
            subtitles = st.selectbox("Subtitles enabled", cats["SubtitlesEnabled"])
            gender = st.selectbox("Gender", cats["Gender"])

        submitted = st.form_submit_button("Predict churn", type="primary")

    if submitted:
        st.session_state["payload"] = {
            "AccountAge": int(account_age), "MonthlyCharges": float(monthly), "TotalCharges": float(total),
            "SubscriptionType": subscription, "PaymentMethod": payment, "PaperlessBilling": paperless,
            "ContentType": content_type, "MultiDeviceAccess": multi, "DeviceRegistered": device,
            "ViewingHoursPerWeek": float(hours), "AverageViewingDuration": float(duration),
            "ContentDownloadsPerMonth": int(downloads), "GenrePreference": genre,
            "UserRating": float(rating), "SupportTicketsPerMonth": int(tickets),
            "Gender": gender, "WatchlistSize": int(watchlist), "ParentalControl": parental,
            "SubtitlesEnabled": subtitles,
        }

    payload = st.session_state.get("payload")
    if payload:
        try:
            r = requests.post(f"{API_URL}/predict", params={"threshold": threshold}, json=payload, timeout=30)
            r.raise_for_status()
            res = r.json()
        except Exception as e:
            st.error(f"Prediction failed: {api_error(e)}")
            res = None

        if res:
            prob, risk = res["churn_probability"], res["risk_level"]
            g, d = st.columns([1, 1.4])
            with g:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob * 100,
                    number={"suffix": "%", "valueformat": ".1f"},
                    title={"text": "Churn probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": RISK_COLORS[risk]},
                        "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.8,
                                      "value": threshold * 100},
                    },
                ))
                fig.update_layout(height=320, margin=dict(t=60, b=10))
                show_plot(fig, "gauge")
                if res["predicted_churn"]:
                    st.error(f"**{risk} risk** — predicted to churn (probability ≥ {threshold:.2f})")
                elif risk == "Medium":
                    st.warning(f"**{risk} risk** — below the {threshold:.2f} threshold but close to it")
                else:
                    st.success(f"**{risk} risk** — predicted to stay")
                st.caption("Black marker on the gauge = decision threshold from the sidebar.")
            with d:
                drivers = pd.DataFrame(res["top_drivers"])
                if drivers.empty:
                    st.info("Per-customer explanations aren't available for this model.")
                else:
                    drivers = drivers.iloc[::-1]
                    labels = [f"{f} = {v:,.2f}" for f, v in zip(drivers["feature"], drivers["value"])]
                    fig = go.Figure(go.Bar(
                        x=drivers["contribution"], y=labels, orientation="h",
                        marker_color=["#d64545" if v > 0 else "#3b7dd8" for v in drivers["contribution"]],
                    ))
                    fig.update_layout(title="What drives this prediction", height=420,
                                      xaxis_title="Contribution to churn score (log-odds)",
                                      margin=dict(l=10, r=10))
                    show_plot(fig, "drivers")
                    st.caption("🔴 pushes toward churn · 🔵 pushes toward staying. "
                               "Computed with XGBoost's built-in SHAP values. Categorical values show 1 when active.")
    else:
        st.info("Fill in the customer details above and press **Predict churn**.")


# ----------------------------------------------------------------------------- TAB 2: batch
with tab_batch:
    st.subheader("Score many customers from a CSV")
    st.write(
        "Upload a CSV with the raw customer columns. If it also contains a `Churn` column (0/1), "
        "the ROC, precision-recall, confusion-matrix and gains graphs are drawn live from your data — "
        "for example your merged `test.csv` + `test_sol.csv`."
    )

    template = {}
    for col in schema["raw_columns"]:
        if col in cats:
            template[col] = cats[col][0]
        else:
            template[col] = DEFAULTS.get(col, 0)
    template_df = pd.DataFrame([{"CustomerID": "C0001", **template}])
    st.download_button("⬇️ Download CSV template", template_df.to_csv(index=False), "churn_template.csv", "text/csv")

    upload = st.file_uploader("CSV file", type="csv")
    if upload is not None and st.button("Run predictions", type="primary"):
        with st.spinner("Scoring..."):
            try:
                r = requests.post(
                    f"{API_URL}/predict/batch", params={"threshold": threshold},
                    files={"file": (upload.name, upload.getvalue(), "text/csv")}, timeout=300,
                )
                r.raise_for_status()
                st.session_state["batch"] = r.json()
            except Exception as e:
                st.session_state.pop("batch", None)
                st.error(f"Batch scoring failed: {api_error(e)}")

    batch = st.session_state.get("batch")
    if batch:
        if abs(batch["threshold"] - threshold) > 1e-9:
            st.warning(f"Results were computed at threshold {batch['threshold']:.2f}; "
                       "press **Run predictions** again to apply the current sidebar value.")
        if batch["unseen_categories"]:
            st.warning(f"Some category values weren't seen in training and were treated as 'unknown': "
                       f"{batch['unseen_categories']}")

        res = pd.DataFrame({
            "CustomerID": batch["customer_id"],
            "ChurnProbability": batch["churn_probability"],
            "PredictedChurn": batch["predicted_churn"],
            "RiskLevel": batch["risk_level"],
        }).sort_values("ChurnProbability", ascending=False)

        k = st.columns(4)
        k[0].metric("Customers scored", f"{batch['n_rows']:,}")
        k[1].metric("Predicted to churn", f"{int(res['PredictedChurn'].sum()):,}")
        k[2].metric("Predicted churn rate", f"{res['PredictedChurn'].mean():.1%}")
        k[3].metric("Average probability", f"{res['ChurnProbability'].mean():.1%}")

        left, right = st.columns([1.6, 1])
        with left:
            fig = px.histogram(res, x="ChurnProbability", nbins=40, color="RiskLevel",
                               color_discrete_map=RISK_COLORS, range_x=[0, 1])
            fig.add_vline(x=batch["threshold"], line_dash="dash", annotation_text="threshold")
            fig.update_layout(title="Distribution of churn probabilities", height=360, bargap=0.05)
            show_plot(fig, "batch_hist")
        with right:
            counts = res["RiskLevel"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
            counts.columns = ["RiskLevel", "Customers"]
            fig = px.bar(counts, x="RiskLevel", y="Customers", color="RiskLevel",
                         color_discrete_map=RISK_COLORS)
            fig.update_layout(title="Customers by risk level", height=360, showlegend=False)
            show_plot(fig, "batch_risk")

        st.markdown("**Predictions (riskiest first)**")
        stretch(st.dataframe, res.style.format({"ChurnProbability": "{:.1%}"}), hide_index=True)
        st.download_button("⬇️ Download predictions", res.to_csv(index=False), "churn_predictions.csv", "text/csv")

        if batch["evaluation"]:
            st.divider()
            render_evaluation(batch["evaluation"], "batch")
        elif batch["evaluation_note"]:
            st.info(batch["evaluation_note"])


# ----------------------------------------------------------------------------- TAB 3: performance
with tab_perf:
    st.subheader("Held-out test performance (from the training notebook)")
    test, train = metrics["test"], metrics["train"]
    rep = metrics["classification_report"]

    k = st.columns(4)
    k[0].metric("Test ROC-AUC", f"{test['roc_auc']:.4f}", f"{test['roc_auc'] - train['roc_auc']:+.4f} vs train", delta_color="off")
    k[1].metric("Test accuracy", f"{test['accuracy']:.4f}", f"{test['accuracy'] - train['accuracy']:+.4f} vs train", delta_color="off")
    k[2].metric("Churn recall", f"{rep['Churned (1)']['recall']:.2f}")
    k[3].metric("Churn precision", f"{rep['Churned (1)']['precision']:.2f}")

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("**Train vs test**")
        st.dataframe(pd.DataFrame(
            {"Train": [train["roc_auc"], train["accuracy"]], "Test": [test["roc_auc"], test["accuracy"]]},
            index=["ROC-AUC", "Accuracy"]).style.format("{:.4f}"))
        ds = metrics["dataset"]
        st.caption(f"Train: {ds['train_shape'][0]:,} rows · Test: {ds['test_shape'][0]:,} rows · "
                   f"{ds['train_shape'][1]} columns")
    with right:
        st.markdown("**Classification report (threshold 0.50)**")
        st.dataframe(pd.DataFrame(rep).T.rename(columns={"f1": "f1-score"}).style.format(
            {"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:,.0f}"}))

    churn_share = rep["Churned (1)"]["support"] / rep["Weighted avg"]["support"]
    st.info(
        f"Only about **{churn_share:.0%}** of test customers churn, so at 68% recall the model also flags many "
        f"loyal customers — that is why churn precision is {rep['Churned (1)']['precision']:.2f}. "
        "Use the sidebar threshold to trade recall for precision."
    )

    st.markdown("### Evaluation graphs")
    a, b = st.columns(2)
    with a:
        show_image("roc_curve.png", "ROC curve")
    with b:
        show_image("pr_curve.png", "Precision-recall curve")
    a, b = st.columns(2)
    with a:
        show_image("confusion_matrix.png", "Confusion matrix (test set)")
    with b:
        show_image("cumulative_gains.png", "Cumulative gains curve")


# ----------------------------------------------------------------------------- TAB 4: insights
with tab_insights:
    st.subheader("What the model pays attention to")

    c1, c2 = st.columns([1, 1])
    imp_type = c1.selectbox(
        "Importance measure", ["weight", "gain", "cover"],
        help="weight = how often a feature is used to split · gain = average improvement per split · "
             "cover = average number of samples affected.",
    )
    top_n = c2.slider("Features to show", 5, 40, 15)
    try:
        imp = pd.DataFrame(get_importance(imp_type, top_n)["features"]).iloc[::-1]
        fig = px.bar(imp, x="score", y="feature", orientation="h")
        fig.update_layout(title=f"Top {len(imp)} features by {imp_type} (live from model)",
                          height=max(380, 26 * len(imp) + 120), xaxis_title=imp_type, yaxis_title="")
        show_plot(fig, "importance")
    except Exception as e:
        st.error(f"Could not load feature importance: {api_error(e)}")

    st.markdown("### Notebook graphs")
    a, b = st.columns(2)
    with a:
        show_image("feature_importance.png", "Top 10 feature importances (notebook)")
    with b:
        show_image("correlation_heatmap.png", "Correlation between numeric features")
    show_image("churn_by_subscription.png", "Churn by subscription type")
    show_image("churn_by_categorical.png", "Churn across categorical features")
    with st.expander("Last decision tree of the model (large)"):
        show_image("last_tree.png", "Tree #1499 — one of 1,500 trees in the ensemble")


# ----------------------------------------------------------------------------- TAB 5: about
with tab_about:
    st.subheader("About this project")
    ds = metrics["dataset"]
    st.markdown(f"""
**Goal:** predict whether a subscriber of a video-streaming platform will churn.
**Data:** {ds['source']}. Evaluation metric: ROC-AUC.

**Model:** {metrics['model']['algorithm']}.
Class imbalance handled with `scale_pos_weight`. Hyper-parameters found with {metrics['model']['tuning']}.
""")
    with st.expander("Final hyper-parameters"):
        st.json(metrics["model"]["params"])

    st.markdown("**Inputs the API expects (raw columns)**")
    st.write(", ".join(f"`{c}`" for c in schema["raw_columns"]))

    st.markdown("**Features engineered automatically by the API** (same code as the notebook)")
    st.write(", ".join(f"`{c}`" for c in schema["engineered"]))

    st.markdown("**API endpoints**")
    st.markdown(f"""
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service status |
| GET | `/schema` | Input columns and valid categories |
| GET | `/metrics` | Notebook scores |
| GET | `/feature-importance` | Live importance from the model |
| POST | `/predict` | Score one customer (+ top drivers) |
| POST | `/predict/batch` | Score a CSV (+ live evaluation if `Churn` is present) |

Interactive docs: [{API_URL}/docs]({API_URL}/docs)
""")
