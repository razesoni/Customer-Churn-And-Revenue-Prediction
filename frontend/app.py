"""Streamlit dashboard for the customer churn model. Talks to the FastAPI service."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

API_URL = os.getenv("CHURN_API_URL", "http://localhost:8000").rstrip("/")
ASSETS = Path(__file__).parent / "assets"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TEST_FEATURES_PATH = DATA_DIR / "test.csv"
TEST_LABELS_PATH = DATA_DIR / "test_sol.csv"
MODEL_CARD_PATH = Path(__file__).resolve().parents[1] / "model_card.json"

RISK_COLORS = {"Low": "#2e9e6b", "Medium": "#e3a008", "High": "#d64545"}
BUSINESS_ASSUMPTIONS = {"horizon_months": 12, "save_rate": 0.15, "offer_cost": 15.0}
if MODEL_CARD_PATH.exists():
    try:
        BUSINESS_ASSUMPTIONS.update(json.loads(MODEL_CARD_PATH.read_text()).get("business_assumptions", {}))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass

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


@st.cache_data(ttl=3600, show_spinner=False)
def get_current_test_predictions(api_url, test_path, labels_path, test_mtime, labels_mtime):
    test = pd.read_csv(test_path)
    labels = pd.read_csv(labels_path)
    if not {"CustomerID", "Churn"}.issubset(labels.columns):
        raise ValueError("test_sol.csv must contain CustomerID and Churn columns.")

    labels = labels[["CustomerID", "Churn"]].drop_duplicates("CustomerID", keep="first")
    labelled = test.merge(labels, on="CustomerID", how="inner").dropna(subset=["Churn"])
    if labelled.empty:
        raise ValueError("No labelled test customers were found.")

    probabilities = []
    for start in range(0, len(labelled), 25_000):
        chunk = labelled.iloc[start:start + 25_000]
        response = requests.post(
            f"{api_url}/predict/batch",
            params={"threshold": 0.5},
            files={"file": ("test.csv", chunk.drop(columns="Churn").to_csv(index=False), "text/csv")},
            timeout=300,
        )
        response.raise_for_status()
        probabilities.extend(response.json()["churn_probability"])

    return {
        "customer_id": labelled["CustomerID"].astype(str).tolist(),
        "actual": labelled["Churn"].astype(int).tolist(),
        "probability": probabilities,
        "monthly_charges": pd.to_numeric(labelled["MonthlyCharges"], errors="raise").tolist(),
    }


def net_campaign_value(y_true, probabilities, monthly_charges, threshold):
    contacted = probabilities >= threshold
    saved = contacted & (y_true == 1)
    customer_value = monthly_charges * BUSINESS_ASSUMPTIONS["horizon_months"]
    return (
        BUSINESS_ASSUMPTIONS["save_rate"] * customer_value[saved].sum()
        - BUSINESS_ASSUMPTIONS["offer_cost"] * contacted.sum()
    )


def render_current_test_results(data, threshold):
    y_true = np.asarray(data["actual"], dtype=int)
    probabilities = np.asarray(data["probability"], dtype=float)
    monthly_charges = np.asarray(data["monthly_charges"], dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    prevalence = float(y_true.mean())
    net_value = net_campaign_value(y_true, probabilities, monthly_charges, threshold)

    st.subheader("Current model on the held-out test set")
    metrics = st.columns(7)
    metrics[0].metric("Customers", f"{len(y_true):,}")
    metrics[1].metric("ROC-AUC", f"{roc_auc_score(y_true, probabilities):.4f}")
    metrics[2].metric("PR-AUC", f"{average_precision_score(y_true, probabilities):.4f}")
    metrics[3].metric("Accuracy", f"{accuracy_score(y_true, predicted):.3f}")
    metrics[4].metric("Churn precision", f"{precision_score(y_true, predicted, zero_division=0):.3f}")
    metrics[5].metric("Churn recall", f"{recall_score(y_true, predicted, zero_division=0):.3f}")
    metrics[6].metric("Net campaign value", f"${net_value:,.0f}")
    st.caption(
        f"Threshold {threshold:.2f} · assumptions: ${BUSINESS_ASSUMPTIONS['offer_cost']:,.0f} offer, "
        f"{BUSINESS_ASSUMPTIONS['save_rate']:.0%} save rate, "
        f"{BUSINESS_ASSUMPTIONS['horizon_months']} month value horizon."
    )

    left, right = st.columns(2)
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    with left:
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name="Current model",
                        line=dict(color="#246b62", width=3))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                        line=dict(color="#777777", dash="dash"))
        fig.update_layout(title=f"ROC curve · AUC {roc_auc_score(y_true, probabilities):.3f}",
                          xaxis_title="False positive rate", yaxis_title="True positive rate", height=380)
        show_plot(fig, "current_test_roc")
    with right:
        fig = go.Figure()
        fig.add_scatter(x=recall, y=precision, mode="lines", name="Current model",
                        line=dict(color="#bd5a37", width=3))
        fig.add_hline(y=prevalence, line_dash="dot", line_color="#777777",
                      annotation_text=f"Churn prevalence {prevalence:.1%}")
        fig.update_layout(title=f"Precision-recall · AP {average_precision_score(y_true, probabilities):.3f}",
                          xaxis_title="Recall", yaxis_title="Precision", height=380)
        show_plot(fig, "current_test_pr")

    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    offer_cost = BUSINESS_ASSUMPTIONS["offer_cost"]
    save_rate = BUSINESS_ASSUMPTIONS["save_rate"]
    customer_value = monthly_charges * BUSINESS_ASSUMPTIONS["horizon_months"]
    dollars = [
        [0, -offer_cost * fp],
        [-customer_value[(y_true == 1) & (predicted == 0)].sum(),
         save_rate * customer_value[(y_true == 1) & (predicted == 1)].sum() - offer_cost * tp],
    ]
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    cell_text = [
        [f"{labels[row][col]}<br>{matrix[row, col]:,} ({matrix[row, col] / len(y_true):.1%})"
         f"<br>{'-' if dollars[row][col] < 0 else '+'}${abs(dollars[row][col]):,.0f}"
         for col in range(2)]
        for row in range(2)
    ]
    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Heatmap(
            z=matrix / len(y_true), x=["Predicted stay", "Predicted churn"],
            y=["Actual stay", "Actual churn"], text=cell_text, texttemplate="%{text}",
            colorscale="Blues", showscale=False, hovertemplate="%{text}<extra></extra>",
        ))
        fig.update_layout(title=f"Confusion matrix · threshold {threshold:.2f}", height=390)
        show_plot(fig, "current_test_confusion")
    with right:
        threshold_grid = np.linspace(0.02, 0.95, 94)
        values = [net_campaign_value(y_true, probabilities, monthly_charges, value) for value in threshold_grid]
        fig = go.Figure(go.Scatter(x=threshold_grid, y=values, mode="lines",
                                   line=dict(color="#246b62", width=3), name="Held-out test"))
        fig.add_vline(x=threshold, line_dash="dot", line_color="#bd5a37",
                      annotation_text=f"Current {threshold:.2f}")
        fig.add_hline(y=0, line_color="#777777", line_width=1)
        fig.update_layout(title="Retention campaign value by threshold", xaxis_title="Decision threshold",
                          yaxis_title="Net value vs. doing nothing ($)", height=390)
        show_plot(fig, "current_test_profit")

    order = np.argsort(-probabilities, kind="stable")
    sorted_y = y_true[order]
    decile_rows = []
    cumulative = 0
    churners = max(int(y_true.sum()), 1)
    for decile, group in enumerate(np.array_split(sorted_y, 10), start=1):
        count = int(group.sum())
        cumulative += count
        rate = float(group.mean()) if len(group) else 0.0
        decile_rows.append({"Decile": decile, "Customers": len(group), "Churn rate": rate,
                            "Lift": rate / prevalence if prevalence else 0.0,
                            "Cumulative churn captured": cumulative / churners})
    deciles = pd.DataFrame(decile_rows)
    left, right = st.columns(2)
    with left:
        fig = px.bar(deciles, x="Decile", y="Churn rate", hover_data=["Lift", "Customers"],
                     color="Churn rate", color_continuous_scale=["#c8ddd5", "#246b62"])
        fig.add_hline(y=prevalence, line_dash="dot", line_color="#bd5a37",
                      annotation_text=f"Baseline {prevalence:.1%}")
        fig.update_layout(title="Churn rate by risk decile · 1 is highest", height=380,
                          coloraxis_showscale=False)
        fig.update_yaxes(tickformat=".0%")
        show_plot(fig, "current_test_deciles")
    with right:
        population = np.linspace(0, 1, 11)
        capture = [0.0] + deciles["Cumulative churn captured"].tolist()
        fig = go.Figure()
        fig.add_scatter(x=population, y=capture, mode="lines+markers", name="Current model",
                        line=dict(color="#246b62", width=3))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random targeting",
                        line=dict(color="#777777", dash="dash"))
        fig.update_layout(title="Cumulative churn capture", xaxis_title="Share of customers contacted",
                          yaxis_title="Share of churners captured", height=380)
        show_plot(fig, "current_test_gains")

    observed, predicted_probability = calibration_curve(y_true, probabilities, n_bins=15, strategy="quantile")
    fig = go.Figure()
    fig.add_scatter(x=predicted_probability, y=observed, mode="lines+markers", name="Current model",
                    line=dict(color="#246b62", width=3))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                    line=dict(color="#777777", dash="dash"))
    fig.update_layout(title="Probability calibration", xaxis_title="Mean predicted probability",
                      yaxis_title="Observed churn rate", height=380)
    show_plot(fig, "current_test_calibration")

    predictions = pd.DataFrame({
        "CustomerID": data["customer_id"],
        "ChurnProbability": probabilities,
        "PredictedChurn": predicted,
        "RiskLevel": np.where(probabilities >= threshold, "High",
                               np.where(probabilities >= 0.8 * threshold, "Medium", "Low")),
        "ActualChurn": y_true,
    }).sort_values("ChurnProbability", ascending=False)
    st.markdown("**Highest-risk test customers**")
    stretch(st.dataframe, predictions.head(50).style.format({"ChurnProbability": "{:.1%}"}), hide_index=True)
    st.download_button("Download all test predictions", predictions.to_csv(index=False),
                        "current_test_predictions.csv", "text/csv", key="download_current_test_predictions")


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
default_threshold = min(0.95, max(0.05, float(schema.get("default_threshold", 0.50))))
with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Decision threshold", 0.05, 0.95, default_threshold, 0.01,
        help="A customer is flagged as a churner when their probability is at or above this value. "
             "Lower = catch more churners (higher recall) but more false alarms. "
             "The initial value is the notebook's profit-selected threshold when available.",
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
    st.subheader("Live evaluation of the served model")
    st.caption("Scores the held-out test customers through the running API; cached for one hour.")
    if not TEST_FEATURES_PATH.exists() or not TEST_LABELS_PATH.exists():
        st.warning("The local data/test.csv and data/test_sol.csv files are needed for held-out evaluation.")
    elif st.button("Score held-out test set", type="primary", key="score_current_test"):
        try:
            with st.spinner("Scoring the held-out customers and preparing evaluation charts..."):
                st.session_state["current_test_results"] = get_current_test_predictions(
                    API_URL,
                    str(TEST_FEATURES_PATH),
                    str(TEST_LABELS_PATH),
                    TEST_FEATURES_PATH.stat().st_mtime,
                    TEST_LABELS_PATH.stat().st_mtime,
                )
        except Exception as e:
            st.error(f"Could not evaluate the current model: {api_error(e)}")

    current_test_results = st.session_state.get("current_test_results")
    if current_test_results:
        render_current_test_results(current_test_results, threshold)
    else:
        st.info("Run the held-out evaluation to see current metrics, notebook-aligned graphs, and customer predictions.")

    with st.expander("Saved training snapshot and notebook graphs"):
        st.caption("Metric tables come from the API's /metrics endpoint; graphs are static exports from the notebook. "
                   "Live results above use the model currently served.")
        test, train = metrics["test"], metrics["train"]
        rep = metrics["classification_report"]
        k = st.columns(4)
        k[0].metric("Saved test ROC-AUC", f"{test['roc_auc']:.4f}")
        k[1].metric("Saved test accuracy", f"{test['accuracy']:.4f}")
        k[2].metric("Saved churn recall", f"{rep['Churned (1)']['recall']:.2f}")
        k[3].metric("Saved churn precision", f"{rep['Churned (1)']['precision']:.2f}")
        left, right = st.columns([1, 1.3])
        with left:
            st.markdown("**Saved train vs test**")
            st.dataframe(pd.DataFrame(
                {"Train": [train["roc_auc"], train["accuracy"]], "Test": [test["roc_auc"], test["accuracy"]]},
                index=["ROC-AUC", "Accuracy"]).style.format("{:.4f}"))
            ds = metrics["dataset"]
            st.caption(f"Train: {ds['train_shape'][0]:,} rows · Test: {ds['test_shape'][0]:,} rows · "
                       f"{ds['train_shape'][1]} columns")
        with right:
            st.markdown("**Saved classification report (threshold 0.50)**")
            st.dataframe(pd.DataFrame(rep).T.rename(columns={"f1": "f1-score"}).style.format(
                {"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:,.0f}"}))
        st.markdown("**Notebook evaluation graphs**")
        show_image("roc_pr_vs_baselines.png", "ROC and precision-recall: XGBoost vs. logistic regression and no-skill baselines")
        a, b = st.columns(2)
        with a:
            show_image("profit_vs_threshold.png", "Retention campaign value by threshold (out-of-fold vs. held-out test)")
        with b:
            show_image("confusion_matrix_profit.png", "Confusion matrix with % of customers and $ impact")
        show_image("decile_lift_gains.png", "Decile lift and cumulative gains on the held-out test set")
        a, b = st.columns(2)
        with a:
            show_image("calibration.png", "Probability calibration curve")
        with b:
            show_image("learning_curve.png", "Learning curve with early stopping")


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
    st.caption("Static figures exported from the final run of Churn_Prediction.ipynb.")

    st.markdown("#### Explainability")
    a, b = st.columns(2)
    with a:
        show_image("importance_gain_vs_shap.png", "XGBoost built-in importance (gain) vs. mean |SHAP value|")
    with b:
        show_image("shap_beeswarm.png", "SHAP summary: direction and size of each driver's effect")
    a, b = st.columns(2)
    with a:
        show_image("shap_waterfall.png", "Why the highest-risk customer in the SHAP sample was flagged")
    with b:
        show_image("shap_dependence.png", "SHAP dependence plot for the strongest driver")

    st.markdown("#### Exploratory data analysis")
    a, b = st.columns(2)
    with a:
        show_image("target_balance.png", "Target balance in the train and test sets")
    with b:
        show_image("correlation_heatmap.png", "Correlation between numeric features and churn (train)")
    with st.expander("Numeric feature distributions by churn class"):
        show_image("numeric_distributions.png", "Numeric feature distributions, churned vs. retained (train sample)")
    with st.expander("Churn rate by category"):
        show_image("churn_by_categorical.png", "Churn rate for each category of every categorical feature")


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