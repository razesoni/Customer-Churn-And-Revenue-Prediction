# Customer Churn Predictor (Streamlit + FastAPI)

```
churn_app/
├── model.joblib          your trained pipeline (ColumnTransformer -> XGBClassifier)
├── metrics.json          scores copied from your notebook
├── backend/              FastAPI
│   ├── main.py           endpoints
│   ├── features.py       feature engineering (exact copy of the notebook)
│   ├── model_service.py  loads the pipeline, predictions, importance, SHAP-style drivers
│   └── evaluation.py     ROC / PR / confusion matrix / gains / decile lift
└── frontend/
    ├── app.py            Streamlit dashboard
    └── assets/           plots extracted from your notebook
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
./run.sh                                              # or the two commands below
```

Manually, in two terminals:

```bash
cd backend && uvicorn main:app --port 8000     # API docs at http://localhost:8000/docs
streamlit run frontend/app.py                  # dashboard at http://localhost:8501
```

To point the dashboard at a remote API: `CHURN_API_URL=https://your-api streamlit run frontend/app.py`

## Dashboard tabs

1. **Predict** – form for one customer -> probability gauge, risk level, and the features pushing the score up/down.
2. **Batch & live evaluation** – upload a CSV to score many customers. If it has a 0/1 `Churn` column
   (e.g. `test.csv` merged with `test_sol.csv`), ROC, PR, confusion matrix, cumulative gains and
   decile lift are computed live.
3. **Model performance** – the notebook's accuracy, ROC-AUC, classification report and plots.
4. **Feature insights** – live feature importance (weight / gain / cover) plus the EDA plots.
5. **About** – model, features, endpoints.

The sidebar threshold (default 0.50, same as the notebook) controls what counts as a predicted churner.
