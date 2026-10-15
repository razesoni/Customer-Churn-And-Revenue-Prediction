# Customer Churn & Revenue Prediction

A machine learning project designed to predict customer churn for a video streaming platform and estimate the financial impact of a retention campaign. The repository includes a reproducible notebook, a Python API, and a Streamlit dashboard for scoring single customers and batches of customers in real time.

## Overview

Customer churn is one of the most important operational metrics for subscription businesses. In streaming platforms, losing customers reduces recurring revenue and increases the cost of customer acquisition. This project builds a predictive model to identify likely churners before they leave, enabling the business to target interventions strategically.

The solution uses:

- XGBoost in a scikit-learn pipeline
- Automated feature engineering for customer behavior and billing signals
- FastAPI for model-serving endpoints
- Streamlit for interactive exploration and inference
- Docker-based deployment for local production-like execution

## Business problem

The goal is to answer the following question:

- Which customers are most likely to cancel their subscription?
- Which customers should receive retention offers?
- What is the expected value of a churn-prevention campaign?

The model is designed to balance recall and precision so that retention teams can act on the highest-risk accounts while managing cost and campaign efficiency.

## Dataset
Coursera Data Science Challenge

## Key findings

The project includes a saved model snapshot and evaluation metrics from the training notebook. The most relevant results are:

- Dataset size:
  - Train: 243,787 rows
  - Test: 104,480 rows
- Source: Coursera data science challenge on video streaming churn
- Model type: XGBClassifier inside a scikit-learn Pipeline with a ColumnTransformer
- Tuning: RandomizedSearchCV with 5-fold cross-validation, scored by ROC-AUC
- Test ROC-AUC: 0.7486
- Train ROC-AUC: 0.7868
- Test accuracy: 0.6822
- Train accuracy: 0.6988
- Classification report (held-out test set):
  - Retained (0): precision 0.91, recall 0.68, F1 0.78
  - Churned (1): precision 0.32, recall 0.68, F1 0.44

The model is strongest for identifying churn risk at a portfolio level and ranking customers by risk likelihood. The project also includes a business-value analysis using a retention campaign threshold and expected savings assumptions, which means the model is not only evaluated on statistical accuracy but also on business impact.

## Repository structure

```text
Customer-Churn-Prediction/
├── Churn_Prediction.ipynb        # End-to-end notebook for exploration, training, and evaluation
├── README.md                     # Project documentation
├── requirements.txt              # Root Python dependencies
├── run.sh                        # Convenience startup script
├── docker-compose.yml            # Multi-container orchestration for backend + frontend
├── model.joblib                  # Trained model artifact
├── backend/
│   ├── Dockerfile                # Backend container configuration
│   ├── __init__.py               # Package marker
│   ├── evaluation.py             # Evaluation logic for ROC, PR, confusion matrix, gains, lift
│   ├── features.py               # Shared feature engineering logic
│   ├── main.py                   # FastAPI application exposing model endpoints
│   ├── model_service.py          # Model loading, preprocessing, and prediction helpers
│   └── requirements.txt          # Backend dependencies
├── frontend/
│   ├── Dockerfile                # Frontend container configuration
│   ├── app.py                    # Streamlit dashboard
│   ├── requirements.txt          # Frontend dependencies
│   └── assets/                   # Static plots exported from the notebook
├── data/
│   ├── train.csv                 # Training dataset
│   ├── test.csv                  # Held-out test set
│   ├── test_sol.csv              # Test labels for evaluation
│   ├── data_descriptions.csv     # Feature dictionary / dataset description
│   └── metrics.json              # Saved model metrics and dataset summary
└── .gitignore
```

## Data and feature engineering

The project uses a video-streaming customer dataset with behavioral, subscription, and engagement features such as:

- AccountAge
- MonthlyCharges
- TotalCharges
- ViewingHoursPerWeek
- AverageViewingDuration
- ContentDownloadsPerMonth
- UserRating
- SupportTicketsPerMonth
- SubscriptionType
- PaymentMethod
- GenrePreference
- WatchlistSize
- ParentalControl
- SubtitlesEnabled
- Gender

The feature engineering layer creates interaction and ratio variables that capture behavioral intensity and churn risk patterns, for example:

- ChargesPerMonthAccount
- DownloadsPerViewingHour
- SupportTicketsPerMonthAge
- EngagementScore
- ValueRatio
- ActivityIntensity
- CostPerRating
- SupportRisk
- CostPerViewingHour
- ValueIndex
- SupportTicketsPerViewingHour
- UnproductiveViewing
- DownloadsPerSupportTicket
- IsNewUser
- IsLoyalUser

This is implemented in `backend/features.py` and mirrors the logic used in the notebook to keep the trained pipeline and deployed system aligned.

## Project workflow

The end-to-end workflow implemented in the repository is:

1. Load customer data from `data/train.csv` and `data/test.csv`
2. Clean and validate raw input fields
3. Engineer additional behavioral/financial features
4. Train an XGBoost classifier with hyperparameter optimization
5. Evaluate the model on train/test splits using ROC-AUC, accuracy, precision, recall, and F1
6. Save the trained model and metrics
7. Expose the model through FastAPI
8. Build an interactive Streamlit dashboard for real-time inference
9. Run both services locally or with Docker

## Technical stack

- Python
- pandas and NumPy
- scikit-learn
- XGBoost
- FastAPI
- Streamlit
- Plotly
- Docker and Docker Compose

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/razesoni/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the project locally

The repository provides a convenience startup script:

```bash
./run.sh
```

Alternatively, start the services manually in two terminals:

### Terminal 1: backend API

```bash
cd backend
uvicorn main:app --port 8000
```

API docs will be available at:

```text
http://localhost:8000/docs
```

### Terminal 2: Streamlit dashboard

```bash
streamlit run frontend/app.py
```

Dashboard URL:

```text
http://localhost:8501
```

## Running with Docker

From the project root, run:

```bash
docker compose up --build
```

This launches both the backend and frontend services. The dashboard is served on:

```text
http://localhost:8501
```

The API documentation is available at:

```text
http://localhost:8000/docs
```

## Dashboard capabilities

The Streamlit app includes several tabs and workflows:

1. Predict
   - Score a single customer
   - View churn probability, risk level, and top drivers

2. Batch & live evaluation
   - Upload a CSV for batch inference
   - If a `Churn` column is present, compute live ROC, PR, confusion matrix, cumulative gains, and decile lift

3. Model performance
   - Review current held-out test metrics
   - Compare live and saved evaluation outputs

4. Feature insights
   - Explore feature importance by weight, gain, and cover
   - Review explanatory and EDA plots exported from the notebook

5. About
   - Project description, model inputs, feature engineering, and endpoints

## API endpoints

The backend exposes the following endpoints:

- `GET /health` — service health check
- `GET /schema` — input schema and category values
- `GET /metrics` — saved notebook metrics
- `GET /feature-importance` — feature importance values for the served model
- `POST /predict` — predict churn for a single customer
- `POST /predict/batch` — score a CSV and compute live evaluation if labels are present

## Model evaluation details

The saved notebook metrics illustrate the model's expected performance on the test set:

- ROC-AUC: 0.7486
- Accuracy: 0.6822
- Churn recall: 0.68
- Churn precision: 0.32
- F1-score for churned customers: 0.44

These metrics suggest that the model is useful for prioritizing high-risk accounts, but it still struggles to separate churners from non-churners with high precision. This is common in churn prediction, where the class is imbalanced and cost-sensitive decisions matter. The repository therefore also evaluates campaign value by threshold, allowing teams to choose operating points based on business outcomes rather than accuracy alone.

## Interpretation of the model

The project includes explainability tools and plots to identify which features most strongly influence churn probability. Examples include:

- Feature importance plots
- SHAP-style driver analysis
- Probability calibration curves
- Confusion matrices
- Cumulative gains and decile lift charts

This is especially valuable for retaining customers in a subscription business, where interventions should be applied to the riskiest customers with the highest expected return.

## Recommended next steps

For a real production rollout, the following extensions would be valuable:

- Retrain the model with a more recent dataset
- Tune the decision threshold against business cost and campaign value
- Add monitoring for model drift over time
- Integrate with CRM or marketing automation tools
- Add A/B testing for retention offers

## License

This project is provided as-is for educational and portfolio use. If you intend to use it in a production or commercial environment, review the licensing terms of the dataset and code dependencies before deployment.

## Conclusion

This repository provides a complete end-to-end churn prediction workflow: from data exploration to feature engineering, model training, evaluation, deployment, and user-facing analytics. It demonstrates how machine learning can be turned into a business-facing decision-support system for subscription retention.

For questions or improvements, please refer to the project notebook, API documentation, and Streamlit dashboard included in this repository.
