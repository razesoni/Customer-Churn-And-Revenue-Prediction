"""Wraps the trained sklearn Pipeline (ColumnTransformer -> XGBClassifier)."""
import logging
import re
import sys
from typing import Any
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from .features import ENGINEERED_COLUMNS, engineer_features
except ImportError:  # pragma: no cover
    from features import ENGINEERED_COLUMNS, engineer_features

log = logging.getLogger("churn.model")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ModelService:
    def __init__(self, path: str):
        self.pipeline = joblib.load(path)
        steps = self.pipeline.named_steps
        self.has_pipeline_features = "fe" in steps
        self.pre = steps.get("prep", steps.get("preprocessor"))
        if self.pre is None:
            raise ValueError("Model pipeline must contain a 'prep' or 'preprocessor' step.")
        self.clf = self.pipeline.named_steps["model"]

        fitted = {name: cols for name, _, cols in self.pre.transformers_ if name != "remainder"}
        self.num_cols: list[str] = list(fitted["num"])
        self.cat_cols: list[str] = list(fitted["cat"])

        cat_transformer = self.pre.named_transformers_["cat"]
        encoder = getattr(cat_transformer, "named_steps", {}).get("ohe", cat_transformer)
        self.categories: dict[str, list[str]] = {
            col: [str(v) for v in cats] for col, cats in zip(self.cat_cols, encoder.categories_)
        }
        self.feature_names: list[str] = list(self.pre.get_feature_names_out())

        if self.has_pipeline_features:
            self.raw_columns = list(self.pipeline.feature_names_in_)
            self.model_input_columns = self.raw_columns
        else:
            self.model_input_columns = list(self.pre.feature_names_in_)
            legacy_engineered = set(ENGINEERED_COLUMNS) | {"TotalChargesPerMonth"}
            self.raw_columns = [c for c in self.model_input_columns if c not in legacy_engineered]
        self.raw_numeric = [c for c in self.raw_columns if c not in self.cat_cols]

    # ------------------------------------------------------------------ inputs
    def prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.raw_columns if c not in raw.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")
        df = raw[self.raw_columns].copy()
        for col in self.raw_numeric:
            df[col] = pd.to_numeric(df[col], errors="raise")
        for col in self.cat_cols:
            if col in df.columns:
                values = df[col].astype("string").str.strip()
                df[col] = values.astype(object).where(values.notna(), np.nan)

        if self.has_pipeline_features:
            return df

        engineered = engineer_features(df)
        if "TotalChargesPerMonth" in self.model_input_columns:
            engineered["TotalChargesPerMonth"] = engineered["ChargesPerMonthAccount"]
        return engineered[self.model_input_columns]

    def unseen_categories(self, raw: pd.DataFrame) -> dict[str, list[str]]:
        out = {}
        for col, known in self.categories.items():
            if col in raw.columns:
                values = raw[col].dropna().astype(str).str.strip()
                extra = sorted(set(values) - set(known))
                if extra:
                    out[col] = extra[:10]
        return out

    # ------------------------------------------------------------- predictions
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def contributions(self, X: pd.DataFrame, top_k: int = 10) -> list[dict[str, Any]]:
        """Per-customer SHAP-style contributions (log-odds) via XGBoost's native pred_contribs."""
        try:
            import xgboost as xgb

            Xt = self.pipeline[:-1].transform(X.iloc[[0]])
            booster = self.clf.get_booster()
            names = list(booster.feature_names or self.feature_names)
            contribs = booster.predict(xgb.DMatrix(Xt, feature_names=names), pred_contribs=True)[0]
            dense = Xt.toarray()[0] if hasattr(Xt, "toarray") else np.asarray(Xt)[0]

            rows = [
                {"feature": names[i], "contribution": float(contribs[i]), "value": float(dense[i])}
                for i in range(len(names))
            ]
            rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
            return rows[:top_k]
        except Exception:  # explanations are a bonus; never fail the prediction because of them
            log.exception("Could not compute feature contributions")
            return []

    # ------------------------------------------------------------- importance
    def importance(self, importance_type: str = "weight", top_n: int = 20) -> list[dict[str, Any]]:
        booster = self.clf.get_booster()
        score = booster.get_score(importance_type=importance_type)
        names = self.feature_names
        mapped: dict[str, float] = {}
        for key, val in score.items():
            if re.fullmatch(r"f\d+", key) and int(key[1:]) < len(names):
                key = names[int(key[1:])]
            mapped[key] = float(val)
        ranked = sorted(mapped.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return [{"feature": k, "score": v} for k, v in ranked]
