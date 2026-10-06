"""Wraps the trained sklearn Pipeline (ColumnTransformer -> XGBClassifier)."""
import logging
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from .features import ENGINEERED_COLUMNS, engineer_features
except ImportError:  # pragma: no cover
    from features import ENGINEERED_COLUMNS, engineer_features

log = logging.getLogger("churn.model")


class ModelService:
    def __init__(self, path: str):
        self.pipeline = joblib.load(path)
        self.pre = self.pipeline.named_steps["preprocessor"]
        self.clf = self.pipeline.named_steps["model"]

        # Column layout is read from the fitted pipeline, so it can never drift from training.
        self.input_columns: list[str] = list(self.pre.feature_names_in_)
        fitted = {name: cols for name, _, cols in self.pre.transformers_ if name != "remainder"}
        self.num_cols: list[str] = list(fitted["num"])
        self.cat_cols: list[str] = list(fitted["cat"])

        encoder = self.pre.named_transformers_["cat"]
        self.categories: dict[str, list[str]] = {
            col: [str(v) for v in cats] for col, cats in zip(self.cat_cols, encoder.categories_)
        }
        self.feature_names: list[str] = self.num_cols + list(encoder.get_feature_names_out(self.cat_cols))

        self.raw_numeric = [c for c in self.num_cols if c not in ENGINEERED_COLUMNS]
        self.raw_columns = self.raw_numeric + self.cat_cols

    # ------------------------------------------------------------------ inputs
    def prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.raw_columns if c not in raw.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")
        df = raw[self.raw_columns].copy()
        for col in self.raw_numeric:
            df[col] = pd.to_numeric(df[col], errors="raise")
        if df[self.raw_numeric].isna().any().any():
            bad = df[self.raw_numeric].columns[df[self.raw_numeric].isna().any()].tolist()
            raise ValueError(f"Empty values found in numeric column(s): {', '.join(bad)}")
        return engineer_features(df)[self.input_columns]

    def unseen_categories(self, raw: pd.DataFrame) -> dict[str, list[str]]:
        out = {}
        for col, known in self.categories.items():
            if col in raw.columns:
                extra = sorted(set(raw[col].dropna().astype(str)) - set(known))
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

            Xt = self.pre.transform(X.iloc[[0]])
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
