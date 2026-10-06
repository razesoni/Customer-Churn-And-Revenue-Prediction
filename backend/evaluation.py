"""Compute every score/curve the dashboard draws, from labels + predicted probabilities."""
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _thin(*arrays: np.ndarray, n: int = 400) -> list[list[float]]:
    """Down-sample curves so the JSON payload stays small."""
    length = len(arrays[0])
    if length <= n:
        return [np.asarray(a, dtype=float).tolist() for a in arrays]
    idx = np.unique(np.linspace(0, length - 1, n).astype(int))
    return [np.asarray(a, dtype=float)[idx].tolist() for a in arrays]


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Evaluation needs both churned (1) and retained (0) customers in the 'Churn' column.")

    y_pred = (proba >= threshold).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fpr, tpr, _ = roc_curve(y_true, proba)
    precision, recall, _ = precision_recall_curve(y_true, proba)

    # Cumulative gains: share of all churners captured by contacting the top X% riskiest customers
    order = np.argsort(-proba, kind="stable")
    sorted_y = y_true[order]
    gain = np.concatenate([[0.0], np.cumsum(sorted_y) / sorted_y.sum()])
    population = np.concatenate([[0.0], np.arange(1, len(sorted_y) + 1) / len(sorted_y)])

    # Decile lift: churn rate in each 10% slice of customers, riskiest first
    overall_rate = float(y_true.mean())
    deciles = []
    for i, chunk in enumerate(np.array_split(sorted_y, 10), start=1):
        rate = float(chunk.mean())
        deciles.append({"decile": i, "churn_rate": rate, "lift": rate / overall_rate if overall_rate else 0.0})

    fpr_t, tpr_t = _thin(fpr, tpr)
    rec_t, prec_t = _thin(recall, precision)
    pop_t, gain_t = _thin(population, gain)

    return {
        "n": int(len(y_true)),
        "churn_rate": overall_rate,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "precision": float(report["1"]["precision"]),
        "recall": float(report["1"]["recall"]),
        "f1": float(report["1"]["f1-score"]),
        "report": report,
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
        "roc": {"fpr": fpr_t, "tpr": tpr_t},
        "pr": {"recall": rec_t, "precision": prec_t},
        "gains": {"population": pop_t, "gain": gain_t},
        "deciles": deciles,
    }
