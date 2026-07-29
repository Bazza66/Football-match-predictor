from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from app.ml.labels import LABEL_IDS, LABELS, predict_class_ids, validate_labels


def confusion_matrix_list(
    y_true: list[int], y_pred: list[int], label_ids: list[int]
) -> list[list[int]]:
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    return matrix.astype(int).tolist()


def brier_score_multiclass(y_true: list[int], y_prob: list[list[float]]) -> float:
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    n_classes = y_prob_arr.shape[1]
    one_hot = np.eye(n_classes)[y_true_arr]
    return float(np.mean(np.sum((y_prob_arr - one_hot) ** 2, axis=1)))


def calibration_summary(
    y_true: list[int],
    y_prob: list[list[float]],
    labels: list[str],
    n_bins: int = 10,
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    per_class: dict[str, list[dict[str, float]]] = {}
    ece_values: list[float] = []

    for class_idx, label in enumerate(labels):
        prob = y_prob_arr[:, class_idx]
        y_bin = (y_true_arr == class_idx).astype(float)
        bin_ids = np.digitize(prob, bins, right=True) - 1
        class_bins: list[dict[str, float]] = []
        ece = 0.0
        for bin_idx in range(n_bins):
            mask = bin_ids == bin_idx
            count = int(mask.sum())
            if count == 0:
                continue
            avg_pred = float(prob[mask].mean())
            avg_true = float(y_bin[mask].mean())
            class_bins.append(
                {
                    "bin": float(bin_idx),
                    "avg_pred": avg_pred,
                    "avg_true": avg_true,
                    "count": float(count),
                }
            )
            ece += abs(avg_pred - avg_true) * (count / len(prob))
        per_class[str(label)] = class_bins
        ece_values.append(float(ece))

    ece_mean = float(np.mean(ece_values)) if ece_values else float("nan")
    return {
        "per_class": per_class,
        "ece": ece_mean,
    }


def class_priors(y_true: list[int], label_ids: list[int]) -> list[float]:
    counts = {label_id: 0 for label_id in label_ids}
    for value in y_true:
        counts[int(value)] = counts.get(int(value), 0) + 1
    total = sum(counts.values()) or 1
    return [counts[label_id] / total for label_id in label_ids]


def classification_metrics(
    name: str,
    y_true: pd.Series,
    y_pred: list[int],
    y_prob: list[list[float]] | None,
    label_ids: list[int],
    labels: list[str],
) -> dict[str, Any]:
    validate_labels(y_true)
    validate_labels(y_pred)
    if label_ids != LABEL_IDS or labels != LABELS:
        raise RuntimeError("Label mapping mismatch detected in classification_metrics.")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )
    per_class = []
    for idx, label in enumerate(labels):
        per_class.append(
            {
                "label": str(label),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )

    metrics: dict[str, Any] = {
        f"{name}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{name}_f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        f"{name}_f1_per_class": per_class,
        f"{name}_confusion_matrix": confusion_matrix_list(
            list(y_true), list(y_pred), label_ids
        ),
    }
    if y_prob is not None:
        try:
            metrics[f"{name}_log_loss"] = float(
                log_loss(y_true, y_prob, labels=label_ids)
            )
        except ValueError:
            metrics[f"{name}_log_loss"] = float("nan")
        metrics[f"{name}_brier_score"] = brier_score_multiclass(list(y_true), y_prob)
        metrics[f"{name}_calibration"] = calibration_summary(
            list(y_true), y_prob, labels=labels
        )
    return metrics


def evaluate_predictions(
    name: str,
    y_true: pd.Series,
    y_prob: list[list[float]],
) -> dict[str, Any]:
    y_pred = predict_class_ids(y_prob)
    return classification_metrics(name, y_true, y_pred, y_prob, LABEL_IDS, LABELS)
