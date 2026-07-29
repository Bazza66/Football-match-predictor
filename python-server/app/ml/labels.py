from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

LABELS = ["H", "D", "A"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
LABEL_IDS = list(range(len(LABELS)))


def encode_labels(values: Sequence[str] | pd.Series) -> list[int]:
    if isinstance(values, pd.Series):
        values = values.tolist()
    encoded: list[int] = []
    unknown: set[str] = set()
    for value in values:
        if value not in LABEL_TO_ID:
            unknown.add(str(value))
            continue
        encoded.append(LABEL_TO_ID[str(value)])
    if unknown:
        raise RuntimeError(f"Unknown labels encountered: {sorted(unknown)}.")
    return encoded


def decode_labels(values: Sequence[int] | np.ndarray) -> list[str]:
    decoded: list[str] = []
    for value in values:
        if int(value) not in ID_TO_LABEL:
            raise RuntimeError(f"Unknown label id encountered: {value}.")
        decoded.append(ID_TO_LABEL[int(value)])
    return decoded


def validate_labels(values: Sequence[int] | np.ndarray | pd.Series) -> None:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise RuntimeError("Label array must be 1-dimensional.")
    if arr.dtype.kind not in {"i", "u"}:
        raise RuntimeError(f"Labels must be integers. Got dtype={arr.dtype}.")
    unique = set(int(val) for val in np.unique(arr))
    invalid = unique.difference(LABEL_IDS)
    if invalid:
        raise RuntimeError(f"Labels contain invalid ids: {sorted(invalid)}.")


def predict_class_ids(scores: Sequence[Sequence[float]] | np.ndarray) -> list[int]:
    arr = np.asarray(scores)
    if arr.ndim != 2:
        raise RuntimeError("Scores must be a 2D array of shape (n_samples, n_classes).")
    preds = np.argmax(arr, axis=1).astype(int)
    validate_labels(preds)
    return preds.tolist()


def label_counts(values: Sequence[int] | np.ndarray | pd.Series) -> dict[str, int]:
    arr = np.asarray(values).astype(int)
    validate_labels(arr)
    counts = Counter(int(val) for val in arr)
    return {label: int(counts.get(idx, 0)) for idx, label in enumerate(LABELS)}
