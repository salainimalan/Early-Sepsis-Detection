"""Evaluation and threshold-selection utilities for the upgraded LSTM."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_probabilities(model, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return one probability per window."""
    probabilities = np.asarray(model.predict(inputs, verbose=0)).reshape(-1)
    return np.clip(probabilities, 0.0, 1.0)


def select_f1_threshold(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    thresholds: Iterable[float] | None = None,
) -> tuple[float, float]:
    """Select the validation threshold with the highest F1 score."""
    labels = np.asarray(list(y_true), dtype=np.int8)
    scores = np.asarray(list(probabilities), dtype=float)
    candidates = np.asarray(
        list(thresholds) if thresholds is not None else np.linspace(0.05, 0.95, 91)
    )
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        predictions = (scores >= threshold).astype(np.int8)
        score = f1_score(labels, predictions, zero_division=0)
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(score)
    return best_threshold, best_f1


def calculate_metrics(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Calculate threshold-independent and threshold-dependent metrics."""
    labels = np.asarray(list(y_true), dtype=np.int8)
    scores = np.asarray(list(probabilities), dtype=float)
    predictions = (scores >= threshold).astype(np.int8)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    try:
        auroc = float(roc_auc_score(labels, scores))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(labels, scores))
    except ValueError:
        auprc = float("nan")
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auroc": auroc,
        "auprc": auprc,
        "true_negatives": int(matrix[0, 0]),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "true_positives": int(matrix[1, 1]),
        "positive_prevalence": float(labels.mean()),
    }


def evaluate_and_save(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    threshold: float,
    output_path: Path,
    split: str,
) -> dict[str, float | int | str]:
    """Calculate metrics and save one split's results as a one-row CSV."""
    metrics = calculate_metrics(y_true, probabilities, threshold)
    row = {"split": split, **metrics}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output_path, index=False)
    return row

