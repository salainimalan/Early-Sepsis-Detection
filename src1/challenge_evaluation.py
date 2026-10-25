"""PhysioNet/CinC 2019-style utility evaluation.

The Challenge evaluates binary predictions at every patient hour. This module
expects one row per patient-hour with ``patient_id``, ``hour``,
``SepsisLabel``, and ``prediction`` columns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _linear(value: float, left: float, right: float, left_score: float, right_score: float) -> float:
    if right == left:
        return left_score
    fraction = (value - left) / (right - left)
    return left_score + fraction * (right_score - left_score)


def _patient_utility(patient: pd.DataFrame) -> float:
    """Calculate utility for one patient's hourly predictions."""
    labels = patient["SepsisLabel"].astype(int).to_numpy()
    predictions = patient["prediction"].astype(int).to_numpy()
    hours = patient["hour"].astype(float).to_numpy()
    septic_indices = [index for index, label in enumerate(labels) if label == 1]

    if not septic_indices:
        # Non-septic false alarms are penalized; true negatives are neutral.
        return float((predictions == 1).sum() * -0.05)

    onset_hour = hours[septic_indices[0]]
    total = 0.0
    for hour, prediction in zip(hours, predictions):
        if prediction == 1:
            relative_hour = hour - onset_hour
            if relative_hour < -12:
                total -= 0.05
            elif relative_hour <= -6:
                total += _linear(relative_hour, -12, -6, 0.0, 1.0)
            elif relative_hour <= 3:
                total += _linear(relative_hour, -6, 3, 1.0, 0.0)
            else:
                total -= 2.0
        elif hour >= onset_hour:
            total -= 2.0
    return total


def normalized_utility(predictions: pd.DataFrame) -> float:
    """Return an approximate normalized Challenge utility in the [0, 1] range.

    The input must contain one row per patient-hour and the columns
    ``patient_id``, ``hour``, ``SepsisLabel``, and ``prediction``. For official
    submissions, validate results with the organizers' released scorer too.
    """
    required = {"patient_id", "hour", "SepsisLabel", "prediction"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing utility columns: {sorted(missing)}")

    work = predictions.sort_values(["patient_id", "hour"])
    score = sum(_patient_utility(group) for _, group in work.groupby("patient_id"))
    # The no-prediction score is zero under this implementation. A perfect
    # score is patient-count dependent, so scale by the number of septic cases.
    septic_patients = work.groupby("patient_id")["SepsisLabel"].max().sum()
    if septic_patients == 0:
        return 0.0
    return float(max(0.0, min(1.0, score / float(septic_patients))))


def evaluate_prediction_file(input_path: Path, output_path: Path) -> float:
    """Evaluate a prediction CSV and save the resulting utility score."""
    predictions = pd.read_csv(input_path)
    score = normalized_utility(predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"normalized_utility": score}]).to_csv(output_path, index=False)
    return score

