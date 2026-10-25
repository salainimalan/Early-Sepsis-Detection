"""End-to-end runner for the upgraded LSTM experiment."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from .lstm_config import DEFAULT_CONFIG, LSTMConfig
    from .lstm_dataset import (
        WindowedDataset,
        build_windows,
        discover_patient_files,
        load_patients,
        split_patient_records,
    )
    from .lstm_evaluation import evaluate_and_save, predict_probabilities, select_f1_threshold
    from .lstm_features import FeatureStatistics, fit_feature_statistics
    from .lstm_model import build_lstm_model
    from .lstm_training import train_model
except ImportError:
    from lstm_config import DEFAULT_CONFIG, LSTMConfig
    from lstm_dataset import WindowedDataset, build_windows, discover_patient_files, load_patients, split_patient_records
    from lstm_evaluation import evaluate_and_save, predict_probabilities, select_f1_threshold
    from lstm_features import FeatureStatistics, fit_feature_statistics
    from lstm_model import build_lstm_model
    from lstm_training import train_model


def set_seed(seed: int) -> None:
    """Set reproducible seeds for Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def fit_sequence_statistics(
    dataset: WindowedDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit median, mean, and standard deviation on training windows only."""
    values = dataset.X.reshape(-1, dataset.X.shape[-1])
    medians = np.nanmedian(values, axis=0)
    medians = np.nan_to_num(medians, nan=0.0)
    filled = np.where(np.isnan(values), medians, values)
    means = np.nanmean(filled, axis=0)
    stds = np.nanstd(filled, axis=0)
    return medians, np.nan_to_num(means), np.where(stds == 0, 1.0, stds)


def transform_windows(
    dataset: WindowedDataset,
    medians: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize sequences and create summary features from each window."""
    raw = dataset.X
    missing = np.isnan(raw).astype(np.float32)
    filled = np.where(np.isnan(raw), medians.reshape(1, 1, -1), raw)
    sequence = ((filled - means.reshape(1, 1, -1)) / stds.reshape(1, 1, -1)).astype(np.float32)
    summary = np.concatenate(
        [
            sequence.mean(axis=1),
            sequence.std(axis=1),
            sequence[:, -1, :],
            missing.mean(axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    return sequence, summary


def save_split_metadata(config: LSTMConfig, splits: dict[str, WindowedDataset]) -> None:
    """Save split sizes and class prevalence for reproducibility."""
    rows = []
    for name, dataset in splits.items():
        rows.append(
            {
                "split": name,
                "windows": len(dataset.y),
                "positive_windows": int(dataset.y.sum()),
                "negative_windows": int((dataset.y == 0).sum()),
                "positive_prevalence": float(dataset.y.mean()),
            }
        )
    path = config.output.metrics_dir / "split_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_pipeline(config: LSTMConfig = DEFAULT_CONFIG) -> dict[str, dict]:
    """Run one complete new-model experiment."""
    set_seed(config.data.random_state)
    config.output.create_directories()

    paths = discover_patient_files(config.data.data_dir)
    patients = load_patients(paths)
    train_patients, validation_patients, test_patients = split_patient_records(
        patients,
        test_size=config.data.test_size,
        validation_size=config.data.validation_size,
        random_state=config.data.random_state,
    )

    feature_columns = config.data.sequence_features
    train_data = build_windows(train_patients, feature_columns, config.data.window_hours, config.data.prediction_horizon_hours)
    validation_data = build_windows(validation_patients, feature_columns, config.data.window_hours, config.data.prediction_horizon_hours)
    test_data = build_windows(test_patients, feature_columns, config.data.window_hours, config.data.prediction_horizon_hours)
    splits = {"train": train_data, "validation": validation_data, "test": test_data}
    save_split_metadata(config, splits)

    medians, means, stds = fit_sequence_statistics(train_data)
    train_sequence, train_summary = transform_windows(train_data, medians, means, stds)
    validation_sequence, validation_summary = transform_windows(validation_data, medians, means, stds)
    test_sequence, test_summary = transform_windows(test_data, medians, means, stds)

    model = build_lstm_model(
        window_hours=config.data.window_hours,
        sequence_feature_count=train_sequence.shape[-1],
        summary_feature_count=train_summary.shape[-1],
        lstm_units=config.model.lstm_units,
        dense_units=config.model.dense_units,
        dropout=config.model.dropout,
        recurrent_dropout=config.model.recurrent_dropout,
        l2_strength=config.model.l2_strength,
        learning_rate=config.training.learning_rate,
    )
    train_model(
        model,
        {"sequence_input": train_sequence, "summary_input": train_summary},
        train_data.y,
        {"sequence_input": validation_sequence, "summary_input": validation_summary},
        validation_data.y,
        config.training,
        config.output.checkpoints_dir / "best_lstm.keras",
        config.output.metrics_dir / "training_history.csv",
    )

    validation_probabilities = predict_probabilities(model, {"sequence_input": validation_sequence, "summary_input": validation_summary})
    test_probabilities = predict_probabilities(model, {"sequence_input": test_sequence, "summary_input": test_summary})
    threshold, _ = select_f1_threshold(validation_data.y, validation_probabilities)
    validation_metrics = evaluate_and_save(validation_data.y, validation_probabilities, threshold, config.output.metrics_dir / "validation_metrics.csv", "validation")
    test_metrics = evaluate_and_save(test_data.y, test_probabilities, threshold, config.output.metrics_dir / "test_metrics.csv", "test")
    pd.DataFrame({"patient_id": test_data.patient_ids, "hour": test_data.end_hours, "label": test_data.y, "probability": test_probabilities, "prediction": (test_probabilities >= threshold).astype(int)}).to_csv(config.output.predictions_dir / "test_predictions.csv", index=False)
    return {"validation": validation_metrics, "test": test_metrics}


if __name__ == "__main__":
    results = run_pipeline()
    print(results)

