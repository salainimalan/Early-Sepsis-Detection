"""Training utilities for the upgraded LSTM model."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

try:  # Package import, e.g. ``from src1.lstm_training import train_model``.
    from .lstm_config import TrainingConfig
    from .lstm_losses import focal_binary_cross_entropy, weighted_binary_cross_entropy
except ImportError:  # Direct execution, e.g. ``python src1/lstm_training.py``.
    from lstm_config import TrainingConfig
    from lstm_losses import focal_binary_cross_entropy, weighted_binary_cross_entropy


def compile_for_training(
    model: keras.Model,
    config: TrainingConfig,
) -> keras.Model:
    """Compile the model with the configured imbalance-aware loss."""
    if config.use_focal_loss:
        loss = focal_binary_cross_entropy(
            positive_weight=config.positive_class_weight,
            gamma=config.focal_loss_gamma,
        )
    else:
        loss = weighted_binary_cross_entropy(
            positive_weight=config.positive_class_weight,
        )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=loss,
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def train_model(
    model: keras.Model,
    train_inputs: Mapping[str, np.ndarray],
    y_train: np.ndarray,
    validation_inputs: Mapping[str, np.ndarray],
    y_validation: np.ndarray,
    config: TrainingConfig,
    checkpoint_path: Path,
    history_path: Path | None = None,
) -> keras.callbacks.History:
    """Train the model and save the best validation-PR-AUC checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)

    compile_for_training(model, config)
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor=config.monitor,
            mode="max",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=config.monitor,
            mode="max",
            patience=config.reduce_lr_patience,
            factor=0.5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=config.monitor,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_inputs,
        y_train,
        validation_data=(validation_inputs, y_validation),
        epochs=config.max_epochs,
        batch_size=config.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    if history_path is not None:
        pd.DataFrame(history.history).to_csv(history_path, index=False)
    return history


if __name__ == "__main__":
    print(
        "lstm_training.py provides training functions. "
        "Run the end-to-end pipeline to load data and train the model."
    )

