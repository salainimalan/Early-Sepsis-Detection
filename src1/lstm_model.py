"""Two-branch Bidirectional LSTM model for early sepsis prediction."""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_lstm_model(
    window_hours: int,
    sequence_feature_count: int,
    summary_feature_count: int,
    lstm_units: Sequence[int] = (64, 32),
    dense_units: int = 32,
    dropout: float = 0.30,
    recurrent_dropout: float = 0.10,
    l2_strength: float = 1e-4,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Build and compile the upgraded two-branch LSTM."""
    if len(lstm_units) < 1:
        raise ValueError("lstm_units must contain at least one layer size")
    if window_hours < 1 or sequence_feature_count < 1:
        raise ValueError("Window and sequence feature counts must be positive")

    sequence_input = keras.Input(
        shape=(window_hours, sequence_feature_count),
        name="sequence_input",
    )
    sequence_branch = layers.Masking(mask_value=0.0, name="sequence_masking")(
        sequence_input
    )

    for index, units in enumerate(lstm_units):
        sequence_branch = layers.Bidirectional(
            layers.LSTM(
                units,
                return_sequences=index < len(lstm_units) - 1,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=regularizers.l2(l2_strength),
            ),
            name=f"bidirectional_lstm_{index + 1}",
        )(sequence_branch)

    sequence_branch = layers.LayerNormalization(name="sequence_normalization")(
        sequence_branch
    )

    summary_input = keras.Input(
        shape=(summary_feature_count,),
        name="summary_input",
    )
    summary_branch = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_strength),
        name="summary_dense",
    )(summary_input)
    summary_branch = layers.Dropout(dropout, name="summary_dropout")(summary_branch)

    merged = layers.Concatenate(name="merge_sequence_and_summary")(
        [sequence_branch, summary_branch]
    )
    merged = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_strength),
        name="merged_dense",
    )(merged)
    merged = layers.Dropout(dropout, name="merged_dropout")(merged)
    output = layers.Dense(1, activation="sigmoid", name="sepsis_probability")(merged)

    model = keras.Model(
        inputs={"sequence_input": sequence_input, "summary_input": summary_input},
        outputs=output,
        name="upgraded_two_branch_bidirectional_lstm",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model

