"""Imbalance-aware losses for the sepsis LSTM."""

from __future__ import annotations

import tensorflow as tf


def weighted_binary_cross_entropy(
    positive_weight: float = 8.0,
) -> tf.keras.losses.Loss:
    """Return BCE with an increased penalty for false-negative errors."""
    if positive_weight <= 0:
        raise ValueError("positive_weight must be positive")

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_float = tf.cast(y_true, tf.float32)
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        positive_term = y_true_float * tf.math.log(y_pred_clipped)
        negative_term = (1.0 - y_true_float) * tf.math.log(1.0 - y_pred_clipped)
        weighted = -(positive_weight * positive_term + negative_term)
        return tf.reduce_mean(weighted)

    loss.__name__ = f"weighted_bce_{positive_weight:g}"
    return loss


def focal_binary_cross_entropy(
    positive_weight: float = 8.0,
    gamma: float = 2.0,
) -> tf.keras.losses.Loss:
    """Return weighted focal loss for hard and minority examples."""
    if positive_weight <= 0 or gamma < 0:
        raise ValueError("positive_weight must be positive and gamma non-negative")

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_float = tf.cast(y_true, tf.float32)
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_t = y_true_float * y_pred_clipped + (1.0 - y_true_float) * (
            1.0 - y_pred_clipped
        )
        alpha_t = y_true_float * positive_weight + (1.0 - y_true_float)
        focal_factor = tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(-alpha_t * focal_factor * tf.math.log(p_t))

    loss.__name__ = f"focal_bce_{positive_weight:g}_gamma_{gamma:g}"
    return loss

