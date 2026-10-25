"""Leakage-safe feature engineering for the upgraded LSTM pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass
class FeatureStatistics:
    """Training-only statistics reused unchanged on validation and test data."""

    medians: pd.Series
    means: pd.Series
    standard_deviations: pd.Series


def _numeric_series(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return requested columns as numeric values, creating missing columns."""
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def add_missingness_indicators(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Add one binary feature per measurement indicating original missingness."""
    result = frame.copy()
    for column in feature_columns:
        if column not in result.columns:
            result[column] = np.nan
        result[f"{column}__missing"] = result[column].isna().astype(np.float32)
    return result


def add_temporal_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Add first differences and short rolling means without using future rows."""
    result = _numeric_series(frame, feature_columns)
    for column in feature_columns:
        result[f"{column}__delta"] = result[column].diff().fillna(0.0)
        result[f"{column}__rolling_mean"] = (
            result[column].rolling(window=3, min_periods=1).mean().fillna(0.0)
        )
    return result


def build_feature_frame(
    frame: pd.DataFrame,
    sequence_features: Sequence[str],
    summary_features: Sequence[str] = (),
) -> pd.DataFrame:
    """Build continuous, missingness, trend, and summary features for one patient."""
    base_columns = tuple(dict.fromkeys((*sequence_features, *summary_features)))
    result = _numeric_series(frame, base_columns)
    result = add_missingness_indicators(result, base_columns)
    result = add_temporal_features(result, sequence_features)
    return result


def fit_feature_statistics(
    frames: Iterable[pd.DataFrame],
    feature_columns: Sequence[str],
) -> FeatureStatistics:
    """Fit imputation and scaling statistics using training frames only."""
    combined = pd.concat(
        [_numeric_series(frame, feature_columns) for frame in frames],
        axis=0,
        ignore_index=True,
    )
    medians = combined.median().fillna(0.0)
    filled = combined.fillna(medians)
    means = filled.mean().fillna(0.0)
    standard_deviations = filled.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    return FeatureStatistics(medians, means, standard_deviations)


def transform_feature_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    statistics: FeatureStatistics,
) -> pd.DataFrame:
    """Impute and standardize a feature frame using pre-fitted statistics."""
    result = _numeric_series(frame, feature_columns)
    result = result.fillna(statistics.medians)
    result = (result - statistics.means) / statistics.standard_deviations
    return result.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)


def feature_column_names(
    sequence_features: Sequence[str],
    summary_features: Sequence[str] = (),
) -> list[str]:
    """Return the deterministic feature order used by downstream model code."""
    names: list[str] = []
    for column in (*sequence_features, *summary_features):
        names.extend((column, f"{column}__missing"))
    for column in sequence_features:
        names.extend((f"{column}__delta", f"{column}__rolling_mean"))
    return list(dict.fromkeys(names))

