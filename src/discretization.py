"""Configurable, reproducible clinical discretization for symbolic sequences."""

from typing import Dict, Mapping, Optional

import pandas as pd


# Thresholds are deliberately explicit and versionable. Values are in the
# units used by the PhysioNet/CinC 2019 Training Set A files.
DEFAULT_THRESHOLDS: Dict[str, Dict[str, Optional[float]]] = {
	"HR": {"low": 60, "high": 100},
	"O2Sat": {"low": 92, "high": 100},
	"Temp": {"low": 36, "high": 38},
	"SBP": {"low": 90, "high": 140},
	"MAP": {"low": 65, "high": 100},
	"Resp": {"low": 12, "high": 20},
	"FiO2": {"low": 0.21, "high": 0.50},
	"pH": {"low": 7.35, "high": 7.45},
	"Lactate": {"low": 2, "high": 4},
	"WBC": {"low": 4, "high": 12},
	"Creatinine": {"low": None, "high": 1.3},
}


def discretize_value(value: float, thresholds: Mapping[str, Optional[float]]) -> Optional[str]:
	"""Map a numeric value to LOW, NORMAL, or HIGH; preserve missingness."""
	if pd.isna(value):
		return None
	low = thresholds.get("low")
	high = thresholds.get("high")
	if low is not None and value < low:
		return "LOW"
	if high is not None and value > high:
		return "HIGH"
	return "NORMAL"


def row_to_event(row: pd.Series, thresholds: Mapping[str, Mapping[str, Optional[float]]]) -> list[str]:
	"""Create one time-ordered itemset, omitting unavailable measurements."""
	event = []
	for feature, bounds in thresholds.items():
		state = discretize_value(row.get(feature), bounds)
		if state is not None:
			event.append(f"{feature}_{state}")
	return event


def frame_to_sequence(frame: pd.DataFrame, thresholds: Mapping[str, Mapping[str, Optional[float]]]) -> list[list[str]]:
	"""Convert hourly rows to ordered PrefixSpan itemsets."""
	return [row_to_event(row, thresholds) for _, row in frame.iterrows()]
