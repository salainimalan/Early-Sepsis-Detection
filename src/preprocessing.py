"""Patient-local loading, inspection, and leakage-safe window preparation."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


def find_patient_files(data_dir: Path) -> List[Path]:
	"""Return deterministically ordered Training Set A patient files."""
	return sorted(Path(data_dir).glob("*.psv"))


def read_patient_file(path: Path) -> pd.DataFrame:
	"""Read one PhysioNet patient file without combining patients in memory."""
	frame = pd.read_csv(path, sep="|")
	if "SepsisLabel" not in frame.columns:
		raise ValueError(f"{path.name} does not contain SepsisLabel")
	frame["SepsisLabel"] = pd.to_numeric(frame["SepsisLabel"], errors="coerce").fillna(0).astype(int)
	frame = frame.reset_index(drop=True)
	frame["patient_id"] = path.stem
	frame["hour"] = frame.index
	return frame


def inspect_patient_files(paths: Iterable[Path]) -> Dict[str, object]:
	"""Collect lightweight, patient-wise inspection statistics."""
	paths = list(paths)
	label_counts: Dict[int, int] = {0: 0, 1: 0}
	missing_counts: Dict[str, int] = {}
	columns: List[str] = []
	row_count = 0
	for path in paths:
		frame = read_patient_file(path)
		if not columns:
			columns = list(frame.columns[:-2])
		row_count += len(frame)
		counts = frame["SepsisLabel"].value_counts().to_dict()
		for label, count in counts.items():
			label_counts[int(label)] = label_counts.get(int(label), 0) + int(count)
		for column, count in frame.isna().sum().items():
			missing_counts[column] = missing_counts.get(column, 0) + int(count)
	missing = pd.DataFrame(
		{"missing_values": missing_counts, "missing_rate": {k: v / row_count for k, v in missing_counts.items()}}
	).sort_values("missing_rate", ascending=False)
	return {
		"patient_count": len(paths),
		"row_count": row_count,
		"columns": columns,
		"label_distribution": pd.Series(label_counts, name="count"),
		"missing_statistics": missing,
	}


def select_patient_window(
	frame: pd.DataFrame,
	window_hours: int,
) -> Optional[Tuple[pd.DataFrame, str, Optional[int]]]:
	"""Select a fixed-length pre-onset or non-septic window from one patient.

	Positive windows are the ``window_hours`` rows immediately before the first
	septic row. Negative windows use the first ``window_hours`` rows from a
	patient whose label is always zero. This deterministic choice avoids overlap
	and does not use observations after the prediction window.
	"""
	if window_hours < 1:
		raise ValueError("window_hours must be positive")
	onset_positions = frame.index[frame["SepsisLabel"].eq(1)].tolist()
	if onset_positions:
		onset = onset_positions[0]
		start = onset - window_hours
		if start < 0:
			return None
		window = frame.iloc[start:onset].copy()
		return window, "positive", int(onset)
	if len(frame) < window_hours:
		return None
	return frame.iloc[:window_hours].copy(), "negative", None


def prepare_window_values(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
	"""Forward-fill only from earlier rows in the selected patient window."""
	available = [column for column in feature_columns if column in frame.columns]
	values = frame[available].apply(pd.to_numeric, errors="coerce")
	return values.ffill()
