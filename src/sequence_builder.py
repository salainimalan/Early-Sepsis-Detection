"""Build patient-level symbolic sequences and metadata."""

from pathlib import Path
from typing import Dict, Mapping, Sequence

import pandas as pd

from .discretization import frame_to_sequence
from .preprocessing import prepare_window_values, read_patient_file, select_patient_window


def build_sequence_record(
	path: Path,
	window_hours: int,
	feature_columns: Sequence[str],
	thresholds: Mapping[str, Mapping[str, float]],
) -> Dict[str, object] | None:
	"""Process one patient and return a serializable sequence record."""
	frame = read_patient_file(path)
	selected = select_patient_window(frame, window_hours)
	if selected is None:
		return None
	window, cohort, onset = selected
	values = prepare_window_values(window, feature_columns)
	sequence = frame_to_sequence(values, thresholds)
	if not any(sequence):
		return None
	return {
		"patient_id": path.stem,
		"cohort": cohort,
		"sepsis_status": int(cohort == "positive"),
		"sequence": sequence,
		"sequence_length": len(sequence),
		"onset_hour": onset,
		"window_start_hour": int(window["hour"].iloc[0]),
		"window_end_hour": int(window["hour"].iloc[-1]),
	}
