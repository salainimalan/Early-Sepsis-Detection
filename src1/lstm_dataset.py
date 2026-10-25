"""Patient-level temporal-window construction for the upgraded LSTM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class PatientRecord:
    """One loaded patient recording."""

    patient_id: str
    frame: pd.DataFrame
    has_positive_label: int


@dataclass
class WindowedDataset:
    """Window tensors and metadata for one split."""

    X: np.ndarray
    y: np.ndarray
    patient_ids: np.ndarray
    end_hours: np.ndarray


def discover_patient_files(data_dir: Path) -> list[Path]:
    """Return deterministically ordered PSV files."""
    return sorted(Path(data_dir).glob("*.psv"))


def read_patient(path: Path) -> PatientRecord:
    """Read one patient file and normalize its label column."""
    frame = pd.read_csv(path, sep="|")
    if "SepsisLabel" not in frame.columns:
        raise ValueError(f"Missing SepsisLabel in {path}")
    frame["SepsisLabel"] = (
        pd.to_numeric(frame["SepsisLabel"], errors="coerce")
        .fillna(0)
        .astype(np.int8)
    )
    frame = frame.reset_index(drop=True)
    return PatientRecord(
        patient_id=path.stem,
        frame=frame,
        has_positive_label=int(frame["SepsisLabel"].eq(1).any()),
    )


def load_patients(paths: Iterable[Path]) -> list[PatientRecord]:
    """Load patients one at a time while preserving deterministic order."""
    return [read_patient(path) for path in sorted(paths)]


def split_patient_records(
    patients: Sequence[PatientRecord],
    test_size: float,
    validation_size: float,
    random_state: int,
) -> tuple[list[PatientRecord], list[PatientRecord], list[PatientRecord]]:
    """Create stratified train/validation/test patient splits."""
    if not patients:
        raise ValueError("At least one patient is required")
    indices = np.arange(len(patients))
    labels = np.asarray([patient.has_positive_label for patient in patients])
    train_val, test = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    train_val_labels = labels[train_val]
    relative_validation_size = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_val,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=train_val_labels,
    )

    def select(selected: np.ndarray) -> list[PatientRecord]:
        return [patients[int(index)] for index in selected]

    return select(train), select(validation), select(test)


def build_windows(
    patients: Sequence[PatientRecord],
    feature_columns: Sequence[str],
    window_hours: int,
    prediction_horizon_hours: int,
) -> WindowedDataset:
    """Build fixed-length windows without crossing patient boundaries."""
    if window_hours < 1 or prediction_horizon_hours < 1:
        raise ValueError("Window and prediction horizon must be positive")

    windows: list[np.ndarray] = []
    labels: list[int] = []
    patient_ids: list[str] = []
    end_hours: list[int] = []

    for patient in patients:
        frame = patient.frame
        values = frame.reindex(columns=feature_columns).apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float32)
        target = frame["SepsisLabel"].to_numpy(dtype=np.int8)
        # The target is located after the final row in the input window.
        final_end = len(frame) - prediction_horizon_hours - 1
        for end in range(window_hours - 1, final_end + 1):
            target_index = end + prediction_horizon_hours
            windows.append(values[end - window_hours + 1 : end + 1])
            labels.append(int(target[target_index]))
            patient_ids.append(patient.patient_id)
            end_hours.append(end)

    if not windows:
        raise ValueError("No valid windows were created")
    return WindowedDataset(
        X=np.stack(windows).astype(np.float32),
        y=np.asarray(labels, dtype=np.int8),
        patient_ids=np.asarray(patient_ids),
        end_hours=np.asarray(end_hours, dtype=np.int32),
    )

