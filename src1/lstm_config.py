"""Configuration for the upgraded patient-level LSTM pipeline.

This module contains configuration only. It does not read data, train models,
or create output files. Keeping these values in one place makes experiments
reproducible and prevents train/test settings from being scattered across
notebooks or scripts.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "training_setA"
OUTPUT_ROOT = PROJECT_ROOT / "output1"


@dataclass(frozen=True)
class DataConfig:
    """Patient-window and feature settings."""

    data_dir: Path = DATA_ROOT
    window_hours: int = 24
    prediction_horizon_hours: int = 1
    test_size: float = 0.20
    validation_size: float = 0.15
    random_state: int = 42

    # Sequential measurements passed to the recurrent branch.
    sequence_features: Tuple[str, ...] = (
        "HR",
        "O2Sat",
        "Temp",
        "SBP",
        "MAP",
        "DBP",
        "Resp",
        "FiO2",
        "pH",
        "Lactate",
        "WBC",
        "Creatinine",
    )

    # Static/window summary features passed to the dense branch.
    summary_features: Tuple[str, ...] = (
        "Age",
        "Gender",
        "HospAdmTime",
        "ICULOS",
    )


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings for the two-branch Bidirectional LSTM."""

    lstm_units: Tuple[int, ...] = (64, 32)
    dense_units: int = 32
    dropout: float = 0.30
    recurrent_dropout: float = 0.10
    l2_strength: float = 1e-4


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization, imbalance, and checkpoint settings."""

    batch_size: int = 64
    max_epochs: int = 50
    learning_rate: float = 1e-3
    positive_class_weight: float = 8.0
    focal_loss_gamma: float = 2.0
    use_focal_loss: bool = False
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 3
    monitor: str = "val_pr_auc"


@dataclass(frozen=True)
class OutputConfig:
    """Locations for new-model artifacts."""

    output_dir: Path = OUTPUT_ROOT
    prepared_data_dir: Path = OUTPUT_ROOT / "prepared_data"
    checkpoints_dir: Path = OUTPUT_ROOT / "checkpoints"
    metrics_dir: Path = OUTPUT_ROOT / "metrics"
    predictions_dir: Path = OUTPUT_ROOT / "predictions"
    figures_dir: Path = OUTPUT_ROOT / "figures"

    def create_directories(self) -> None:
        """Create only directories belonging to the new model."""
        for directory in (
            self.output_dir,
            self.prepared_data_dir,
            self.checkpoints_dir,
            self.metrics_dir,
            self.predictions_dir,
            self.figures_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class LSTMConfig:
    """Complete configuration object used by the new pipeline."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


DEFAULT_CONFIG = LSTMConfig()

