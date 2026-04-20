"""
Training / data / IO configuration. Mutable singleton pattern for CLI overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

from oste_screening.model import ModelConfig


def _default_data_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


@dataclass
class RunConfig:
    # Data paths — replace via CLI or env for your site
    IMAGE_ROOT_STAGE1: str = field(
        default_factory=lambda: os.path.join(_default_data_root(), "Stage1", "Preprocessed")
    )
    IMAGE_ROOT_STAGE2: str = field(
        default_factory=lambda: os.path.join(_default_data_root(), "Stage2", "Preprocessed")
    )
    CSV_STAGE1: str = field(
        default_factory=lambda: os.path.join(_default_data_root(), "Stage1", "splits.csv")
    )
    CSV_STAGE2: str = field(
        default_factory=lambda: os.path.join(_default_data_root(), "Stage2", "splits.csv")
    )
    OUTPUT_DIR: str = field(
        default_factory=lambda: os.path.join(_default_data_root(), "..", "outputs", "checkpoints_binaryV10")
    )

    COL_IMAGE_PATH: str = "path_x"
    COL_PATIENT_ID: str = "patient_id"
    COL_LABEL: str = "label_cls"
    COL_TSCORE: str = "t_score"
    COL_AGE: str = "age"
    COL_SEX: str = "sex"
    COL_BMI: str = "bmi"
    COL_SPLIT: str = "split"

    random_seed: int = 42
    val_ratio: float = 0.2
    img_size: int = 512
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 20
    early_stop_patience: int = 30
    max_grad_norm: float = 1.0
    dropout: float = 0.3
    neck_dim: int = 128
    use_neck: bool = True
    use_severe_head: bool = True
    use_tscore_branch: bool = True
    use_tarr: bool = True
    lambda_severe: float = 0.3
    lambda_tscore: float = 0.1
    use_fp_suppression: bool = True
    lambda_fp: float = 0.1
    fp_margin: float = 0.30
    fp_warmup_epochs: int = 5
    tscore_warmup_epochs: int = 5
    tarr_hidden_dim: int | None = None
    tarr_adapter_dim: int = 64
    use_threshold_search: bool = True
    threshold_search_mode: str = "constraint_sens_max_spec"
    threshold_min: float = 0.20
    threshold_max: float = 0.80
    threshold_step: float = 0.01
    target_sensitivity: float = 0.86
    default_screen_threshold: float = 0.5
    experiment_id: str = "default"
    experiment_category: str = "manual"

    gray_mean: tuple[float, ...] = (0.5,)
    gray_std: tuple[float, ...] = (0.25,)

    use_foreground_crop: bool = True
    foreground_margin_ratio: float = 0.03
    intensity_clip_p_low: float = 1.0
    intensity_clip_p_high: float = 99.0
    use_clahe: bool = False

    use_border_masking: bool = True
    border_mask_ratio: float = 0.04
    border_mask_prob: float = 0.5
    use_background_dropout: bool = False
    background_dropout_ratio: float = 0.12
    background_dropout_prob: float = 0.3
    background_dropout_percentile: float = 30.0

    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def __post_init__(self):
        self.refresh_output_dirs()

    def refresh_output_dirs(self):
        self.DIR_LOGS = os.path.join(self.OUTPUT_DIR, "logs")
        self.DIR_CKPTS = os.path.join(self.OUTPUT_DIR, "checkpoints")
        self.DIR_SUMS = os.path.join(self.OUTPUT_DIR, "summaries")

    def to_model_config(self) -> ModelConfig:
        th = self.tarr_hidden_dim
        if th is None:
            th = int(self.neck_dim if self.use_neck else 2048)
        return ModelConfig(
            dropout=float(self.dropout),
            use_neck=bool(self.use_neck),
            neck_dim=int(self.neck_dim),
            use_severe_head=bool(self.use_severe_head),
            use_tscore_branch=bool(self.use_tscore_branch),
            use_tarr=bool(self.use_tarr),
            tarr_hidden_dim=int(th),
            tarr_adapter_dim=int(self.tarr_adapter_dim),
        )


# Default instance used by training script (CLI mutates this).
cfg = RunConfig()
