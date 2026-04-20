"""
Osteoporosis opportunistic screening — package root.

Use `BinaryScreeningModel` + `ModelConfig` for inference in your own code;
use `oste_screening.cli:main` or the `oste-screening-train` console script to train.
"""

from oste_screening.config import RunConfig, cfg
from oste_screening.model import BinaryScreeningModel, ModelConfig

__all__ = [
    "BinaryScreeningModel",
    "ModelConfig",
    "RunConfig",
    "cfg",
    "__version__",
]

__version__ = "0.1.0"
