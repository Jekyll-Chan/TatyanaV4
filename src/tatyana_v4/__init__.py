"""Public, data-free TatyanaV4 model implementation."""

from .config import ModelConfig
from .losses import LossWeights, SurrogateLoss
from .model import TatyanaV4, reconstruct_physical_outputs

__all__ = [
    "LossWeights",
    "ModelConfig",
    "SurrogateLoss",
    "TatyanaV4",
    "reconstruct_physical_outputs",
]

__version__ = "0.4.2"
