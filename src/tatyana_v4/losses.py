"""Data-agnostic objectives for spectral surrogate training."""

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import reconstruct_physical_outputs


@dataclass(frozen=True)
class LossWeights:
    potential: float = 1.0
    electron_heat: float = 0.5
    ion_heat: float = 0.5
    sign: float = 0.10
    integrated_flux: float = 0.05


class SurrogateLoss(nn.Module):
    """Log-spectral, sign, and integrated-flux objective.

    Targets are raw physical fields. No private normalization statistics are
    embedded in this implementation.
    """

    def __init__(self, weights: LossWeights | None = None, eps: float = 1e-8) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.eps = eps

    def _log_abs(self, value: Tensor) -> Tensor:
        return torch.log(value.abs() + self.eps)

    def _integrated_log_error(self, prediction: Tensor, target: Tensor) -> Tensor:
        pred_flux = prediction.sum(dim=(-2, -1))
        true_flux = target.sum(dim=(-2, -1))
        return F.smooth_l1_loss(self._log_abs(pred_flux), self._log_abs(true_flux))

    def forward(self, outputs: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Dict[str, Tensor]:
        required = {"phi_power", "electron_heat_flux", "ion_heat_flux"}
        missing = required.difference(targets)
        if missing:
            raise KeyError(f"Missing training targets: {sorted(missing)}")

        phi_target = torch.log(targets["phi_power"].clamp_min(0) + self.eps)
        electron_target = self._log_abs(targets["electron_heat_flux"])
        ion_target = self._log_abs(targets["ion_heat_flux"])

        phi = F.smooth_l1_loss(outputs["phi_log_power"], phi_target)
        electron = F.smooth_l1_loss(outputs["electron_heat_log_abs"], electron_target)
        ion = F.smooth_l1_loss(outputs["ion_heat_log_abs"], ion_target)

        electron_sign = (targets["electron_heat_flux"] >= 0).to(outputs["electron_heat_sign_logit"].dtype)
        ion_sign = (targets["ion_heat_flux"] >= 0).to(outputs["ion_heat_sign_logit"].dtype)
        sign = F.binary_cross_entropy_with_logits(
            outputs["electron_heat_sign_logit"], electron_sign
        ) + F.binary_cross_entropy_with_logits(outputs["ion_heat_sign_logit"], ion_sign)

        physical = reconstruct_physical_outputs(outputs)
        integrated = self._integrated_log_error(
            physical["electron_heat_flux"], targets["electron_heat_flux"]
        ) + self._integrated_log_error(
            physical["ion_heat_flux"], targets["ion_heat_flux"]
        )

        total = (
            self.weights.potential * phi
            + self.weights.electron_heat * electron
            + self.weights.ion_heat * ion
            + self.weights.sign * sign
            + self.weights.integrated_flux * integrated
        )
        return {
            "total": total,
            "potential": phi,
            "electron_heat": electron,
            "ion_heat": ion,
            "sign": sign,
            "integrated_flux": integrated,
        }
