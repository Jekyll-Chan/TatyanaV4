"""Iterative conditioned spectral operator for gyrokinetic surrogate modelling.

This module contains no data loader, normalization constants, trained weights, or
database-derived values. Inputs are expected to be normalized by an authorized
downstream data adapter.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ModelConfig


class ParameterGeometryEncoder(nn.Module):
    """Fuse scalar plasma controls and a compact geometry representation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.condition_dim
        self.scalar_encoder = nn.Sequential(
            nn.Linear(config.n_scalars, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Linear(d, d),
            nn.SiLU(),
        )
        self.geometry_encoder = nn.Sequential(
            nn.Conv2d(config.geometry_channels, 24, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, d),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * d, 2 * d),
            nn.SiLU(),
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
        )

    def forward(self, scalars: Tensor, geometry: Tensor) -> Tensor:
        if scalars.ndim != 2:
            raise ValueError(f"scalars must have shape [B, P], got {tuple(scalars.shape)}")
        if geometry.ndim != 4:
            raise ValueError(f"geometry must have shape [B, C, H, W], got {tuple(geometry.shape)}")
        if scalars.shape[0] != geometry.shape[0]:
            raise ValueError("scalars and geometry must have the same batch size")
        return self.fusion(
            torch.cat(
                [self.scalar_encoder(scalars), self.geometry_encoder(geometry)],
                dim=-1,
            )
        )


class SpectralConv2d(nn.Module):
    """Two-sided Fourier operator with explicit input-to-output channel mixing.

    The first spatial FFT axis has positive and negative low-frequency modes.
    The second axis uses ``rfft2`` and is therefore one-sided. Separate complex
    weights are learned for the two sides of the first axis.
    """

    def __init__(self, channels: int, modes_height: int, modes_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.modes_height = modes_height
        self.modes_width = modes_width
        scale = channels**-0.5
        shape = (channels, channels, modes_height, modes_width)
        self.weight_positive = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_negative = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def _mix(coefficients: Tensor, weights: Tensor) -> Tensor:
        return torch.einsum("bihw,iohw->bohw", coefficients, weights)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"spectral input must have shape [B, C, H, W], got {tuple(x.shape)}")
        _, _, height, width = x.shape
        modes_h = min(self.modes_height, height // 2)
        modes_w = min(self.modes_width, width // 2 + 1)
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :modes_h, :modes_w] = self._mix(
            x_ft[:, :, :modes_h, :modes_w],
            self.weight_positive[:, :, :modes_h, :modes_w],
        )
        out_ft[:, :, -modes_h:, :modes_w] = self._mix(
            x_ft[:, :, -modes_h:, :modes_w],
            self.weight_negative[:, :, :modes_h, :modes_w],
        )
        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")


class ConditionedThoughtBlock(nn.Module):
    """Shared spectral update with FiLM conditioning and a learned step gate.

    FiLM is used deliberately here. Cross-attention to a single condition token
    has a one-element softmax and cannot express query-dependent attention.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        c = config.latent_channels
        self.norm = nn.GroupNorm(1, c)
        self.spectral = SpectralConv2d(
            c,
            config.spectral_modes_height,
            config.spectral_modes_width,
        )
        self.local = nn.Conv2d(c, c, 1)
        self.film = nn.Linear(config.condition_dim, 2 * c)
        self.gate = nn.Linear(config.condition_dim, c)
        self.dropout = nn.Dropout2d(config.dropout)

    def forward(self, latent: Tensor, condition: Tensor) -> Tensor:
        normalized = self.norm(latent)
        scale, shift = self.film(condition).chunk(2, dim=-1)
        conditioned = normalized * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        update = F.silu(self.spectral(conditioned) + self.local(conditioned))
        gate = torch.sigmoid(self.gate(condition))[:, :, None, None]
        return latent + gate * self.dropout(update)


class LatentInitializer(nn.Module):
    """Initialize a spatial latent with a bounded, learnable global scale."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.shape = (
            config.latent_channels,
            config.latent_height,
            config.latent_width,
        )
        n_values = self.shape[0] * self.shape[1] * self.shape[2]
        self.direction = nn.Linear(config.condition_dim, n_values)
        self.log_scale = nn.Linear(config.condition_dim, 1)
        nn.init.zeros_(self.log_scale.weight)
        nn.init.zeros_(self.log_scale.bias)

    def forward(self, condition: Tensor) -> Tensor:
        direction = self.direction(condition).reshape(condition.shape[0], *self.shape)
        scale = torch.exp(2.0 * torch.tanh(self.log_scale(condition)))
        return direction * scale[:, :, None, None]


class TargetBranch(nn.Module):
    def __init__(self, channels: int, outputs: int) -> None:
        super().__init__()
        hidden = max(24, channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, outputs, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SpectralDecoder(nn.Module):
    """Decode separate potential and heat-flux branches in log space."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        c = config.latent_channels
        self.pre = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.SiLU())
        self.phi = TargetBranch(c, 1)
        self.electron_heat = TargetBranch(c, 2)
        self.ion_heat = TargetBranch(c, 2)

    def forward(self, latent: Tensor) -> Dict[str, Tensor]:
        x = F.interpolate(
            self.pre(latent),
            size=(self.config.output_height, self.config.output_width),
            mode="bilinear",
            align_corners=False,
        )
        phi_log_power = self.phi(x).squeeze(1)
        if self.config.enforce_kx_reflection:
            phi_log_power = 0.5 * (phi_log_power + phi_log_power.flip(-2))
        electron_log_abs, electron_sign_logit = self.electron_heat(x).chunk(2, dim=1)
        ion_log_abs, ion_sign_logit = self.ion_heat(x).chunk(2, dim=1)
        return {
            "phi_log_power": phi_log_power,
            "electron_heat_log_abs": electron_log_abs.squeeze(1),
            "electron_heat_sign_logit": electron_sign_logit.squeeze(1),
            "ion_heat_log_abs": ion_log_abs.squeeze(1),
            "ion_heat_sign_logit": ion_sign_logit.squeeze(1),
        }


class TatyanaV4(nn.Module):
    """Conditioned, shared-weight iterative surrogate.

    The repeated thought block acts as a learned relaxation process. It is an
    architectural analogy, not a claim that the latent trajectory is a physical
    time integration of the gyrokinetic equation.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = ParameterGeometryEncoder(self.config)
        self.initializer = LatentInitializer(self.config)
        self.thought_block = ConditionedThoughtBlock(self.config)
        self.final_norm = nn.GroupNorm(1, self.config.latent_channels)
        self.decoder = SpectralDecoder(self.config)

    def forward(
        self,
        scalars: Tensor,
        geometry: Tensor,
        *,
        thought_steps: int | None = None,
        return_trajectory: bool = False,
    ) -> Tuple[Dict[str, Tensor], Tuple[Tensor, ...]]:
        steps = self.config.thought_steps if thought_steps is None else thought_steps
        if steps <= 0:
            raise ValueError("thought_steps must be positive")
        condition = self.encoder(scalars, geometry)
        latent = self.initializer(condition)
        trajectory = [latent] if return_trajectory else []
        for _ in range(steps):
            latent = self.thought_block(latent, condition)
            if return_trajectory:
                trajectory.append(latent)
        outputs = self.decoder(self.final_norm(latent))
        return outputs, tuple(trajectory)


def reconstruct_physical_outputs(
    outputs: Dict[str, Tensor],
    *,
    log_value_clip: float = 30.0,
) -> Dict[str, Tensor]:
    """Convert stable log-space outputs to non-negative power and signed flux."""

    phi = torch.exp(outputs["phi_log_power"].clamp(-log_value_clip, log_value_clip))

    def signed_field(log_abs: Tensor, sign_logit: Tensor) -> Tensor:
        magnitude = torch.exp(log_abs.clamp(-log_value_clip, log_value_clip))
        return torch.tanh(sign_logit) * magnitude

    return {
        "phi_power": phi,
        "electron_heat_flux": signed_field(
            outputs["electron_heat_log_abs"],
            outputs["electron_heat_sign_logit"],
        ),
        "ion_heat_flux": signed_field(
            outputs["ion_heat_log_abs"],
            outputs["ion_heat_sign_logit"],
        ),
    }
