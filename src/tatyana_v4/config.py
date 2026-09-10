"""Typed configuration for the public TatyanaV4 architecture."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Architecture dimensions independent of any private dataset statistics."""

    n_scalars: int = 6
    geometry_channels: int = 1
    condition_dim: int = 128
    latent_channels: int = 24
    latent_height: int = 32
    latent_width: int = 32
    spectral_modes_height: int = 8
    spectral_modes_width: int = 8
    thought_steps: int = 6
    dropout: float = 0.10
    output_height: int = 127
    output_width: int = 48
    enforce_kx_reflection: bool = False
    log_value_clip: float = 30.0

    def __post_init__(self) -> None:
        positive = {
            "n_scalars": self.n_scalars,
            "geometry_channels": self.geometry_channels,
            "condition_dim": self.condition_dim,
            "latent_channels": self.latent_channels,
            "latent_height": self.latent_height,
            "latent_width": self.latent_width,
            "spectral_modes_height": self.spectral_modes_height,
            "spectral_modes_width": self.spectral_modes_width,
            "thought_steps": self.thought_steps,
            "output_height": self.output_height,
            "output_width": self.output_width,
            "log_value_clip": self.log_value_clip,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Configuration values must be positive: {invalid}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
