"""Run a forward/backward pass without any research data."""

import torch

from tatyana_v4 import ModelConfig, SurrogateLoss, TatyanaV4


def main() -> None:
    config = ModelConfig()
    model = TatyanaV4(config)
    scalars = torch.randn(2, config.n_scalars)
    geometry = torch.randn(2, config.geometry_channels, 16, 16)
    outputs, trajectory = model(scalars, geometry, return_trajectory=True)
    targets = {
        "phi_power": torch.rand(2, config.output_height, config.output_width),
        "electron_heat_flux": torch.randn(2, config.output_height, config.output_width),
        "ion_heat_flux": torch.randn(2, config.output_height, config.output_width),
    }
    loss = SurrogateLoss()(outputs, targets)["total"]
    loss.backward()
    print(f"output shape: {tuple(outputs['phi_log_power'].shape)}")
    print(f"thought states: {len(trajectory)}")
    print(f"finite loss: {bool(torch.isfinite(loss))}")


if __name__ == "__main__":
    main()
