import unittest

import torch

from tatyana_v4 import ModelConfig, SurrogateLoss, TatyanaV4, reconstruct_physical_outputs
from tatyana_v4.model import SpectralConv2d


def small_config(**overrides):
    values = dict(
        condition_dim=32,
        latent_channels=8,
        latent_height=8,
        latent_width=8,
        spectral_modes_height=3,
        spectral_modes_width=3,
        thought_steps=2,
        output_height=31,
        output_width=12,
        dropout=0.0,
    )
    values.update(overrides)
    return ModelConfig(**values)


class ModelTests(unittest.TestCase):
    def test_forward_shapes_and_finite_gradients(self):
        torch.manual_seed(7)
        config = small_config()
        model = TatyanaV4(config)
        scalars = torch.randn(2, config.n_scalars)
        geometry = torch.randn(2, config.geometry_channels, 16, 16)
        outputs, trajectory = model(scalars, geometry, return_trajectory=True)

        expected = (2, config.output_height, config.output_width)
        self.assertEqual(
            set(outputs),
            {
                "phi_log_power",
                "electron_heat_log_abs",
                "electron_heat_sign_logit",
                "ion_heat_log_abs",
                "ion_heat_sign_logit",
            },
        )
        self.assertTrue(all(value.shape == expected for value in outputs.values()))
        self.assertEqual(len(trajectory), config.thought_steps + 1)
        self.assertTrue(all(torch.isfinite(value).all() for value in outputs.values()))

        physical = reconstruct_physical_outputs(outputs)
        loss = sum(value.square().mean() for value in physical.values())
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_optional_kx_reflection_is_exact(self):
        config = small_config(enforce_kx_reflection=True)
        model = TatyanaV4(config).eval()
        outputs, _ = model(
            torch.randn(1, config.n_scalars),
            torch.randn(1, config.geometry_channels, 16, 16),
        )
        phi = outputs["phi_log_power"]
        torch.testing.assert_close(phi, phi.flip(-2))

    def test_spectral_operator_mixes_channels_and_both_kx_sides(self):
        layer = SpectralConv2d(channels=4, modes_height=2, modes_width=2)
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        y = layer(x)
        self.assertEqual(y.shape, x.shape)
        y.square().mean().backward()
        self.assertIsNotNone(layer.weight_positive.grad)
        self.assertIsNotNone(layer.weight_negative.grad)
        self.assertGreater(torch.count_nonzero(layer.weight_positive.grad).item(), 0)
        self.assertGreater(torch.count_nonzero(layer.weight_negative.grad).item(), 0)

    def test_training_loss_is_finite(self):
        config = small_config()
        model = TatyanaV4(config)
        outputs, _ = model(
            torch.randn(2, config.n_scalars),
            torch.randn(2, config.geometry_channels, 16, 16),
        )
        targets = {
            "phi_power": torch.rand(2, config.output_height, config.output_width),
            "electron_heat_flux": torch.randn(2, config.output_height, config.output_width),
            "ion_heat_flux": torch.randn(2, config.output_height, config.output_width),
        }
        losses = SurrogateLoss()(outputs, targets)
        self.assertTrue(torch.isfinite(losses["total"]))


if __name__ == "__main__":
    unittest.main()
