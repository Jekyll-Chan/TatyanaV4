# TatyanaV4

TatyanaV4 is a data-free research implementation of an iterative spectral neural surrogate for nonlinear gyrokinetic turbulence. It maps normalized plasma controls and an encoded magnetic-geometry field to two-dimensional electrostatic-potential power and electron/ion heat-flux spectra.

This public repository contains model and training-objective code only. It deliberately excludes every research database and every artifact derived from those databases.

## Research role and data provenance

The private simulation databases used during internal development were supplied through the **NTU Plasma Theory group** and by **Youngwoo Cho at the Korea Institute of Fusion Energy (KFE)**.

**Tingyi Chen is the surrogate-model developer.** The contribution represented here is the design and implementation of the neural surrogate, the leakage-aware training and evaluation methodology, and the numerical diagnostics. No claim is made to database authorship or ownership.

See [DATA_POLICY.md](DATA_POLICY.md) for the full publication boundary. In particular, this repository contains no database files, fitted normalization statistics, trained weights, checkpoints, logs, metrics, or database-derived figures.

## Scientific objective

Gyrokinetic transport is governed by more than a scalar instability label. A useful surrogate must preserve how drive, magnetic geometry, nonlinear transfer, and scale coupling shape the saturated spectrum. TatyanaV4 therefore treats the prediction target as a field over a discretized perpendicular-wave-number plane rather than reducing the problem to a single integrated heat flux.

The public interface accepts:

- a vector of normalized dimensionless plasma controls;
- a compact tensor representation of magnetic geometry; and
- a configurable output grid for potential power and signed electron/ion heat flux.

The exact private preprocessing transform and simulation campaign are not published. An authorized downstream adapter is responsible for constructing normalized tensors and for computing all fitted statistics from the training partition only.

## Gyrokinetic and saturation perspective

In a local gyrokinetic picture, background gradients inject free energy into unstable fluctuations. Nonlinear \(E\times B\) advection transfers that free energy across perpendicular scales, while dissipation and phase-space mixing remove it. Zonal flows and their shearing of drift-wave eddies can strongly regulate the saturated state. Magnetic shear, safety factor, aspect-ratio effects, and equilibrium geometry modify both the linear eigenstructure and the nonlinear coupling pathways.

A mixing-length estimate such as

\[
D_{\mathrm{ML}} \sim \frac{\gamma}{k_\perp^2}
\]

is a valuable dimensional guide: saturation is expected when a nonlinear decorrelation rate becomes comparable to the effective linear drive. It is not a universal closure. Eigenfunction structure, multiple competing branches, zonal-flow response, geometry, collisions, and the chosen spectral normalization all affect the mapping from a linear growth rate to nonlinear transport.

TatyanaV4 consequently does **not** hard-code a single saturation amplitude. Instead, it uses an iterative, geometry-conditioned spectral operator to learn a nonlinear map while retaining several physical checks:

- spectral outputs are modelled explicitly rather than inferred from one scalar amplitude;
- signed heat flux is separated into log-magnitude and sign channels;
- integrated heat flux can enter the objective as a global consistency term;
- normalization must be fitted on the training partition only; and
- reflection symmetry in \(k_x\) is configurable, because it is valid only when supported by the database's Fourier and averaging conventions.

The repeated latent update is a learned relaxation mechanism. It should not be interpreted as a time integrator for the gyrokinetic equation.

## Architecture

```text
normalized plasma controls ── scalar encoder ─┐
                                             ├─ fused condition
normalized geometry ──────── geometry CNN ───┘
                                                     │
                                                     ▼
                                      bounded latent initialization
                                                     │
                                                     ▼
                         shared conditioned spectral block × T iterations
                                                     │
                          ┌──────────────────────────┼──────────────────────────┐
                          ▼                          ▼                          ▼
                 potential log-power      electron heat branch         ion heat branch
                                           log|Q| + sign logit          log|Q| + sign logit
```

The public v0.4.2 implementation makes three corrections to the earlier experimental code:

1. Fourier coefficients are contracted across input and output channels, so the spectral operator performs genuine channel mixing.
2. Separate complex kernels cover the positive and negative low-frequency modes of the first FFT axis.
3. Condition injection uses FiLM and a learned gate. Cross-attention to one key/value token was removed because its one-element softmax cannot depend on the spatial query.

Log-space amplitude composition is also kept in log space until a numerically bounded reconstruction step. Potential and heat-flux targets use separate decoder branches to reduce destructive coupling between physically different observables.

These changes are structurally tested, but no claim of improved empirical accuracy is made here. A comparison with the previous generation requires authorized retraining on the private databases.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

Run the data-free smoke example:

```bash
python examples/synthetic_smoke.py
```

## Minimal model use

```python
import torch
from tatyana_v4 import ModelConfig, TatyanaV4, reconstruct_physical_outputs

config = ModelConfig()
model = TatyanaV4(config)

scalars = torch.randn(2, config.n_scalars)
geometry = torch.randn(2, config.geometry_channels, 16, 16)
log_outputs, _ = model(scalars, geometry)
physical_outputs = reconstruct_physical_outputs(log_outputs)
```

The random tensors above are only an API demonstration. They are not representative samples and encode no information about the private databases.

## Evaluation principles

Internal model selection should report complementary quantities rather than one flattering score:

- pointwise and per-case errors in log and physical space;
- rank correlation and coefficient of determination for each spectral target;
- integrated electron and ion heat-flux error;
- spectral-shape diagnostics across \(k_x\) and \(k_y\);
- robustness across grouped splits and random seeds; and
- ablations in which data partition and optimization budget are held fixed.

High rank correlation with poor magnitude calibration is not sufficient for a transport surrogate. It indicates that ordering or morphology has been learned while absolute spectral power remains unreliable. The appropriate response is to diagnose scale calibration, decoder coupling, loss balance, and out-of-distribution coverage rather than report rank correlation alone.

## Repository status

This is research software under active development. The public code is suitable for architecture inspection and data-free tests. It is not a released predictive model, and no trained checkpoint is distributed.
