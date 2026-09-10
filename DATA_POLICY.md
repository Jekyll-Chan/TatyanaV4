# Data policy

This public repository is intentionally data-free.

## Provenance and roles

The private simulation databases used during the internal research were supplied through the NTU Plasma Theory group and by Youngwoo Cho at the Korea Institute of Fusion Energy (KFE). Tingyi Chen's role is surrogate-model development, including model architecture, training and evaluation methodology, and software implementation. This repository does not claim authorship or ownership of the databases.

## Content that is not distributed

The repository excludes:

- raw, converted, sampled, or synthetic replicas of the research databases;
- database keys, case identifiers, parameter ranges, and private metadata;
- normalization statistics and other fitted preprocessing state;
- trained weights, checkpoints, optimizer state, and serialized objects;
- logs, metrics, plots, reports, and other database-derived artifacts; and
- local paths, credentials, cluster scripts, and institutional infrastructure details.

The public tests use randomly generated tensors only. They verify software behavior and do not reproduce the distribution or contents of the private databases.

## Reproduction boundary

The model interface is public, but empirical results cannot be reproduced from this repository alone. Authorized users must provide their own compatible data adapter and compute normalization statistics strictly from their training partition. Access to the original databases must be requested from their respective owners and is not granted by this repository.
