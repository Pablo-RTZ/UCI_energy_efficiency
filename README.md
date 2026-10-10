# UCI Energy Efficiency

[![Matlab Version](https://img.shields.io/badge/MATLAB-R2024%2B-blue)](https://www.mathworks.com/products/matlab.html)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE.md)

A repository on dimensionality reduction and regularization applied to the UCI Energy Efficiency dataset.

It studies how classical linear models and regularized regressors behave on a building energy prediction task, with an emphasis on numerical stability, model interpretability, and the bias-variance tradeoff.

## Scope and audience

This repository is intended for students and practitioners interested in applied regression, regularization, dimensionality reduction, and model comparison in a small but representative ML problem. The emphasis is on clarity and educational value: each method is implemented in MATLAB and paired with a discussion of the underlying assumptions, strengths, and limitations.

The project explores the following core ideas:

- Linear regression as a baseline
- Ridge regression for multicollinearity control
- LASSO regression for sparse feature selection
- K-fold cross-validation for hyperparameter tuning
- Learning curves for comparison across sample sizes

## Quick start / Setup

In order to use the programs in this repository, you have to

1. Clone or download the repository
2. Open the folder in MATLAB
3. Run `startup` from the `MATLAB` directory or add the repository to your path
4. Open any notebook in `MATLAB/` and start experimenting
5. Read the full writeup in the [documentation](Docs/Documentation.md) and [index](Docs/INDEX.md)

The folder contains a `startup.m` file, which adds all subfolders to the MATLAB path for the current session.

### Prerequisites

- MATLAB R2024b or later

The code was developed and tested in MATLAB R2024b and does not require any additional toolboxes for the core workflow.

## Contents

The repository currently includes material on the following topics:

- Linear regression
- Ridge regression
- LASSO regression
- K-fold tuning
- Learning curves

Relevant notebooks and supporting files are stored in the [MATLAB](MATLAB/) folder, while the theoretical discussion is available in [Docs/Documentation.md](Docs/Documentation.md).

## Documentation

The full writeup is available in the documentation files:

- [Docs/Documentation.md](Docs/Documentation.md)
- [Docs/INDEX.md](Docs/INDEX.md)

## License

This project is licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).

See the [LICENSE](LICENSE.md) file for details.

If you use this repository, please cite following the [citation](citation.cff).
