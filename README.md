# PINN Implementation for Pendulum Dynamics

## Project Overview

This project implements Physics-Informed Neural Networks (PINNs) using PyTorch to model the dynamics of a simple pendulum, potentially extending to damped systems and other physical phenomena like the heat equation.

## Goals

- Implement a base PINN structure.
- Model simple and damped pendulum systems.
- Validate results against numerical methods (e.g., RK4).
- Explore hyperparameter tuning and optimization techniques.
- Investigate generalization to other PDEs (e.g., Heat Equation).

## Structure

- `src/`: Core source code (models, physics definitions, utilities).
- `scripts/`: Runnable scripts for training, evaluation, data generation.
- `notebooks/`: Jupyter notebooks for exploration and visualization.
- `data/`: Data files (if any generated/used).
- `tests/`: Unit and integration tests.

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707. 