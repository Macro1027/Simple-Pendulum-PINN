# PINN Implementation for Pendulum Dynamics

## Project Overview

This project implements Physics-Informed Neural Networks (PINNs) using PyTorch to model the dynamics of a simple pendulum, potentially extending to damped systems and other physical phenomena like the heat equation.

## Goals

- Implement a base PINN structure.
- Model simple and damped pendulum systems.
- Validate results against numerical methods (e.g., RK4).
- Explore hyperparameter tuning and optimization techniques.
- Investigate generalization to other PDEs (e.g., Heat Equation).

## Mathematical Formulation

The simple pendulum without damping is governed by the second-order ordinary differential equation (ODE):

\[ \frac{d^2 \theta}{dt^2} + \frac{g}{L} \sin(\theta) = 0 \]

Where:
- \( \theta \) is the angle of displacement
- \( t \) is time
- \( g \) is the acceleration due to gravity
- \( L \) is the length of the pendulum

We typically solve this as an Initial Value Problem (IVP) with given \( \theta(0) \) and \( \frac{d\theta}{dt}(0) \).

## Structure

- `src/`: Core source code (models, physics definitions, utilities).
- `scripts/`: Runnable scripts for training, evaluation, data generation.
- `notebooks/`: Jupyter notebooks for exploration and visualization.
- `data/`: Data files (if any generated/used).
- `tests/`: Unit and integration tests.

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://doi.org/10.1016/j.jcp.2018.10.045). *Journal of Computational Physics*, 378, 686-707. 