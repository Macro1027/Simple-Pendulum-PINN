#!/usr/bin/env python
"""Generates training/validation data for the pendulum problem.

Generates:
- Collocation points within the time domain.
- Initial condition points.
"""

import torch
import numpy as np
import os
import sys

# Add src directory to path to import physics module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from physics import SimplePendulum

def generate_pendulum_data(t_min: float = 0.0, t_max: float = 10.0, n_collocation: int = 1000, n_initial: int = 1):
    """Generates data points for the simple pendulum problem."""

    # Collocation points
    t_collocation = torch.linspace(t_min, t_max, n_collocation, requires_grad=True).unsqueeze(1)

    # Initial conditions
    pendulum = SimplePendulum() # Use default L and g
    initial_conditions = pendulum.get_initial_conditions()
    # For now, just use the first IC set provided
    t_initial = initial_conditions[0]['t']
    theta_initial = initial_conditions[0]['theta']
    dtheta_dt_initial = initial_conditions[0]['dtheta_dt']

    print(f"Generated {t_collocation.shape[0]} collocation points between t={t_min} and t={t_max}")
    print(f"Initial condition at t={t_initial.item()}: theta={theta_initial.item():.4f}, dtheta/dt={dtheta_dt_initial.item()}")

    # In a real scenario, you would save these tensors or return them
    return {
        'collocation': t_collocation,
        'initial': {'t': t_initial, 'theta': theta_initial, 'dtheta_dt': dtheta_dt_initial}
    }

if __name__ == "__main__":
    generate_pendulum_data()

# Future implementation details:
# - Use Latin Hypercube Sampling for collocation points
# - Handle multiple initial/boundary conditions systematically
# - Generate high-fidelity validation data using an ODE solver
# - Save data to files (e.g., .pt or .csv) 