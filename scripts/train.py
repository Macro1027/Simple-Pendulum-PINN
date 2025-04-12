#!/usr/bin/env python
"""Basic training script for the PINN model."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from models import MLP
from physics import SimplePendulum
from generate_data import generate_pendulum_data # Re-use data generation logic

# Configuration (move to config file later)
SEED = 42
LEARNING_RATE = 1e-3
EPOCHS = 5000 # Start with a moderate number
HIDDEN_LAYERS = 4
HIDDEN_DIM = 50
T_MIN = 0.0
T_MAX = 10.0
N_COLLOCATION = 1000
LOSS_WEIGHT_RESIDUAL = 1.0
LOSS_WEIGHT_IC_POS = 1.0
LOSS_WEIGHT_IC_VEL = 1.0
GRAD_CLIP_MAX_NORM = 1.0 # Added for gradient clipping

def train_pinn():
    # Seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = MLP(
        input_dim=1,
        output_dim=1, # Output is theta(t)
        hidden_layers=HIDDEN_LAYERS,
        hidden_dim=HIDDEN_DIM,
        activation=nn.Tanh
    ).to(device)

    # Physics Problem
    physics = SimplePendulum() # Default L, g

    # Data
    data = generate_pendulum_data(t_min=T_MIN, t_max=T_MAX, n_collocation=N_COLLOCATION)
    t_collocation = data['collocation'].to(device)
    ic_data = data['initial']
    t_initial = ic_data['t'].to(device)
    theta_initial_true = ic_data['theta'].to(device)
    dtheta_dt_initial_true = ic_data['dtheta_dt'].to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Loss function (Mean Squared Error)
    loss_fn = nn.MSELoss()

    print(f"Starting training for {EPOCHS} epochs...")

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()

        # 1. Calculate Residual Loss (Physics Loss)
        theta_pred_colloc = model(t_collocation)
        residuals = physics.compute_residuals(t_collocation, model)
        loss_residual = loss_fn(residuals, torch.zeros_like(residuals))

        # 2. Calculate Initial Condition Loss
        # Position IC: theta(0)
        theta_pred_initial = model(t_initial)
        loss_ic_pos = loss_fn(theta_pred_initial, theta_initial_true)

        # Velocity IC: theta'(0)
        # Need to compute gradient of model output w.r.t. input t
        t_initial.requires_grad_(True) # Ensure gradient tracking for this specific input
        theta_pred_ic_for_grad = model(t_initial)
        dtheta_dt_pred_initial = torch.autograd.grad(
            theta_pred_ic_for_grad, t_initial, grad_outputs=torch.ones_like(theta_pred_ic_for_grad), create_graph=True
        )[0]
        loss_ic_vel = loss_fn(dtheta_dt_pred_initial, dtheta_dt_initial_true)

        # 3. Total Loss (Weighted Sum)
        total_loss = (LOSS_WEIGHT_RESIDUAL * loss_residual +
                      LOSS_WEIGHT_IC_POS * loss_ic_pos +
                      LOSS_WEIGHT_IC_VEL * loss_ic_vel)

        # 4. Backpropagation and Optimization
        optimizer.zero_grad()
        total_loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()

        # Print Loss (every N epochs)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss.item():.6f} "
                  f"(Res: {loss_residual.item():.4e}, IC_pos: {loss_ic_pos.item():.4e}, IC_vel: {loss_ic_vel.item():.4e})")

    print("Training finished.")
    # TODO: Add evaluation and saving model steps

if __name__ == "__main__":
    train_pinn() 