#!/usr/bin/env python
"""Basic training script for the PINN model."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import matplotlib.pyplot as plt # Added for plotting

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from models import MLP
from physics import SimplePendulum
# from generate_data import generate_pendulum_data # Keep for collocation points generation for now
# PINA related imports (will need more later)
from pina import LabelTensor, Condition
from pina.problem import Problem
from pina.model import FeedForward

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
PLOT_FREQUENCY = 500 # How often to update plot during training (if desired)
SAVE_PLOT = True
PLOT_FILENAME = "training_loss.png"

def train_pinn():
    # Seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model (Using PINA's FeedForward potentially, or keep MLP and wrap)
    # model = FeedForward(...)
    model = MLP(
        input_dim=1,
        output_dim=1, # Output is theta(t)
        hidden_layers=HIDDEN_LAYERS,
        hidden_dim=HIDDEN_DIM,
        activation=nn.Tanh
    ).to(device)

    # Physics Problem
    physics = SimplePendulum() # Default L, g

    # Data (Adapt to PINA's way of handling data)
    # Generate collocation points using LabelTensor
    t_range = [T_MIN, T_MAX]
    input_variables = {'t': t_range}
    collocation_pts = LabelTensor(torch.linspace(T_MIN, T_MAX, N_COLLOCATION, requires_grad=True).unsqueeze(1), labels=['t']).to(device)

    # Get PINA Conditions for ICs
    ic_conditions = physics.get_initial_conditions()
    ic_pos_condition = ic_conditions['ic_pos'].to(device)
    ic_vel_condition = ic_conditions['ic_vel'].to(device)

    # PINA PROBLEM DEFINITION (Conceptual - Full setup deferred)
    # problem = Problem(...)
    # locations = {'collocation': collocation_pts, 'ic_pos': ic_pos_condition.input_points, ...}

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Loss function (Mean Squared Error)
    loss_fn = nn.MSELoss()

    print(f"Starting training for {EPOCHS} epochs...")

    # Store loss history
    history = {'epoch': [], 'total_loss': [], 'loss_residual': [], 'loss_ic_pos': [], 'loss_ic_vel': []}

    # Training Loop (Simplified - PINA Trainer will handle this later)
    for epoch in range(EPOCHS):
        model.train()

        # 1. Calculate Residual Loss (Physics Loss)
        # PINA's compute_residuals expects LabelTensor
        residuals = physics.compute_residuals(collocation_pts, model)
        loss_residual = loss_fn(residuals, torch.zeros_like(residuals))

        # 2. Calculate Initial Condition Loss (using PINA Conditions)
        # Position IC: theta(0)
        theta_pred_initial = model(ic_pos_condition.input_points)
        loss_ic_pos = loss_fn(theta_pred_initial, ic_pos_condition.output_points)

        # Velocity IC: theta'(0)
        # Get the operator and evaluate it
        # NOTE: This part is tricky without the full PINA Trainer context which usually assigns the model to the operator.
        # We simulate it here by assigning the model manually to the Grad operator.
        # This might NOT be the standard PINA way.
        dtheta_dt_operator = ic_vel_condition.equation
        dtheta_dt_operator.model = model # Manually assign model for this calculation
        dtheta_dt_pred_initial = dtheta_dt_operator(ic_vel_condition.input_points)
        loss_ic_vel = loss_fn(dtheta_dt_pred_initial, ic_vel_condition.output_points)

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
            # Adjust printing if LabelTensors are used in losses
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss.item():.6f} "
                  f"(Res: {loss_residual.item():.4e}, IC_pos: {loss_ic_pos.item():.4e}, IC_vel: {loss_ic_vel.item():.4e})")

        # Record history
        history['epoch'].append(epoch + 1)
        history['total_loss'].append(total_loss.item())
        history['loss_residual'].append(loss_residual.item())
        history['loss_ic_pos'].append(loss_ic_pos.item())
        history['loss_ic_vel'].append(loss_ic_vel.item())

        # Optional: Update plot periodically (can slow down training)
        # if (epoch + 1) % PLOT_FREQUENCY == 0:
        #     plot_loss_history(history, PLOT_FILENAME, show=True)

    print("Training finished.")
    print("Note: Training loop adapted for PINA concepts, but full PINA Trainer integration is pending.")
    # TODO: Replace manual loop with PINA Trainer

    # Plot final loss history
    plot_loss_history(history, PLOT_FILENAME, save=SAVE_PLOT, show=True)

def plot_loss_history(history: dict, filename: str, save: bool = False, show: bool = True):
    """Plots the training loss history."""
    plt.figure(figsize=(12, 6))

    plt.semilogy(history['epoch'], history['total_loss'], label='Total Loss', alpha=0.8)
    plt.semilogy(history['epoch'], history['loss_residual'], label='Residual Loss', linestyle=':', alpha=0.7)
    plt.semilogy(history['epoch'], history['loss_ic_pos'], label='IC Position Loss', linestyle=':', alpha=0.7)
    plt.semilogy(history['epoch'], history['loss_ic_vel'], label='IC Velocity Loss', linestyle=':', alpha=0.7)

    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Loss History")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    if save:
        plt.savefig(filename)
        print(f"Loss plot saved to {filename}")
    if show:
        plt.show()
    plt.close()

if __name__ == "__main__":
    train_pinn() 