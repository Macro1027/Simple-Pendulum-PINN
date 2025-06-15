#!/usr/bin/env python3
import os
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import argparse
from scipy.integrate import solve_ivp
from pina.label_tensor import LabelTensor
from pina.optim import TorchOptimizer

# Import constants
from constants import (
    GRAVITY, 
    PENDULUM_LENGTH, 
    INITIAL_ANGLE, 
    INITIAL_VELOCITY,
    SIMULATION_TIME
)

# Import model components from train_pina
from train_pina import create_neural_network, PendulumEquations, Pendulum

# Add config loading function (copied from train_pina.py for self-containment)
def load_config(config_path="pina_config/config.yaml"):
    """Loads configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_trained_model(model_path, input_dim=1, output_dim=2, config_path="pina_config/config.yaml"):
    """
    Load the trained pendulum model.
    
    Args:
        model_path (str): Path to the saved model state dict
        input_dim (int): Number of input dimensions (time)
        output_dim (int): Number of output dimensions (theta, omega)
        config_path (str): Path to the main YAML configuration file.
        
    Returns:
        torch.nn.Module: Loaded model
    """
    # Load the main configuration to get model architecture details
    config = load_config(config_path)
    model_config = config.get('model', {}) # Get model section, default to empty dict

    # Pass model_config to create_neural_network
    model = create_neural_network(input_dim, output_dim, model_config=model_config) 
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    return model

def generate_time_points(t_min=0.0, t_max=10.0, num_points=1000):
    """
    Generate uniformly spaced time points for prediction.
    
    Args:
        t_min (float): Start time
        t_max (float): End time
        num_points (int): Number of time points
        
    Returns:
        torch.Tensor: Tensor of time points with shape [num_points, 1]
    """
    t = torch.linspace(t_min, t_max, num_points, dtype=torch.float32).unsqueeze(1)
    return t

def predict_pendulum_states(model, t):
    """
    Predict theta and omega for given time points.
    
    Args:
        model (torch.nn.Module): Trained pendulum model
        t (torch.Tensor): Time points tensor [num_points, 1]
        
    Returns:
        tuple: (theta, omega) predictions as numpy arrays
    """
    with torch.no_grad():
        # Wrap the input tensor t into a LabelTensor before passing to the model
        t_labeled = LabelTensor(t, labels=['t'])
        y_pred_labeled = model(t_labeled)
    
    # Extract the raw tensor values - Use direct slicing on LabelTensor
    # y_pred_raw = y_pred_labeled.values() # REMOVE this incorrect call
    
    # First column is theta, second is omega
    theta = y_pred_labeled[:, 0].squeeze().detach().cpu().numpy()
    omega = y_pred_labeled[:, 1].squeeze().detach().cpu().numpy()
    
    return theta, omega

def plot_physics_results(t, theta, omega, save_path_base='results/physics/'):
    """
    Plot the time series of theta, omega, energy, and the phase space.
    
    Args:
        t (torch.Tensor): Time points
        theta (numpy.ndarray): Predicted theta values
        omega (numpy.ndarray): Predicted omega values
        save_path_base (str): Base path for saving the plot
    """
    t_np = t.squeeze().numpy()
    
    # Calculate Energy per unit mass E/m = 0.5 * L^2 * omega^2 + g * L * (1 - cos(theta))
    KE_per_m = 0.5 * (PENDULUM_LENGTH**2) * (omega**2)
    PE_per_m = GRAVITY * PENDULUM_LENGTH * (1 - np.cos(theta))
    E_per_m = KE_per_m + PE_per_m
    
    fig = plt.figure(figsize=(18, 12))
    
    # Plot theta(t)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(t_np, theta, 'b-', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('θ(t) [rad]')
    ax1.set_title('PINN-predicted Angular Position')
    ax1.grid(True)
    
    # Plot omega(t)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(t_np, omega, 'r-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('ω(t) [rad/s]')
    ax2.set_title('PINN-predicted Angular Velocity')
    ax2.grid(True)
    
    # Plot Energy E(t)/m
    ax_energy = fig.add_subplot(2, 3, 3)
    ax_energy.plot(t_np, E_per_m, 'g-', linewidth=2)
    ax_energy.set_xlabel('Time (s)')
    ax_energy.set_ylabel('Energy/Mass [J/kg]')
    ax_energy.set_title('PINN-predicted Energy per Mass')
    # Check variation to set appropriate y-limits if needed
    energy_variation = E_per_m.max() - E_per_m.min()
    energy_mean = E_per_m.mean()
    ax_energy.set_ylim(energy_mean - energy_variation * 1.5, energy_mean + energy_variation * 1.5) # Dynamic ylim
    ax_energy.grid(True)
    
    # Plot phase space (theta vs omega)
    ax3 = fig.add_subplot(2, 1, 2)
    
    # Create a colormap based on time to show trajectory evolution
    norm = Normalize(t_np.min(), t_np.max())
    colors = cm.viridis(norm(t_np))
    
    # Plot as points with color gradient based on time
    for i in range(len(t_np)-1):
        ax3.plot(theta[i:i+2], omega[i:i+2], '-', color=colors[i], linewidth=1.5)
    
    # Add arrow to show direction of time
    arrow_idx = len(theta) // 2
    ax3.arrow(theta[arrow_idx], omega[arrow_idx], 
              theta[arrow_idx+1]-theta[arrow_idx], omega[arrow_idx+1]-omega[arrow_idx],
              head_width=0.05, head_length=0.1, fc='black', ec='black', length_includes_head=True)
    
    # Add colorbar to show time progression
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3)
    cbar.set_label('Time (s)')
    
    ax3.set_xlabel('θ [rad]')
    ax3.set_ylabel('ω [rad/s]')
    ax3.set_title('Phase Space Trajectory (θ vs ω)')
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Ensure directory exists and save
    os.makedirs(save_path_base, exist_ok=True)
    save_name = os.path.join(save_path_base, 'pinn_dynamics.png')
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Dynamics plots saved to {save_name}")
    
    return fig

def calculate_residuals(model, t, g=9.81, L=1.0):
    """
    Calculate the residuals of the pendulum equation |θ̈ + (g/L)sin(θ)| for each time point.
    
    Args:
        model (torch.nn.Module): Trained pendulum model
        t (torch.Tensor): Time points tensor [num_points, 1]
        g (float): Gravitational acceleration
        L (float): Pendulum length
        
    Returns:
        numpy.ndarray: Residual values at each time point
    """
    # We need to use requires_grad for autograd to work
    t_grad = t.clone().detach().requires_grad_(True)
    
    # Forward pass to get theta
    with torch.enable_grad(): # Need grad enabled for autograd later
        # First get regular output for reference
        t_labeled_ref = LabelTensor(t_grad, labels=['t']) # Use grad-enabled tensor
        output_regular_labeled = model(t_labeled_ref)
        # output_regular_raw = output_regular_labeled.values() # REMOVE this incorrect call
        theta_regular = output_regular_labeled[:, 0].clone().detach().cpu().numpy() # Slice directly
    
    # Now compute the gradients
    residuals = []
    
    # Process in smaller batches to avoid memory issues
    batch_size = 100
    num_batches = len(t) // batch_size + (1 if len(t) % batch_size > 0 else 0)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(t))
        
        batch_t = t_grad[start_idx:end_idx]
        
        # Wrap input for the model
        batch_t_labeled = LabelTensor(batch_t, labels=['t'])

        # Forward pass with gradients enabled
        output_labeled = model(batch_t_labeled)
        theta = output_labeled.extract(['theta']) # Now use extract
        
        # Calculate first derivative (dθ/dt)
        dtheta_dt = torch.autograd.grad(
            theta, batch_t, 
            grad_outputs=torch.ones_like(theta),
            create_graph=True
        )[0].squeeze()
        
        # Calculate second derivative (d²θ/dt²)
        d2theta_dt2 = torch.autograd.grad(
            dtheta_dt, batch_t,
            grad_outputs=torch.ones_like(dtheta_dt),
            retain_graph=True
        )[0].squeeze()
        
        # Calculate residual |θ̈ + (g/L)sin(θ)|
        sin_theta = torch.sin(theta)
        residual = torch.abs(d2theta_dt2 + (g/L) * sin_theta)
        
        # Store the residuals
        residuals.append(residual.detach().cpu().numpy())
    
    # Concatenate all batches
    residuals = np.concatenate(residuals)
    
    return residuals, theta_regular

def plot_residual_evolution(t, residuals, theta, save_path_base='results/physics/'):
    """
    Plot the residual evolution |θ̈ + (g/L)sin(θ)| vs. t.
    
    Args:
        t (torch.Tensor): Time points
        residuals (numpy.ndarray): Residual values
        theta (numpy.ndarray): Theta values for reference
        save_path_base (str): Base path for saving the plot
    """
    t_np = t.squeeze().numpy()
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})
    
    # Plot theta for reference
    ax1.plot(t_np, theta, 'b-', linewidth=1.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('θ(t) [rad]')
    ax1.set_title('Angular Position (Reference)')
    ax1.grid(True)
    
    # Plot residual evolution
    ax2.semilogy(t_np, residuals, 'r-', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('|θ̈ + (g/L)sin(θ)|')
    ax2.set_title('Residual Evolution (Log Scale)')
    ax2.grid(True, which="both", ls="-")
    
    # Set y-axis to log scale for better visualization of small residuals
    # (already done with semilogy)
    
    plt.tight_layout()
    
    # Ensure directory exists and save
    os.makedirs(save_path_base, exist_ok=True)
    save_name = os.path.join(save_path_base, 'residual_evolution.png')
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Residual evolution plot saved to {save_name}")
    
    return fig

def compare_with_analytical(model, t_max=SIMULATION_TIME, num_points=1000, save_path_base='results/physics/'):
    """Compare PINN solution with both linear and nonlinear analytical solutions."""
    # Generate time points and get PINN predictions
    t = generate_time_points(t_max=t_max, num_points=num_points)
    theta_pinn, omega_pinn = predict_pendulum_states(model, t)
    t_np = t.squeeze().numpy()
    
    # 1. Linear SHM solution (small angle approximation)
    def linear_shm(t, y):
        theta, omega = y
        return [omega, -(GRAVITY/PENDULUM_LENGTH) * theta]
    
    sol_linear = solve_ivp(
        linear_shm, 
        [0, t_max], 
        [INITIAL_ANGLE, INITIAL_VELOCITY],
        t_eval=t_np,
        method='RK45'
    )
    
    # 2. Full nonlinear pendulum solution
    def nonlinear_pendulum(t, y):
        theta, omega = y
        return [omega, -(GRAVITY/PENDULUM_LENGTH) * np.sin(theta)]
    
    sol_nonlinear = solve_ivp(
        nonlinear_pendulum,
        [0, t_max],
        [INITIAL_ANGLE, INITIAL_VELOCITY],
        t_eval=t_np,
        method='RK45',
        rtol=1e-8,  # Tighter tolerances for accuracy
        atol=1e-8
    )
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Main plot
    plt.subplot(2, 1, 1)
    plt.plot(t_np, theta_pinn, 'b-', label='PINN', linewidth=2)
    plt.plot(t_np, sol_nonlinear.y[0], 'r--', label='Nonlinear Analytical', linewidth=2)
    plt.plot(t_np, sol_linear.y[0], 'g:', label='Linear SHM', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('θ(t) [rad]')
    plt.title('Pendulum Angle: PINN vs Analytical Solutions')
    plt.legend()
    plt.grid(True)
    
    # Error plot
    plt.subplot(2, 1, 2)
    plt.plot(t_np, np.abs(theta_pinn - sol_nonlinear.y[0]), 'r-', label='PINN vs Nonlinear', linewidth=2)
    plt.plot(t_np, np.abs(theta_pinn - sol_linear.y[0]), 'g-', label='PINN vs Linear', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Absolute Error [rad]')
    plt.title('Error Analysis')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Log scale for better error visualization
    
    plt.tight_layout()
    
    # Ensure directory exists and save
    os.makedirs(save_path_base, exist_ok=True)
    save_name = os.path.join(save_path_base, 'analytical_comparison.png')
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print some statistics
    print("\nAnalytical Comparison Statistics:")
    print(f"Initial Angle: {INITIAL_ANGLE:.4f} rad ({np.degrees(INITIAL_ANGLE):.2f}°)")
    print(f"Initial Velocity: {INITIAL_VELOCITY:.4f} rad/s")
    print("\nMaximum Errors:")
    print(f"PINN vs Nonlinear: {np.max(np.abs(theta_pinn - sol_nonlinear.y[0])):.6f} rad")
    print(f"PINN vs Linear: {np.max(np.abs(theta_pinn - sol_linear.y[0])):.6f} rad")
    print(f"Nonlinear vs Linear: {np.max(np.abs(sol_nonlinear.y[0] - sol_linear.y[0])):.6f} rad")
    
    print(f"\nAnalytical comparison saved to {save_name}")

def main():
    """Main function to generate pendulum visualizations."""
    parser = argparse.ArgumentParser(description='Visualize pendulum dynamics')
    parser.add_argument('--model_path', type=str, default='results/pendulum_model.pt',
                        help='Path to the trained model')
    parser.add_argument('--t_max', type=float, default=10.0,
                        help='Maximum time for simulation')
    parser.add_argument('--num_points', type=int, default=1000,
                        help='Number of time points')
    parser.add_argument('--output_dir', type=str, default='results/physics',
                        help='Output directory for visualization plots')
    # Add argument for main config file path
    parser.add_argument('--config', type=str, default='pina_config/config.yaml',
                        help='Path to the main configuration YAML file.')
    
    args = parser.parse_args()
    
    # Load the trained model (which is ResNet) - pass config path
    model = load_trained_model(args.model_path, input_dim=1, output_dim=2, config_path=args.config) # Ensure correct dims and pass config path
    
    # Generate time points
    t = generate_time_points(t_min=0.0, t_max=args.t_max, num_points=args.num_points)
    
    # Predict theta and omega
    theta, omega = predict_pendulum_states(model, t)
    
    # Plot and save dynamics results (theta, omega, energy, phase space)
    plot_physics_results(t, theta, omega, save_path_base=args.output_dir)
    
    # Calculate and plot residuals
    print("Calculating residuals (this may take a moment)...")
    residuals, theta_ref = calculate_residuals(model, t, g=GRAVITY, L=PENDULUM_LENGTH) # Pass constants
    plot_residual_evolution(t, residuals, theta_ref, save_path_base=args.output_dir)

    # Compare with analytical solution
    print("Comparing with analytical solution...")
    compare_with_analytical(model, t_max=args.t_max, num_points=args.num_points, save_path_base=args.output_dir)

if __name__ == "__main__":
    main()

