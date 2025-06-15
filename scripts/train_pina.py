import pina
import torch
import numpy as np
import torch.nn as nn # Import nn
from torch.nn import Tanh
import matplotlib.pyplot as plt
import yaml # Import YAML loader
import os   # For path joining
# --- Imports for Callback ---
from lightning.pytorch.callbacks import Callback, EarlyStopping
# --------------------------
# --- Imports for Scheduler ---
from torch.optim.lr_scheduler import ReduceLROnPlateau
# ---------------------------

# Import constants
from constants import (
    GRAVITY, 
    PENDULUM_LENGTH, 
    INITIAL_ANGLE, 
    INITIAL_VELOCITY,
    THETA_SCALE,
    OMEGA_SCALE
)

from pina.problem import TimeDependentProblem
from pina.operator import grad
from pina.condition import Condition
from pina.equation import Equation, FixedValue
from pina.domain import CartesianDomain
from pina.model import FeedForward
from pina.label_tensor import LabelTensor
from pina.optim import TorchOptimizer
from pina.solver import PINN
from pina.trainer import Trainer

# --- Configuration Loading ---
def load_config(config_path="pina_config/config.yaml"):
    """Loads configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Add derived values or defaults if needed here
    return config
# ---------------------------

# --- Define enhanced LossHistoryCallback class ---
class LossHistoryCallback(Callback):
    """Callback to record data, physics, and total loss at the end of each epoch."""
    def __init__(self):
        super().__init__()
        self.data_losses = []
        self.physics_losses = []
        self.total_losses = []
        self.prediction_history = []  # Store predictions every 500 epochs
        self.epochs = []  # Store corresponding epochs

    def on_train_epoch_end(self, trainer, pl_module):
        # Access loss from callback metrics directly
        metrics = trainer.callback_metrics
        
        # Extract total loss
        total_loss = metrics.get('train_loss')
        if total_loss is not None:
            self.total_losses.append(total_loss.item())
        
        # Extract initial condition losses (data loss)
        data_loss = 0.0
        if 'initial_theta_loss' in metrics:
            data_loss += metrics['initial_theta_loss'].item()
        if 'initial_omega_loss' in metrics:
            data_loss += metrics['initial_omega_loss'].item()
        self.data_losses.append(data_loss)
        
        # Extract physics loss
        physics_loss = 0.0
        if 'dynamics_loss' in metrics:
            physics_loss = metrics['dynamics_loss'].item()
        self.physics_losses.append(physics_loss)

        # Store predictions every 500 epochs
        current_epoch = trainer.current_epoch
        if current_epoch % 500 == 0:
            self.epochs.append(current_epoch)
            # Generate predictions for the full time domain
            t = torch.linspace(0, 1, 1000).reshape(-1, 1)
            t_label = LabelTensor(t, labels=['t'])
            with torch.no_grad():
                predictions = pl_module.model(t_label)
                theta_pred = predictions.extract(['theta']).cpu().numpy() * THETA_SCALE
            self.prediction_history.append(theta_pred)

def plot_training_predictions(loss_history, config):
    """
    Plot and save the training predictions history showing how the model's predictions
    evolve during training.
    
    Args:
        loss_history (LossHistoryCallback): Callback object containing prediction history
        config: Configuration object containing paths and other settings
    """
    save_path = os.path.join(os.path.dirname(config['paths']['loss_plot']), 'training_predictions.png')
    plt.figure(figsize=(12, 7))
    
    # Generate time points for x-axis
    t = np.linspace(0, config['simulation']['time'], 1000)
    
    # Plot predictions for each stored epoch
    for epoch, pred in zip(loss_history.epochs, loss_history.prediction_history):
        plt.plot(t, pred, alpha=0.5, label=f'Epoch {epoch}')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Evolution of PINN Predictions During Training')
    plt.grid(True)
    
    # Add legend with fewer entries to avoid overcrowding
    handles, labels = plt.gca().get_legend_handles_labels()
    # Show only every 3rd entry in legend
    plt.legend(handles[::3], labels[::3], loc='upper right')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training predictions plot saved to {save_path}")

# --- Define Custom PINN for Scheduler Configuration ---
class CustomPINN(PINN):
    """
    Custom PINN class to correctly handle optimizer and LR scheduler configuration
    in the standard PyTorch Lightning way. Creates optimizer/scheduler within configure_optimizers.
    """
    def __init__(self, problem, model, dummy_pina_optimizer, # Just to satisfy super init check
                 optimizer_class, optimizer_kwargs, 
                 scheduler_class, scheduler_kwargs, scheduler_monitor="train_loss"):
         # Pass the DUMMY PINA optimizer wrapper to the parent class to satisfy type checks
         super().__init__(problem, model, optimizer=dummy_pina_optimizer)
         
         # Store the actual configuration needed by configure_optimizers
         self.model_ref = model # Need access to model.parameters() later
         self.optimizer_class = optimizer_class
         self.optimizer_kwargs = optimizer_kwargs
         self.scheduler_class = scheduler_class
         self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs is not None else {}
         self.scheduler_monitor = scheduler_monitor

    def configure_optimizers(self):
         # Create the actual torch optimizer instance using the stored config and model parameters
         torch_optimizer = self.optimizer_class(
             self.model_ref.parameters(), # Access parameters via stored model reference
             **self.optimizer_kwargs
         )
         
         # Create the scheduler instance using the actual torch_optimizer
         scheduler_instance = self.scheduler_class(torch_optimizer, **self.scheduler_kwargs)

         # Configure the scheduler dictionary for Lightning
         scheduler_config = {
             "scheduler": scheduler_instance,
             "monitor": self.scheduler_monitor,
             "interval": "epoch",
             "frequency": 1,
             "name": "lr_scheduler" # Optional: Name for logging
         }
         return {"optimizer": torch_optimizer, "lr_scheduler": scheduler_config}
# ----------------------------------------------------

class PendulumEquations:
    """Class containing the physics and initial condition equations."""

    # WEIGHT_INIT will be passed from config now

    @staticmethod
    # Accept weight_init as argument
    def weighted_initial_theta(input_, output_):
        """Weighted equation for initial theta condition."""
        theta_scaled = output_.extract(['theta'])
        target_theta_scaled = torch.tensor([INITIAL_ANGLE / THETA_SCALE], dtype=torch.float32, device=theta_scaled.device)
        # Scale the residual before PINA squares it
        return (theta_scaled - target_theta_scaled) * np.sqrt(PendulumEquations.WEIGHT_INIT)

    @staticmethod
    # Accept weight_init as argument
    def weighted_initial_omega(input_, output_):
        """Weighted equation for initial omega condition."""
        omega_scaled = output_.extract(['omega'])
        target_omega_scaled = torch.tensor([INITIAL_VELOCITY / OMEGA_SCALE], dtype=torch.float32, device=omega_scaled.device)
        # Scale the residual before PINA squares it
        return (omega_scaled - target_omega_scaled) * np.sqrt(PendulumEquations.WEIGHT_INIT)

    """Class containing the physics equations for the pendulum system."""
    
    @staticmethod
    def system_equations(input_, output_, sim_time, theta_scale, omega_scale): # Accept constants
        """
        Define the system of first-order ODEs for the pendulum using
        normalized time and scaled output variables.
        
        Args:
            input_: Normalized time input tensor (t_norm)
            output_: Neural network output containing scaled theta_scaled and omega_scaled
            
        Returns:
            torch.Tensor: Concatenated scaled equations for theta_scaled and omega_scaled
        """
        # Extract scaled variables
        theta_scaled = output_.extract(['theta'])
        omega_scaled = output_.extract(['omega'])
        
        # Calculate derivatives with respect to normalized time t_norm
        dtheta_scaled_dt_norm = grad(output_, input_, components=['theta'], d=['t'])
        domega_scaled_dt_norm = grad(output_, input_, components=['omega'], d=['t'])

        # Physical equations:
        # d(theta)/dt = omega
        # d(omega)/dt = -(GRAVITY/PENDULUM_LENGTH) * sin(theta)
        
        # Scaled equations (derived based on t_norm = t/T, theta = theta_s * theta_scaled):
        # 1. d(theta_scaled)/d(t_norm) - (OMEGA_SCALE * SIMULATION_TIME / THETA_SCALE) * omega_scaled = 0
        # 2. d(omega_scaled)/d(t_norm) + (SIMULATION_TIME * GRAVITY / (PENDULUM_LENGTH * OMEGA_SCALE)) * sin(theta_scaled * THETA_SCALE) = 0
        eq1 = dtheta_scaled_dt_norm - (omega_scaled * omega_scale * sim_time / theta_scale)
        eq2 = domega_scaled_dt_norm + (sim_time * GRAVITY / (PENDULUM_LENGTH * OMEGA_SCALE)) * torch.sin(theta_scaled * THETA_SCALE)

        return torch.cat([eq1, eq2], dim=1)


class Pendulum(TimeDependentProblem):
    """Physics-informed neural network problem definition for a simple pendulum."""
    
    # Define the output variables (state variables)
    output_variables = ['theta', 'omega']
    
    # Define the time domain
    temporal_domain = CartesianDomain({'t': [0.0, 1.0]})

    # Define domains for initial conditions and the differential equation
    domains = {
        'initial': CartesianDomain({'t': 0.0}),
        'domain': CartesianDomain({'t': [0.0, 1.0]})
    }

    # Define conditions as a class attribute (placeholder) to satisfy ABC check
    conditions = {} 

    # Define the initial conditions and ODE constraints
    # We need to instantiate this with config values, so define it in __init__
    def __init__(self, config):
        super().__init__() # Initialize the parent class
        self.config = config
        
        # Set weight in PendulumEquations class based on config
        PendulumEquations.WEIGHT_INIT = config['problem']['initial_condition_weight']
        
        # Populate self.conditions for the instance, overwriting the class placeholder
        self.conditions = {
            'initial_theta': Condition(
                domain='initial', # Use the string key 'initial'
                equation=Equation(PendulumEquations.weighted_initial_theta)
            ),
            'initial_omega': Condition(
                domain='initial', # Use the string key 'initial'
                equation=Equation(PendulumEquations.weighted_initial_omega)
            ),
            'dynamics': Condition(
                domain='domain', # Use the string key 'domain'
                equation=Equation(lambda i, o: PendulumEquations.system_equations(i, o, sim_time=self.config['simulation']['time'], theta_scale=THETA_SCALE, omega_scale=OMEGA_SCALE)) # Use self.config
            )
        }

# --- Define Residual Network Architecture ---
class ResidualBlock(nn.Module):
    """Simple Residual Block."""
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.fc(x)

class ResNet(nn.Module):
    """Network with Residual Blocks and Input Feature Expansion."""
    def __init__(self, input_dim=1, output_dim=2, hidden_dim=64, num_blocks=3):
        super().__init__()
        # Input layer should accept the original input dimension
        self.input_layer = nn.Linear(input_dim, hidden_dim) # Use input_dim

        # Residual blocks operate on the hidden dimension
        res_blocks = [ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        self.res_layers = nn.Sequential(*res_blocks)

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in):
        # Reverting temporary modification - Back to original code
        # Original code expecting LabelTensor
        t = x_in.extract(['t']) # Extract original time input 
        # t = x_in # Assume x_in is standard Tensor for plotting
        
        # Feature Expansion: [t] -> [t, sin(2*pi*t), cos(2*pi*t)] (Added 2*pi for better periodicity)
        # features = torch.cat([t, torch.sin(t), torch.cos(t)], dim=1) # REMOVE THIS
        # Pass original input features through the network
        x = torch.tanh(self.input_layer(t)) # Use t directly, Use Tanh after first layer too
        x = self.res_layers(x)
        x = self.output_layer(x)
        # Reverting temporary modification - Back to original code
        # Return original LabelTensor
        # return x 
        return LabelTensor(x, labels=['theta', 'omega'])
# -----------------------------------------

def create_neural_network(input_dim, output_dim, model_config):
    """
    Create a feedforward neural network for the pendulum system.
    
    Args:
        input_dim (int): Number of input dimensions
        output_dim (int): Number of output dimensions
        
    Returns:
        FeedForward: Configured neural network model
    """
    # Use the new ResNet architecture
    # Extract params from config
    hidden_dim = model_config.get('hidden_dim', 64)
    num_blocks = model_config.get('num_blocks', 3)
    network = ResNet(
        input_dim=input_dim, # Should be 1 ('t')
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks
    )

    return network


def plot_training_loss(loss_history, config):
    """
    Plot and save the training loss history with separate curves for
    data loss, physics loss, and total loss.
    
    Args:
        loss_history (LossHistoryCallback): Callback object containing loss history
        config: Configuration object containing paths and other settings
    """
    save_path = config['paths']['loss_plot'] # Get path from config
    plt.figure(figsize=(12, 7))
    
    epochs = range(1, len(loss_history.total_losses) + 1)
    
    # Plot all three loss components
    plt.plot(epochs, loss_history.data_losses, 'b-', label='Data Loss (Ldata)')
    plt.plot(epochs, loss_history.physics_losses, 'r-', label='Physics Loss (Lphysics)')
    plt.plot(epochs, loss_history.total_losses, 'g-', label='Total Loss (Ltotal)')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Components History')
    plt.grid(True)
    plt.legend()
    
    # Use log scale for y-axis as losses often span multiple orders of magnitude
    plt.yscale('log')
    
    # Add minor gridlines
    plt.grid(True, which="minor", ls="-", alpha=0.2)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure results/model directory exists
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training loss components plot saved to {save_path}")


def setup_training(problem, model, config):
    """Setup training components based on configuration."""
    # Create dummy PINA optimizer wrapper (will be replaced by actual torch optimizer)
    dummy_pina_optimizer = TorchOptimizer(torch.optim.Adam, lr=config['optimizer']['lr'])
    
    # Create the custom PINN instance
    pinn = CustomPINN(
        problem=problem,
        model=model,
        dummy_pina_optimizer=dummy_pina_optimizer,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={
            'lr': config['optimizer']['lr'],
            'weight_decay': config['optimizer']['weight_decay']
        },
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={
            'mode': config['scheduler']['mode'],
            'factor': config['scheduler']['factor'],
            'patience': config['scheduler']['patience'],
            'threshold': config['scheduler']['threshold'],
            'threshold_mode': config['scheduler']['threshold_mode'],
            'cooldown': config['scheduler']['cooldown'],
            'min_lr': config['scheduler']['min_lr']
        }
    )
    
    # Create callbacks
    loss_history = LossHistoryCallback()
    early_stop = EarlyStopping(
        monitor='train_loss',
        patience=500,
        min_delta=1e-4,
        mode='min',
        verbose=True
    )
    
    # Create trainer
    trainer = Trainer(
        max_epochs=config['trainer']['max_epochs'],
        callbacks=[loss_history, early_stop],
        gradient_clip_val=config['trainer']['gradient_clip_val'],
        enable_model_summary=config['trainer']['enable_model_summary']
    )
    
    return trainer, pinn, loss_history


def main():
    """Main execution function."""
    # Load configuration
    config = load_config()
    # Access simulation time from config
    # Note: SIMULATION_TIME variable is no longer globally defined from constants
    # simulation_time = config['simulation']['time'] # We pass this to equations via lambda now
    
    # Initialize problem
    problem = Pendulum(config) # Pass config to problem init
    problem.discretise_domain(n=config['problem']['discretization_points']) # Use config value

    # Create model
    model = create_neural_network(
        input_dim=len(problem.input_variables),
        output_dim=len(problem.output_variables),
        model_config=config['model'] # Pass model config section
    )

    # Setup and run training using the adjusted setup function
    trainer, pinn, loss_history = setup_training(problem, model, config) # Pass full config
    trainer.train()
    
    # Plot and save training loss history with all components
    plot_training_loss(loss_history, config)
    
    # Plot and save training predictions history
    plot_training_predictions(loss_history, config)
    
    # --- Verification Test ---
    print("\nRunning verification test...")
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        # Prepare input point (normalized time t=0)
        test_input = LabelTensor(torch.tensor([[0.0]], dtype=torch.float32), labels=['t']) # Instantiate the class
        pred_scaled = model(test_input) # Get scaled predictions (LabelTensor)
        # LabelTensor should behave like a tensor here. Access row 0, then elements.
        # Note: Directly using [0,0] failed before, so access row then elements.
        pred_row_0 = pred_scaled[0] # Get the first (and only) row
        theta_pred = pred_row_0[0].item() * THETA_SCALE # Get first element (theta)
        omega_pred = pred_row_0[1].item() * OMEGA_SCALE # Get second element (omega)
        print(f"Predicted Initial Conditions (t=0): θ={theta_pred:.4f} rad, ω={omega_pred:.4f} rad/s")
        print(f"Target Initial Conditions:          θ={INITIAL_ANGLE:.4f} rad, ω={INITIAL_VELOCITY:.4f} rad/s")
    # --------------------------

    # Save the trained model
    model_save_path = config['paths']['model_save']
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True) # Ensure save directory exists
    torch.save(model.state_dict(), model_save_path)
    print("Model saved to results/pendulum_model.pt")


if __name__ == "__main__":
    main()