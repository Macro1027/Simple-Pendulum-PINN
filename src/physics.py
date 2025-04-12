import torch
import numpy as np
import torch.nn as nn # Added for type hint

class SimplePendulum:
    """
    Defines the physics of a simple pendulum.
    theta'' + (g/L) * sin(theta) = 0
    """
    def __init__(self, L: float = 1.0, g: float = 9.81):
        self.L = L
        self.g = g
        self.name = "SimplePendulum"

    def compute_residuals(self, t: torch.Tensor, nn_approximator: nn.Module) -> torch.Tensor:
        """
        Computes the residual of the governing ODE.

        Args:
            t: Input tensor (time), requires gradients.
            nn_approximator: The neural network model approximating theta(t).

        Returns:
            The residual of the ODE.
        """
        t.requires_grad_(True)
        theta = nn_approximator(t)

        # First derivative: d(theta)/dt
        dtheta_dt = torch.autograd.grad(
            theta, t, grad_outputs=torch.ones_like(theta), create_graph=True
        )[0]

        # Second derivative: d^2(theta)/dt^2
        d2theta_dt2 = torch.autograd.grad(
            dtheta_dt, t, grad_outputs=torch.ones_like(dtheta_dt), create_graph=True
        )[0]

        # ODE residual: theta'' + (g/L) * sin(theta)
        residual = d2theta_dt2 + (self.g / self.L) * torch.sin(theta)
        return residual

    def get_initial_conditions(self) -> list[dict]:
        """ Returns typical initial conditions: theta(0)=pi/4, theta'(0)=0 """
        # NOTE: Returning tensors directly for now. Will refine data structure later.
        t0 = torch.tensor([[0.0]], requires_grad=True)
        theta0 = torch.tensor([[np.pi / 4.0]])
        dtheta0_dt = torch.tensor([[0.0]])
        return [{'t': t0, 'theta': theta0, 'dtheta_dt': dtheta0_dt}]

    def get_boundary_conditions(self) -> list:
        # Simple pendulum usually defined by IVP, no BCs needed here
        return []

# --- Placeholder for future extension ---
class DampedPendulum(SimplePendulum):
    def __init__(self, L: float = 1.0, g: float = 9.81, b: float = 0.0): # b = damping coeff
        super().__init__(L, g)
        self.b = b
        self.name = "DampedPendulum"
        print("DampedPendulum placeholder - residual calculation not implemented yet.")

    # Override compute_residuals later
    # def compute_residuals(self, t: torch.Tensor, nn_approximator: nn.Module) -> torch.Tensor:
    #     # Implementation for damped pendulum ODE residual
    #     pass 