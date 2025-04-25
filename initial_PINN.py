import pina
import torch
from torch.nn import Softplus
import matplotlib.pyplot as plt
import numpy as np

theta_0 = torch.tensor([np.pi/4], dtype=float)  # Initial angle (π/4 radians)
omega_0 = torch.tensor([0.0], dtype=float)      # Initial angular velocity (0 radians/s)

from pina.problem import TimeDependentProblem
from pina.operator import grad
from pina.condition import Condition
from pina.equation import Equation, FixedValue
from pina.domain import CartesianDomain

def pendulum_equations(input_, output_):
        # Extract variables
        theta = output_.extract(['theta'])
        omega = output_.extract(['omega'])

        # First-order system of equations
        dtheta_dt = grad(output_, input_, components=['theta'], d=['t'])
        domega_dt = grad(output_, input_, components=['omega'], d=['t'])

        # Pendulum equations
        eq1 = dtheta_dt - omega
        eq2 = domega_dt + (Pendulum.g/Pendulum.L) * torch.sin(theta)

        return torch.cat([eq1, eq2], dim=1)


class Pendulum(TimeDependentProblem):
    # Define the output variables (state variables)
    output_variables = ['theta', 'omega']  # angular position and velocity

    # Define the time domain
    temporal_domain = CartesianDomain({'t': [0.0, 10.0]})  # 10 seconds simulation

    # Physical parameters
    g = torch.tensor(9.81, dtype=torch.float32)
    L = torch.tensor(1.0, dtype=torch.float32)

    # Define domains for initial conditions and the differential equation
    domains = {
        'initial': CartesianDomain({'t': 0.0}),
        'domain': CartesianDomain({'t': [0.0, 10.0]})
    }

    # Define the initial conditions and ODE constraints
    conditions = {
        'initial_theta': Condition(
            domain='initial',
            equation=FixedValue(theta_0)
        ),
        'initial_omega': Condition(
            domain='initial',
            equation=FixedValue(omega_0)
        ),
        'dynamics': Condition(
            domain='domain',
            equation=Equation(pendulum_equations)
        )
    }

from pina.model import FeedForward
from pina.optim import TorchOptimizer
from pina.solver import PINN
from pina.trainer import Trainer

problem = Pendulum()

problem.discretise_domain(n=500)

# Make model + solver + trainer
model = FeedForward(
    layers=[20, 20, 20],
    func=Softplus,                 # Activation function
    output_dimensions=len(problem.output_variables),  # 2 outputs: theta and omega
    input_dimensions=len(problem.input_variables)     # 1 input: time t
)

# Configure optimizer with Adam
optimizer = TorchOptimizer(torch.optim.Adam, lr=0.006, weight_decay=1e-8)

# Create Physics-Informed Neural Network
pinn = PINN(problem, model, optimizer=optimizer)

# Set up trainer
trainer = Trainer(
    pinn,
    max_epochs=100,
    enable_model_summary=False,
    batch_size=8
)

# Train the model
trainer.train()