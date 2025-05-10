import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the Physics-Informed Neural Architecture (PINA)
class AdaptiveActivation(nn.Module):
    def __init__(self):
        super().__init__()
        # Add a learnable parameter 'a'. Initialize it to 1.
        self.a = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # Apply adaptive Tanh: tanh(a * x)
        return torch.tanh(self.a * x)

class PINA(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                # Use the adaptive activation function
                self.net.add_module(f"adaptive_activation_{i}", AdaptiveActivation())

    def forward(self, t):
        return self.net(t)

# Define the physics loss function
def physics_loss(model, t):
    t.requires_grad_(True)
    theta = model(t)
    
    # Compute derivatives d(theta)/dt and d^2(theta)/dt^2
    d_theta_dt = torch.autograd.grad(theta, t, grad_outputs=torch.ones_like(theta), create_graph=True)[0]
    d2_theta_dt2 = torch.autograd.grad(d_theta_dt, t, grad_outputs=torch.ones_like(d_theta_dt), create_graph=True)[0]
    
    # Pendulum parameters (example: g=9.81, L=1.0)
    g = 9.81
    L = 1.0
    
    # Residual of the ODE: d^2(theta)/dt^2 + (g/L) * sin(theta) = 0
    residual = d2_theta_dt2 + (g / L) * torch.sin(theta)
    
    # Return the mean squared error of the residual
    return torch.mean(residual**2)

# Training parameters
layers = [1, 20, 20, 1] # Input: time (1D), Output: theta (1D)
learning_rate = 5e-4
epochs = 12000

# Create the model and optimizer
pina = PINA(layers)
optimizer = torch.optim.Adam(pina.parameters(), lr=learning_rate)

# Training data (collocation points)
t_train = torch.linspace(0, 10, 100, requires_grad=True).unsqueeze(-1) # Time points

# --- Training Loop ---
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Calculate physics loss
    loss = physics_loss(pina, t_train)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        # Also print the learned 'a' parameters (if multiple layers have them)
        learned_a_params = [p.item() for name, p in pina.named_parameters() if 'adaptive_activation' in name and '.a' in name]
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Learned a: {learned_a_params}')

# --- Basic Validation/Visualization (Example) ---
# Generate test data
t_test = torch.linspace(0, 10, 200).unsqueeze(-1)
pina.eval()
with torch.no_grad():
    theta_pred = pina(t_test)

# Plotting (requires matplotlib)
plt.figure()
plt.plot(t_test.numpy(), theta_pred.numpy(), label='PINA Prediction')
# Add analytical solution or simulated data here for comparison if available
plt.xlabel('Time (s)')
plt.ylabel('Theta (rad)')
plt.title('Pendulum Angle Prediction using PINA')
plt.legend()
plt.grid(True)
# Comment out plt.show() if running in a non-interactive environment
# plt.show() 
# Or save the figure
plt.savefig('pina_prediction.png')
print("Training finished. Plot saved to pina_prediction.png")

# --- Validation Section ---
def analytical_solution(t, theta0=0.1, omega0=0.0, g=9.81, L=1.0):
    """Analytical solution for small angle approximation."""
    # Small angle approximation: theta(t) = theta0*cos(sqrt(g/L)*t) + (omega0/sqrt(g/L))*sin(sqrt(g/L)*t)
    # This is a simplified example, assumes theta0 is small.
    omega = np.sqrt(g/L)
    return theta0 * np.cos(omega * t) + (omega0 / omega) * np.sin(omega * t)

# Compare PINA prediction with analytical solution (for small angles)
t_test_np = t_test.detach().numpy()
theta_analytical = analytical_solution(t_test_np)

plt.figure(figsize=(10, 6))
plt.plot(t_test_np, theta_pred.numpy(), label='PINA Prediction', linestyle='--')
plt.plot(t_test_np, theta_analytical, label='Analytical Solution (Small Angle)', linestyle=':')
plt.xlabel('Time (s)')
plt.ylabel('Theta (rad)')
plt.title('PINA vs Analytical Solution Comparison')
plt.legend()
plt.grid(True)
plt.savefig('pina_validation_comparison.png')
print("Validation plot saved to pina_validation_comparison.png") 