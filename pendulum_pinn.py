import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the Physics-Informed Neural Network
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.net.add_module(f"activation_{i}", nn.Tanh())

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
learning_rate = 1e-3
epochs = 10000

# Create the model and optimizer
pinn = PINN(layers)
optimizer = torch.optim.Adam(pinn.parameters(), lr=learning_rate)

# Training data (collocation points)
t_train = torch.linspace(0, 10, 100, requires_grad=True).unsqueeze(-1) # Time points

# --- Training Loop ---
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Calculate physics loss
    loss = physics_loss(pinn, t_train)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# --- Basic Validation/Visualization (Example) ---
# Generate test data
t_test = torch.linspace(0, 10, 200).unsqueeze(-1)
pinn.eval() # Set model to evaluation mode
with torch.no_grad():
    theta_pred = pinn(t_test)

# Plotting (requires matplotlib)
plt.figure()
plt.plot(t_test.numpy(), theta_pred.numpy(), label='PINN Prediction')
# Add analytical solution or simulated data here for comparison if available
plt.xlabel('Time (s)')
plt.ylabel('Theta (rad)')
plt.title('Pendulum Angle Prediction using PINN')
plt.legend()
plt.grid(True)
# Comment out plt.show() if running in a non-interactive environment
# plt.show() 
# Or save the figure
plt.savefig('pinn_prediction.png')
print("Training finished. Plot saved to pinn_prediction.png") 