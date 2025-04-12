"""Physical and simulation constants for the pendulum system."""

import numpy as np

# Physical constants
GRAVITY = 9.81  # gravitational acceleration (m/s^2)
PENDULUM_LENGTH = 1.0  # length of pendulum (m)

# Initial conditions
INITIAL_ANGLE = np.pi / 4  # initial angle (radians) - pi/4
INITIAL_VELOCITY = 0.0  # initial angular velocity (rad/s)

# Simulation parameters
SIMULATION_TIME = 5.0  # total simulation time (s)

# Normalization/Scaling constants
THETA_SCALE = np.pi        # Characteristic angle scale
# Estimate max omega based on energy conservation: 0.5*L^2*omega_max^2 = g*L*(1-cos(theta0))
OMEGA_MAX_ESTIMATE = np.sqrt(2 * GRAVITY / PENDULUM_LENGTH * (1 - np.cos(INITIAL_ANGLE)))
OMEGA_SCALE = OMEGA_MAX_ESTIMATE * 1.1 # Use a value slightly larger than estimate
if OMEGA_SCALE == 0: # Avoid division by zero if initial angle is 0
    OMEGA_SCALE = 1.0 