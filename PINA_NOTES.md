# PINA (Physics-Informed Neural Architecture) Notes

## Goal
Extend the PINN concept by making parts of the neural network architecture itself learnable or adaptable based on the physics.

## Initial Ideas

1.  **Adaptive Activation Functions:** Instead of fixed activation functions (like Tanh), use activation functions with learnable parameters. These parameters could be trained globally or vary with the input (e.g., time).
    *   Reference: [Jagrap et al., Adaptive activation functions](https://arxiv.org/abs/1906.01170)

2.  **Neural ODEs:** Frame the problem as learning the vector field of the ODE system directly using a neural network. This inherently incorporates the physics into the architecture.
    *   Reference: [Chen et al., Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)

3.  **Learnable Differential Operators:** Explore architectures that explicitly learn or approximate differential operators within the network structure.

4.  **Architecture Search (NAS):** Use Neural Architecture Search techniques guided by the physics loss to find optimal network structures.

## Pendulum Application

*   Start with adaptive activation functions for the pendulum ODE.
*   Potentially explore Neural ODEs for comparison.
*   The input will still be time `t`.
*   The output will be `theta(t)`.
*   The loss function will involve the pendulum ODE residual, similar to PINN, but the network structure itself might be different or have learnable components beyond weights/biases.

## Next Steps

*   Research implementations of adaptive activation functions (e.g., Swish, Maxout with learnable parameters).
*   Design the PINA model structure incorporating these ideas.
*   Adapt the training script (`pendulum_pinn.py` -> `pendulum_pina.py`?) to handle the new architecture. 