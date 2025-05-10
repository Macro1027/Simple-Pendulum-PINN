# PINN and PINA for Pendulum Simulation

This project explores the use of Physics-Informed Neural Networks (PINNs) and Physics-Informed Neural Architectures (PINAs) to simulate the motion of a simple pendulum.

## Project Structure

-   `pendulum_pinn.py`: Implementation of a standard PINN model.
-   `pendulum_pina.py`: Implementation of a PINA model with adaptive activation functions.
-   `compare_models.py`: Script to compare the performance of PINN and PINA models.
-   `PINA_NOTES.md`: Design notes and ideas for the PINA architecture.
-   `requirements.txt`: Python dependencies.
-   `.gitignore`: Specifies intentionally untracked files that Git should ignore.

## Setup

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

-   To train the PINN model:
    ```bash
    python pendulum_pinn.py
    ```
-   To train the PINA model:
    ```bash
    python pendulum_pina.py
    ```
-   To run the comparison (after training and saving models - note: saving logic is currently placeholder):
    ```bash
    python compare_models.py
    ```

Predictions and comparison plots will be saved as PNG files in the root directory.

## Concepts

### PINN (Physics-Informed Neural Network)
A neural network trained to solve supervised learning tasks while respecting laws of physics described by general nonlinear partial differential equations. The loss function includes both the data mismatch and the physics residual.

### PINA (Physics-Informed Neural Architecture)
An extension where the neural network architecture itself has components that are learnable or adaptable based on the physics. This project explores a simple PINA with adaptive activation functions.

## Future Work

-   Implement more sophisticated adaptive activation functions.
-   Explore Neural ODEs for the pendulum problem.
-   Conduct a rigorous hyperparameter search for both PINN and PINA models.
-   Expand the comparison to include metrics like training time, convergence speed, and robustness to noise.
-   Apply the PINA concept to more complex physical systems.
-   Develop more comprehensive unit and integration tests.
-   Investigate the effect of different collocation point sampling strategies.
-   Explore transfer learning: e.g., train on a simple pendulum and adapt to a damped or forced pendulum. 