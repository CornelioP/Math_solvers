# Optimization Algorithms for Model Fitting

This repository contains implementations of three optimization algorithms widely used in machine learning and numerical analysis:

- **Levenberg-Marquardt (LM)**
- **Stochastic Gradient Descent (SGD)**
- **ADAM (Adaptive Moment Estimation)**

Each algorithm is implemented as a Python class with methods for optimizing model parameters, including optional real-time visualization of the fitting process.

---

## Features

- **LM (Levenberg-Marquardt):**  
  Ideal for non-linear least squares problems. Uses a numerical Jacobian computed via finite differences to update parameters and adaptively adjusts a damping factor.

- **SGD (Stochastic Gradient Descent):**  
  Implements mini-batch gradient descent with numerical gradient estimation via central differences. Suitable for various optimization tasks.

- **ADAM:**  
  Combines momentum and adaptive learning rates using bias-corrected first and second moment estimates for parameter updates.

---

## Classes Overview
### LM (Levenberg-Marquardt)
- **Constructor Parameters**:

- model: The model function taking (params, x) as inputs.
- init_guess: Initial guess for the parameters.
- max_iter: Maximum number of iterations (default: 100).
- tol: Tolerance for convergence (default: 1e-6).
- lamda_init: Initial damping factor (default: 1e-3).
**Key Methods**:

-numerical_jacobian: Computes the Jacobian matrix of residuals with respect to parameters using finite differences.
-optimize: Runs the LM algorithm. Supports optional real-time visualization.
### SGD (Stochastic Gradient Descent)
- **Constructor Parameters**:

- model: Model function.
- init_params: Initial parameter guess.
- loss_fn: Loss function (e.g., mean squared error).
- lr: Learning rate (default: 0.001).
- max_iter: Maximum number of epochs (default: 1000).
- batch_size: Mini-batch size (default: 20).
- tol: Tolerance for convergence (default: 1e-6).
- h: Step size for numerical gradient estimation (default: 1e-7).
- verbose: If True, prints progress information (default: True).
**Key Methods**:

- numerical_grad: Approximates the gradient of the loss using central differences.
- optimize: Executes the SGD optimization with optional visualization.
### ADAM (Adaptive Moment Estimation)
- **Constructor Parameters**:

- Same as SGD, plus:
- beta1: Exponential decay rate for the first moment estimates (default: 0.9).
- beta2: Exponential decay rate for the second moment estimates (default: 0.999).
- epsilon: Small constant for numerical stability (default: 1e-8).
**Key Methods**:

- numerical_grad: Similar to SGD's numerical gradient computation.
- optimize: Runs the ADAM algorithm with real-time visualization optional.
