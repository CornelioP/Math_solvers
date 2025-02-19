import numpy as np
import matplotlib.pyplot as plt

class LM:
    def __init__(self, model, init_guess, max_iter=100, tol=1e-6, lamda_init=1e-3):
        self.model = model
        self.init_guess = init_guess
        self.max_iter = max_iter
        self.tol = tol
        self.lamda_init = lamda_init
        
    def numerical_jacobian(self, params, x, y, h=1e-8):
        """
        Compute the Jacobian matrix of residuals w.r.t. parameters.
        J[i, j] = ∂r_i/∂p_j where r = model(params, x) - y.
        """
        residual_0 = self.model(params, x) - y
        # Correctly initialize the Jacobian matrix with shape (n_data, n_params)
        J = np.zeros((len(residual_0), len(params)))
        
        # Compute each column of the Jacobian by perturbing each parameter in turn
        for j in range(len(params)):
            params_perturbed = params.copy()
            params_perturbed[j] += h
            residual_perturbed = self.model(params_perturbed, x) - y
            # The derivative is approximated as the finite difference divided by h
            J[:, j] = (residual_perturbed - residual_0) / h
        
        return J
                
    def optimize(self, x, y, x_plot=None, visualize=False):
        """
        Optimize the parameters using the Levenberg-Marquardt algorithm.
        
        Parameters:
            x: The input data for the model.
            y: The observed data.
            x_plot: A set of x values for plotting the model estimate.
            visualize: If True, update a plot at each iteration.
        """
        current_params = self.init_guess.copy()
        current_lambda = self.lamda_init
        model = self.model
        tol = self.tol
        cost_history = []
        
        # Set up visualization if desired
        if visualize:
            if x_plot is None:
                # If no plotting grid is provided, use the x data for visualization.
                x_plot = x
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
        
        for iter in range(self.max_iter):
            # 1. Compute residuals and cost 
            residuals = model(current_params, x) - y
            cost = np.sum(residuals**2)
            
            print(f"Iteration {iter}: cost = {cost}, params = {current_params}")

            # 2. Check convergence
            if iter > 0 and abs(cost_history[-1] - cost) < tol:
                print(f"Converged at iteration {iter}")
                break
            
            # 3. Compute Jacobian 
            J = self.numerical_jacobian(current_params, x, y)
            
            # 4. Form the LM system: (JᵀJ + λ*diag(JᵀJ))Δ = -Jᵀr
            J_transpose = J.T
            JtJ = J_transpose @ J
            Jt_r = J_transpose @ residuals
            # Use lambda * diag(JtJ) for damping
            damping = current_lambda * np.diag(np.diag(JtJ))
            A = JtJ + damping
            
            try:
                delta = np.linalg.solve(A, -Jt_r)
            except np.linalg.LinAlgError:
                print("Linear system could not be solved.")
                break
                
            # 5. Trial update 
            trial_params = current_params + delta
            trial_residuals = model(trial_params, x) - y 
            trial_cost = np.sum(trial_residuals**2)
            
            # 6. Accept/reject step
            if trial_cost < cost:
                current_params = trial_params
                current_lambda /= 10
            else:
                current_lambda *= 10
            
            cost_history.append(cost)
            
            # 7. Visualization update
            if visualize:
                ax.clear()
                # Plot the observed data
                ax.scatter(x, y, label="Data", color='blue')
                # Compute and plot the current model estimate over x_plot
                y_est = model(current_params, x_plot)
                ax.plot(x_plot, y_est, label="Model Estimate", color='red')
                ax.set_title(f"Iteration {iter} - Cost: {cost:.4f}")
                ax.legend()
                plt.pause(0.1)
        
        if visualize:
            plt.ioff()
            plt.show()
            
        return current_params