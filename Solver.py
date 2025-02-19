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
            
            print(f"Iteration {iter}: cost = {cost}, params = {current_params}, lambda = {current_lambda}")

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


class SGD:
    def __init__(self, model, init_params, loss_fn, lr=0.001, max_iter=1000, batch_size=20, tol=1e-6, h=1e-7, verbose=True):
        """
        Parameters:
            model: function
                Model function
            init_params: float array
                Array of inital param guess
            loss_fn: function
                Loss function
            lr : float
                Learning rate.
            max_iter : int
                Maximum number of iterations (epochs).
            batch_size : int
                Number of samples per mini-batch. If set to 1, it's pure SGD.
            tol : float
                Tolerance for stopping criterion based on loss improvement.
            verbose : bool
                If True, print loss info during training.
        """
        self.model = model
        self.init_params = init_params
        self.loss_fn = loss_fn
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.verbose = verbose
        self.h = h
        
    def numerical_grad(self, params, x, y):
        """
        Compute the gradient of the loss with respect to parameters numerically
        using central differences.

        Parameters:
            params : np.ndarray
                Current parameters.
            x : np.ndarray
                Input data for the current batch.
            y : np.ndarray
                Observed targets for the current batch.
            loss_fn : callable
                Function that computes the loss given (predictions, y).

        Returns:
            grad : np.ndarray
                Approximated gradient vector.
        """
        
        grad = np.zeros_like(params)
        loss = self.loss_fn(self.model(params,x),y)
        
        for i in range(len(params)):
            params_perturbed = params.copy()
            params_perturbed[i] += self.h
            loss_perturbed = self.loss_fn(self.model(params_perturbed,x),y)
            grad[i] = (loss_perturbed - loss)/ self.h
        
        return grad
    
    def optimize(self, x, y, x_plot=None, visualize=False):
        
        
        # Set up visualization if desired
        if visualize:
            if x_plot is None:
                # If no plotting grid is provided, use the x data for visualization.
                x_plot = x
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
        
        n_samples = x.shape[0]
        loss_history = []
        params = self.init_params
        
        for epoch in range(self.max_iter):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            for i in range(0, n_samples, self.batch_size):
                x_batch =x_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                
                #Compute predictions and loss for current batch 
                preds = self.model(params,x_batch)
                loss = self.loss_fn(preds,y_batch)
                epoch_loss += loss
                
                #Compute gradient
                grad = self.numerical_grad(params,x_batch,y_batch)
                
                #Update param
                params = params - self.lr * grad
            
            #Avarage loss over batch in this epoch 
            epoch_loss /= (n_samples/ self.batch_size)
            loss_history.append(epoch_loss)
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.max_iter} - Loss: {epoch_loss:.6f}")
                
            # Visualization update (if requested)
            if x_plot is not None:
                ax.clear()
                ax.scatter(x, y, label="Data", color="blue")
                # Compute model predictions on the plotting grid
                y_est = self.model(params, x_plot)
                ax.plot(x_plot, y_est, label="Model Estimate", color="red")
                ax.set_title(f"Epoch {epoch+1} - Loss: {epoch_loss:.6f}")
                ax.legend()
                plt.pause(0.1)
            
              # Early stopping based on loss improvement
            if epoch > 0 and abs(loss_history[-2] - epoch_loss) < self.tol:
                print(f"Converged at epoch {epoch+1}")
                break
        
        if x_plot is not None:
            plt.ioff()
            plt.show()
        
        return params, loss_history            
        

class ADAM:
    def __init__(self, model, init_params, loss_fn, lr=0.001, max_iter=1000, batch_size=20, tol=1e-6, h=1e-7, beta1=0.9, beta2=0.999, epsilon=1e-8, verbose=True):
        """
        Parameters:
            model: function
                Model function
            init_params: float array
                Array of inital param guess
            loss_fn: function
                Loss function
            lr : float
                Learning rate.
            max_iter : int
                Maximum number of iterations (epochs).
            batch_size : int
                Number of samples per mini-batch. If set to 1, it's pure SGD.
            tol : float
                Tolerance for stopping criterion based on loss improvement.
             beta1: float
                Exponential decay rate for the first moment estimates.
            beta2: float
                Exponential decay rate for the second moment estimates.
            epsilon: float
                Small constant for numerical stability.
            verbose : bool
                If True, print loss info during training.
        """
        self.model = model
        self.init_params = init_params
        self.loss_fn = loss_fn
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.verbose = verbose
        self.h = h
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
    def numerical_grad(self, params, x, y):
        """
        Compute the gradient of the loss with respect to parameters numerically
        using central differences.

        Parameters:
            params : np.ndarray
                Current parameters.
            x : np.ndarray
                Input data for the current batch.
            y : np.ndarray
                Observed targets for the current batch.
            loss_fn : callable
                Function that computes the loss given (predictions, y).

        Returns:
            grad : np.ndarray
                Approximated gradient vector.
        """
        
        grad = np.zeros_like(params)
        loss = self.loss_fn(self.model(params,x),y)
        
        for i in range(len(params)):
            params_perturbed = params.copy()
            params_perturbed[i] += self.h
            loss_perturbed = self.loss_fn(self.model(params_perturbed,x),y)
            grad[i] = (loss_perturbed - loss)/ self.h
        
        return grad
    
    def optimize(self, x, y, x_plot=None, visualize=False):
        
        
        # Set up visualization if desired
        if visualize:
            if x_plot is None:
                # If no plotting grid is provided, use the x data for visualization.
                x_plot = x
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
        
        n_samples = x.shape[0]
        loss_history = []
        params = self.init_params
        
        # Initialize first and second moment estimates
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        t = 0
        
        
        for epoch in range(self.max_iter):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            for i in range(0, n_samples, self.batch_size):
                x_batch =x_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                
                #Compute predictions and loss for current batch 
                preds = self.model(params,x_batch)
                loss = self.loss_fn(preds,y_batch)
                epoch_loss += loss
                
                #Compute gradient
                grad = self.numerical_grad(params,x_batch,y_batch)
                
                t += 1
                # Update biased first moment estimate
                m = self.beta1 * m + (1 - self.beta1) * grad
                # Update biased second raw moment estimate
                v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
                # Compute bias-corrected first moment estimate
                m_hat = m / (1 - self.beta1 ** t)
                # Compute bias-corrected second raw moment estimate
                v_hat = v / (1 - self.beta2 ** t )
                
                # Update parameters
                params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            #Avarage loss over batch in this epoch 
            epoch_loss /= (n_samples/ self.batch_size)
            loss_history.append(epoch_loss)
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.max_iter} - Loss: {epoch_loss:.6f}")
                
            # Visualization update (if requested)
            if x_plot is not None:
                ax.clear()
                ax.scatter(x, y, label="Data", color="blue")
                # Compute model predictions on the plotting grid
                y_est = self.model(params, x_plot)
                ax.plot(x_plot, y_est, label="Model Estimate", color="red")
                ax.set_title(f"Epoch {epoch+1} - Loss: {epoch_loss:.6f}")
                ax.legend()
                plt.pause(0.1)
            
              # Early stopping based on loss improvement
            if epoch > 0 and abs(loss_history[-2] - epoch_loss) < self.tol:
                print(f"Converged at epoch {epoch+1}")
                break
        
        if x_plot is not None:
            plt.ioff()
            plt.show()
        
        return params, loss_history     