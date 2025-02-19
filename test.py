from LM import LM
import numpy as np

if __name__ == "__main__":
    # Define model
    def model(params, x):
        a, b, c = params
        return a * np.exp(-b * x) + c

    # Generate synthetic data
    true_params = [4.0, 2.5, -1.0]
    x_data = np.linspace(0, 10, 50)
    y_data = model(true_params, x_data) + np.random.normal(scale=0.5, size=x_data.shape)

    # Initial guess for the parameters
    init_guess = np.array([1.0, 1.0, 1.0])

    # Create the LM optimizer instance
    lm = LM(model=model, init_guess=init_guess, max_iter=100, tol=1e-9)

    # Optimize with visualization (using a finer grid for plotting)
    x_plot = np.linspace(0, 10, 200)
    estimated_params = lm.optimize(x_data, y_data, x_plot=x_plot, visualize=True)
    
    print("Estimated Parameters:", estimated_params)