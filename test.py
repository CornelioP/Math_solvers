from Solver import LM, SGD
import numpy as np

# Define model
def model(params, x):
    a, b, c = params
    return a *x  + c + np.exp(-b * x)

def gen_syntetic_data(n_samples,true_params):

    x_data = np.linspace(-1, 10, n_samples)
    y_data = model(true_params, x_data) + np.random.normal(scale=0.5, size=x_data.shape)
    
    return x_data, y_data


def test_LM():
    
    # Generate synthetic data
    true_params = [4.0, 2.5, -1.0]
    n_samples = 100
    x_data , y_data = gen_syntetic_data(n_samples, true_params)

    # Initial guess for the parameters
    init_guess = np.array([3.0, 1.0, -1.0])

    # Create the LM optimizer instance
    lm = LM(model=model, init_guess=init_guess, max_iter=100, tol=1e-9)

    # Optimize with visualization (using a finer grid for plotting)
    x_plot = np.linspace(0, 10, 200)
    estimated_params = lm.optimize(x_data, y_data, x_plot=x_plot, visualize=True)
    
    print("Estimated Parameters:", estimated_params)

def test_SGD():
    
    # Generate synthetic data
    true_params = [4.0, 2.5, -1.0]
    n_samples = 100
    x_data , y_data = gen_syntetic_data(n_samples, true_params)
    
    # Initial guess
    init_guess = [1, 1, 1]
    
    # Loss fn
    def mse_loss(y_pred, y_true):
        return np.mean((y_pred.flatten() - y_true.flatten())**2)
    
    # Create instance of SGD
    sgd = SGD(model,init_guess,mse_loss)
    
    #Optimize
    estimated_params, loss = sgd.optimize(x_data,y_data,visualize=True)
    
    print("Estimated Parameters:", estimated_params)

if __name__ == "__main__":
    
    test_LM()
    test_SGD()