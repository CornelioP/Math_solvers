import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the objective function
def f(x, y):
    "Objective function"
    return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)

# Define the gradient of f (analytical derivatives)
def grad_f(x, y):
    "Gradient of f with respect to x and y"
    df_dx = 2*(x-3.14) + 3*np.cos(3*x+1.41)
    df_dy = 2*(y-2.72) + 4*np.cos(4*y-1.73)
    return np.array([df_dx, df_dy])

# Create a grid and compute f for contour plotting
x_grid, y_grid = np.meshgrid(np.linspace(0,5,100), np.linspace(0,5,100))
z = f(x_grid, y_grid)

# Compute the global minimum on the grid for reference
z_flat = z.ravel()
x_flat = x_grid.ravel()
y_flat = y_grid.ravel()
x_min = x_flat[z_flat.argmin()]
y_min = y_flat[z_flat.argmin()]

# Hyper-parameters for SGD
lr = 0.05  # Learning rate

# Initialize the point (starting from a random location)
np.random.seed(100)
current_point = np.random.rand(2) * 5
points_history = [current_point.copy()]  # To store the trajectory

def update():
    "Perform one SGD iteration"
    global current_point
    grad = grad_f(current_point[0], current_point[1])
    # Update the point by moving in the negative gradient direction
    current_point = current_point - lr * grad
    points_history.append(current_point.copy())

# Set up the base figure: the contour map
fig, ax = plt.subplots(figsize=(8,6))
fig.set_tight_layout(True)
img = ax.imshow(z, extent=[0, 5, 0, 5], origin='lower', cmap='viridis', alpha=0.5)
fig.colorbar(img, ax=ax)
ax.plot([x_min], [y_min], marker='x', markersize=5, color="white", label="Global Minimum")
contours = ax.contour(x_grid, y_grid, z, 10, colors='black', alpha=0.4)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
point_plot, = ax.plot([], [], marker='o', color='blue', markersize=8, label="Current Point")
traj_plot, = ax.plot([], [], marker='o', color='red', linestyle='--', alpha=0.7, label="Trajectory")
ax.set_xlim([0,5])
ax.set_ylim([0,5])
ax.set_title('SGD Optimization')
ax.legend()

def animate(i):
    "Animate one iteration of SGD and update the plot"
    update()
    title = 'Iteration {:02d}'.format(i)
    ax.set_title(title)
    # Update current point and trajectory
    traj = np.array(points_history)
    traj_plot.set_data(traj[:,0], traj[:,1])
    point_plot.set_data([current_point[0]], [current_point[1]])
    return point_plot, traj_plot

# Use Pillow writer instead of ImageMagick
anim = FuncAnimation(fig, animate, frames=range(1, 50), interval=500, blit=False, repeat=True)
anim.save("images/SGD.gif", dpi=120, writer="pillow")

final_obj = f(current_point[0], current_point[1])
print("SGD found best solution at f({})={}".format(current_point, final_obj))
print("Global optimal on grid at f({})={}".format([x_min, y_min], f(x_min, y_min)))
