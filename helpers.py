import os
import numpy as numpy
import matplotlib.pyplot as plt
import tqdm

def plot_trajectory_on_landscape(model, triplets, weight_history, resolution=30, range_=1.0):
    """
    weight_history: list of weight arrays saved every N epochs
    plots the 2D loss landscape and overlays the optimisation trajectory
    """
    # use first and last weights to define the trajectory direction
    w_start = weight_history[0]
    w_end = weight_history[-1]
    
    # direction 1: from start to end (main trajectory direction)
    d1 = w_end - w_start
    d1 = d1 / numpy.linalg.norm(d1)
    
    # direction 2: random orthogonal direction
    d2 = numpy.random.randn(*d1.shape)
    d2 = d2 - numpy.dot(d2.flatten(), d1.flatten()) * d1
    d2 = d2 / numpy.linalg.norm(d2)
    
    # project each checkpoint onto (d1, d2) plane
    trajectory = []
    for w in weight_history:
        delta = w - w_start
        x = numpy.dot(delta.flatten(), d1.flatten())
        y = numpy.dot(delta.flatten(), d2.flatten())
        trajectory.append((x, y))
    
    # compute loss landscape around the trajectory
    # centre the grid on the midpoint of the trajectory
    mid = weight_history[len(weight_history)//2]
    alphas = numpy.linspace(-range_, range_, resolution)
    betas  = numpy.linspace(-range_, range_, resolution)
    loss_grid = numpy.zeros((resolution, resolution))
    
    batch_idx = numpy.random.randint(0, len(triplets), 16)
    batch = [triplets[i] for i in batch_idx]
    
    for i, alpha in enumerate(tqdm(alphas, desc="Landscape")):
        for j, beta in enumerate(betas):
            w = mid + alpha * d1 + beta * d2
            loss_grid[i, j] = float(model.loss(w, batch)[0])
    
    # plot
    fig, ax = plt.subplots(figsize=(10, 8))
    A, B = numpy.meshgrid(alphas, betas)
    cf = ax.contourf(A, B, loss_grid.T, levels=40, cmap='viridis', alpha=0.8)
    ax.contour(A, B, loss_grid.T, levels=40, colors='white', alpha=0.2, linewidths=0.5)
    plt.colorbar(cf, ax=ax)
    
    # overlay trajectory
    xs = [t[0] for t in trajectory]
    ys = [t[1] for t in trajectory]
    ax.plot(xs, ys, 'r-', linewidth=2, alpha=0.8, label='Optimisation path')
    ax.scatter(xs[0],  ys[0],  color='lime',   s=150, zorder=5, label='Start')
    ax.scatter(xs[-1], ys[-1], color='red',    s=150, zorder=5, label='End')
    
    # colour points by epoch
    scatter = ax.scatter(xs, ys, c=range(len(xs)), cmap='autumn', s=50, zorder=6)
    plt.colorbar(scatter, ax=ax, label='Epoch checkpoint')
    
    ax.set_xlabel('Direction 1 (start→end)')
    ax.set_ylabel('Direction 2 (orthogonal)')
    ax.set_title('Loss landscape with optimisation trajectory')
    ax.legend()
    plt.tight_layout()
    
    path = os.path.join(model.results_dir, 'trajectory.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")