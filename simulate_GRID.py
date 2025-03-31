import numpy as np
import matplotlib.pyplot as plt
from os.path import join

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

def plot_grid(grid, title, filename):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='hot', origin='lower')
    plt.colorbar(label='Temperature (°C)')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)
    for i in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if delta < atol:
            break
    return u

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    BUILDING_ID = '9991'  # here goes building id to visualize
    
    u0, interior_mask = load_data(LOAD_DIR, BUILDING_ID)
    plot_grid(u0, 'Initial Condition', 'initial_condition.png')
    
    u_final = jacobi(u0, interior_mask, max_iter=10000)
    plot_grid(u_final, 'Final Temperature Distribution', 'final_temperature.png')