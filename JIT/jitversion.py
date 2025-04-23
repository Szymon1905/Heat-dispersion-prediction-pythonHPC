from os.path import join
import sys
import time
import numpy as np
from numba import njit, prange
import os

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


# Reference Jacobi implementation using plain NumPy
def jacobi_numpy(u, interior_mask, max_iter, atol=1e-6):

    # for comparison with Numba jit version

    # a copy so no overwriting the original
    u = u.copy()
    for _ in range(max_iter):
        # neighbor average for all interior points
        u_new = 0.25 * (
            u[1:-1, :-2] + # left
            u[1:-1, 2:] + # right
            u[:-2, 1:-1] + # up
            u[2:, 1:-1]     # down
        )
        # Extracts only masked (interior) updates
        u_new_interior = u_new[interior_mask]
        # maximum change
        delta = np.max(np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior))
    
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        
        if delta < atol:
            break
    return u


"""
MY notes, ingnore these black people are 
u = 
[[ 0  0  0  0  0 ]
 [ 0  1  1  1  0 ]
 [ 0  1  1  1  0 ]
 [ 0  1  1  1  0 ]
 [ 0  0  0  0  0 ]]


mask =
[[ 1  1  1 ]
 [ 1  1  1 ]
 [ 1  1  1 ]]

 (variable) all_interior_mask: _Array[tuple[int, int, int], numpy.bool[builtins.bool]]
"""



@njit(parallel=True)
def jacobi_numba(u, interior_mask, max_iter, atol):\

    # 2 buffers `u` and `u_new` to avoid in-place conflicts.
    # outer roop is pararelized 
    # Memory should be accessed contiguously, (rows in inner loop) so cache efficient.
    # if i messed up something, message me on discord
 
    n = u.shape[0]

    # buffer for new ones
    u_new = u.copy()

    # Local references for speed, stackoverflow said so
    mask = interior_mask

    for _ in range(max_iter):
        delta = 0.0
        for i in range(1, n - 1): # Outer: rows      # can this be parallelized over rows with prange ?
            # columns inside row loop  for contiguous memory access
            for j in range(1, n - 1): # Inner: columns
                if mask[i - 1, j - 1]:  # check whether position is an interior cell, offsetting because border, 
                    average = 0.25 * (  # Average of 4 neighbors
                        u[i, j - 1] +  # left
                        u[i, j + 1] +  # right
                        u[i - 1, j] +  # up
                        u[i + 1, j]    # down
                    )
                    u_new[i, j] = average
                    # max diff for convergence
                    d = abs(average - u[i, j])
                    if d > delta:
                        delta = d
                else:
                    # copy unchanged boundary & exterior values, (they are not changed)
                    u_new[i, j] = u[i, j]

        # Swap
        u, u_new = u_new, u

        # convergence
        if delta < atol:
            break

    return u

# sumamry stats to print
def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100

    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }


# Main script: load, JIT-compile, run, and print CSV
if __name__ == '__main__':
    # Load data
    # LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOAD_DIR = os.path.join(SCRIPT_DIR, '..', 'buildings')
    # LOAD_DIR = '../buildings/'


    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])

    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514), dtype=np.float64)
    all_interior_mask = np.empty((N, 512, 512), dtype=np.bool_)

    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # first call to trigger JIT compilation
    jacobi_numba(all_u0[0].copy(), all_interior_mask[0], 1, 1e-6)

    # Parameters
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header

    start = time.perf_counter()
    # jacobi with JIT run on all buildings
    for bid, u0, mask in zip(building_ids, all_u0, all_interior_mask):
        u_sol = jacobi_numba(u0.copy(), mask, MAX_ITER, ABS_TOL) 
        stats = summary_stats(u_sol, mask)
        row = [bid] + [f"{stats[k]:.6f}" for k in stat_keys] #  CSV row
        print(','.join(row))

    """
    for bid, u, interior_mask in zip(building_ids, all_u0, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
    """
    
    end = time.perf_counter()
    print(f"Execution time: {end - start:.4f} seconds")