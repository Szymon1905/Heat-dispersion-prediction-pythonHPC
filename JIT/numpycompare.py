from os.path import join
import sys
import time
import numpy as np
import os

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

def jacobi_numpy(u, interior_mask, max_iter, atol=1e-6):

    # for comparison with Numba jit version

    # a copy so no overwriting the original
    u = u.copy()
    for _ in range(max_iter):
        # neighbor average for all interior points
        u_new = 0.25 * (
            u[1:-1, :-2] + u[1:-1, 2:] +  # left, right
            u[:-2, 1:-1] + u[2:, 1:-1]     # up, down
        )
        # Extracts only masked (interior) updates
        u_new_interior = u_new[interior_mask]
        # maximum change
        delta = np.max(np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior))
    
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        
        if delta < atol:
            break
    return u

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

if __name__ == '__main__':
    # Load data
    # LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOAD_DIR = os.path.join(SCRIPT_DIR, '..', 'buildings')

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')

    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    start = time.perf_counter()

    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi_numpy(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header

    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))

    end = time.perf_counter()
    print(f"Execution time of original: {end - start:.2f} seconds")