from os.path import join
import sys
import time
import numpy as np
from numba import njit, prange


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


# Reference Jacobi implementation using plain NumPy
def jacobi_numpy(u, interior_mask, max_iter, atol=1e-6):
    """
    Performs Jacobi relaxation using array slicing and boolean masking.
    This serves as a reference for correctness and rough performance.
    """
    # a copy so we don't overwrite the original
    u = u.copy()
    for _ in range(max_iter):
        # Compute neighbor average for all interior points
        u_new = 0.25 * (
            u[1:-1, :-2] + u[1:-1, 2:] +  # left, right
            u[:-2, 1:-1] + u[2:, 1:-1]     # up, down
        )
        # Extract only masked (interior) updates
        new_vals = u_new[interior_mask]
        # Compute maximum change
        delta = np.max(np.abs(u[1:-1, 1:-1][interior_mask] - new_vals))
        # Write back only interior cells
        u[1:-1, 1:-1][interior_mask] = new_vals
        # Check convergence
        if delta < atol:
            break
    return u

# ----------------------------------------------------------------------------
# Accelerated Jacobi using Numba JIT on CPU
# ----------------------------------------------------------------------------
@njit(parallel=True)
def jacobi_numba(u, interior_mask, max_iter, atol):
    """
    Numba-compiled Jacobi relaxation.

    - Uses two buffers `u` and `u_new` to avoid in-place conflicts.
    - Parallelizes outer loop with `prange`.
    - Accesses contiguous rows in inner loop for cache efficiency.
    """
    n = u.shape[0]
    # Allocate secondary buffer
    u_new = u.copy()
    # Local references for speed
    mask = interior_mask

    for _ in range(max_iter):
        delta = 0.0
        # Parallel over rows (1..n-2)
        for i in prange(1, n - 1):
            # Row-major inner loop for contiguous memory access
            for j in range(1, n - 1):
                if mask[i - 1, j - 1]:  # interior cell?
                    # Average of 4 neighbors
                    val = 0.25 * (
                        u[i, j - 1] + u[i, j + 1] +  # left, right
                        u[i - 1, j] + u[i + 1, j]     # up, down
                    )
                    u_new[i, j] = val
                    # Track max difference for convergence
                    d = abs(val - u[i, j])
                    if d > delta:
                        delta = d
                else:
                    # Preserve boundary & exterior values
                    u_new[i, j] = u[i, j]

        # Swap buffers
        u, u_new = u_new, u
        # Check convergence
        if delta < atol:
            break

    return u

# ----------------------------------------------------------------------------
# Compute summary statistics over the interior points
# ----------------------------------------------------------------------------
def summary_stats(u, interior_mask):
    """
    Compute mean, std, and percentage of points above/below thresholds.

    Returns:
        dict with keys: mean_temp, std_temp, pct_above_18, pct_below_15
    """
    # Extract interior (unpadded) region and apply mask
    grid = u[1:-1, 1:-1]
    vals = grid[interior_mask]

    return {
        'mean_temp': np.mean(vals),
        'std_temp': np.std(vals),
        'pct_above_18': np.sum(vals > 18) / vals.size * 100,
        'pct_below_15': np.sum(vals < 15) / vals.size * 100,
    }

# ----------------------------------------------------------------------------
# Main script: load, JIT-compile, run, and print CSV
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # Directory containing data files
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

    # Read building IDs
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    # Number of buildings to process (default: 1)
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    building_ids = building_ids[:N]

    # Preload all initial states and masks
    all_u0 = np.empty((N, 514, 514), dtype=np.float64)
    all_masks = np.empty((N, 512, 512), dtype=np.bool_)
    for idx, bid in enumerate(building_ids):
        u0, mask = load_data(LOAD_DIR, bid)
        all_u0[idx] = u0
        all_masks[idx] = mask

    # Warm up Numba (first call triggers JIT compilation)
    jacobi_numba(all_u0[0].copy(), all_masks[0], 1, 1e-6)

    # Parameters
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    # Print CSV header
    keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id,' + ','.join(keys))

    # Run JIT-accelerated Jacobi on each building
    for bid, u0, mask in zip(building_ids, all_u0, all_masks):
        # Run solver
        u_sol = jacobi_numba(u0.copy(), mask, MAX_ITER, ABS_TOL)
        # Compute stats
        stats = summary_stats(u_sol, mask)
        # Output as CSV row
        row = [bid] + [f"{stats[k]:.6f}" for k in keys]
        print(','.join(row))
