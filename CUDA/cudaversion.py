from os.path import join
import sys
import time
import numpy as np
from numba import cuda
import os


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

@cuda.jit
def jacobi_kernel(u, interior_mask, u_new):
    x, y = cuda.grid(2)

    # 1) check for out of bounds
    if 1 <= x < u.shape[0]-1 and 1 <= y < u.shape[1]-1:
        # only write to interior cells
        if interior_mask[x-1, y-1]:
            # per thread, not vectorized like numpy
            left  = u[x, y-1]
            right = u[x, y+1]
            up    = u[x-1, y]
            down  = u[x+1, y]

            u_new[x, y] = 0.25 * (left + right + up + down)
            """
            u_new[x,y] = 0.25 * (
                u[1:-1, :-2] + u[1:-1, 2:] +  # left, right
                u[:-2, 1:-1] + u[2:, 1:-1]     # up, down
            )
            """
            
    # computed u_new is written to memory of param 


def jacobi_helper(u, interior_mask, max_iter):
    # copy data to gpu
    cuda_u = cuda.to_device(u)
    cuda_interior_mask = cuda.to_device(interior_mask)
    cuda_u_new = cuda.device_array_like(cuda_u)

    # copy all data because exterior not carry over later and device_array_like does not make copy apparently
    cuda_u_new[:] = cuda_u[:]

    # setup params
    threadsperblock = (16, 16)
    blockspergrid_x = (cuda_u.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (cuda_u.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    # Round up division to cover full domain
    blockspergrid = (blockspergrid_x,blockspergrid_y)

    # loop with kernel, one kernel does one iteration
    for i in range(max_iter):
        jacobi_kernel[blockspergrid, threadsperblock](cuda_u, cuda_interior_mask, cuda_u_new)
        cuda_u, cuda_u_new = cuda_u_new, cuda_u
    
    # data retued to host
    return cuda_u.copy_to_host()

# summary stats to print
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
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    #SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    #LOAD_DIR = os.path.join(SCRIPT_DIR, '..', 'buildings')

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()
    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]
    all_u0 = np.empty((N, 514, 514), dtype=np.float64)
    all_interior_mask = np.empty((N, 512, 512), dtype=np.bool_)
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask
    
    MAX_ITER = 20_000
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header

    start = time.perf_counter()

    # loop to measure
    for bid, u0, mask in zip(building_ids, all_u0, all_interior_mask):

        u_sol = jacobi_helper(u0, mask, MAX_ITER) 

        stats = summary_stats(u_sol, mask)
        row = [bid] + [f"{stats[k]:.6f}" for k in stat_keys] #  CSV row
        print(','.join(row))

    end = time.perf_counter()
    print(f"Execution time: {end - start:.4f} seconds")
    
