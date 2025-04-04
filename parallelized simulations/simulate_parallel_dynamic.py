from os.path import join
import sys
import csv
import numpy as np
import multiprocessing

#### Global variables
# Run jacobi iterations for each floor plan
MAX_ITER = 20_000
ABS_TOL = 1e-4
LOCAL = True
CSV = False
num_processes = max(int(sys.argv[2]),1)
if (len(sys.argv) > 3):
    CSV = True if sys.argv[3].lower() == "y" else False


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(zipped_args):
    u, interior_mask = zipped_args
    u = np.copy(u)
    max_iter = MAX_ITER
    atol = ABS_TOL

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
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
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    ##### FOR TESTING
    LOAD_DIR = "buildings/" if LOCAL else LOAD_DIR

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


    chunksize = 1 # Keep chunksize as 1 for dynamic scheduling
    with multiprocessing.Pool(num_processes) as pool:
        all_u = pool.map(jacobi, zip(all_u0, all_interior_mask), chunksize=chunksize)
    
    # Print summary statistics in CSV format
    data = []

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    header = ['building_id'] +stat_keys
    print(header)  # CSV header
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
        stats.update({"building_id": bid})
        data.append(stats)
    # if CSV option is true
    if (CSV):
        with open('floorplan_results.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            writer.writerows(data)