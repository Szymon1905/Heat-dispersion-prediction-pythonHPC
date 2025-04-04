from os.path import join
import numpy as np
import csv
import sys
from multiprocessing import Pool, cpu_count

LOCAL = False
to_csv = False

LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
LOAD_DIR = "buildings/" if LOCAL else LOAD_DIR


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


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


def process_building(bid):
    u0, interior_mask = load_data(LOAD_DIR, bid)
    u = jacobi(u0, interior_mask, max_iter=20_000, atol=1e-4)
    stats = summary_stats(u, interior_mask)
    return bid, stats


if __name__ == '__main__':
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 5
    else:
        N = int(sys.argv[1])

    building_ids = building_ids[:N]

    num_workers = int(sys.argv[2]) #min(cpu_count(), N)
    chunk_size = (N + num_workers - 1) // num_workers  # static scheduling

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_building, building_ids, chunksize=chunk_size)

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))
    for bid, stats in results:
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))

    if to_csv:
        with open('stats.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['building_id'] + stat_keys)
            for bid, stats in results:
                writer.writerow([bid] + [stats[k] for k in stat_keys])
