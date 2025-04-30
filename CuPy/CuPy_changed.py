from os.path import join
import sys
import csv
import cupy as cp

LOCAL = False
CSV = True


def load_data(load_dir, bid):
    SIZE = 512
    u = cp.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = cp.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = cp.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u_batch, mask_batch, max_iter, atol=1e-6):
    u = cp.copy(u_batch)

    for _ in range(max_iter):
        u_new = 0.25 * (
            u[:, 1:-1, :-2] + u[:, 1:-1, 2:] +
            u[:, :-2, 1:-1] + u[:, 2:, 1:-1]
        )
        u_interior = u[:, 1:-1, 1:-1]
        delta = cp.abs(u_interior[mask_batch] - u_new[mask_batch]).max()
        u_interior[mask_batch] = u_new[mask_batch]

        if delta < atol:
            break
    return u


def summary_stats(all_u, all_mask):
    u_interior = all_u[:, 1:-1, 1:-1]
    masked = cp.where(all_mask, u_interior, cp.nan)

    mean_temp = cp.nanmean(masked, axis=(1, 2))
    std_temp = cp.nanstd(masked, axis=(1, 2))
    pct_above_18 = cp.nanmean(masked > 18, axis=(1, 2)) * 100
    pct_below_15 = cp.nanmean(masked < 15, axis=(1, 2)) * 100

    return [
        {
            'mean_temp': float(mean_temp[i]),
            'std_temp': float(std_temp[i]),
            'pct_above_18': float(pct_above_18[i]),
            'pct_below_15': float(pct_below_15[i]),
        }
        for i in range(all_u.shape[0])
    ]



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
    all_u0 = cp.empty((N, 514, 514))
    all_interior_mask = cp.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    all_u = jacobi(all_u0, all_interior_mask, MAX_ITER, ABS_TOL)


    # Print summary statistics in CSV format
    data =[]
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    header = ['building_id'] + stat_keys
    print(header)  # CSV header
    print('building_id, ' + ', '.join(stat_keys))  # CSV header

    stats_list = summary_stats(all_u, all_interior_mask)
    for bid, stats in zip(building_ids, stats_list):
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
        stats.update({"building_id": bid})
        data.append(stats)

    # if CSV option is true
    if (CSV):
        with open('floorplan_results.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            writer.writerows(data)