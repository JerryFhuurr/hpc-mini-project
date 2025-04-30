from os.path import join
import sys
from multiprocessing import Pool
import numpy as np
import time


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

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

def process_building(args):
    load_dir, bid, max_iter, atol = args
    u, interior_mask = load_data(load_dir, bid)
    u = jacobi(u, interior_mask, max_iter, atol)
    stats = summary_stats(u, interior_mask)
    return bid, stats


if __name__ == '__main__':
    # Load data
    LOAD_DIR = 'modified_swiss_dwelling'
    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    N = 16
    n_processor_list = np.array((1, 2, 4, 8))
    runtime = np.zeros((np.size(n_processor_list)))

    for num in range(np.size(n_processor_list)):
        n_processor = n_processor_list[num]

        building_ids = building_ids[:N]


        tasks_per_worker = len(building_ids) // n_processor
        tasks = [(LOAD_DIR, bid, MAX_ITER, ABS_TOL) for bid in building_ids]
        print(f"n_processors: {n_processor}, total tasks: {N}")
        start = time.perf_counter()
        with Pool(processes=n_processor) as pool:
            results = pool.map(process_building, tasks)
        end = time.perf_counter()
        runtime[num] = end - start
        print(f"run time: {end - start}")












    # Print summary statistics in CSV format
    #stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    #print('building_id, ' + ', '.join(stat_keys))  # CSV header
    #for bid, stats in results:
        #print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))