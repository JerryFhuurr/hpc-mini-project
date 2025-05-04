from os.path import join
import sys
from multiprocessing import Pool
import numpy as np
import time
import numba


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2), dtype=np.float32)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy")).astype(np.float32)
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@numba.jit(nopython=True, cache=True, fastmath=True)
def jacobi(u_flat, J, flat_idx, max_iter, atol=1e-6):
    #n = u.size
    n = u_flat.size
    #u_flat = u.reshape(n)    
    u_new  = u_flat.copy()
    #J = u.shape[1]
    #flat_idx = I_idx * 514 + J_idx
    npts = flat_idx.shape[0]

    for _ in range(max_iter):
        delta = 0.0
        for idx in range(npts):
            p = flat_idx[idx]
            val = 0.25 * (u_flat[p-J] + u_flat[p+J] + u_flat[p-1] + u_flat[p+1])
            u_new[p] = val
            diff = abs(val - u_flat[p])
            if diff > delta:
                delta = diff
 
        u_flat, u_new = u_new, u_flat
        if delta < atol:
            break
    #return u_flat.reshape(u.shape)
    return u_flat


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
    mi, mj = np.where(interior_mask)
    I_idx = mi + 1
    J_idx = mj + 1  
    flat_idx = I_idx * 514 + J_idx
    u_flat = u.ravel().astype(np.float32)    
    u_flat = jacobi(u_flat, u.shape[1], flat_idx, max_iter, atol)
    u = u_flat.reshape(u.shape)
    stats = summary_stats(u, interior_mask)
    return bid, stats


if __name__ == '__main__':
    # Load data
    LOAD_DIR = 'C:/Users/14349/hpc/123/modified_swiss_dwellings/'
    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    N = 4571
    n_processor_list = np.array((32, ))
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
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, stats in results:
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))