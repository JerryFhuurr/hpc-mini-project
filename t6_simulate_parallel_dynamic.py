import numpy as np
import os
import time
from multiprocessing import Pool, cpu_count

LOAD_DIR = "modified_swiss_dwellings"
MAX_ITER = 20000
ABS_TOL = 1e-4

def get_building_ids(n):
    files = os.listdir(LOAD_DIR)
    ids = sorted(set(f.split('_')[0] for f in files if f.endswith("_domain.npy")))
    return ids[:n]

def load_data(bid):
    u = np.zeros((514, 514))
    u[1:-1, 1:-1] = np.load(os.path.join(LOAD_DIR, f"{bid}_domain.npy"))
    interior = np.load(os.path.join(LOAD_DIR, f"{bid}_interior.npy"))
    return u, interior

def jacobi(u, interior_mask, max_iter=MAX_ITER, atol=ABS_TOL):
    u = np.copy(u)
    for _ in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if delta < atol:
            break
    return u

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    return {
        "mean_temp": u_interior.mean(),
        "std_temp": u_interior.std(),
        "pct_above_18": (u_interior > 18).sum() / u_interior.size * 100,
        "pct_below_15": (u_interior < 15).sum() / u_interior.size * 100,
    }

def process_building(bid):
    u0, interior_mask = load_data(bid)
    u = jacobi(u0, interior_mask)
    stats = summary_stats(u, interior_mask)
    return (bid, stats)

if __name__ == "__main__":
    N = 100  # Max number of buildings to test
    building_ids = get_building_ids(N)

    print(f"Using {cpu_count()} CPUs...")
    start = time.time()
    with Pool(processes=cpu_count()) as pool:
        results = list(pool.imap_unordered(process_building, building_ids))
    end = time.time()

    print("building_id, mean_temp, std_temp, pct_above_18, pct_below_15")
    for bid, stats in results:
        print(f"{bid},", ", ".join(f"{v:.2f}" for v in stats.values()))
    print(f"\nProcessed {N} buildings in {end - start:.2f} seconds")
