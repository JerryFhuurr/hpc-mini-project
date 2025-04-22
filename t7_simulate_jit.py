import numpy as np
import os
import time
from numba import njit

LOAD_DIR = "modified_swiss_dwellings"
building_ids = sorted(set(f.split('_')[0] for f in os.listdir(LOAD_DIR) if f.endswith("_domain.npy")))
N = 10  # How many buildings to process
building_ids = building_ids[:N]

@njit
def jacobi_jit(u, interior_mask, max_iter=20000, atol=1e-4):
    u = u.copy()
    for _ in range(max_iter):
        delta = 0.0
        u_new = u.copy()

        for i in range(1, u.shape[0] - 1):
            for j in range(1, u.shape[1] - 1):
                if interior_mask[i - 1, j - 1]:
                    val = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
                    delta = max(delta, abs(val - u[i, j]))
                    u_new[i, j] = val

        u = u_new
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

def load_data(bid):
    u = np.zeros((514, 514))
    u[1:-1, 1:-1] = np.load(os.path.join(LOAD_DIR, f"{bid}_domain.npy"))
    interior_mask = np.load(os.path.join(LOAD_DIR, f"{bid}_interior.npy"))
    return u, interior_mask

# Run and time
start = time.time()
for bid in building_ids:
    u0, interior_mask = load_data(bid)
    u = jacobi_jit(u0, interior_mask)
    stats = summary_stats(u, interior_mask)
    print(f"{bid},", ", ".join(f"{v:.2f}" for v in stats.values()))
print(f"\nProcessed {N} buildings in {time.time() - start:.2f} seconds")
