import os
import cupy as cp
import numpy as np  # Used only for np.load
import time

# Configuration
LOAD_DIR = "modified_swiss_dwellings"
MAX_ITER = 20000
ABS_TOL = 1e-4
N = 10  # Number of buildings to simulate

# Get N building IDs
building_ids = sorted({
    fname.split('_')[0] for fname in os.listdir(LOAD_DIR)
    if fname.endswith('_domain.npy')
})[:N]

def load_data(bid):
    """Load simulation grid and interior mask for one building."""
    u = cp.zeros((514, 514), dtype=cp.float32)
    domain = np.load(os.path.join(LOAD_DIR, f"{bid}_domain.npy"))
    mask = np.load(os.path.join(LOAD_DIR, f"{bid}_interior.npy"))

    u[1:-1, 1:-1] = cp.array(domain, dtype=cp.float32)
    interior_mask = cp.array(mask, dtype=cp.bool_)
    return u, interior_mask

def jacobi_cupy(u, interior_mask, max_iter=MAX_ITER, atol=ABS_TOL):
    """Run the Jacobi iteration on GPU using CuPy."""
    for _ in range(max_iter):
        u_new = 0.25 * (
            u[1:-1, :-2] + u[1:-1, 2:] +
            u[:-2, 1:-1] + u[2:, 1:-1]
        )
        delta = cp.abs(u[1:-1, 1:-1][interior_mask] - u_new[interior_mask]).max()
        u[1:-1, 1:-1][interior_mask] = u_new[interior_mask]
        if delta < atol:
            break
    return u

def summary_stats(u, interior_mask):
    """Calculate summary statistics on CPU."""
    u_interior = cp.asnumpy(u[1:-1, 1:-1][interior_mask])
    return {
        "mean_temp": u_interior.mean(),
        "std_temp": u_interior.std(),
        "pct_above_18": (u_interior > 18).sum() / u_interior.size * 100,
        "pct_below_15": (u_interior < 15).sum() / u_interior.size * 100,
    }

# === Run Simulation ===
start = time.time()
print("building_id, mean_temp, std_temp, pct_above_18, pct_below_15")
for bid in building_ids:
    u0, mask = load_data(bid)
    u_final = jacobi_cupy(u0, mask)
    stats = summary_stats(u_final, mask)
    print(f"{bid},", ", ".join(f"{v:.2f}" for v in stats.values()))
print(f"\nProcessed {N} buildings in {time.time() - start:.2f} seconds")
