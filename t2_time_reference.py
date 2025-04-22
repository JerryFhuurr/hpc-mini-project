import os
import time
import numpy as np

LOAD_DIR = "modified_swiss_dwellings"
building_ids = [fname.split('_')[0] for fname in os.listdir(LOAD_DIR) if fname.endswith("_domain.npy")]
building_ids = sorted(set(building_ids))[:20]  # Adjust N here

def load_data(bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(os.path.join(LOAD_DIR, f"{bid}_domain.npy"))
    interior_mask = np.load(os.path.join(LOAD_DIR, f"{bid}_interior.npy"))
    return u, interior_mask

def jacobi(u, interior_mask, max_iter=20000, atol=1e-4):
    u = np.copy(u)
    for _ in range(max_iter):
        u_new = 0.25 * (
            u[1:-1, :-2] + u[1:-1, 2:] +
            u[:-2, 1:-1] + u[2:, 1:-1]
        )
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

# Run simulation and time it
start = time.time()
for bid in building_ids:
    u0, mask = load_data(bid)
    u = jacobi(u0, mask)
    stats = summary_stats(u, mask)
    print(f"{bid},", ", ".join(f"{v:.2f}" for v in stats.values()))
elapsed = time.time() - start
print(f"\nProcessed {len(building_ids)} buildings in {elapsed:.2f} seconds")
