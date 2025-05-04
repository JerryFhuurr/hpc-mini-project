import numpy as np
import os
import time
#from numba import njit
import numba

LOAD_DIR = "C:/Users/14349/hpc/123/modified_swiss_dwellings"
building_ids = sorted(set(f.split('_')[0] for f in os.listdir(LOAD_DIR) if f.endswith("_domain.npy")))
N = 10  # How many buildings to process
building_ids = building_ids[:N]

@numba.jit(nopython=True, cache=True, fastmath=True)
def jacobi_jit(u, I_idx, J_idx, max_iter=20000, atol=1e-4):
    n = u.size
    u_flat = u.reshape(n)    
    u_new  = u_flat.copy()
    J = u.shape[1]
    flat_idx = I_idx * 514 + J_idx
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
    return u_flat.reshape(u.shape)

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
    mi, mj = np.where(interior_mask)
    I_idx = mi + 1
    J_idx = mj + 1  
    u = jacobi_jit(u0, I_idx, J_idx)
    stats = summary_stats(u, interior_mask)
    print(f"{bid},", ", ".join(f"{v:.2f}" for v in stats.values()))
print(f"\nProcessed {N} buildings in {time.time() - start:.2f} seconds")
