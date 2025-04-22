import numpy as np
import matplotlib.pyplot as plt
import os

LOAD_DIR = "modified_swiss_dwellings"
building_ids = ["114", "132", "201"]

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

# Plot results
for bid in building_ids:
    u0, mask = load_data(bid)
    u = jacobi(u0, mask)

    plt.figure(figsize=(6, 5))
    plt.imshow(u, cmap="hot", origin="upper")
    plt.colorbar(label="Temperature (°C)")
    plt.title(f"Final Temperature Distribution (Building {bid})")
    plt.tight_layout()
    plt.show()
