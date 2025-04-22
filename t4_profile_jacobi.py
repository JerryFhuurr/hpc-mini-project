import numpy as np

@profile
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

# Add simple test run
if __name__ == '__main__':
    domain = np.load("modified_swiss_dwellings/114_domain.npy")
    interior = np.load("modified_swiss_dwellings/114_interior.npy")

    u = np.zeros((514, 514))
    u[1:-1, 1:-1] = domain

    jacobi(u, interior)
