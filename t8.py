from os.path import join
import sys
import time

import numpy as np
from numba import cuda

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

@cuda.jit
def jacobi_kernel(u, u_new, interior_mask):
    # Get thread indices
    i, j = cuda.grid(2)
    
    # Check if indices are in bounds and if the point is an interior point
    if (i > 0 and i < u.shape[0]-1 and 
        j > 0 and j < u.shape[1]-1 and 
        interior_mask[i-1, j-1]):
        
        # Compute average of neighbors
        u_new[i, j] = 0.25 * (u[i, j-1] + u[i, j+1] + u[i-1, j] + u[i+1, j])

def jacobi_cuda(u, interior_mask, max_iter, atol=None):
    """
    CUDA implementation of the Jacobi method for solving Laplace's equation.
    This function runs a fixed number of iterations without checking for convergence.
    
    Args:
        u: Initial temperature grid (includes boundary)
        interior_mask: Boolean mask indicating interior points
        max_iter: Maximum number of iterations
        atol: Not used in this implementation
    
    Returns:
        Updated temperature grid after all iterations
    """
    # Make a copy of the input
    u = np.copy(u)
    
    # Allocate device memory and copy data to device
    d_u = cuda.to_device(u)
    d_u_new = cuda.device_array_like(d_u)
    d_interior_mask = cuda.to_device(interior_mask)
    
    # Define grid and block dimensions
    # Using 16x16 threads per block is a common choice for 2D problems
    block_dim = (16, 16)
    grid_dim = ((u.shape[0] + block_dim[0] - 1) // block_dim[0],
                (u.shape[1] + block_dim[1] - 1) // block_dim[1])
    
    # Initial copy of u to u_new to ensure non-interior points are set correctly
    d_u_new.copy_to_device(d_u)
    
    # Run kernel for specified number of iterations
    for i in range(max_iter):
        # Odd iterations: u -> u_new
        if i % 2 == 0:
            jacobi_kernel[grid_dim, block_dim](d_u, d_u_new, d_interior_mask)
        # Even iterations: u_new -> u
        else:
            jacobi_kernel[grid_dim, block_dim](d_u_new, d_u, d_interior_mask)
    
    # Copy result back to host
    # If max_iter is odd, the final result is in u_new, otherwise it's in u
    if max_iter % 2 == 1:
        result = d_u_new.copy_to_host()
    else:
        result = d_u.copy_to_host()
    
    return result

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

if __name__ == '__main__':
    # Load data
    LOAD_DIR = 'modified_swiss_dwellings/'  # replace it with ur own path
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()
    
    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]
    
    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask
    
    # Run jacobi iterations for each floor plan
    MAX_ITER = 20000  # Using the full 20,000 iterations as per task requirements
    ABS_TOL = 1e-4    # Not used in the CUDA implementation but kept for reference
    
    start_time = time.time()
    
    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        # Use the CUDA implementation
        u = jacobi_cuda(u0, interior_mask, MAX_ITER)
        all_u[i] = u
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid}, " + ", ".join(str(stats[k]) for k in stat_keys))



'''
answer to the question:
a) 
A CUDA kernel that performs one iteration of the Jacobi method.

A helper function that repeatedly calls the kernel and handles memory transfers.

A fixed number of iterations (5000) instead of checking for convergence, simplifying execution.

Alternating between two arrays (d_u and d_u_new) to avoid race conditions.

A 2D thread grid (16×16 threads per block) for efficient parallelism.

b) 
You ran one building (ID: 10000) in Task 8 with CUDA in 0.46 seconds, while the CPU JIT solution in Task 7 took 26.81 seconds for 10 buildings → ~2.68 sec/building.

Your CUDA implementation is roughly 5.8× faster than the optimized CPU JIT version.

c)
With 4,571 buildings, the estimated time is:
4571 x 0.46 sec = 2106.66 sec = 35.11 min.

Compared to the CPU JIT solution, which would take:
4571 x 2.68 sec = 12283.08 sec = 204.72 min.
'''
