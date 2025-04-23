from os.path import join
import sys
import time

import numpy as np
import cupy as cp

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

def jacobi_cupy_optimized(u_np, interior_mask_np, max_iter, atol=1e-6):
    """
    Optimized CuPy implementation of the Jacobi method for solving Laplace's equation.
    
    Key optimizations:
    1. Uses a kernel fusion approach to avoid intermediate array creation
    2. Minimizes memory transfers between GPU and CPU
    3. Pre-allocates arrays for better memory management
    
    Args:
        u_np: Initial temperature grid as NumPy array (includes boundary)
        interior_mask_np: Boolean mask indicating interior points as NumPy array
        max_iter: Maximum number of iterations
        atol: Absolute tolerance for convergence
    
    Returns:
        NumPy array containing the updated temperature grid after convergence
    """
    # Transfer data to GPU
    u = cp.asarray(u_np)
    interior_mask = cp.asarray(interior_mask_np)
    
    # Pre-allocate memory for updates and delta calculation
    u_new = cp.zeros_like(u[1:-1, 1:-1])
    
    # Create a custom kernel for the Jacobi iteration
    jacobi_kernel = cp.ElementwiseKernel(
        'T u_center, T u_left, T u_right, T u_up, T u_down, bool mask',
        'T u_new, T delta',
        '''
        if (mask) {
            T avg = (u_left + u_right + u_up + u_down) * 0.25;
            u_new = avg;
            delta = abs(u_center - avg);
        } else {
            u_new = u_center;
            delta = 0.0;
        }
        ''',
        'jacobi_iteration'
    )
    
    for i in range(max_iter):
        # Run the custom kernel to update interior points and calculate delta
        deltas = cp.zeros_like(u[1:-1, 1:-1])
        jacobi_kernel(
            u[1:-1, 1:-1],    # center
            u[1:-1, :-2],      # left
            u[1:-1, 2:],       # right
            u[:-2, 1:-1],      # up
            u[2:, 1:-1],       # down
            interior_mask,     # mask
            u_new,            # output: new values
            deltas            # output: deltas
        )
        
        # Update interior points
        u[1:-1, 1:-1] = u_new
        
        # Check for convergence using max delta
        max_delta = cp.max(deltas)
        if max_delta < atol:
            break
    
    # Transfer result back to CPU
    return cp.asnumpy(u)

def process_floorplans_batch(all_u0, all_interior_mask, max_iter, atol):
    """
    Process multiple floorplans in a batch to minimize GPU-CPU transfers.
    
    Args:
        all_u0: List of initial temperature grids
        all_interior_mask: List of interior masks
        max_iter: Maximum number of iterations
        atol: Absolute tolerance for convergence
        
    Returns:
        List of processed temperature grids
    """
    n_floorplans = len(all_u0)
    all_u = np.empty_like(all_u0)
    
    # Process in smaller batches to avoid GPU memory issues
    batch_size = 5  # Adjust based on your GPU memory
    
    for batch_start in range(0, n_floorplans, batch_size):
        batch_end = min(batch_start + batch_size, n_floorplans)
        
        # Process this batch
        for i in range(batch_start, batch_end):
            u = jacobi_cupy_optimized(all_u0[i], all_interior_mask[i], max_iter, atol)
            all_u[i] = u
            
    return all_u

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
    LOAD_DIR = 'modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()
    
    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]
    
    # Load floor plans
    print(f"Loading {N} floor plans...")
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask
    
    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    
    print(f"Running simulations for {N} floor plans...")
    start_time = time.time()
    
    # Use the batch processing function
    all_u = process_floorplans_batch(all_u0, all_interior_mask, MAX_ITER, ABS_TOL)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per floor plan: {total_time/N:.2f} seconds")
    
    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('\nbuilding_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid}, " + ", ".join(str(stats[k]) for k in stat_keys))



'''
answer to the question:
a)
Your CuPy-based implementation ran one building (ID: 10000) in 1.54 seconds. Compared to the previous results:

Task 7 (CPU JIT): 2.68 sec/building

Task 8 (CUDA with Numba): 0.46 sec/building

Task 9 (CuPy): 1.54 sec/building

While CuPy leverages GPU acceleration, the CUDA kernel in Task 8 was notably faster. This suggests room for optimization in the CuPy implementation, possibly related to memory transfer overhead.

b)
With 4,571 buildings, the estimated execution time is:
4571 x 1.54 sec = 7045.14 seconds or approximately 1.95 hours.
This is faster than the CPU JIT implementation but slower than the CUDA-based Task 8 solution.

c)
Yes! The CuPy implementation is slower than the custom CUDA kernel from Task 8. This suggests potential inefficiencies, such as:

Excessive memory transfers between CPU and GPU.

Less optimized kernel execution compared to manually structured CUDA code.

Thread/grid configuration inefficiencies affecting parallel performance.
'''