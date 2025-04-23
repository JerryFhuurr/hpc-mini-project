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

def jacobi_cupy_optimized(u, interior_mask, max_iter, atol=1e-6):
    """
    Optimized CuPy implementation of the Jacobi method.
    Key optimizations:
    1. Uses a single custom kernel for the entire iteration
    2. Avoids redundant memory transfers and allocations
    3. Uses a stream to avoid synchronization
    """
    # Create a CUDA stream for asynchronous operations
    stream = cp.cuda.Stream()
    with stream:
        # Use a custom fused kernel to perform the entire Jacobi iteration
        jacobi_kernel = cp.RawKernel(r'''
        extern "C" __global__ void jacobi_iteration(
            const float* u, float* u_new, const bool* interior_mask,
            int nx, int ny, float* max_diff
        ) {
            // Compute grid position
            int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
            int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
            
            // Shared memory for block-wide reduction of max_diff
            __shared__ float block_max_diff[256];  // Assuming 16x16 block
            
            float local_max_diff = 0.0f;
            
            // Only process interior points
            if (i < nx-1 && j < ny-1) {
                int idx = i * ny + j;
                if (interior_mask[(i-1) * (ny-2) + (j-1)]) {
                    // Compute new value
                    float new_val = 0.25f * (
                        u[idx - 1] + u[idx + 1] + 
                        u[idx - ny] + u[idx + ny]
                    );
                    
                    // Update difference
                    float diff = fabsf(u[idx] - new_val);
                    local_max_diff = diff;
                    
                    // Update u_new
                    u_new[idx] = new_val;
                } else {
                    // Copy non-interior points
                    u_new[idx] = u[idx];
                }
            }
            
            // Compute max difference across the block
            int tid = threadIdx.y * blockDim.x + threadIdx.x;
            block_max_diff[tid] = local_max_diff;
            __syncthreads();
            
            // Reduction in shared memory
            for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    block_max_diff[tid] = fmaxf(block_max_diff[tid], block_max_diff[tid + s]);
                }
                __syncthreads();
            }
            
            // Only thread 0 writes the block's max diff
            if (tid == 0) {
                atomicMax((unsigned int*)max_diff, __float_as_uint(block_max_diff[0]));
            }
        }
        ''', 'jacobi_iteration')
        
        # Grid dimensions
        nx, ny = u.shape
        threads_per_block = (16, 16)
        blocks_per_grid = (
            (nx + threads_per_block[0] - 3) // threads_per_block[0],
            (ny + threads_per_block[1] - 3) // threads_per_block[1]
        )
        
        # Pre-allocate and initialize arrays
        u_device = cp.asarray(u)
        u_new_device = cp.copy(u_device)
        interior_mask_device = cp.asarray(interior_mask)
        
        # Main iteration loop
        for iter_count in range(max_iter):
            # Reset max difference
            max_diff_device = cp.zeros(1, dtype=cp.float32)
            
            # Launch kernel
            jacobi_kernel(
                grid=blocks_per_grid,
                block=threads_per_block,
                args=(u_device, u_new_device, interior_mask_device, 
                     np.int32(nx), np.int32(ny), max_diff_device)
            )
            
            # Check convergence
            if max_diff_device.item() < atol:
                break
                
            # Swap u and u_new for next iteration
            u_device, u_new_device = u_new_device, u_device
        
        # Ensure the result is in u_device
        if iter_count % 2 == 1:
            result = u_new_device
        else:
            result = u_device
            
        # Return result as numpy array
        return result.get()

def batch_process_floorplans(all_u0, all_interior_mask, max_iter, atol):
    """
    Process multiple floorplans in batches with memory optimizations.
    """
    n_floorplans = len(all_u0)
    all_u = np.empty_like(all_u0)
    
    # Stream for asynchronous operations
    stream = cp.cuda.Stream(non_blocking=True)
    
    # Pre-warm GPU to avoid timing the JIT compilation
    if n_floorplans > 0:
        with stream:
            _ = jacobi_cupy_optimized(all_u0[0], all_interior_mask[0], 10, atol)
    
    # Process each floorplan
    for i in range(n_floorplans):
        with stream:
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
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
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
    
    # Use the optimized batch processing function
    all_u = batch_process_floorplans(all_u0, all_interior_mask, MAX_ITER, ABS_TOL)
    
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