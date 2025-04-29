from os.path import join
import sys
import time
import numpy as np

# Try to import CuPy with better exception handling
try:
    import cupy as cp
    # Test if CUDA is actually working properly
    test_array = cp.array([1, 2, 3])
    HAS_CUPY = True
    print("CuPy and CUDA available and working")
except Exception as e:
    print(f"CuPy or CUDA error: {e}")
    print("Falling back to NumPy implementation")
    HAS_CUPY = False

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

def jacobi_numpy(u, interior_mask, max_iter, atol=1e-6):
    """CPU implementation of the Jacobi method using NumPy."""
    u_new = np.copy(u)
    nx, ny = u.shape
    
    # Create interior mask with same shape as u for direct indexing
    full_mask = np.zeros((nx, ny), dtype=bool)
    full_mask[1:-1, 1:-1] = interior_mask
    
    for iter_count in range(max_iter):
        # Update interior points (element-wise)
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                if interior_mask[i-1, j-1]:
                    u_new[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
        
        # Check convergence
        diff = np.abs(u_new - u)
        max_diff = np.max(diff)
        
        # Copy for next iteration (avoid reallocation)
        u[:] = u_new[:]
        
        if max_diff < atol:
            print(f"Converged after {iter_count+1} iterations")
            break
    
    return u

def jacobi_cupy_simplified(u, interior_mask, max_iter, atol=1e-6):
    """
    Simplified CuPy implementation of the Jacobi method without using streams.
    """
    if not HAS_CUPY:
        return jacobi_numpy(u, interior_mask, max_iter, atol)
    
    try:
        # Transfer data to GPU
        u_device = cp.asarray(u, dtype=cp.float32)
        u_new_device = cp.copy(u_device)
        interior_mask_device = cp.asarray(interior_mask, dtype=cp.bool_)
        
        # Create a version of interior_mask with same shape as u
        full_mask_device = cp.zeros_like(u_device, dtype=cp.bool_)
        full_mask_device[1:-1, 1:-1] = interior_mask_device
        
        # Grid and block dimensions - keeping it simple
        nx, ny = u.shape
        
        # Main iteration loop using standard CuPy operations (no custom kernel)
        for iter_count in range(max_iter):
            # Update interior points using standard CuPy operations
            # This is less efficient but more likely to work with CUDA compatibility issues
            for i in range(1, nx-1):
                for j in range(1, ny-1):
                    if interior_mask_device[i-1, j-1]:
                        u_new_device[i, j] = 0.25 * (
                            u_device[i-1, j] + u_device[i+1, j] +
                            u_device[i, j-1] + u_device[i, j+1]
                        )
            
            # Check convergence
            diff = cp.abs(u_new_device - u_device)
            max_diff = cp.max(diff).get()
            
            # Copy for next iteration
            u_device[:] = u_new_device[:]
            
            if max_diff < atol:
                print(f"Converged after {iter_count+1} iterations")
                break
        
        # Return result as numpy array
        return u_device.get()
    
    except Exception as e:
        print(f"CuPy execution failed: {e}")
        print("Falling back to NumPy implementation")
        return jacobi_numpy(u, interior_mask, max_iter, atol)

def batch_process_floorplans(all_u0, all_interior_mask, max_iter, atol):
    """
    Process multiple floorplans in batches.
    """
    n_floorplans = len(all_u0)
    all_u = np.empty_like(all_u0)
    
    # Check if CuPy is available
    if not HAS_CUPY:
        print("Using NumPy implementation for all floorplans")
        for i in range(n_floorplans):
            print(f"Processing floorplan {i+1}/{n_floorplans}")
            all_u[i] = jacobi_numpy(all_u0[i], all_interior_mask[i], max_iter, atol)
        return all_u

    try:
        # Process each floorplan
        for i in range(n_floorplans):
            print(f"Processing floorplan {i+1}/{n_floorplans}")
            all_u[i] = jacobi_cupy_simplified(all_u0[i], all_interior_mask[i], max_iter, atol)
    except Exception as e:
        print(f"CuPy batch processing failed: {e}")
        print("Falling back to NumPy for all floorplans")
        for i in range(n_floorplans):
            all_u[i] = jacobi_numpy(all_u0[i], all_interior_mask[i], max_iter, atol)
    
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
    
    # Use the batch processing function
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