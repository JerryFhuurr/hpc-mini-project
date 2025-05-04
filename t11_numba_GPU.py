import sys, time
from os.path import join
import numpy as np
from numba import cuda, float32
import os

SIZE = 512         # compute part size
PAD  = 2           # boundary
NYP  = SIZE + PAD  # 514
DTYPE = np.float32 # float32

def load_data(load_dir, bid):
    """read .npy file with (514,514)"""
    u = np.zeros((NYP, NYP), dtype=DTYPE)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy")).astype(DTYPE)
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy")).astype(np.bool_)
    return u, interior_mask


@cuda.jit
def jacobi_kernel(batch_u, batch_u_new, batch_mask):
    k, i, j = cuda.grid(3)

    # mask = 512×512，for u[1:-1,1:-1]
    # Indices for mask are offset by 1
    mask_i = i - 1
    mask_j = j - 1

    if batch_mask[k, mask_i, mask_j]:
        batch_u_new[k, i, j] = 0.25 * (
            batch_u[k, i, j - 1] + batch_u[k, i, j + 1] +
            batch_u[k, i - 1, j] + batch_u[k, i + 1, j]
        )
    else:
        # Keep boundary or non-interior points unchanged from previous iteration
        batch_u_new[k, i, j] = batch_u[k, i, j]

# check kernel
@cuda.jit
def check_convergence_kernel(cur, nxt, mask, d_max_diff_out):
    k, i, j = cuda.grid(3)
    if k >= cur.shape[0] or \
       i == 0 or i >= cur.shape[1] - 1 or \
       j == 0 or j >= cur.shape[2] - 1:
        return
    mask_i = i - 1
    mask_j = j - 1
    if mask[k, mask_i, mask_j]:
        diff = abs(cur[k, i, j] - nxt[k, i, j])
        cuda.atomic.max(d_max_diff_out, 0, diff)


def jacobi_cuda(d_u, d_mask, max_iter, tolerance, check_interval, stream): # Added check_interval
    """
    Performs Jacobi iteration with periodic convergence check.

    Args:
        d_u (DeviceNDArray): Initial state on GPU.
        d_mask (DeviceNDArray): Interior mask on GPU.
        max_iter (int): Maximum number of iterations.
        tolerance (float): Convergence threshold for max absolute difference.
        check_interval (int): How often (in iterations) to check convergence.
        stream (cuda.Stream): CUDA stream for operations.

    Returns:
        tuple: (final_device_array, iterations_done)
    """
    d_u_new = cuda.device_array_like(d_u, stream=stream)
    d_max_diff = cuda.device_array(1, dtype=DTYPE, stream=stream)
    h_zero = np.zeros(1, dtype=DTYPE)

    BLOCK = (1, 16, 16)
    grid  = (
        d_u.shape[0],
        (NYP + BLOCK[1] - 1) // BLOCK[1],
        (NYP + BLOCK[2] - 1) // BLOCK[2],
    )

    cur, nxt = d_u, d_u_new
    iterations_done = 0
    converged = False
    last_checked_diff = float('inf') # Store the diff from the last check

    for i in range(max_iter):
        # take Jacobi loop**
        jacobi_kernel[grid, BLOCK, stream](cur, nxt, d_mask)
        iterations_done = i + 1

        # check converge for CHECK_INTERVAL loops
        if iterations_done % check_interval == 0 and iterations_done >= 5000:
            # max_diff = 0
            d_max_diff.copy_to_device(h_zero, stream=stream)
            # compare nxt and cur
            check_convergence_kernel[grid, BLOCK, stream](nxt, cur, d_mask, d_max_diff) # Note: arguments are nxt, cur
            # copy to Host
            h_max_diff = d_max_diff.copy_to_host(stream=stream)
            # synchronisation
            stream.synchronize()
            last_checked_diff = h_max_diff[0] # Update last checked difference

            #print(f"Checked at Iter {iterations_done}, Max Diff: {last_checked_diff:.6e}")

            # Determine whether it converges.
            if last_checked_diff < tolerance:
                converged = True
                print(f"Convergence reached after {iterations_done} iterations. Max Diff: {last_checked_diff:.2e} < {tolerance:.2e}")
                return nxt, iterations_done 

        # replace cur and nxt pointer
        cur, nxt = nxt, cur

    if not converged:
        # Report the last measured difference if available, otherwise state max_iter reached
        if last_checked_diff != float('inf'):
              print(f"Warning: Jacobi did not converge within {max_iter} iterations.** Last checked diff at iter ~{iterations_done // check_interval * check_interval}: {last_checked_diff:.2e}", file=sys.stderr)
        else: # This happens if max_iter < check_interval
              print(f"Warning: Jacobi finished {max_iter} iterations without reaching the first convergence check.", file=sys.stderr)

    return cur, iterations_done


if __name__ == "__main__":
    LOAD_DIR = r"C:/Users/14349/hpc/123/modified_swiss_dwellings"

    # parameters
    N           = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
    MAX_ITER    = 20000
    TOLERANCE   = 1e-4
    CHECK_INTERVAL = 1000
    BATCH_SIZE  = 10

    print(f"Running Jacobi for {N} buildings. Max Iter: {MAX_ITER}, Tolerance: {TOLERANCE:.2e}, Check Interval: {CHECK_INTERVAL}")

    with open(join(LOAD_DIR, 'building_ids.txt')) as f:
        building_ids = f.read().splitlines()[:N]

    all_msk = np.empty((N, SIZE, SIZE), dtype=np.bool_)
    total_iterations = 0

    # Batching + dual stream overlap
    num_batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
    streams = [cuda.stream(), cuda.stream()]
    all_u   = np.empty((N, NYP, NYP), dtype=DTYPE)

    print(f"Processing in {num_batches} batches of size up to {BATCH_SIZE}...")
    batch_start_time = time.time()
    total_iterations = 0 # Reset for batch mode

    for batch_idx in range(num_batches):
        current_stream_idx = batch_idx & 1
        s = streams[current_stream_idx]
        beg = batch_idx * BATCH_SIZE
        end = min(beg + BATCH_SIZE, N)
        curB = end - beg

        print(f"Batch {batch_idx+1}/{num_batches} (Buildings {beg}-{end-1}) ---")

        h_u0  = cuda.pinned_array((curB, NYP, NYP), dtype=DTYPE)
        h_msk = cuda.pinned_array((curB, SIZE, SIZE), dtype=np.bool_)
        load_s = time.time()
        for i, bid in enumerate(building_ids[beg:end]):
            h_u0[i], h_msk[i] = load_data(LOAD_DIR, bid)

        all_msk[beg:end] = h_msk

        transfer_s = time.time()
        d_u0   = cuda.to_device(h_u0, stream=s)
        d_mask = cuda.to_device(h_msk, stream=s)
        del h_u0, h_msk

        print(f"batch {batch_idx+1} (Stream {current_stream_idx})...")
        compute_s = time.time()
        # **传入 check_interval**
        d_res, iters_done = jacobi_cuda(d_u0, d_mask, MAX_ITER, TOLERANCE, CHECK_INTERVAL, s)
        # Note: total_iterations sum might be less meaningful now as each batch converges independently
        # total_iterations += iters_done
        print(f"Batch {batch_idx+1} computation finished in {time.time()-compute_s:.2f} s ({iters_done} iterations)")

        copy_back_s = time.time()
        d_res.copy_to_host(all_u[beg:end], stream=s)

        del d_u0, d_mask, d_res

        prev_stream_idx = (batch_idx - 1) & 1
        if batch_idx > 0:
                streams[prev_stream_idx].synchronize()

        print("\nWaiting for all streams to synchronize...")
        cuda.synchronize()
        print(f"All batches finished in {time.time()-batch_start_time:.2f} s")


    def summary(u, m):
        m_bool = m.astype(np.bool_)
        data = u[1:-1, 1:-1][m_bool]
        if data.size == 0:
             return (np.nan, np.nan, 0.0, 0.0)
        mean_val = data.mean()
        std_val = data.std()
        pct_gt_18 = (data > 18).sum() / data.size * 100
        pct_lt_15 = (data < 15).sum() / data.size * 100
        return (mean_val, std_val, pct_gt_18, pct_lt_15)

    header = ["mean", "std", "pct>18", "pct<15"]
    print("Results Summary")
    print("building_id," + ",".join(header))
    for bid, u, m in zip(building_ids, all_u, all_msk):
        stat = summary(u, m)
        stat_str = ",".join(f"{x:.4f}" if not np.isnan(x) else "nan" for x in stat)
        print(f"{bid},{stat_str}")

