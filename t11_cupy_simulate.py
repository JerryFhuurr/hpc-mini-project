from os.path import join
from concurrent.futures import ThreadPoolExecutor
import cupy as cp
import numpy as np
import time
import sys

SIZE = 512                
PAD  = 1                  
H = SIZE + 2              
W = SIZE + 2              

def load_single(load_dir: str, bid: str):
    u = np.zeros((H, W), dtype=np.float32)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy")).astype(np.float32)
    mask = np.load(join(load_dir, f"{bid}_interior.npy")).astype(bool)
    return u, mask


def load_batch(load_dir: str, ids: list[str]):
    u_list, m_list = [], []
    for bid in ids:
        u, m = load_single(load_dir, bid)
        u_list.append(u)
        m_list.append(m)
    u_arr = np.stack(u_list, axis=0)          # (B,H,W)
    m_arr = np.stack(m_list, axis=0)          # (B,512,512)
    return ids, u_arr, m_arr


def jacobi_batch_gpu(u_cpu: np.ndarray,
                     mask_cpu: np.ndarray,
                     max_iter: int = 20_000,
                     atol: float = 1e-4,
                     CHECK_EVERY: int = 5000) -> np.ndarray:
    """
    Batched Jacobi：u_cpu shape (B,H,W);
    mask_cpu shape (B,512,512)  ( u_cpu[:,1:-1,1:-1])
    return u_final CPU (B,H,W)
    """
    u  = cp.asarray(u_cpu)          # (B,H,W)
    m  = cp.asarray(mask_cpu)       # (B,512,512)
    B, H, W = u.shape
    u_new = u.copy()
    jacobi_src = r'''
    extern "C" __global__
    void jacobi_nstep(float* u_cur, float* u_next,
                    const unsigned char* mask,
                    float* g_max,
                    int H, int W, int Nsub)
    {
        extern __shared__ float s_buf[];          // blockDim.x*blockDim.y 个 float
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int b   = blockIdx.z;
        int i   = blockIdx.y * blockDim.y + threadIdx.y + 1;
        int j   = blockIdx.x * blockDim.x + threadIdx.x + 1;

        int stride = H * W;
        float* cur = u_cur;
        float* nxt = u_next;
        float local_max = 0.f;

        #pragma unroll 8
        for (int step = 0; step < Nsub; ++step) {
            if (i < H-1 && j < W-1) {
                int idx = b * stride + i * W + j;
                int mid = b * (H-2) * (W-2) + (i-1) * (W-2) + (j-1);

                float oldv = cur[idx];
                float newv = 0.25f * (cur[idx-1] + cur[idx+1] +
                                    cur[idx-W] + cur[idx+W]);

                nxt[idx] = mask[mid] ? newv : oldv;
                if (step == Nsub-1) {
                    local_max = fmaxf(local_max, fabsf(newv - oldv));
                }
            }
            __syncthreads();        // prepare next sub-step
            float* tmp = cur; cur = nxt; nxt = tmp;
        }

        s_buf[tid] = local_max;
        __syncthreads();
        for (int s = blockDim.x*blockDim.y/2; s > 0; s >>= 1) {
            if (tid < s) s_buf[tid] = fmaxf(s_buf[tid], s_buf[tid+s]);
            __syncthreads();
        }

        if (tid == 0) {
            atomicMax((int*)g_max, __float_as_int(s_buf[0]));
        }
    }
    '''
    jacobi_kernel = cp.RawKernel(jacobi_src, 'jacobi_nstep')
    block  = (16, 16, 1)
    grid   = ((W-2 + block[0]-1)//block[0],
            (H-2 + block[1]-1)//block[1],
            B)

    shared_bytes = block[0] * block[1] * 4   # 256 * 4 = 1 KB
    Nsub   = 50                              
    g_max  = cp.zeros(1, dtype=cp.float32)

    stream = cp.cuda.Stream()
    with stream:
        n_outer = (max_iter + Nsub - 1)//Nsub
        for outer in range(n_outer):
            g_max.fill(0)                     
            jacobi_kernel(grid, block,
                        (u, u_new, m, g_max, H, W, Nsub),
                        shared_mem=shared_bytes)
            u, u_new = u_new, u               

            if ((outer+1)*Nsub) % CHECK_EVERY == 0 and n_outer <= 100:
                stream.synchronize()
                if float(g_max.get()[0]) < atol:
                    break

    stream.synchronize()
    return cp.asnumpy(u)


def summary_batch(u_batch: np.ndarray, mask_batch: np.ndarray, ids: list[str]):
    """
    u_batch : (B,H,W)
    mask_batch : (B,512,512)
    """
    stats = {}
    B = u_batch.shape[0]
    for i in range(B):
        bid = ids[i]
        interior = u_batch[i, 1:-1, 1:-1][mask_batch[i]]
        stats[bid] = {
            "mean_temp":   float(interior.mean()),
            "std_temp":    float(interior.std()),
            "pct_above_18": float((interior > 18).sum() / interior.size * 100),
            "pct_below_15": float((interior < 15).sum() / interior.size * 100),
        }
    return stats


def pipeline_solver(load_dir: str,
                    building_ids: list[str],
                    batch_size: int = 32,
                    max_iter: int = 20_000,
                    atol: float = 1e-4):

    results = {}

    pool = ThreadPoolExecutor(max_workers=1)

    next_off = 0
    future_load = pool.submit(
        load_batch, load_dir, building_ids[next_off: next_off + batch_size]
    )
    next_off += batch_size

    while future_load:
        ids, u_cpu, m_cpu = future_load.result()

        if next_off < len(building_ids):
            future_load = pool.submit(
                load_batch, load_dir, building_ids[next_off: next_off + batch_size]
            )
            next_off += batch_size
        else:
            future_load = None     

        u_final_cpu = jacobi_batch_gpu(u_cpu, m_cpu, max_iter, atol)

        stats = summary_batch(u_final_cpu, m_cpu, ids)
        results.update(stats)

    pool.shutdown(wait=True)
    return results


if __name__ == "__main__":
    N           = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
    LOAD_DIR = r"C:/Users/14349/hpc/123/modified_swiss_dwellings"
    MAX_ITER = 20_000
    ABS_TOL  = 1e-4
    BATCH    = 8                 

    with open(join(LOAD_DIR, "building_ids.txt")) as f:
        ids_all = f.read().splitlines()[:N]

    t0 = time.perf_counter()
    stats_dict = pipeline_solver(LOAD_DIR,
                                 ids_all,         
                                 batch_size=BATCH,
                                 max_iter=MAX_ITER,
                                 atol=ABS_TOL)
    t1 = time.perf_counter()
    print(f"Total runtime: {t1 - t0:.2f} s  "
          f"(batch = {BATCH}, buildings = {len(ids_all)})")

    import csv
    with open("stats.csv", "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["building_id", "mean_temp",
                         "std_temp", "pct_above_18", "pct_below_15"])
        for bid in ids_all:
            s = stats_dict[bid]
            writer.writerow([bid, s["mean_temp"], s["std_temp"],
                             s["pct_above_18"], s["pct_below_15"]])
