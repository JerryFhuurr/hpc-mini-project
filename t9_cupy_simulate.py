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

d_converged  = cp.zeros(1, dtype=cp.uint8)     
d_max_delta  = cp.zeros(1, dtype=cp.float32)   

jacobi_solve_kernel = cp.RawKernel(r'''
extern "C" __global__
void jacobi_solve(float* u, float* u_new,
                  const unsigned char* mask,
                  int h, int w,
                  int max_iter, float atol,
                  unsigned char* converged,
                  float* g_max_delta)
{
    // Computes the thread's coordinates in the 3D batch grid
    int b = blockIdx.z;                                    // batch id
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;     // row   1~h-2
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;     // col   1~w-2
    int tid = threadIdx.y * blockDim.x + threadIdx.x;      // 0~255

    // Each block uses 256 floats as shared memory protocol
    extern __shared__ float sdata[];   // 编译时大小由 python 侧传入
                                       // = blockDim.x*blockDim.y*4 字节
    for (int it = 0; it < max_iter; ++it)
    {
        float delta = 0.0f;

        if (i < h-1 && j < w-1)        
        {
            int pitch  = w;
            int idx    = b*pitch*h + i*pitch + j;
            int up     = idx - pitch;
            int down   = idx + pitch;
            int left   = idx - 1;
            int right  = idx + 1;

            int midx   = b*(h-2)*(w-2) + (i-1)*(w-2) + (j-1);

            float oldv = u[idx];
            float newv = 0.25f * (u[up]+u[down]+u[left]+u[right]);

            if (mask[midx]) u_new[idx] = newv;      // 只更新内部
            else             u_new[idx] = oldv;

            delta = fabsf(newv - oldv);
        }
        else {
            int pitch  = w;
            int idx    = b*pitch*h + i*pitch + j;
            if (i<h && j<w) u_new[idx] = u[idx];
        }

        sdata[tid] = delta;
        __syncthreads();

        // 256 → 128 → … → 1
        for (int s = blockDim.x*blockDim.y/2; s>0; s >>= 1){
            if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]);
            __syncthreads();
        }

        if (tid == 0){
            unsigned int ival = __float_as_uint(sdata[0]);
            atomicMax((unsigned int*)g_max_delta, ival);
        }

        __syncthreads();

        if (blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && tid==0){
            float cur_max = g_max_delta[0];
            if (cur_max < atol){
                *converged = 1;      
            }
            g_max_delta[0] = 0.0f;
        }

        __syncthreads();

        if (*converged) break;

        if (blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && tid==0){
            float* tmp = u;
            u  = u_new;
            u_new = tmp;
        }
        __syncthreads();
    }
}
''', 'jacobi_solve')

def load_single(load_dir: str, bid: str):
    """return (u, mask)， np.float32 / np.bool_"""
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


def jacobi_batch_gpu(u_cpu, mask_cpu,
                     max_iter=20_000, atol=1e-4):

    u  = cp.asarray(u_cpu)    # (B,H,W)
    m  = cp.asarray(mask_cpu) # (B,512,512)

    B,H,W = u.shape
    u_new = u.copy()

    block = (16,16,1)
    grid  = ((W-2+15)//16, (H-2+15)//16, B)
    shmem = 16*16*4                       # 256 floats

    d_converged[...] = 0      
    d_max_delta[...] = 0.0

    jacobi_solve_kernel(grid, block,
        (u, u_new, m, H, W,
         max_iter, atol,
         d_converged, d_max_delta),
        shared_mem = shmem)

    cp.cuda.runtime.deviceSynchronize()

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
    """
    主流程：返回 dict {bid: stats}
    """
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

        # GPU Jacobi
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
        ids = f.read().splitlines()[:N]

    t0 = time.perf_counter()
    stats_dict = pipeline_solver(LOAD_DIR,
                                 ids,          
                                 batch_size=BATCH,
                                 max_iter=MAX_ITER,
                                 atol=ABS_TOL)
    t1 = time.perf_counter()
    print(f"Total runtime: {t1 - t0:.2f} s  "
          f"(batch = {BATCH}, buildings = {len(ids)})")


    # import csv
    # with open("stats.csv", "w", newline="") as fcsv:
    #     writer = csv.writer(fcsv)
    #     writer.writerow(["building_id", "mean_temp",
    #                      "std_temp", "pct_above_18", "pct_below_15"])
    #     for bid in ids:
    #         s = stats_dict[bid]
    #         writer.writerow([bid, s["mean_temp"], s["std_temp"],
    #                          s["pct_above_18"], s["pct_below_15"]])
