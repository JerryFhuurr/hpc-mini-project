"""
pipeline_jacobi_cupy.py
方案 B: I/O 与 GPU Jacobi 计算流水线
"""

from os.path import join
from concurrent.futures import ThreadPoolExecutor
import cupy as cp
import numpy as np
import time
import sys

# ---------------------------
# 1. 读文件——CPU 线程用
# ---------------------------
SIZE = 512                # 网格内部尺寸
PAD  = 1                  # 每边 pad 1
H = SIZE + 2              # 含边界高
W = SIZE + 2              # 含边界宽


def load_single(load_dir: str, bid: str):
    """读单栋楼，返回 (u, mask)，均为 np.float32 / np.bool_"""
    u = np.zeros((H, W), dtype=np.float32)
    # domain.npy 只有内部 512×512；外圈边界为 0
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy")).astype(np.float32)
    mask = np.load(join(load_dir, f"{bid}_interior.npy")).astype(bool)
    return u, mask


def load_batch(load_dir: str, ids: list[str]):
    """把一批楼读进内存并打包"""
    u_list, m_list = [], []
    for bid in ids:
        u, m = load_single(load_dir, bid)
        u_list.append(u)
        m_list.append(m)
    u_arr = np.stack(u_list, axis=0)          # (B,H,W)
    m_arr = np.stack(m_list, axis=0)          # (B,512,512)
    return ids, u_arr, m_arr


# ---------------------------
# 2. GPU Jacobi — 主线程用
# ---------------------------
def jacobi_batch_gpu(u_cpu: np.ndarray,
                     mask_cpu: np.ndarray,
                     max_iter: int = 20_000,
                     atol: float = 1e-4,
                     CHECK_EVERY: int = 5000) -> np.ndarray:
    """
    Batched Jacobi：u_cpu 形状 (B,H,W);
    mask_cpu 形状 (B,512,512)  (对应 u_cpu[:,1:-1,1:-1])
    返回 u_final CPU 数组 (B,H,W)
    """
    # --- 拷到 GPU ---
    u  = cp.asarray(u_cpu)          # (B,H,W)
    m  = cp.asarray(mask_cpu)       # (B,512,512)

    B, H, W = u.shape

    u_new = u.copy()
    jacobi_update_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void jacobi_update(const float* u_in, float* u_out,
                       const unsigned char* mask,
                       float* block_deltas,
                       int H, int W,
                       int Nsub) 
    {
        // 线程／块索引
        int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
        int tx = threadIdx.x, ty = threadIdx.y;
        int b   = bz; 
        int i   = by*blockDim.y + ty + 1;  // 跳过边界
        int j   = bx*blockDim.x + tx + 1;

        int tid = ty * blockDim.x + tx;
        int numX = gridDim.x, numY = gridDim.y;
        int bid = bz * (numY*numX) + by * numX + bx;

        extern __shared__ float s_delta[]; // 大小 = blockDim.x*blockDim.y*sizeof(float)
        
        // ping-pong 指针
        const float* cur = u_in;
        float*       nxt = u_out;
        int stride = H*W;

        // 每次迭代 4 读 1 写，Nsub 步合并
        for(int step=0; step < Nsub; ++step) {
            __syncthreads();   // 确保上一次写入 nxt 已经可见

            float local_delta = 0.0f;
            s_delta[tid] = 0.0f;

            if (i < H-1 && j < W-1) {
                int idx     = b*stride + i*W + j;
                int mask_id = b*(H-2)*(W-2) + (i-1)*(W-2) + (j-1);
                float oldv = cur[idx];
                float newv = 0.25f*( cur[idx- W] + cur[idx+ W]
                                   + cur[idx- 1] + cur[idx+ 1] );
                // 根据 mask 只在内部点更新
                nxt[idx] = mask[mask_id] ? newv : oldv;
                // 仅在最后一次子迭代里算误差
                if (step == Nsub-1) {
                    local_delta = fabsf(newv - oldv);
                }
            }

            // 只在最后一次子迭代才做归约
            if (step == Nsub-1) {
                s_delta[tid] = local_delta;
                __syncthreads();
                // block 内归约
                for (int s=blockDim.x*blockDim.y/2; s>0; s>>=1) {
                    if (tid < s) {
                        s_delta[tid] = fmaxf(s_delta[tid], s_delta[tid+s]);
                    }
                    __syncthreads();
                }
                // 写入 block_deltas
                if (tid == 0) block_deltas[bid] = s_delta[0];
            }

            // ping-pong swap
            const float* tmp = cur; 
            cur = nxt; 
            nxt = (float*)tmp;
        }

        // 上面做了 Nsub 次 swap：
        // 如果 Nsub 是偶数，最终结果在 cur = u_in；否则在 cur = u_out
        // 我们统一要求结果写回 u_out，因此如果 Nsub 是偶数，再额外把 cur->u_out
        if ((Nsub & 1) == 0) {
            __syncthreads();
            if (i < H-1 && j < W-1) {
                int idx = b*stride + i*W + j;
                u_out[idx] = cur[idx];
            }
        }
    }
    ''', 'jacobi_update')


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

        // --- block 内归约 ---
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
    # 定义全局最大值归约内核
    global_max_kernel = cp.RawKernel(r'''
    extern "C" __global__ void global_max(const float* block_deltas, 
                                        float* max_delta, int n) {
        // 块内归约的共享内存
        __shared__ float shared_max[256];
        
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + tid;
        
        // 初始化共享内存
        shared_max[tid] = 0.0f;
        
        // 每个线程从全局内存加载一个元素到共享内存
        if (idx < n) {
            shared_max[tid] = block_deltas[idx];
        }
        __syncthreads();
        
        // 共享内存内归约
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
            }
            __syncthreads();
        }
        
        // 第0线程用原子操作更新全局最大值
        if (tid == 0) {
            atomicMax((unsigned int*)max_delta, 
                      __float_as_uint(shared_max[0]));
        }
    }
    ''', 'global_max')

    # 内核执行参数
    block  = (16, 16, 1)
    grid   = ((W-2 + block[0]-1)//block[0],
            (H-2 + block[1]-1)//block[1],
            B)

    shared_bytes = block[0] * block[1] * 4   # 256 * 4 = 1 KB
    Nsub   = 50                              # 试试 128/256
    g_max  = cp.zeros(1, dtype=cp.float32)

    stream = cp.cuda.Stream()
    with stream:
        n_outer = (max_iter + Nsub - 1)//Nsub
        for outer in range(n_outer):
            g_max.fill(0)                     # 清全局Δ
            jacobi_kernel(grid, block,
                        (u, u_new, m, g_max, H, W, Nsub),
                        shared_mem=shared_bytes)
            u, u_new = u_new, u               # Python 侧保持最新 → u

            # 每 CHECK_EVERY 步检查一次
            if ((outer+1)*Nsub) % CHECK_EVERY == 0 and ((outer+1)*Nsub)!= 0:
                stream.synchronize()
                if float(g_max.get()[0]) < atol:
                    break

    stream.synchronize()
    return cp.asnumpy(u)


# ---------------------------
# 3. 统计量 — CPU
# ---------------------------
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


# ---------------------------
# 4. 主控制循环（I/O ↔ GPU 计算流水线）
# ---------------------------
def pipeline_solver(load_dir: str,
                    building_ids: list[str],
                    batch_size: int = 16,
                    max_iter: int = 20_000,
                    atol: float = 1e-4):
    """
    主流程：返回 dict {bid: stats}
    """
    results = {}

    # 线程池：1 个后台线程做磁盘 I/O
    pool = ThreadPoolExecutor(max_workers=1)

    # 先把第 1 批提交给线程池
    next_off = 0
    future_load = pool.submit(
        load_batch, load_dir, building_ids[next_off: next_off + batch_size]
    )
    next_off += batch_size

    while future_load:
        # 等这一批读完
        ids, u_cpu, m_cpu = future_load.result()

        # 同时把下一批（如果还有）继续丢给线程池
        if next_off < len(building_ids):
            future_load = pool.submit(
                load_batch, load_dir, building_ids[next_off: next_off + batch_size]
            )
            next_off += batch_size
        else:
            future_load = None     # 没有更多楼了

        # ---------- GPU Jacobi ----------
        u_final_cpu = jacobi_batch_gpu(u_cpu, m_cpu, max_iter, atol)

        # ---------- 统计 ----------
        stats = summary_batch(u_final_cpu, m_cpu, ids)
        results.update(stats)

    pool.shutdown(wait=True)
    return results


# ---------------------------
# 5. 运行示例
# ---------------------------
if __name__ == "__main__":
    N           = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
    LOAD_DIR = r"modified_swiss_dwellings/"
    MAX_ITER = 20_000
    ABS_TOL  = 1e-4
    BATCH    = 8                 # 可调成 8–16 之间测试

    # 所有楼 ID
    with open(join(LOAD_DIR, "building_ids.txt")) as f:
        ids_all = f.read().splitlines()[:N]

    t0 = time.perf_counter()
    stats_dict = pipeline_solver(LOAD_DIR,
                                 ids_all,          # 全部 4571 栋
                                 batch_size=BATCH,
                                 max_iter=MAX_ITER,
                                 atol=ABS_TOL)
    t1 = time.perf_counter()
    print(f"Total runtime: {t1 - t0:.2f} s  "
          f"(batch = {BATCH}, buildings = {len(ids_all)})")

    # 如需写 CSV
    import csv
    with open("stats.csv", "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["building_id", "mean_temp",
                         "std_temp", "pct_above_18", "pct_below_15"])
        for bid in ids_all:
            s = stats_dict[bid]
            writer.writerow([bid, s["mean_temp"], s["std_temp"],
                             s["pct_above_18"], s["pct_below_15"]])
