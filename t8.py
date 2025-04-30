import sys, time
from os.path import join

import numpy as np
from numba import cuda, float32

# ──────────────────────── 数据读入 ──────────────────────────
SIZE = 512         # 真实平面尺寸
PAD  = 2           # 边框
NYP  = SIZE + PAD  # 514
DTYPE = np.float32 # 统一使用 float32，减半带宽

def load_data(load_dir, bid):
    """读入单栋 .npy -> (514,514)"""
    u = np.zeros((NYP, NYP), dtype=DTYPE)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy")).astype(DTYPE)
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy")).astype(np.bool_)
    return u, interior_mask


# ──────────────────────── CUDA Kernel ──────────────────────
@cuda.jit
def jacobi_kernel(batch_u, batch_u_new, batch_mask):
    k, i, j = cuda.grid(3)

    if k >= batch_u.shape[0]:
        return
    # 注意 batch_u.shape[1]==514，所以 i,j 范围 [0,513]
    if i == 0 or i == batch_u.shape[1] - 1:
        return
    if j == 0 or j == batch_u.shape[2] - 1:
        return

    # mask 是 512×512，对应 u 的 [1:-1,1:-1]
    if batch_mask[k, i - 1, j - 1]:
        batch_u_new[k, i, j] = 0.25 * (
            batch_u[k, i, j - 1] + batch_u[k, i, j + 1] +
            batch_u[k, i - 1, j] + batch_u[k, i + 1, j]
        )
    else:
        batch_u_new[k, i, j] = batch_u[k, i, j]


# ──────────────────────── Jacobi 驱动 ──────────────────────
def jacobi_cuda(d_u, d_mask, max_iter, stream):
    d_u_new = cuda.device_array_like(d_u, stream=stream)

    # <<<grid,block,stream>>> 配置
    BLOCK = (1, 16, 16)  # (k,i,j) 维；楼栋维放 1
    grid  = (
        d_u.shape[0],                       # k 方向
        (NYP + BLOCK[1] - 1) // BLOCK[1],   # i
        (NYP + BLOCK[2] - 1) // BLOCK[2],   # j
    )

    cur, nxt = d_u, d_u_new
    for _ in range(max_iter):
        jacobi_kernel[grid, BLOCK, stream](cur, nxt, d_mask)
        cur, nxt = nxt, cur
    return cur  # 返回当前有效的 device array


# ──────────────────────── 主程序 ──────────────────────────
if __name__ == "__main__":
    LOAD_DIR = r"modified_swiss_dwellings/"

    # ------- 参数 -------
    N           = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
    MAX_ITER    = 20000
    BATCH_SIZE  = 10          # 须根据显存自行调整
    TOTAL_IN_GPU = False      # 如果显存足够可切 True

    # ------- 读 building id -------
    with open(join(LOAD_DIR, 'building_ids.txt')) as f:
        building_ids = f.read().splitlines()[:N]

    # 预分配 pinned host 内存（all_u 最终结果可普通 np.array）
    all_msk = np.empty((N, SIZE, SIZE), dtype=np.bool_)
    if TOTAL_IN_GPU:
        # 一次性载入全部
        hu0  = cuda.pinned_array((N, NYP, NYP), dtype=DTYPE)
        hmsk = cuda.pinned_array((N, SIZE, SIZE), dtype=np.bool_)
        for idx, bid in enumerate(building_ids):
            hu0[idx], hmsk[idx] = load_data(LOAD_DIR, bid)
        all_msk[:] = hmsk 

        # 上传到 GPU
        d_u0   = cuda.to_device(hu0)
        d_mask = cuda.to_device(hmsk)

        # 计算
        stream0 = cuda.stream()
        d_res = jacobi_cuda(d_u0, d_mask, MAX_ITER, stream0)
        all_u = d_res.copy_to_host(stream=stream0)
        stream0.synchronize()

    else:
        # 分批 + 双 stream 交叠
        num_batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
        streams = [cuda.stream(), cuda.stream()]
        all_u   = np.empty((N, NYP, NYP), dtype=DTYPE)  # 最终结果
        all_msk = np.empty((N, SIZE, SIZE), dtype=np.bool_)

        start_time = time.time()

        for batch_idx in range(num_batches):
            s = streams[batch_idx & 1]  # 0/1 交替
            beg = batch_idx * BATCH_SIZE
            end = min(beg + BATCH_SIZE, N)
            curB = end - beg

            # pinned host 批数据
            h_u0  = cuda.pinned_array((curB, NYP, NYP), dtype=DTYPE)
            h_msk = cuda.pinned_array((curB, SIZE, SIZE), dtype=np.bool_)
            for i, bid in enumerate(building_ids[beg:end]):
                h_u0[i], h_msk[i] = load_data(LOAD_DIR, bid)

            all_msk[beg:end] = h_msk  # 为后面统计

            # 异步复制到 GPU
            d_u0   = cuda.to_device(h_u0, stream=s)
            d_mask = cuda.to_device(h_msk, stream=s)

            # 计算
            d_res = jacobi_cuda(d_u0, d_mask, MAX_ITER, s)

            # 异步将结果拷回
            d_res.copy_to_host(all_u[beg:end], stream=s)

            # （可选）马上释放 GPU 内存，减少峰值占用
            del d_u0, d_mask, d_res

        # 等待所有 stream 完成
        cuda.synchronize()
        print(f"All batches finished in {time.time()-start_time:.2f} s")


    # ─────── 结果统计 ───────
    def summary(u, m):
        data = u[1:-1, 1:-1][m]
        return (data.mean(), data.std(),
                (data > 18).sum() / data.size * 100,
                (data < 15).sum() / data.size * 100)

    header = ["mean", "std", "pct>18", "pct<15"]
    print("building_id," + ",".join(header))
    for bid, u, m in zip(building_ids, all_u, all_msk):
        stat = summary(u, m)
        print(bid + "," + ",".join(f"{x:.4f}" for x in stat))
