"""筛选并固化边界函数 B（不改动 B_function.py / svd.py 的原始构造）。

背景
----
B 由齐次约束 A·coeff = 0 的零空间确定。该零空间维数高达 140（324 个系数、
rank 184），而 B_function.solve_nullspace 只取 SVD 的 *第一个* 零空间方向
V[:,184]。这个方向是否可用（内部不穿零、燃料区量级足够）没有任何保证——
实测 140 个零空间基向量里约 22% 在内部穿零或近零，一旦命中，φ=B·NN 会被
强迫在内部某处为零，永远无法逼近处处为正的真实通量。

本模块在零空间内生成大量候选 B，按明确准则打分，选出最优并保存：

  1. 硬性条件：B 在内部（排除 Dirichlet 边界窄层）不穿零（符号一致）；
  2. 择优准则：归一化 max|B|=1 后，最大化 *燃料区* 的最小 |B|（燃料区远离
     Dirichlet 边界，能有效衡量 B 是否在通量集中的地方保持量级）。

应用层（train.build_networks / network.make_pspn）通过 get_B() 优先加载固化
B，找不到时才回退到原始 build_B，因此完全向后兼容。

用法
----
    python B_screen.py                     # 默认 n=5，保存 B_fixed.pt
    python B_screen.py --n 5 --n_random 4000 --path B_fixed.pt
"""

import argparse

import numpy as np
import torch

from B_function import (_build_constraint_matrix, _make_B_func, build_B,
                        N_CELLS, NX, NY, XS, YS)
from svd import nullspace_basis


DEFAULT_PATH = "B_fixed.pt"


# ==================== 网格与掩码 ====================
def _grid(N):
    t = torch.linspace(0.0, 0.8, N, dtype=torch.float32)
    return torch.meshgrid(t, t, indexing="ij")


def _masks(gx, gy):
    """内部掩码（排除 Dirichlet 边界窄层）与燃料区掩码（Region1 + 两臂 Region2）。"""
    interior = (gx <= 0.79) & (gy <= 0.79)
    center = (gx > 0.24) & (gx < 0.56) & (gy > 0.24) & (gy < 0.56)
    left = (gx < 0.24) & (gy > 0.24) & (gy < 0.56)
    bottom = (gx > 0.24) & (gx < 0.56) & (gy < 0.24)
    fuel = center | left | bottom
    return interior, fuel


def _value_matrix(xf, yf, n):
    """由坐标构造值矩阵 M：B(x,y) = M @ coeffs（coeffs 为 9 块拼接向量）。"""
    nt = (n + 1) ** 2
    N = xf.numel()
    M = torch.zeros(N, N_CELLS * nt, dtype=torch.float32)
    pows = torch.arange(n + 1, dtype=torch.float32)
    mono = (xf[:, None].pow(pows)[:, :, None] * yf[:, None].pow(pows)[:, None, :])
    mono = mono.reshape(N, nt)
    bnd_x = torch.tensor(XS[1:-1], dtype=torch.float32)
    bnd_y = torch.tensor(YS[1:-1], dtype=torch.float32)
    ix = torch.bucketize(xf, bnd_x)
    iy = torch.bucketize(yf, bnd_y)
    cid = iy * NX + ix
    M[torch.arange(N)[:, None], cid[:, None] * nt + torch.arange(nt)[None, :]] = mono
    return M


def _score(coeffs, gx, gy, interior, fuel, n):
    """归一化 max|B|=1 后打分，返回 (是否可用, 燃料区min|B|, 内部min|B|)。"""
    B = _make_B_func(coeffs, n)(gx, gy).numpy()
    s = float(np.abs(B).max())
    if s < 1e-12:
        return False, 0.0, 0.0
    B = B / s
    it = B[interior]
    if float(it.min()) * float(it.max()) <= 0.0:
        return False, 0.0, 0.0          # 内部穿零 -> 硬性拒绝
    return True, float(np.abs(B[fuel]).min()), float(np.abs(it).min())


# ==================== 筛选 ====================
def screen_and_save(n=5, Nd=None, tol=1e-6, path=DEFAULT_PATH,
                    n_random=4000, scoring_N=161, seed=0, verbose=True):
    """生成候选 B，筛选并保存最优系数，返回 (coeffs, (fuel_min, interior_min))。"""
    if Nd is None:
        Nd = n - 1
    nt = (n + 1) ** 2
    ntot = N_CELLS * nt

    A = _build_constraint_matrix(n, Nd)
    V = nullspace_basis(A, tol=tol)          # (ntot, d)
    d = V.shape[1]
    if verbose:
        print(f"零空间维数 d = {d}（系数总数 {ntot}）")

    gx, gy = _grid(scoring_N)
    interior, fuel = _masks(gx, gy)

    best = None

    def consider(coeffs, tag):
        nonlocal best
        ok, sf, si = _score(coeffs, gx, gy, interior, fuel, n)
        if not ok:
            return
        key = (sf, si)
        if best is None or key > best[0]:
            best = (key, coeffs.clone())
            if verbose:
                print(f"  [新最优] {tag:>12s}: 燃料区min|B|={sf:.3f}  内部min|B|={si:.3f}")

    # 候选 1：最小二乘拟合“内部 B≈1”（ridge 正则，抑制不可见方向）
    tgx, tgy = _grid(61)
    xf, yf = tgx.reshape(-1), tgy.reshape(-1)
    keep = (xf < 0.8) & (yf < 0.8)           # 内部点（Dirichlet 边界由零空间约束）
    M = _value_matrix(xf[keep], yf[keep], n)
    G = (M @ V).T @ (M @ V)                  # (d, d)
    b = (M @ V).T @ torch.ones(M.shape[0], dtype=torch.float32)
    lam = 1e-6 * G.diagonal().mean().clamp(min=1e-12)
    w_ls = torch.linalg.solve(G + lam * torch.eye(d), b)
    consider((V @ w_ls).reshape(N_CELLS, nt), "LS拟合B~1")

    # 候选 2：零空间正交基各方向
    for i in range(d):
        consider((V[:, i] / V[:, i].abs().max()).reshape(N_CELLS, nt), f"基向量{i}")

    # 候选 3：随机线性组合
    rng = np.random.default_rng(seed)
    Vn = V.numpy()
    for k in range(n_random):
        w = rng.standard_normal(d).astype(np.float32)
        w = w / np.abs(w).max()
        consider(torch.from_numpy(Vn @ w).reshape(N_CELLS, nt), f"随机{k}")

    if best is None:
        raise RuntimeError("未找到符号一致且量级足够的 B，请增大 n_random 或检查约束。")

    (sf, si), coeffs = best
    Bmax = _make_B_func(coeffs, n)(gx, gy).abs().max()
    coeffs = coeffs / Bmax                     # 归一化到 max|B|=1（与 build_B 约定一致）

    torch.save({"n": n, "coeffs": coeffs}, path)
    if verbose:
        resid = (A @ coeffs.reshape(-1)).abs().max().item()
        print(f"\n已保存最优 B 到 {path}")
        print(f"  n={n}, 系数形状={tuple(coeffs.shape)}")
        print(f"  燃料区min|B|={sf:.3f}  内部min|B|={si:.3f}  约束残差max={resid:.2e}")
    return coeffs, (sf, si)


# ==================== 加载 / 应用层入口 ====================
def get_B(n=5, Nd=None, tol=1e-6, path=DEFAULT_PATH):
    """应用层入口：优先加载固化 B，找不到或 n 不匹配时回退到原始 build_B。"""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        coeffs = ckpt["coeffs"].to(torch.float32)
        if int(ckpt["n"]) == n and tuple(coeffs.shape) == (N_CELLS, (n + 1) ** 2):
            return _make_B_func(coeffs, n), coeffs
    except (FileNotFoundError, KeyError):
        pass
    return build_B(n=n, Nd=Nd, tol=tol)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="筛选并固化边界函数 B")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--n_random", type=int, default=4000)
    ap.add_argument("--scoring_N", type=int, default=161)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    screen_and_save(n=args.n, path=args.path, n_random=args.n_random,
                    scoring_N=args.scoring_N, seed=args.seed)
