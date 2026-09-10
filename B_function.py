"""构造分片多项式函数 B(x, y) 的模块。

B 用于通量重构 φ = B * NNs，其中 B 满足：
  - 外边界条件：右边界 (x=0.8) 与上边界 (y=0.8) 上 B = 0（Dirichlet）；
                左边界 (x=0) 与下边界 (y=0) 上 ∂B/∂x = 0、∂B/∂y = 0（Neumann）。
  - 内部界面：所有子矩形之间的直线上 B 连续，且法向导数 ∂B/∂n = 0（两侧均为 0）。

几何（1/4 对称，总长 0.8，坐标范围 [0, 0.8] × [0, 0.8]）：
  分割线 x=0.24、x=0.56，y=0.24、y=0.56，构成 3×3 = 9 个子矩形。
  区域划分（四分之一反应堆）：
    - Region 1：中心 (0.24<x<0.56, 0.24<y<0.56)
    - Region 2：左臂 (x<0.24, 0.24<y<0.56) 与下臂 (0.24<x<0.56, y<0.24)
    - Region 3：其余六个子矩形（反射层/blanket）
  每个子矩形内 B 为最高 x^a y^b（a, b ≤ n）的多项式，TWIGL 中 n = 5。

构造方法：
  每个子矩形有 (n+1)^2 个未知系数，总计 9×(n+1)^2。
  在所有边界与界面上取 Nd 个离散点（Nd ≤ n-1），对每个约束点列线性方程，
  形成齐次方程组 A · coeff = 0，再调用 svd.solve_nullspace 求零空间解。

仅依赖 PyTorch，使用 float32。
"""

import torch

from svd import solve_nullspace


# ==================== 几何与区域划分 ====================
XS = (0.0, 0.24, 0.56, 0.8)   # x 方向分割点（3 段）
YS = (0.0, 0.24, 0.56, 0.8)   # y 方向分割点（3 段）
NX = len(XS) - 1              # x 方向子矩形数 = 3
NY = len(YS) - 1              # y 方向子矩形数 = 3
N_CELLS = NX * NY             # 子矩形总数 = 9


def cell_id(ix, iy):
    """子矩形 (ix, iy) 的线性索引 (0..8)，ix/iy 取 0..2。"""
    return iy * NX + ix


def region_of(ix, iy):
    """返回子矩形 (ix, iy) 所属的材料区域编号 (1/2/3)。

    四分之一反应堆（1/4 对称）：
      Region 1：中心 (1,1)；
      Region 2：左臂 (0,1) 与下臂 (1,0)，与 Region 1 同为种子燃料；
      Region 3：其余 6 个子矩形（反射层/blanket）。
    """
    if ix == 1 and iy == 1:
        return 1
    if (ix == 0 and iy == 1) or (ix == 1 and iy == 0):
        return 2
    return 3


# 材料参数（2 群扩散）。B 的构造不依赖这些参数，仅作为区域元数据保存，
# 供通量方程残差（φ = B * NNs）使用。
REGION_PARAMS = {
    1: dict(D1=1.4, D2=0.4, Sr1=0.02, Sr2=0.15,
            nuSf1=0.007, nuSf2=0.2, Ss12=0.01, Ss21=0.0),
    2: dict(D1=1.4, D2=0.4, Sr1=0.02, Sr2=0.15,
            nuSf1=0.007, nuSf2=0.2, Ss12=0.01, Ss21=0.0),
    3: dict(D1=1.3, D2=0.5, Sr1=0.018, Sr2=0.05,
            nuSf1=0.003, nuSf2=0.06, Ss12=0.01, Ss21=0.0),
}


# ==================== 多项式基 ====================
def _basis_row(x, y, n, deriv="value"):
    """返回长度 (n+1)^2 的系数梯度行向量，对应单个多项式在某点的约束。

    deriv: "value" -> P(x,y)；"dx" -> ∂P/∂x；"dy" -> ∂P/∂y。
    系数排列 index = a*(n+1)+b，对应 x^a y^b。
    """
    nt = (n + 1) * (n + 1)
    row = torch.zeros(nt, dtype=torch.float32)
    idx = 0
    for a in range(n + 1):
        xa = x ** a
        xa_dx = (a * x ** (a - 1)) if a > 0 else 0.0
        for b in range(n + 1):
            if deriv == "value":
                row[idx] = xa * (y ** b)
            elif deriv == "dx":
                row[idx] = xa_dx * (y ** b)
            elif deriv == "dy":
                row[idx] = xa * ((b * y ** (b - 1)) if b > 0 else 0.0)
            else:
                raise ValueError(f"未知的导数类型: {deriv}")
            idx += 1
    return row


# ==================== 约束矩阵 ====================
def _build_constraint_matrix(n, Nd):
    """构造齐次约束矩阵 A，满足 A @ coeff = 0。"""
    nt = (n + 1) * (n + 1)
    ntot = N_CELLS * nt
    rows = []

    def add_point_constraint(ix, iy, x, y, deriv):
        """对子矩形 (ix, iy) 的多项式在某点施加齐次约束（= 0）。"""
        row = torch.zeros(ntot, dtype=torch.float32)
        off = cell_id(ix, iy) * nt
        row[off:off + nt] = _basis_row(x, y, n, deriv)
        rows.append(row)

    def add_pair_constraint(ix1, iy1, ix2, iy2, x, y, d1, d2):
        """界面连续性：cell1 与 cell2 在点上的 (导数)值之差为 0。"""
        row = torch.zeros(ntot, dtype=torch.float32)
        o1 = cell_id(ix1, iy1) * nt
        o2 = cell_id(ix2, iy2) * nt
        row[o1:o1 + nt] = _basis_row(x, y, n, d1)
        row[o2:o2 + nt] -= _basis_row(x, y, n, d2)
        rows.append(row)

    def interior_pts(lo, hi):
        """在 (lo, hi) 内取 Nd 个等距内部点。"""
        return torch.linspace(lo, hi, Nd + 2, dtype=torch.float32)[1:-1]

    # ---- 外边界 ----
    # 右边界 x=0.8: B=0
    for iy in range(NY):
        for y in interior_pts(YS[iy], YS[iy + 1]):
            add_point_constraint(NX - 1, iy, XS[-1], y.item(), "value")
    # 上边界 y=0.8: B=0
    for ix in range(NX):
        for x in interior_pts(XS[ix], XS[ix + 1]):
            add_point_constraint(ix, NY - 1, x.item(), YS[-1], "value")
    # 左边界 x=0: dB/dx=0
    for iy in range(NY):
        for y in interior_pts(YS[iy], YS[iy + 1]):
            add_point_constraint(0, iy, XS[0], y.item(), "dx")
    # 下边界 y=0: dB/dy=0
    for ix in range(NX):
        for x in interior_pts(XS[ix], XS[ix + 1]):
            add_point_constraint(ix, 0, x.item(), YS[0], "dy")

    # ---- 内部界面 ----
    # 竖直界面 x=0.24, 0.56：连续 + 两侧法向导数 = 0
    for xi in (1, 2):
        x_int = XS[xi]
        for iy in range(NY):
            for y in interior_pts(YS[iy], YS[iy + 1]):
                yv = y.item()
                add_pair_constraint(xi - 1, iy, xi, iy, x_int, yv, "value", "value")
                add_point_constraint(xi - 1, iy, x_int, yv, "dx")
                add_point_constraint(xi, iy, x_int, yv, "dx")
    # 水平界面 y=0.24, 0.56：连续 + 两侧法向导数 = 0
    for yi in (1, 2):
        y_int = YS[yi]
        for ix in range(NX):
            for x in interior_pts(XS[ix], XS[ix + 1]):
                xv = x.item()
                add_pair_constraint(ix, yi - 1, ix, yi, xv, y_int, "value", "value")
                add_point_constraint(ix, yi - 1, xv, y_int, "dy")
                add_point_constraint(ix, yi, xv, y_int, "dy")

    A = torch.stack(rows, dim=0)
    return A


# ==================== B 函数 ====================
def _make_B_func(coeffs, n):
    """由系数张量构造支持批量的可调用 B(x, y)。"""
    nt = (n + 1) * (n + 1)

    def B_func(x, y):
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        x, y = torch.broadcast_tensors(x, y)
        shape = x.shape
        xf = x.reshape(-1).contiguous()
        yf = y.reshape(-1).contiguous()
        dev = xf.device

        # 单点各次幂与单项式矩阵 (N, nt)
        pows = torch.arange(n + 1, dtype=torch.float32, device=dev)
        xp = xf[:, None].pow(pows)                       # (N, n+1)
        yp = yf[:, None].pow(pows)                       # (N, n+1)
        mono = (xp[:, :, None] * yp[:, None, :]).reshape(-1, nt)  # (N, nt)

        # 每个子矩形单独求值 (N, N_CELLS)，再按点所在子矩形选取
        vals = mono @ coeffs.to(dev).T                    # (N, N_CELLS)
        bnd_x = torch.tensor(XS[1:-1], dtype=torch.float32, device=dev)
        bnd_y = torch.tensor(YS[1:-1], dtype=torch.float32, device=dev)
        ix = torch.bucketize(xf, bnd_x)
        iy = torch.bucketize(yf, bnd_y)
        cid = iy * NX + ix
        result = torch.gather(vals, 1, cid.unsqueeze(1)).squeeze(1)
        return result.reshape(shape)

    return B_func


# ==================== 主入口 ====================
def build_B(n=5, Nd=None, tol=1e-6):
    """构造分片多项式 B(x, y)，返回 (B_func, coeffs)。

    参数
    ----
    n : int
        多项式阶数（每维最高次数），TWIGL 中 n=5。
    Nd : int 或 None
        每条边界/界面线段上的离散点数，默认 n-1（满足 Nd ≤ n-1）。
    tol : float
        SVD 奇异值阈值，传给 svd.solve_nullspace。

    返回
    ----
    B_func : callable
        B_func(x, y) -> torch.Tensor，支持批量（自动广播）；
        已按函数值归一化（max|B| ≈ 1）。
    coeffs : torch.Tensor
        形状 (9, (n+1)^2)，coeffs[c][a*(n+1)+b] 为子矩形 c 中 x^a y^b 的系数。
    """
    if Nd is None:
        Nd = n - 1
    if Nd < 1 or Nd > n - 1:
        raise ValueError(f"Nd 应满足 1 ≤ Nd ≤ n-1，当前 Nd={Nd}, n={n}")

    A = _build_constraint_matrix(n, Nd)
    # solve_nullspace 返回零空间方向（内部已做 max|coeff|=1 的初步归一化）
    coeff = solve_nullspace(A, tol=tol)
    coeffs = coeff.reshape(N_CELLS, (n + 1) * (n + 1))

    # 按 B 的函数值归一化：使 max|B(x, y)| ≈ 1（B 的量级归一）
    B_tmp = _make_B_func(coeffs, n)
    gx, gy = torch.meshgrid(
        torch.linspace(XS[0], XS[-1], 201, dtype=torch.float32),
        torch.linspace(YS[0], YS[-1], 201, dtype=torch.float32),
        indexing="ij",
    )
    scale = B_tmp(gx, gy).abs().max()
    if scale <= 0:
        raise RuntimeError("B 函数恒为零，无法进行量级归一。")
    coeffs = coeffs / scale

    B_func = _make_B_func(coeffs, n)
    return B_func, coeffs


# ==================== 自检 ====================
if __name__ == "__main__":
    n = 5
    Nd = n - 1
    B_func, coeffs = build_B(n=n, Nd=Nd)

    print("== B 函数构造完成 ==")
    print(f"多项式阶数 n = {n}, 每线段离散点数 Nd = {Nd}")
    print(f"系数张量形状: {tuple(coeffs.shape)}")

    print("\n== 区域划分 (ix, iy) -> Region ==")
    for iy in reversed(range(NY)):
        print("   " + " ".join(str(region_of(ix, iy)) for ix in range(NX)))

    # 每个子矩形系数是否非零
    print("\n== 各子矩形系数最大绝对值 ==")
    for iy in reversed(range(NY)):
        vals = [coeffs[cell_id(ix, iy)].abs().max().item() for ix in range(NX)]
        print("   " + " ".join(f"{v:.2e}" for v in vals))

    # 约束残差校验
    A = _build_constraint_matrix(n, Nd)
    resid = A @ coeffs.reshape(-1)
    print(f"\n约束残差 ||A@coeff||_max = {resid.abs().max().item():.3e}")

    # 边界值校验
    xs = torch.linspace(0.0, 0.8, 40, dtype=torch.float32)
    ys = torch.linspace(0.0, 0.8, 40, dtype=torch.float32)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    B = B_func(gx, gy)
    print(f"B 值范围: [{B.min().item():.4g}, {B.max().item():.4g}]")

    B_right = B_func(torch.full_like(ys, 0.8), ys)
    B_top = B_func(xs, torch.full_like(xs, 0.8))
    print(f"右边界 B 最大绝对值 = {B_right.abs().max().item():.3e}")
    print(f"上边界 B 最大绝对值 = {B_top.abs().max().item():.3e}")

    eps = 1e-5
    dB_dx0 = (B_func(eps, ys) - B_func(0.0, ys)) / eps
    dB_dy0 = (B_func(xs, eps) - B_func(xs, 0.0)) / eps
    print(f"左边界 dB/dx 最大绝对值 = {dB_dx0.abs().max().item():.3e}")
    print(f"下边界 dB/dy 最大绝对值 = {dB_dy0.abs().max().item():.3e}")
