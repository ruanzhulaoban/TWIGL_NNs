"""构造分片多项式向量函数 s(x, y) = (u(x, y), v(x, y)) 的模块。

s 用于神经网络专门层，每个能群 g 独立构造一个 s_g（界面条件涉及扩散系数 D_g）。

几何与区域划分同 B_function（真实坐标 [0, 0.8]^2，分割线 x=0.24、x=0.56，
y=0.24、y=0.56，共 9 个子区域，材料映射与 B 一致）。

s 的约束条件（仅内部界面，无外边界约束）：
  - 每个内部界面上 u、v 均连续；
  - 法向导数连续（含扩散系数）：
      水平界面（法向 y）：D_left·∂u/∂y = D_right·∂u/∂y（v 同理）；
      竖直界面（法向 x）：D_left·∂u/∂x = D_right·∂u/∂x（v 同理）；
  - 每个子区域内 u、v 均为最高 x^a y^b（a, b ≤ n）的多项式。

构造方法：
  每个子区域 u 有 (n+1)^2 个系数，v 同样，总计 9×2×(n+1)^2。
  由于 u、v 满足完全相同的界面约束（D_g 只与区域、能群有关），且二者互不耦合，
  故构造单分量约束矩阵 A，取其零空间的两个独立基向量分别作为 u、v 的系数；
  在界面离散点（Nd ≤ n-1）上列出连续性与法向导数方程，调用 SVD 求零空间，
  再按函数值归一化（max|u| ≈ 1、max|v| ≈ 1）。

仅依赖 PyTorch，使用 float32。
"""

import torch

from svd import nullspace_basis
from B_function import (
    XS, YS, NX, NY, N_CELLS, cell_id, region_of, REGION_PARAMS,
    _basis_row, _make_B_func as _make_piecewise_func,
)


def D_of(region, g):
    """区域 region 在能群 g 下的扩散系数 D_g。"""
    return REGION_PARAMS[region][f"D{g}"]


def _build_constraint_matrix(n, Nd, g):
    """构造单分量的齐次约束矩阵 A（形状 (24*Nd, 9*(n+1)^2)）。

    u、v 满足相同的约束，故只需构造单分量矩阵，再取其零空间的两个
    独立基向量分别作为 u 与 v 的系数。
    """
    nt = (n + 1) * (n + 1)
    ntot = N_CELLS * nt
    rows = []

    def add_continuity(c1, c2, x, y):
        """值连续性：P(c1) - P(c2) = 0。"""
        row = torch.zeros(ntot, dtype=torch.float32)
        row[c1 * nt:(c1 + 1) * nt] = _basis_row(x, y, n, "value")
        row[c2 * nt:(c2 + 1) * nt] -= _basis_row(x, y, n, "value")
        rows.append(row)

    def add_flux(c1, c2, x, y, D1, D2, deriv):
        """D 加权法向导数连续：D1·∂P(c1) - D2·∂P(c2) = 0。"""
        row = torch.zeros(ntot, dtype=torch.float32)
        row[c1 * nt:(c1 + 1) * nt] = D1 * _basis_row(x, y, n, deriv)
        row[c2 * nt:(c2 + 1) * nt] -= D2 * _basis_row(x, y, n, deriv)
        rows.append(row)

    def interior_pts(lo, hi):
        return torch.linspace(lo, hi, Nd + 2, dtype=torch.float32)[1:-1]

    # 竖直界面 x=0.24, 0.56（法向 x）
    for xi in (1, 2):
        x_int = XS[xi]
        for iy in range(NY):
            cL = cell_id(xi - 1, iy)
            cR = cell_id(xi, iy)
            DL = D_of(region_of(xi - 1, iy), g)
            DR = D_of(region_of(xi, iy), g)
            for y in interior_pts(YS[iy], YS[iy + 1]):
                yv = y.item()
                add_continuity(cL, cR, x_int, yv)
                add_flux(cL, cR, x_int, yv, DL, DR, "dx")

    # 水平界面 y=0.24, 0.56（法向 y）
    for yi in (1, 2):
        y_int = YS[yi]
        for ix in range(NX):
            cB = cell_id(ix, yi - 1)
            cT = cell_id(ix, yi)
            DB = D_of(region_of(ix, yi - 1), g)
            DT = D_of(region_of(ix, yi), g)
            for x in interior_pts(XS[ix], XS[ix + 1]):
                xv = x.item()
                add_continuity(cB, cT, xv, y_int)
                add_flux(cB, cT, xv, y_int, DB, DT, "dy")

    return torch.stack(rows, dim=0)


def build_s(g, n=5, Nd=None, tol=1e-6):
    """构造能群 g 的分片多项式向量函数 s_g(x, y) = (u, v)。

    参数
    ----
    g : int
        能群编号（1 或 2），决定界面条件中的扩散系数 D_g。
        D_g 从 B_function.REGION_PARAMS 读取。
    n : int
        多项式阶数（每维最高次数），默认 5。
    Nd : int 或 None
        每条界面线段上的离散点数，默认 n-1（满足 Nd ≤ n-1）。
    tol : float
        SVD 奇异值阈值，传给 svd.nullspace_basis。

    返回
    ----
    s_func : callable
        s_func(x, y) -> (u, v)，u、v 为 torch.Tensor，支持批量（自动广播）；
        已按函数值归一化（max|u| ≈ 1，max|v| ≈ 1）。
    coeffs : torch.Tensor
        形状 (2, 9, (n+1)^2)，coeffs[0] 为 u 的系数，coeffs[1] 为 v 的系数。
    """
    if g not in (1, 2):
        raise ValueError(f"能群 g 应为 1 或 2，当前 g={g}")
    if Nd is None:
        Nd = n - 1
    if Nd < 1 or Nd > n - 1:
        raise ValueError(f"Nd 应满足 1 ≤ Nd ≤ n-1，当前 Nd={Nd}, n={n}")

    nt = (n + 1) * (n + 1)
    A = _build_constraint_matrix(n, Nd, g)   # 单分量约束矩阵 (24*Nd, 324)
    basis = nullspace_basis(A, tol=tol)      # (324, d)，d ≥ 2

    if basis.shape[1] < 2:
        raise RuntimeError(
            f"零空间维数不足（{basis.shape[1]} < 2），无法构造两个分量。"
        )

    # 取两个独立零空间基向量分别作为 u、v 的系数
    coeff_u = basis[:, 0].reshape(N_CELLS, nt)
    coeff_v = basis[:, 1].reshape(N_CELLS, nt)

    # 按函数值归一化：max|u| ≈ 1，max|v| ≈ 1
    u_tmp = _make_piecewise_func(coeff_u, n)
    v_tmp = _make_piecewise_func(coeff_v, n)
    gx, gy = torch.meshgrid(
        torch.linspace(XS[0], XS[-1], 201, dtype=torch.float32),
        torch.linspace(YS[0], YS[-1], 201, dtype=torch.float32),
        indexing="ij",
    )
    su = u_tmp(gx, gy).abs().max()
    sv = v_tmp(gx, gy).abs().max()
    if su <= 0 or sv <= 0:
        raise RuntimeError("s 的某个分量恒为零，无法进行量级归一。")
    coeff_u = coeff_u / su
    coeff_v = coeff_v / sv

    u_func = _make_piecewise_func(coeff_u, n)
    v_func = _make_piecewise_func(coeff_v, n)

    def s_func(x, y):
        return u_func(x, y), v_func(x, y)

    coeffs = torch.stack([coeff_u, coeff_v], dim=0)  # (2, 9, nt)
    return s_func, coeffs


if __name__ == "__main__":
    n = 5
    Nd = 4
    for g in (1, 2):
        s_func, coeffs = build_s(g=g, n=n, Nd=Nd)
        print(f"\n===== 能群 g = {g} =====")
        print(f"系数张量形状: {tuple(coeffs.shape)}")

        A = _build_constraint_matrix(n, Nd, g)
        r_u = (A @ coeffs[0].reshape(-1)).abs().max().item()
        r_v = (A @ coeffs[1].reshape(-1)).abs().max().item()
        print(f"约束残差 ||A@coeff_u||_max = {r_u:.3e}, ||A@coeff_v||_max = {r_v:.3e}")

        xs = torch.linspace(0.0, 0.8, 60, dtype=torch.float32)
        ys = torch.linspace(0.0, 0.8, 60, dtype=torch.float32)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        u, v = s_func(gx, gy)
        print(f"u 范围: [{u.min().item():.4g}, {u.max().item():.4g}], "
              f"v 范围: [{v.min().item():.4g}, {v.max().item():.4g}]")

        # 界面连续性直观检查：x=0.56（region1|region3 界面）两侧 u、v 跳变
        x0, y0 = 0.56, 0.4
        eps = 1e-5
        uL, vL = s_func(x0 - eps, y0)
        uR, vR = s_func(x0 + eps, y0)
        print(f"界面 x=0.56, y=0.4 处 u 跳变 = {(uL - uR).abs().item():.3e}, "
              f"v 跳变 = {(vL - vR).abs().item():.3e}")
