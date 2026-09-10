"""SVD求解模块：求解齐次线性方程组 A x = 0 的零空间。

给定 m×n 的系数矩阵 A（通常 m < n），提供：
  - nullspace_basis：返回零空间的一组正交基（列向量）；
  - solve_nullspace：返回一个归一化（最大分量为 1）的非零解向量。

仅依赖 PyTorch，使用 float32。
"""

import torch


def nullspace_basis(A: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """返回 A 的零空间的一组正交基（列向量）。

    返回形状 (n, d) 的张量，d 为零空间维数；各列满足 A @ v ≈ 0。
    涵盖 m < n 时 V 的末 n - m 列（对应隐式零奇异值）。

    参数
    ----
    A : torch.Tensor
        系数矩阵，形状 (m, n)，单精度（float32）。
    tol : float
        奇异值判定阈值，小于该值视为零空间基。
    """
    if A.dtype != torch.float32:
        A = A.to(torch.float32)

    if A.ndim != 2:
        raise ValueError(f"输入应为二维矩阵，实际维度为 {A.ndim}")

    m, n = A.shape
    U, S, Vh = torch.linalg.svd(A, full_matrices=True)
    V = Vh.T
    k = S.numel()  # min(m, n)
    null_mask = torch.zeros(n, dtype=torch.bool, device=A.device)
    null_mask[:k] = S < tol
    null_mask[k:] = True  # m < n 时的隐式零奇异值
    null_indices = null_mask.nonzero(as_tuple=False).flatten()
    return V[:, null_indices].contiguous()


def solve_nullspace(A: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """求解齐次线性方程组 A x = 0，返回一个归一化的非零解。

    返回
    ----
    x : torch.Tensor
        非零解向量，形状 (n,)，满足 A @ x ≈ 0，且 max(abs(x)) == 1。

    抛出
    ----
    ValueError
        若零空间维度为 0（无非零解）。
    """
    basis = nullspace_basis(A, tol=tol)
    if basis.shape[1] == 0:
        raise ValueError("零空间维度为 0，齐次方程组 A x = 0 无非零解。")

    # 从零空间基中依次选取非零列作为解
    for i in range(basis.shape[1]):
        x = basis[:, i]
        if torch.any(x != 0):
            return x / torch.max(torch.abs(x))

    raise ValueError("零空间基全为零向量，无法得到非零解。")
