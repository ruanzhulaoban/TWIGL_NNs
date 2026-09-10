"""预测与可视化模块。

加载训练好的两个能群网络（默认 twigl_g1.pt / twigl_g2.pt），在细网格上预测
通量 φ1、φ2，绘制 jet 热力图并保存；同时可视化训练点（高斯点）分布与
材料区域划分并保存。

只做预测/可视化，不训练。依赖 matplotlib（绘图）与 PyTorch（推理）。

用法：
    python predict.py [path_g1] [path_g2] [--grid N] [--n_sub N] [--device cuda|cpu]
"""

import argparse
import os

import torch
import matplotlib
matplotlib.use("Agg")  # 无显示环境直接保存
import matplotlib.pyplot as plt

from B_function import XS, YS, NX, NY, region_of
from network import PSNNNet
from train import make_training_points


# ==================== 加载与预测 ====================
def load_models(path1, path2, device="cuda"):
    """加载两个能群网络（B、s 由 checkpoint 内系数确定性重建），返回
    (net1, net2, keff, meta)。"""
    net1, keff1, meta1 = PSNNNet.load(path1)
    net2, keff2, meta2 = PSNNNet.load(path2)
    net1 = net1.to(device)
    net2 = net2.to(device)
    return net1, net2, keff1, meta1


def predict_fluxes(net1, net2, n_grid=400, device="cuda"):
    """在 [0,0.8]^2 的 n_grid×n_grid 细网格上预测通量。

    返回 (X, Y, phi1, phi2)，均为 numpy 数组，X/Y 为网格坐标。
    """
    xs = torch.linspace(XS[0], XS[-1], n_grid, dtype=torch.float32, device=device)
    ys = torch.linspace(YS[0], YS[-1], n_grid, dtype=torch.float32, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    with torch.no_grad():
        phi1 = net1(gx, gy).cpu().numpy()
        phi2 = net2(gx, gy).cpu().numpy()
    return gx.cpu().numpy(), gy.cpu().numpy(), phi1, phi2


# ==================== 绘图 ====================
def _draw_region_lines(ax):
    """绘制子区域分界线与材料区域背景参考。"""
    for xb in XS[1:-1]:
        ax.axvline(xb, color="white", lw=0.9, ls="--", alpha=0.7)
    for yb in YS[1:-1]:
        ax.axhline(yb, color="white", lw=0.9, ls="--", alpha=0.7)
    ax.set_xlim(XS[0], XS[-1])
    ax.set_ylim(YS[0], YS[-1])
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_aspect("equal", adjustable="box")


def plot_heatmap(X, Y, Z, title, path, cmap="jet", vmin=None, vmax=None):
    """单张 jet 热力图并保存。"""
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    _draw_region_lines(ax)
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("flux")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_fluxes_combined(X, Y, phi1, phi2, path):
    """φ1、φ2 并排组合热力图。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    vmax = max(phi1.max(), phi2.max())
    for ax, Z, t in zip(axes, (phi1, phi2), ("Fast flux φ1", "Thermal flux φ2")):
        im = ax.pcolormesh(X, Y, Z, cmap="jet", shading="auto", vmin=0.0, vmax=vmax)
        _draw_region_lines(ax)
        ax.set_title(t)
        fig.colorbar(im, ax=ax, pad=0.02).set_label("flux")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_training_points(path, n_sub=50):
    """训练点（高斯点）分布与材料区域划分，按区域着色并保存。"""
    x, y, w, region = make_training_points(n_sub=n_sub)
    x = x.numpy()
    y = y.numpy()
    region = region.numpy()

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    colors = {1: "#d62728", 2: "#ff7f0e", 3: "#1f77b4"}
    names = {1: "Region 1 (seed)", 2: "Region 2 (seed arm)", 3: "Region 3 (blanket)"}
    for r in (1, 2, 3):
        m = region == r
        ax.scatter(x[m], y[m], s=4, c=colors[r], label=names[r],
                   alpha=0.55, linewidths=0)
    for xb in XS[1:-1]:
        ax.axvline(xb, color="k", lw=1.1)
    for yb in YS[1:-1]:
        ax.axhline(yb, color="k", lw=1.1)
    ax.set_xlim(XS[0], XS[-1])
    ax.set_ylim(YS[0], YS[-1])
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Training points ({n_sub}x{n_sub} Gauss points per sub-region, {x.size} total)")
    ax.legend(markerscale=5, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ==================== 主流程 ====================
def main(path1, path2, n_grid=400, n_sub=50, outdir=".", device="cuda"):
    os.makedirs(outdir, exist_ok=True)

    net1, net2, keff, meta = load_models(path1, path2, device=device)
    print(f"Loaded models, keff = {keff:.8f}")

    X, Y, phi1, phi2 = predict_fluxes(net1, net2, n_grid=n_grid, device=device)
    print(f"Flux range φ1: [{phi1.min():.4g}, {phi1.max():.4g}], "
          f"φ2: [{phi2.min():.4g}, {phi2.max():.4g}]")

    # B 由 SVD 构造存在符号不定，若通量整体为负则翻转为正（物理通量 ≥ 0）
    if phi1.mean() < 0:
        phi1 = -phi1
    if phi2.mean() < 0:
        phi2 = -phi2

    plot_heatmap(X, Y, phi1, "Fast flux φ1",
                 os.path.join(outdir, "twigl_phi1.png"))
    plot_heatmap(X, Y, phi2, "Thermal flux φ2",
                 os.path.join(outdir, "twigl_phi2.png"))
    plot_fluxes_combined(X, Y, phi1, phi2,
                         os.path.join(outdir, "twigl_fluxes.png"))
    plot_training_points(os.path.join(outdir, "twigl_training_points.png"),
                         n_sub=n_sub)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="TWIGL PINN 预测与可视化")
    p.add_argument("path1", nargs="?", default="twigl_g1.pt")
    p.add_argument("path2", nargs="?", default="twigl_g2.pt")
    p.add_argument("--grid", type=int, default=400, help="预测网格分辨率")
    p.add_argument("--n_sub", type=int, default=50, help="训练点每子区域每维点数")
    p.add_argument("--device", default="cuda", help="cuda 或 cpu")
    p.add_argument("--outdir", default=".", help="输出目录")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, fallback to cpu.")
        args.device = "cpu"

    main(args.path1, args.path2, n_grid=args.grid, n_sub=args.n_sub,
         outdir=args.outdir, device=args.device)
