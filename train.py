"""源迭代训练模块（并行优化）。

实现源迭代算法（功率迭代）求解 TWIGL 两群中子扩散基准：交替求解两个能群的
扩散方程（各自用一个 PSNNNet），并在外层更新有效增殖因子 keff 直至收敛。

物理方程（能群 g，去掉 g 下标书写）：
    -∇·(D ∇φ) + Σr φ = Q
    Q_g = (χ_g / keff) Σ_{g'} νΣf_{g'} φ_{g'} + Σ_{g'≠g} Σs_{g'→g} φ_{g'}
训练点均位于子区域内部，D 为分片常数，故 -∇·(D ∇φ) = -D ∇²φ。

TWIGL 裂变谱：χ = (1.0, 0.0)（裂变中子全部产生于快群）；散射 Ss21 = 0。
于是：
    Q_1 = (1/keff)(νΣf1 φ1 + νΣf2 φ2)
    Q_2 = Ss12 φ1

并行策略：
  - 多 GPU 可将每个能群网络用 nn.DataParallel 包装（批数据切分到多卡）；
    或将两个能群的训练作为独立进程用 torch.multiprocessing 异步并行（Jacobi 式，
    两群源项均取上一轮 φ）。
  - 单卡则在同一个外层循环内依次训练两个网络（Gauss-Seidel 式）：
    先训群 1（源项用上一轮 φ1、φ2），再用本轮更新后的 φ1 计算群 2 源项并训群 2，
    保证散射源项使用最新 φ 值。

注意：源项 Q 用 detach()/no_grad 计算，不参与梯度。

仅依赖 PyTorch，使用 float32。
"""

import json
import os

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")  # 无显示环境直接保存
import matplotlib.pyplot as plt

from B_function import XS, YS, NX, NY, region_of, REGION_PARAMS, build_B
from s_layer import build_s
from network import PSNNNet, reconstruct_net

# TWIGL 裂变谱（χ1=1, χ2=0：裂变中子全部产生于快群）
CHI = (1.0, 0.0)


# ==================== Gauss-Legendre 求积（纯 torch） ====================
def _legendre_val(n, x):
    """返回 (P_n(x), P_{n-1}(x))，x 为张量。"""
    p = torch.ones_like(x)       # P_0
    pm1 = torch.zeros_like(x)    # P_{-1}
    for k in range(1, n + 1):
        pk = ((2 * k - 1) * x * p - (k - 1) * pm1) / k
        pm1 = p
        p = pk
    return p, pm1


def gauss_legendre(n):
    """n 点 Gauss-Legendre 节点与权重（区间 [-1, 1]），纯 torch float32。"""
    i = torch.arange(n, dtype=torch.float32)
    x = torch.cos(torch.pi * (i + 0.75) / (n + 0.5))  # 初值（Chebyshev）
    for _ in range(200):
        pn, pnm1 = _legendre_val(n, x)
        dpn = n * (x * pn - pnm1) / (x * x - 1.0)
        x_new = x - pn / dpn
        if (x_new - x).abs().max() < 1e-6:
            x = x_new
            break
        x = x_new
    pn, pnm1 = _legendre_val(n, x)
    dpn = n * (x * pn - pnm1) / (x * x - 1.0)
    w = 2.0 / ((1.0 - x * x) * dpn * dpn)
    return x, w


def _map_nodes(nodes, weights, a, b):
    """把 [-1,1] 节点/权重仿射映射到 [a, b]。"""
    x = 0.5 * (a + b) + 0.5 * (b - a) * nodes
    w = 0.5 * (b - a) * weights
    return x, w


# ==================== 训练点与材料参数 ====================
def make_training_points(n_sub=50):
    """每个子区域 n_sub×n_sub 个高斯点，共 9*n_sub^2 点。

    返回 (x, y, w, region)：坐标、高斯积分权重、材料区域索引（长整型）。
    """
    nodes, wts = gauss_legendre(n_sub)
    xs, ys, ws, regions = [], [], [], []
    for ix in range(NX):
        xnode, xw = _map_nodes(nodes, wts, XS[ix], XS[ix + 1])
        for iy in range(NY):
            ynode, yw = _map_nodes(nodes, wts, YS[iy], YS[iy + 1])
            X, Y = torch.meshgrid(xnode, ynode, indexing="ij")
            Wx, Wy = torch.meshgrid(xw, yw, indexing="ij")
            W = Wx * Wy
            r = region_of(ix, iy)
            xs.append(X.reshape(-1))
            ys.append(Y.reshape(-1))
            ws.append(W.reshape(-1))
            regions.append(torch.full((n_sub * n_sub,), r, dtype=torch.long))
    x = torch.cat(xs)
    y = torch.cat(ys)
    w = torch.cat(ws)
    region = torch.cat(regions)
    return x, y, w, region


def material_tensors(region, device):
    """按每点的区域索引构造材料参数字段（各字段形状 [N]）。"""
    keys = ["D1", "D2", "Sr1", "Sr2", "nuSf1", "nuSf2", "Ss12", "Ss21"]
    out = {k: torch.empty(region.numel(), dtype=torch.float32, device=device)
           for k in keys}
    for r, p in REGION_PARAMS.items():
        mask = region == r
        for k in keys:
            out[k][mask] = p[k]
    return out


# ==================== 网络构造（共享 B） ====================
def build_networks(n_poly=5, hidden_layers=8, neurons=400, device="cuda"):
    """构造两个能群网络，共享同一个 B_func（B 与能群无关，仅构造一次）。"""
    B_func, B_coeffs = build_B(n=n_poly)
    s1, sc1 = build_s(g=1, n=n_poly)
    s2, sc2 = build_s(g=2, n=n_poly)
    net1 = PSNNNet(B_func, s1, B_coeffs=B_coeffs, s_coeffs=sc1, n=n_poly,
                   hidden_layers=hidden_layers, neurons=neurons).to(device)
    net2 = PSNNNet(B_func, s2, B_coeffs=B_coeffs, s_coeffs=sc2, n=n_poly,
                   hidden_layers=hidden_layers, neurons=neurons).to(device)
    return net1, net2


# ==================== 残差与源项 ====================
def residual(model, x, y, D, Sr, Q):
    """残差 R = -D ∇²φ + Sr φ - Q，返回 [N]（含计算图，用于反传）。"""
    phi = model(x, y)
    ones = torch.ones_like(phi)
    dphi_dx = torch.autograd.grad(phi, x, ones, create_graph=True)[0]
    dphi_dy = torch.autograd.grad(phi, y, ones, create_graph=True)[0]
    d2phi_dx2 = torch.autograd.grad(dphi_dx, x, ones, create_graph=True)[0]
    d2phi_dy2 = torch.autograd.grad(dphi_dy, y, ones, create_graph=True)[0]
    lap = d2phi_dx2 + d2phi_dy2
    return -D * lap + Sr * phi - Q


def residual_loss(model, x, y, D, Sr, Q):
    """损失 = mean(R^2)，标量。"""
    R = residual(model, x, y, D, Sr, Q)
    return (R ** 2).mean()


@torch.no_grad()
def compute_sources(phi1, phi2, keff, mat):
    """计算两个能群的源项（不参与梯度）。

    Q_1 = (χ1/keff)(νΣf1 φ1 + νΣf2 φ2) + Ss21 φ2
    Q_2 = (χ2/keff)(νΣf1 φ1 + νΣf2 φ2) + Ss12 φ1
    返回 (Q1, Q2)。
    """
    fs = mat["nuSf1"] * phi1 + mat["nuSf2"] * phi2
    Q1 = (CHI[0] / keff) * fs + mat["Ss21"] * phi2
    Q2 = (CHI[1] / keff) * fs + mat["Ss12"] * phi1
    return Q1, Q2


@torch.no_grad()
def fission_source_total(mat, phi1, phi2, w):
    """总裂变中子产生率 ∫(νΣf1 φ1 + νΣf2 φ2) dV（高斯加权）。

    等价于 Σ_g ∫ χ_g νΣf_g φ_g dV（χ1+χ2=1，谱权重求和后消去）。
    """
    return ((mat["nuSf1"] * phi1 + mat["nuSf2"] * phi2) * w).sum()


# ==================== 单能群内层训练 ====================
def train_group(net, opt_adam, x, y, D, Sr, Q,
                adam_steps, lbfgs_steps, lbfgs_lr,
                label="", report_every=100, verbose=True):
    """训练单个能群网络：先 Adam 后 L-BFGS，返回最终残差损失（标量）。"""
    for step in range(adam_steps):
        opt_adam.zero_grad()
        loss = residual_loss(net, x, y, D, Sr, Q)
        loss.backward()
        opt_adam.step()
        if verbose and report_every > 0 and (step + 1) % report_every == 0:
            print(f"  [{label}] Adam epoch {step + 1}/{adam_steps} "
                  f"loss={loss.item():.3e}", flush=True)

    opt_lbfgs = torch.optim.LBFGS(
        net.parameters(), lr=lbfgs_lr, max_iter=lbfgs_steps,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt_lbfgs.zero_grad()
        loss = residual_loss(net, x, y, D, Sr, Q)
        loss.backward()
        return loss

    opt_lbfgs.step(closure)
    return residual_loss(net, x, y, D, Sr, Q).detach().item()


# ==================== 训练历史可视化 ====================
def plot_history(history, path, dpi=150):
    """将训练历史（损失与 keff）绘制为折线图并保存。

    左侧：各能群残差损失与总损失（对数纵轴，跨数量级时更清晰）；
    右侧：keff 随外层迭代的变化。history 为 list[dict]，含键
    iter / keff / loss1 / loss2 / total_loss（由 train 主循环逐轮追加）。
    """
    if not history:
        return
    iters = [h["iter"] for h in history]
    loss1 = [h["loss1"] for h in history]
    loss2 = [h["loss2"] for h in history]
    total = [h["total_loss"] for h in history]
    keff = [h["keff"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    ax = axes[0]
    ax.semilogy(iters, total, "o-", lw=1.2, ms=3, color="#1f77b4",
                label="total loss")
    ax.semilogy(iters, loss1, "s-", lw=1.0, ms=3, color="#d62728",
                label="loss g1")
    ax.semilogy(iters, loss2, "^-", lw=1.0, ms=3, color="#2ca02c",
                label="loss g2")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("residual loss")
    ax.set_title("Loss")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend()

    ax = axes[1]
    ax.plot(iters, keff, "o-", lw=1.2, ms=3, color="#1f77b4")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("k_eff")
    ax.set_title("k_eff")
    ax.grid(True, ls="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ==================== 实时落盘（原子写入，避免中断留下半文件） ====================
def save_history(history, path):
    """实时把训练历史（损失/keff）写入 JSON 文件。

    history 为 list[dict]，含 iter/keff/keff_change/loss1/loss2/total_loss/
    loss_change。先写临时文件并 fsync，再 os.replace 原子替换，保证任意时刻
    磁盘上都是一份完整、可读的最新历史。
    """
    if not history:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_ckpt_atomic(net, path, keff, group):
    """原子保存单个能群网络：先写 .tmp 再 os.replace，覆盖为最新权重。"""
    tmp = path + ".tmp"
    net.save(tmp, keff=keff, group=group)
    os.replace(tmp, path)


def save_checkpoint(path, net1, net2, opt1, opt2, n_done, keff, history,
                    total_loss_old, n_sub, max_outer, keff_tol, loss_tol):
    """保存断点续训所需的完整状态（原子写入）。

    与单个模型文件（save_ckpt_atomic）不同，这里额外保存两个 Adam 优化器的
    状态、已完成的外层迭代数 n_done、训练历史与收敛阈值，供中断后精确续训。
    B、s 系数一并保存，恢复时确定性重建而不重做 SVD。
    """
    ckpt = {
        "net1_state_dict": net1.state_dict(),
        "net2_state_dict": net2.state_dict(),
        "opt1_adam": opt1.state_dict(),
        "opt2_adam": opt2.state_dict(),
        # 网络确定性重建信息
        "n": net1.n,
        "hidden_layers": net1.hidden_layers,
        "neurons": net1.neurons,
        "activation": net1.activation_name,
        "B_coeffs": net1.B_coeffs,
        "s1_coeffs": net1.s_coeffs,
        "s2_coeffs": net2.s_coeffs,
        # 训练状态
        "n_done": n_done,
        "keff": keff,
        "history": history,
        "total_loss_old": total_loss_old,
        "n_sub": n_sub,
        "max_outer": max_outer,
        "keff_tol": keff_tol,
        "loss_tol": loss_tol,
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device, adam_lr):
    """载入断点，确定性重建两个网络与 Adam 优化器。

    返回 (net1, net2, opt1, opt2, state)。state 含 n_done/keff/history/
    total_loss_old/n_sub/max_outer/keff_tol/loss_tol 等续训所需字段。
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    act_name = ckpt.get("activation", "Tanh")
    activation = getattr(nn, act_name, nn.Tanh)
    net1 = reconstruct_net(ckpt["B_coeffs"], ckpt["s1_coeffs"], ckpt["n"],
                           hidden_layers=ckpt["hidden_layers"],
                           neurons=ckpt["neurons"],
                           activation=activation, dtype=torch.float32,
                           device=device)
    net2 = reconstruct_net(ckpt["B_coeffs"], ckpt["s2_coeffs"], ckpt["n"],
                           hidden_layers=ckpt["hidden_layers"],
                           neurons=ckpt["neurons"],
                           activation=activation, dtype=torch.float32,
                           device=device)
    net1.load_state_dict(ckpt["net1_state_dict"])
    net2.load_state_dict(ckpt["net2_state_dict"])
    opt1 = torch.optim.Adam(net1.parameters(), lr=adam_lr)
    opt2 = torch.optim.Adam(net2.parameters(), lr=adam_lr)
    opt1.load_state_dict(ckpt["opt1_adam"])
    opt2.load_state_dict(ckpt["opt2_adam"])
    return net1, net2, opt1, opt2, ckpt


# ==================== 主训练流程 ====================
def train(n_sub=50, n_poly=5, hidden_layers=8, neurons=400,
          adam_lr=1e-3, adam_steps=50000, lbfgs_lr=1.0, lbfgs_steps=500,
          max_outer=30, keff_tol=1e-6, loss_tol=1e-4,
          device="cuda", save_path="twigl", plot_every=1, ckpt_every=1,
          resume=True, verbose=True):
    """源迭代求解 TWIGL，返回 (net1, net2, keff, history)。

    参数
    ----
    n_sub : int
        每个子区域每维高斯点数（50 → 共 22500 点）。
    n_poly : int
        B、s 的多项式阶数，TWIGL 中 n=5。
    hidden_layers, neurons : int
        网络结构，默认 8 层、每层 400 神经元。
    adam_lr, adam_steps : float, int
        Adam 学习率与步数（每个外层迭代内），默认 5000 步。
    lbfgs_lr, lbfgs_steps : float, int
        L-BFGS 学习率与步数。
    max_outer : int
        外层最大迭代数，默认 10000。
    keff_tol, loss_tol : float
        keff 与损失的收敛容差。
    device : str
        "cuda" 或 "cpu"，默认 "cuda"。
    save_path : str
        保存前缀，实际保存为 <save_path>_g1.pt / <save_path>_g2.pt（实时覆盖，
        始终为最新权重）与 <save_path>_history.json（实时训练历史）。
    plot_every : int
        每多少个外层迭代更新一次损失/keff 折线图（实时监控），
        折线图保存为 <save_path>_history.png；设为 0 或负数则关闭绘图。
    ckpt_every : int
        每多少个外层迭代实时覆盖保存一次模型权重（默认 1，即每个外层迭代都
        落盘）；设为 0 或负数则仅在训练结束时保存。
    resume : bool
        若为 True 且存在 <save_path>_checkpoint.pt，则自动从断点续训；
        否则从头训练。
    """
    # 1. 训练点
    x, y, w, region = make_training_points(n_sub=n_sub)
    x = x.to(device).requires_grad_(True)
    y = y.to(device).requires_grad_(True)
    w = w.to(device)
    mat = material_tensors(region, device)

    # 实时保存路径：模型权重/断点每次覆盖为最新，历史 JSON 每轮追加写入
    p1 = f"{save_path}_g1.pt"
    p2 = f"{save_path}_g2.pt"
    history_path = f"{save_path}_history.json"
    ckpt_path = f"{save_path}_checkpoint.pt"

    # 2. 网络与优化器：存在断点则恢复续训，否则全新构造
    n_done = 0
    if resume and os.path.exists(ckpt_path):
        net1, net2, opt1_adam, opt2_adam, state = load_checkpoint(
            ckpt_path, device, adam_lr)
        keff = state["keff"]
        history = list(state["history"])
        total_loss_old = state["total_loss_old"]
        n_done = state["n_done"]
        # 用最新权重重算上一轮通量与裂变源（与保存时刻的网络状态一致）
        with torch.no_grad():
            phi1_prev = net1(x, y)
            phi2_prev = net2(x, y)
        F_old = fission_source_total(mat, phi1_prev, phi2_prev, w).item()
        if state.get("n_sub", n_sub) != n_sub and verbose:
            print(f"Warning: 断点 n_sub={state.get('n_sub')} 与当前 {n_sub} 不一致，"
                  f"训练点按当前值重建，续训结果可能偏离。")
        if verbose:
            print(f"已恢复断点 {ckpt_path}：完成 {n_done}/"
                  f"{state.get('max_outer', '?')} 轮，keff={keff:.8f}，"
                  f"续训自第 {n_done + 1} 轮。")
    else:
        net1, net2 = build_networks(n_poly=n_poly, hidden_layers=hidden_layers,
                                    neurons=neurons, device=device)
        opt1_adam = torch.optim.Adam(net1.parameters(), lr=adam_lr)
        opt2_adam = torch.optim.Adam(net2.parameters(), lr=adam_lr)
        # 初始 φ=1.0，初始 keff=1.0，初始裂变源
        phi1_prev = torch.ones_like(x)
        phi2_prev = torch.ones_like(x)
        keff = 1.0
        F_old = fission_source_total(mat, phi1_prev, phi2_prev, w).item()
        history = []
        total_loss_old = float("inf")

    for n in range(n_done + 1, max_outer + 1):
        # ---- 群 1：源项用上一轮 φ1、φ2 ----
        Q1, _ = compute_sources(phi1_prev, phi2_prev, keff, mat)
        loss1 = train_group(net1, opt1_adam, x, y, mat["D1"], mat["Sr1"], Q1,
                            adam_steps, lbfgs_steps, lbfgs_lr,
                            label="g1", verbose=verbose)

        # ---- 群 2：散射源项用本轮最新 φ1（Gauss-Seidel） ----
        with torch.no_grad():
            phi1_new = net1(x, y)
        _, Q2 = compute_sources(phi1_new, phi2_prev, keff, mat)
        loss2 = train_group(net2, opt2_adam, x, y, mat["D2"], mat["Sr2"], Q2,
                            adam_steps, lbfgs_steps, lbfgs_lr,
                            label="g2", verbose=verbose)

        with torch.no_grad():
            phi2_new = net2(x, y)

        # ---- keff 更新 ----
        F_new = fission_source_total(mat, phi1_new, phi2_new, w).item()
        keff_new = keff * (F_new / F_old)

        total_loss = loss1 + loss2
        keff_change = abs(keff_new - keff)
        loss_change = abs(total_loss - total_loss_old)

        info = dict(iter=n, keff=keff_new, keff_change=keff_change,
                    loss1=loss1, loss2=loss2, total_loss=total_loss,
                    loss_change=loss_change)
        history.append(info)
        if verbose:
            pct = 100.0 * n / max_outer
            print(f"[外层 {n}/{max_outer} ({pct:5.2f}%)] keff={keff_new:.8f} "
                  f"(Δ={keff_change:.3e}) | loss1={loss1:.3e} loss2={loss2:.3e} "
                  f"(Δloss={loss_change:.3e})")

        # ---- 实时绘制损失与 keff 曲线 ----
        if plot_every > 0 and (n % plot_every == 0 or n == max_outer):
            plot_history(history, f"{save_path}_history.png")

        # ---- 实时保存训练历史（JSON）与最新模型权重 ----
        save_history(history, history_path)
        if ckpt_every > 0 and (n % ckpt_every == 0 or n == max_outer):
            save_ckpt_atomic(net1, p1, keff_new, group=1)
            save_ckpt_atomic(net2, p2, keff_new, group=2)
            save_checkpoint(ckpt_path, net1, net2, opt1_adam, opt2_adam,
                            n_done=n, keff=keff_new, history=history,
                            total_loss_old=total_loss, n_sub=n_sub,
                            max_outer=max_outer, keff_tol=keff_tol,
                            loss_tol=loss_tol)
            if verbose:
                print(f"    已实时保存检查点：{p1}、{p2}、{ckpt_path}")

        # ---- 收敛判断 ----
        keff = keff_new  # 提前更新：即使收敛 break，也能保存/打印最新 keff
        if keff_change < keff_tol and loss_change < loss_tol:
            if verbose:
                print(f"收敛：|Δkeff|={keff_change:.3e}<{keff_tol} 且 "
                      f"Δloss={loss_change:.3e}<{loss_tol}。")
            break

        phi1_prev = phi1_new.clone()
        phi2_prev = phi2_new.clone()
        F_old = F_new
        total_loss_old = total_loss

    # ---- 最终绘制（覆盖未整除 plot_every 的收尾迭代，含提前收敛） ----
    if plot_every > 0:
        plot_history(history, f"{save_path}_history.png")

    # ---- 最终保存（原子写入；历史 JSON 同步收尾，保证与最终权重一致） ----
    save_ckpt_atomic(net1, p1, keff, group=1)
    save_ckpt_atomic(net2, p2, keff, group=2)
    save_history(history, history_path)
    # 训练正常结束，删除断点文件，避免下次运行时误判为未完成而续训
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        if verbose:
            print(f"训练完成，已删除断点文件 {ckpt_path}。")
    if verbose:
        print(f"已保存：{p1}、{p2}（keff={keff:.8f}）")

    return net1, net2, keff, history


if __name__ == "__main__":
   
    import argparse

    p = argparse.ArgumentParser(description="TWIGL 源迭代训练")
    p.add_argument("--device", default=None, help="cuda 或 cpu，默认自动检测")
    p.add_argument("--save_path", default="twigl",
                   help="保存前缀，模型为 <save_path>_g1.pt / <save_path>_g2.pt")
    p.add_argument("--n_sub", type=int, default=50, help="每子区域每维高斯点数")
    p.add_argument("--n_poly", type=int, default=5, help="B、s 多项式阶数")
    p.add_argument("--hidden_layers", type=int, default=8, help="隐层层数")
    p.add_argument("--neurons", type=int, default=400, help="每层神经元数")
    p.add_argument("--adam_lr", type=float, default=1e-3, help="Adam 学习率")
    p.add_argument("--adam_steps", type=int, default=50000,
                   help="每个外层迭代内 Adam 步数")
    p.add_argument("--lbfgs_lr", type=float, default=1.0, help="L-BFGS 学习率")
    p.add_argument("--lbfgs_steps", type=int, default=500, help="L-BFGS 最大步数")
    p.add_argument("--max_outer", type=int, default=30, help="外层最大迭代数")
    p.add_argument("--ckpt_every", type=int, default=1,
                   help="每多少个外层迭代实时保存一次模型权重")
    p.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="若存在 <save_path>_checkpoint.pt 则自动续训（默认）")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="忽略已有断点，从头训练")
    p.add_argument("--keff_tol", type=float, default=1e-6, help="keff 收敛容差")
    p.add_argument("--loss_tol", type=float, default=1e-4, help="损失收敛容差")
    args = p.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, fallback to cpu.")
        args.device = "cpu"

    print(f"设备: {args.device}, CUDA 可用: {torch.cuda.is_available()}")

    net1, net2, keff, history = train(
        n_sub=args.n_sub, n_poly=args.n_poly,
        hidden_layers=args.hidden_layers, neurons=args.neurons,
        adam_lr=args.adam_lr, adam_steps=args.adam_steps,
        lbfgs_lr=args.lbfgs_lr, lbfgs_steps=args.lbfgs_steps,
        max_outer=args.max_outer, keff_tol=args.keff_tol,
        loss_tol=args.loss_tol, device=args.device,
        save_path=args.save_path, ckpt_every=args.ckpt_every,
        resume=args.resume, verbose=True,
    )
    print(f"\n最终 keff = {keff:.8f}")
