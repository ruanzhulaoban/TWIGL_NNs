"""PSNNNet 神经网络模块。

构造每个能群的预测网络：输入坐标 (x, y)，输出通量
    φ_g(x, y) = B(x, y) * NNs_g(x, y)。

结构：
  1. 专门层（固定、不可训练）：(x, y) -> (u, v) = s_func_g(x, y)；
  2. 拼接层：(x, y, u, v) 拼接为 4 维输入；
  3. 全连接：4 -> 400 -> 400 -> ... -> 400 -> 1（默认 8 个隐藏层，每层 400 神经元，
     每层后接 Tanh）；
  4. 输出层：乘以 B_func(x, y) 得到 φ。

每个能群独立实例化（不同 s_g 与网络参数）。B_func、s_func 由
B_function.build_B 与 s_layer.build_s 构造，均为无训练参数的固定特征变换。

关于保存/载入与确定性：
  B、s 的构造依赖 SVD 求零空间，对退化（重复/零）奇异值存在符号与
  零空间内正交旋转的任意性，重新构造无法保证得到与训练时一致的 B、s。
  因此 save 同时保存 B、s 的系数（B_coeffs、s_coeffs）与阶数 n，
  load 时由系数确定性重建（_make_B_func），不重新做 SVD。
"""

import torch
import torch.nn as nn

from B_function import _make_B_func, build_B
from s_layer import build_s


def _reconstruct_B(B_coeffs, n):
    """由系数确定性重建 B_func（不经过 SVD）。"""
    return _make_B_func(B_coeffs, n)


def _reconstruct_s(s_coeffs, n):
    """由系数确定性重建 s_func（不经过 SVD）。"""
    u_func = _make_B_func(s_coeffs[0], n)
    v_func = _make_B_func(s_coeffs[1], n)

    def s_func(x, y):
        return u_func(x, y), v_func(x, y)

    return s_func


class PSNNNet(nn.Module):
    """单个能群的通量预测网络 φ_g(x, y) = B(x, y) * NNs_g(x, y)。"""

    def __init__(self, B_func, s_func, B_coeffs=None, s_coeffs=None, n=5,
                 hidden_layers=8, neurons=400, activation=nn.Tanh,
                 dtype=torch.float64):
        """
        参数
        ----
        B_func : callable
            边界函数 B(x, y)，支持批量（B_function.build_B 返回）。
        s_func : callable
            专门函数 s_g(x, y) -> (u, v)，支持批量（s_layer.build_s 返回）。
        B_coeffs : torch.Tensor 或 None
            B 的系数 (9, (n+1)^2)，用于 save/load 确定性重建。缺省则无法保存。
        s_coeffs : torch.Tensor 或 None
            s 的系数 (2, 9, (n+1)^2)，用于 save/load 确定性重建。缺省则无法保存。
        n : int
            多项式阶数（每维最高次数），用于从系数重建 B、s。
        hidden_layers : int
            隐藏层数，默认 8。
        neurons : int
            每层神经元数，默认 400。
        activation : type
            激活函数类（如 nn.Tanh），默认 nn.Tanh。
        dtype : torch.dtype
            网络工作精度，默认 float64（与 B、s 一致）。
        """
        super().__init__()
        self.B_func = B_func
        self.s_func = s_func
        self.B_coeffs = B_coeffs
        self.s_coeffs = s_coeffs
        self.n = n
        self.hidden_layers = hidden_layers
        self.neurons = neurons
        self.activation = activation
        self.activation_name = getattr(activation, "__name__", "Tanh")
        self.dtype = dtype

        # 全连接部分：4 -> neurons(×hidden_layers) -> 1
        layers = [nn.Linear(4, neurons, dtype=dtype), activation()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons, neurons, dtype=dtype))
            layers.append(activation())
        layers.append(nn.Linear(neurons, 1, dtype=dtype))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        """前向：x、y 为批量张量（shape [N]），返回 φ（shape [N]）。"""
        dev = self.net[0].weight.device
        dt = self.net[0].weight.dtype
        x = torch.as_tensor(x, dtype=dt, device=dev)
        y = torch.as_tensor(y, dtype=dt, device=dev)

        u, v = self.s_func(x, y)          # 专门层（固定、不可训练）
        B = self.B_func(x, y)

        inp = torch.stack([x, y, u.to(dt), v.to(dt)], dim=-1)   # [N, 4]
        nns = self.net(inp).squeeze(-1)                          # [N]
        return B.to(dt) * nns

    def save(self, path, keff=None, **meta):
        """保存 state_dict、keff，以及 B、s 的系数与阶数（供确定性重建）。

        参数
        ----
        path : str
            保存路径。
        keff : float 或 None
            有效增殖因子等标量结果。
        **meta
            其余需一并保存的元信息（存入 dict）。
        """
        if self.B_coeffs is None or self.s_coeffs is None:
            raise ValueError(
                "B_coeffs / s_coeffs 缺失，无法保存 B、s。"
                "请通过 make_psnn 构造，或显式传入 B_coeffs、s_coeffs。"
            )
        ckpt = {
            "state_dict": self.state_dict(),
            "keff": keff,
            "meta": meta,
            "hidden_layers": self.hidden_layers,
            "neurons": self.neurons,
            "activation": self.activation_name,
            "B_coeffs": self.B_coeffs,
            "s_coeffs": self.s_coeffs,
            "n": self.n,
        }
        torch.save(ckpt, path)

    @classmethod
    def load(cls, path, activation=None, dtype=torch.float64):
        """从 path 载入，返回 (model, keff, meta)。

        B、s 由 checkpoint 中保存的系数确定性重建，不重新做 SVD。
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        hidden_layers = ckpt.get("hidden_layers", 8)
        neurons = ckpt.get("neurons", 400)
        if activation is None:
            act_name = ckpt.get("activation", "Tanh")
            activation = getattr(nn, act_name, nn.Tanh)

        B_coeffs = ckpt.get("B_coeffs")
        s_coeffs = ckpt.get("s_coeffs")
        n = ckpt.get("n", 5)
        if B_coeffs is None or s_coeffs is None:
            raise ValueError("checkpoint 缺少 B_coeffs / s_coeffs，无法重建 B、s。")

        B_func = _reconstruct_B(B_coeffs, n)
        s_func = _reconstruct_s(s_coeffs, n)

        model = cls(B_func, s_func, B_coeffs=B_coeffs, s_coeffs=s_coeffs, n=n,
                    hidden_layers=hidden_layers, neurons=neurons,
                    activation=activation, dtype=dtype)
        model.load_state_dict(ckpt["state_dict"])
        return model, ckpt.get("keff"), ckpt.get("meta", {})


def make_psnn(g, n=5, Nd=None, hidden_layers=8, neurons=400,
              activation=nn.Tanh, dtype=torch.float64, tol=1e-10):
    """便捷构造：构造 B_func、s_func_g 与能群 g 的 PSNNNet。

    B、s 的系数与阶数会被一并保存进模型（供 save/load 确定性重建）。
    返回 model。
    """
    B_func, B_coeffs = build_B(n=n, Nd=Nd, tol=tol)
    s_func, s_coeffs = build_s(g=g, n=n, Nd=Nd, tol=tol)
    return PSNNNet(B_func, s_func, B_coeffs=B_coeffs, s_coeffs=s_coeffs, n=n,
                   hidden_layers=hidden_layers, neurons=neurons,
                   activation=activation, dtype=dtype)


def reconstruct_net(B_coeffs, s_coeffs, n, hidden_layers=8, neurons=400,
                    activation=nn.Tanh, dtype=torch.float64, device="cuda"):
    """由 B、s 系数确定性重建网络（不重做 SVD）。

    用于多进程并行：父进程先算好 B_coeffs / s_coeffs，子进程据此重建网络，
    保证与父进程完全一致。B、s 构造依赖 SVD 零空间的符号/旋转任意性，
    不能在各个进程里各自重新做 SVD。
    """
    B_func = _reconstruct_B(B_coeffs, n)
    s_func = _reconstruct_s(s_coeffs, n)
    return PSNNNet(B_func, s_func, B_coeffs=B_coeffs, s_coeffs=s_coeffs, n=n,
                   hidden_layers=hidden_layers, neurons=neurons,
                   activation=activation, dtype=dtype).to(device)


if __name__ == "__main__":
    import os

    model = make_psnn(g=1, n=5, hidden_layers=8, neurons=400)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数数量: {n_params}")

    # 前向（批量 [N]）
    x = torch.linspace(0.0, 0.8, 100, dtype=torch.float64)
    y = torch.linspace(0.0, 0.8, 100, dtype=torch.float64)
    phi = model(x, y)
    print(f"φ shape: {tuple(phi.shape)}, dtype: {phi.dtype}, "
          f"范围: [{phi.min().item():.4g}, {phi.max().item():.4g}]")

    # 可微性（用于 PDE 残差 / 自动求导）
    xr = torch.linspace(0.2, 0.6, 16, dtype=torch.float64, requires_grad=True)
    yr = torch.linspace(0.2, 0.6, 16, dtype=torch.float64, requires_grad=True)
    loss = (model(xr, yr) ** 2).sum()
    loss.backward()
    w = next(model.parameters())
    print(f"x 梯度非空: {xr.grad is not None}, "
          f"权重梯度非空: {w.grad is not None}")

    # 保存 / 载入（load 不重建 B、s，而是从系数确定性还原）
    path = "_test_ckpt.pt"
    model.save(path, keff=1.0523, note="自检")
    model2, keff, meta = PSNNNet.load(path)
    print(f"载入 keff={keff}, meta={meta}")
    print(f"载入后 φ 一致: {torch.allclose(phi, model2(x, y))}")

    # 重建出的 B、s 与原始逐点一致
    B1 = model.B_func(x, y)
    B2 = model2.B_func(x, y)
    u1, v1 = model.s_func(x, y)
    u2, v2 = model2.s_func(x, y)
    print(f"B 逐点一致: {torch.allclose(B1, B2)}")
    print(f"s(u,v) 逐点一致: {torch.allclose(u1, u2) and torch.allclose(v1, v2)}")
    os.remove(path)
