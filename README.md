# TWIGL_NNs

用 **PINN（物理信息神经网络）** 求解 **TWIGL 两群中子扩散基准**，输出有效增殖因子 k_eff 与中子通量分布。

核心思路：把通量写成 `φ_g = B(x,y) · NNs_g(x,y)`，其中 `B` 和专门层 `s_g` 已经天然满足边界/界面条件，再用**源迭代（功率迭代）**更新 k_eff 直到收敛。

---

## 快速开始

```bash
# 1. 安装依赖
pip install torch numpy matplotlib

# 2. 训练（自动检测 GPU/CPU）
python train.py

# 3. 预测 + 画图
python predict.py
```

或者一键跑完训练 + 预测：

```bash
bash run_experiment.sh
```

冒烟测试（快速验证流程能通，CPU 上跑小规模）：

```bash
bash run_experiment.sh --device cpu --save_path _smoke --n_sub 8 --grid 64
```

---

## 物理问题

四分之一对称反应堆 `[0, 0.8] × [0, 0.8]`，两群扩散方程：

```
-∇·(D_g ∇φ_g) + Σr_g φ_g = Q_g
```

TWIGL 的裂变谱 `χ = (1.0, 0.0)`、散射 `Σs_{2→1} = 0`，源项化简为：

```
Q_1 = (1 / k_eff)(νΣf_1 φ_1 + νΣf_2 φ_2)
Q_2 = Σs_{1→2} φ_1
```

几何是 3×3 = 9 个子区域（分割线 `x = 0.24, 0.56`、`y = 0.24, 0.56`）：

| 区域 | 位置 | 说明 |
|------|------|------|
| Region 1 | 中心 | 种子燃料 |
| Region 2 | 左臂 + 下臂 | 种子燃料 |
| Region 3 | 其余 6 块 | 反射层 blanket |

材料参数见 `B_function.py` 里的 `REGION_PARAMS`。

---

## 方法简介

1. **专门层**：`B(x,y)` 编码外边界（右/上 Dirichlet、左/下 Neumann）与内部界面连续性；`s_g(x,y)=(u,v)` 编码界面处的扩散系数加权连续性（每个能群一个）。
2. **网络**：`(x, y, u, v) → 8×400 全连接（Tanh）→ 1`，输出再乘 `B`。
3. **训练**：内层用 PDE 残差 `loss = mean(R²)`（`R = -D∇²φ + Σr φ - Q`）先 Adam 再 L-BFGS 拟合单群；外层更新 `k_eff ← k_eff × F_new/F_old`，重复到收敛。
4. **并行**：两个能群彼此独立，用多进程并行训练；`--n_gpus 2` 时各占一张 GPU。

---

## 项目结构

```
uploads/
├── train.py          主训练（源迭代、并行、断点续训、实时保存）
├── predict.py        预测 + 可视化（通量热力图）
├── network.py        网络 PSNNNet（φ = B·NNs）
├── B_function.py     B 的原始构造（SVD 零空间）
├── B_screen.py       B 的筛选/固化，生成 B_fixed.pt
├── s_layer.py        专门层 s_g 的构造
├── svd.py            SVD 零空间求解
├── run_experiment.sh 一键运行 train + predict
├── train.slurm       SLURM 提交脚本
└── README.md
```

---

## 常用命令行参数

### `train.py`

| 参数 | 默认 | 说明 |
|------|------|------|
| `--device` | 自动 | `cuda` 或 `cpu` |
| `--save_path` | `twigl` | 保存文件前缀 |
| `--n_sub` | `50` | 每子区域每维高斯点数 |
| `--n_gpus` | `1` | 并行 GPU 数（≥2 时两群各占一张） |
| `--max_outer` | `30` | 外层迭代次数上限 |
| `--adam_steps` | `50000` | 每外层迭代内 Adam 步数 |
| `--lbfgs_steps` | `500` | L-BFGS 步数 |
| `--resume` | 开 | 自动断点续训（`--no-resume` 关闭） |

完整参数见 `python train.py --help`。

### `predict.py`

```bash
python predict.py [path_g1] [path_g2] --grid 400 --outdir results
```

默认读取 `twigl_g1.pt` / `twigl_g2.pt`。

---

## 输出文件

| 文件 | 说明 |
|------|------|
| `twigl_g1.pt` / `twigl_g2.pt` | 两个能群模型权重 |
| `twigl_history.json` | 每轮 k_eff / 损失 |
| `twigl_history.png` | k_eff 与损失曲线 |
| `twigl_phi1.png` / `twigl_phi2.png` | φ1、φ2 热力图 |
| `twigl_fluxes.png` | 双群并排热力图 |
| `twigl_training_points.png` | 训练点分布 |

---

## 在计算节点上运行

```bash
sbatch train.slurm
```

脚本会加载 miniforge、激活 `torch_env`、进入 `~/run` 并执行两 GPU 训练。

> **注意**：`*.sh` / `*.slurm` 必须用 Unix 行尾（LF）。在 Windows 上编辑后上传，请确认行尾仍是 LF，否则会报 `\r` 相关错误。

---

## 断点续训

训练中断后，`twigl_checkpoint.pt` 会保留完整状态，下次运行 `train.py` 自动续训；训练正常收敛后该文件会被自动删除。

---

## 关于 B 的筛选（`B_screen.py`）

`B` 由 SVD 零空间构造，零空间维数高达 140 且无明显的奇异值间隙，直接取第一个方向不可靠（约 22% 的基向量在内部穿零）。`B_screen.py` 会筛选出燃料区量级足够、内部不穿零的最优 B，保存为 `B_fixed.pt`；应用层通过 `get_B()` 优先加载它，找不到时才回退到原始构造。

```bash
python B_screen.py
```
