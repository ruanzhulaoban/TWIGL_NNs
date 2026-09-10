#!/usr/bin/env bash
# ============================================================
# 自动实验脚本：先后运行 train.py 与 predict.py
#
# 用法：
#   bash run_experiment.sh [选项]
#
# 选项：
#   --device  cuda|cpu   训练/预测设备（默认自动检测，有 CUDA 用 cuda）
#   --save_path NAME     模型保存前缀（默认 twigl，产物 twigl_g1.pt/g2.pt）
#   --n_sub    N         每子区域每维高斯点数（默认 50，训练与预测需一致）
#   --grid     N         预测细网格分辨率（默认 400）
#   --outdir   DIR       预测图输出目录（默认 results）
#
# 快速冒烟测试（验证全流程能跑通）：
#   bash run_experiment.sh --device cpu --save_path _smoke \
#       --n_sub 8 --grid 64
#   （训练步数等仍需在 train.py 默认值内；如需缩短请在下方加 --adam_steps 等）
#
# 说明：脚本假设 `python` 已在 PATH（对应 conda 的 torch_env）。
#       若需指定解释器：PYTHON=/path/to/python bash run_experiment.sh
# ============================================================
set -euo pipefail

PYTHON="${PYTHON:-python}"
DEVICE=""
SAVE_PATH="twigl"
N_SUB="50"
GRID="400"
OUTDIR="results"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)    DEVICE="$2";    shift 2 ;;
    --save_path) SAVE_PATH="$2"; shift 2 ;;
    --n_sub)     N_SUB="$2";     shift 2 ;;
    --grid)      GRID="$2";      shift 2 ;;
    --outdir)    OUTDIR="$2";    shift 2 ;;
    -h|--help)   head -n 24 "$0"; exit 0 ;;
    *) echo "未知参数: $1（--help 查看用法）" >&2; exit 1 ;;
  esac
done

# 自动检测设备（与 train.py / predict.py 内部逻辑一致）
if [[ -z "$DEVICE" ]]; then
  if "$PYTHON" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi

echo "======================================================"
echo "自动实验：device=${DEVICE}  save_path=${SAVE_PATH}  n_sub=${N_SUB}"
echo "======================================================"

echo ""
echo "===== [1/2] 训练 train.py ====="
"$PYTHON" train.py --device "$DEVICE" --save_path "$SAVE_PATH" --n_sub "$N_SUB"

G1="${SAVE_PATH}_g1.pt"
G2="${SAVE_PATH}_g2.pt"
if [[ ! -f "$G1" || ! -f "$G2" ]]; then
  echo "错误：训练后未找到 $G1 / $G2，请检查 train.py 输出。" >&2
  exit 1
fi
echo "训练完成：$G1 / $G2"

echo ""
echo "===== [2/2] 预测 predict.py ====="
"$PYTHON" predict.py "$G1" "$G2" --grid "$GRID" --n_sub "$N_SUB" \
  --device "$DEVICE" --outdir "$OUTDIR"

echo ""
echo "===== 全部完成 ====="
echo "模型：$G1 / $G2"
echo "图像：${OUTDIR}/（twigl_phi1.png 等）"
