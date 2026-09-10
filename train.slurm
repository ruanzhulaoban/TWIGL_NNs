#!/bin/bash
#SBATCH --job-name=my_exp
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

# 1. 加载 miniforge 模块
module load miniforge3/26.3.2-3

# 2. 关键：初始化 conda 的 shell 函数
source /data/apps/miniforge3/26.3.2-3/etc/profile.d/conda.sh

# 3. 现在可以正常用 conda activate 了
conda activate torch_env

# 4. 进入代码目录并运行
cd ~/run
bash run_experiment.sh