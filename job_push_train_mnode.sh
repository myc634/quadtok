#!/bin/bash
#SBATCH -p si
#SBATCH -N 2                   # 申请节点数 
#SBATCH --gres=gpu:8            # 每个节点 8 卡
#SBATCH --ntasks-per-node=1     # 每个节点启 1 个 accelerate 实例
#SBATCH --cpus-per-task=16      # CPU核心数 (建议设大一点，例如 16 或 32 以防数据加载卡顿)
#SBATCH -J bash         # 任务名
#SBATCH -o logs/toktrain_vq_%j.out  # 日志输出 (确保 logs 文件夹存在)

# ===========================
# 1. 环境配置 (替换为你现在的环境)
# ===========================
source /mnt/petrelfs/jianglihan/miniforge3/bin/activate 1d

cd /mnt/petrelfs/jianglihan/my_code/quadtok2

# 设置 CUDA 库路径
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export NCCL_DEBUG=INFO

# ===========================
# 2. 节点通信配置 (仿照你的模板)
# ===========================
gpus_per_node=8
# 选第一个节点当主节点
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "Head Node IP: $head_node"
echo "Total Nodes: $SLURM_NNODES"

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1

launcher="accelerate launch \
  --config_file configs/accelerate_config/mnode8gpu.yaml \
  --main_process_ip=$head_node \
  --main_process_port=29500 \
  --num_processes=$((SLURM_NNODES * gpus_per_node)) \
  --num_machines=$SLURM_NNODES \
  --machine_rank=\$SLURM_PROCID \
  scripts/train_tokenizer.py config=configs/training/single_stage/quadtok_ss256_vq.yaml"

echo "Command to run:"
echo "$launcher"

# ===========================
# 4. 分发执行
# ===========================
srun bash -c "$launcher"