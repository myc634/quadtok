#!/bin/bash
#SBATCH -p si
#SBATCH -N 1                    # 申请 1 个节点
#SBATCH --gres=gpu:8            # 申请 8 张 GPU
#SBATCH --ntasks-per-node=1     # 每个节点启 1 个任务 (即 1 个 accelerate 实例)
#SBATCH --cpus-per-task=32      # CPU 核心数 (单机数据加载压力大，建议给足)
#SBATCH -J base         # 任务名称
#SBATCH -o logs/evaltok_vq_%j.out # 日志输出

source /mnt/petrelfs/jianglihan/miniforge3/bin/activate 1d

cd /mnt/petrelfs/jianglihan/my_code/quadtok3

# 设置环境变量
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline 
export NCCL_DEBUG=INFO

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

echo "Master IP: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"


launcher="accelerate launch \
  --num_processes=8 \
  --num_machines=1 \
  --machine_rank=0 \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  --mixed_precision=bf16 \
  scripts/eval_tokenizer.py \
  --work_dir checkpoints/quadtok_sl256_vq_ts8-16kcodebook-2lods-causal-selector-wope"

echo "Command to run:"
echo "$launcher"

srun bash -c "$launcher"