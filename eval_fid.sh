#!/bin/bash
#SBATCH -p si
#SBATCH -N 1                    # 申请 1 个节点
#SBATCH --gres=gpu:1            # 申请 1 张 GPU (FID 评估通常单卡即可)
#SBATCH --ntasks-per-node=1     # 每个节点启 1 个任务
#SBATCH --cpus-per-task=8       # CPU 核心数
#SBATCH -J eval_fid         # 任务名称
#SBATCH -o logs/eval_fid_%j.out # 日志输出

source /mnt/petrelfs/jianglihan/miniforge3/bin/activate eval-gen

cd /mnt/petrelfs/jianglihan/my_code/quadtok2

# 设置环境变量
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

echo "Evaluating FID metrics on inference output..."

launcher="python scripts/eval_imagenet_fid.py \
  --output_dir checkpoints/generator/gpt_quadtree_base_16kcodebook_2lod_base_decay2x_4e-4lr/inference_output/scale_6.9_pow_1.5_decay_power-cosine_temp_1.00_num_50000 \
  --skip_processing \
  --num_samples 50000"

echo "Command to run:"
echo "$launcher"

srun bash -c "$launcher"
