#!/bin/bash
# FSDP training — smoke test on 2 GPUs, change NUM_GPUS=8 for full run

# ===========================
# 1. Environment
# ===========================
source /mnt/petrelfs/jianglihan/miniforge3/bin/activate 1d

cd /mnt/ultracube/zec016/quadtok_train

export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Smoke test: 2 GPUs. Change to 0,1,2,3,4,5,6,7 for full 8-GPU run.
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

# ===========================
# 2. Launch
# ===========================
echo "Launching FSDP on ${NUM_GPUS} GPU(s)"

accelerate launch \
  --config_file configs/accelerate_config/fsdp_single_node.yaml \
  --num_processes=${NUM_GPUS} \
  --num_machines=1 \
  --main_process_port=29500 \
  scripts/train_generator_fsdp.py \
    config=configs/training/generator/gpt_quadtree_base.yaml \
    experiment.output_dir=checkpoints/fsdp_smoke_test
