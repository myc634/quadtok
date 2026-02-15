#!/bin/bash
#SBATCH -p si
#SBATCH -N 1                    # 申请 1 个节点
#SBATCH --gres=gpu:1            # 申请 1 张 GPU (FID 评估通常单卡即可)
#SBATCH --ntasks-per-node=1     # 每个节点启 1 个任务
#SBATCH --cpus-per-task=8       # CPU 核心数
#SBATCH -J eval_fid         # 任务名称
#SBATCH -o logs/upload_%j.out # 日志输出

cd /mnt/petrelfs/jianglihan/my_code/quadtok2_search

export http_proxy=http://jianglihan:iY1NOXbE38du4ZngnG6U9O06wb73sKkBlX3rxYfa5wYH0B3Lo4wvscIOjxfH@10.1.20.50:23128
export https_proxy=http://jianglihan:iY1NOXbE38du4ZngnG6U9O06wb73sKkBlX3rxYfa5wYH0B3Lo4wvscIOjxfH@10.1.20.50:23128

rclone  copy --progress --transfers 200 --checkers 200 --links /mnt/petrelfs/jianglihan/my_code/quadtok2_search/checkpoints/generator/gpt_quadtree_base_16kcodebook_2lod_base_decay2x_4e-4lr_search_pred/checkpoint-70000/ema_model/pytorch_model.bin gdrive:quadtok/gpt_base_tokenizerv2_search_base_70k/
