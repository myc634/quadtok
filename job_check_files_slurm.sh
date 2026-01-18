#!/bin/bash
#SBATCH -p si
#SBATCH -N 1                    # 申请 1 个节点
#SBATCH --ntasks-per-node=1     # 每个节点 1 个任务
#SBATCH --cpus-per-task=1       # 每个任务 1 个 CPU
#SBATCH --array=0-139           # 创建 140 个任务（0-139）
#SBATCH -J check_files          # 任务名称
#SBATCH -o checkfiles/check_files_%A_%a.out  # 日志输出 (任务ID_数组索引)

# ===========================
# 1. 环境配置
# ===========================
source /mnt/petrelfs/jianglihan/miniforge3/bin/activate 1d

cd /mnt/petrelfs/jianglihan/my_code/quadtok

export PYTHONUNBUFFERED=1

# ===========================
# 2. 创建日志目录
# ===========================
mkdir -p logs

# ===========================
# 3. 计算要检查的文件索引
# ===========================
FILE_IDX=$SLURM_ARRAY_TASK_ID

# ===========================
# 4. 执行检查
# ===========================
echo "=========================================="
echo "任务 ID: $SLURM_ARRAY_TASK_ID"
echo "文件索引: $FILE_IDX"
echo "开始时间: $(date)"
echo "=========================================="

python test_files.py --file_idx $FILE_IDX

EXIT_CODE=$?

echo "=========================================="
echo "任务 ID: $SLURM_ARRAY_TASK_ID 完成"
echo "结束时间: $(date)"
echo "退出码: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
