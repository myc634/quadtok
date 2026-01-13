#!/bin/bash
#SBATCH -p si
#SBATCH -N 1                    # 申请 1 个节点
#SBATCH --gres=gpu:4            # 申请 8 张 GPU
#SBATCH --ntasks-per-node=1     # 每个节点启 1 个任务 (即 1 个 accelerate 实例)
#SBATCH --cpus-per-task=32      # CPU 核心数 (单机数据加载压力大，建议给足)
#SBATCH -J extract_code         # 任务名称
#SBATCH -o logs/extract_code_%j.out # 日志输出

# ============================================================================
# Configuration (Modify these parameters as needed)
# ============================================================================

CONFIG_DIR="checkpoints/quadtok_sl256_vq_ts12-4096codebook-base/config.yaml"
TOKENIZER_WEIGHT="checkpoints/quadtok_sl256_vq_ts12-4096codebook-base/checkpoint-200000-save/ema_model/pytorch_model.bin"
OUTPUT_DIR="extract_token_log"
LOCAL_TMP_DIR="/mnt/petrelfs/jianglihan/my_code/tmp_imagenet_codes"
REMOTE_HOSS_PATH="hoss:jianglihan/data/imagenet-debug"
START_SHARD_IDX=0
END_SHARD_IDX=20
GUARANTEED_DEPTH=3
EXPANSION_PROBS="0.3 0.2"
NUM_WORKERS=2
NUM_GPUS=4

# ============================================================================
# Setup
# ============================================================================

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOCAL_TMP_DIR}"
mkdir -p logs

echo "============================================================================"
echo "Batch Extract Code Configuration"
echo "============================================================================"
echo "Config:           ${CONFIG_DIR}"
echo "Tokenizer:        ${TOKENIZER_WEIGHT}"
echo "Output dir:       ${OUTPUT_DIR}"
echo "Local tmp dir:    ${LOCAL_TMP_DIR}"
echo "Remote path:      ${REMOTE_HOSS_PATH}"
echo "Shard range:      ${START_SHARD_IDX} to ${END_SHARD_IDX}"
echo "Guaranteed depth: ${GUARANTEED_DEPTH}"
echo "Expansion probs:  ${EXPANSION_PROBS}"
echo "Number of GPUs:   ${NUM_GPUS}"
echo "============================================================================"
echo ""

# ============================================================================
# Helper Functions
# ============================================================================

process_shard() {
    local shard_idx=$1
    local gpu_id=$2
    local shard_str=$(printf '%06d' ${shard_idx})
    local output_tar_file="${LOCAL_TMP_DIR}/imagenet-train-${shard_str}.tar"
    
    echo "[$(date)] [GPU ${gpu_id}] Starting processing for shard ${shard_str}..."
    
    # Step 1: Extract codes and generate tar file
    CUDA_VISIBLE_DEVICES=${gpu_id} python scripts/extract_code_randomquadtree.py \
        --config_dir "${CONFIG_DIR}" \
        --tokenizer_weight "${TOKENIZER_WEIGHT}" \
        --output_dir "${OUTPUT_DIR}" \
        --shards_index "${shard_idx}" \
        --output_tar_path "${LOCAL_TMP_DIR}" \
        --guaranteed_depth "${GUARANTEED_DEPTH}" \
        --expansion_probs ${EXPANSION_PROBS} \
        --num_workers "${NUM_WORKERS}"
    
    local extract_exit_code=$?
    
    if [ ${extract_exit_code} -ne 0 ]; then
        echo "[$(date)] [GPU ${gpu_id}] ERROR: Code extraction failed for shard ${shard_str} with exit code ${extract_exit_code}"
        return ${extract_exit_code}
    fi
    
    # Check if tar file was created
    if [ ! -f "${output_tar_file}" ]; then
        echo "[$(date)] [GPU ${gpu_id}] ERROR: Output tar file not found: ${output_tar_file}"
        return 1
    fi
    
    echo "[$(date)] [GPU ${gpu_id}] Code extraction completed for shard ${shard_str}"
    echo "[$(date)] [GPU ${gpu_id}] Tar file size: $(du -h ${output_tar_file} | cut -f1)"
    
    # Step 2: Upload to hoss
    echo "[$(date)] [GPU ${gpu_id}] Starting upload to hoss for shard ${shard_str}..."
    
    rclone copy --progress --transfers 200 --checkers 200 --links \
        "${output_tar_file}" "${REMOTE_HOSS_PATH}"
    
    local upload_exit_code=$?
    
    if [ ${upload_exit_code} -ne 0 ]; then
        echo "[$(date)] [GPU ${gpu_id}] ERROR: Upload failed for shard ${shard_str} with exit code ${upload_exit_code}"
        return ${upload_exit_code}
    fi
    
    echo "[$(date)] [GPU ${gpu_id}] Upload completed for shard ${shard_str}"
    
    # Step 3: Clean up local tar file
    echo "[$(date)] [GPU ${gpu_id}] Cleaning up local tar file for shard ${shard_str}..."
    
    if [ -f "${output_tar_file}" ]; then
        rm -f "${output_tar_file}"
        echo "[$(date)] [GPU ${gpu_id}] Local tar file deleted: ${output_tar_file}"
    else
        echo "[$(date)] [GPU ${gpu_id}] WARNING: Local tar file not found (may have been deleted already): ${output_tar_file}"
    fi
    
    echo "[$(date)] [GPU ${gpu_id}] Shard ${shard_str} processing completed successfully"
    return 0
}

wait_for_pid() {
    local pid=$1
    wait ${pid}
    local exit_code=$?
    if [ ${exit_code} -ne 0 ]; then
        echo "WARNING: Process ${pid} exited with code ${exit_code}"
    fi
    return ${exit_code}
}

# ============================================================================
# Main Processing Loop
# ============================================================================

declare -a PIDS
gpu_idx=0
processed_count=0

for ((shard_idx=START_SHARD_IDX; shard_idx<=END_SHARD_IDX; shard_idx++)); do
    # Calculate which GPU to use
    current_gpu=$((gpu_idx % NUM_GPUS))
    
    # Process shard in background
    (
        process_shard ${shard_idx} ${current_gpu}
        exit_code=$?
        if [ ${exit_code} -eq 0 ]; then
            echo "[$(date)] SUCCESS: Shard ${shard_idx} completed on GPU ${current_gpu}" >> ${OUTPUT_DIR}/batch_progress.log
        else
            echo "[$(date)] FAILED: Shard ${shard_idx} failed on GPU ${current_gpu} with exit code ${exit_code}" >> ${OUTPUT_DIR}/batch_progress.log
        fi
        exit ${exit_code}
    ) &
    
    PIDS+=($!)
    gpu_idx=$((gpu_idx + 1))
    processed_count=$((processed_count + 1))
    
    # Wait for oldest process if we've launched NUM_GPUS processes
    if [ ${#PIDS[@]} -ge ${NUM_GPUS} ]; then
        oldest_pid=${PIDS[0]}
        echo "Waiting for process ${oldest_pid} to complete..."
        wait_for_pid ${oldest_pid}
        echo "Process ${oldest_pid} completed."
        PIDS=("${PIDS[@]:1}")
    fi
    
    sleep 1
done

# ============================================================================
# Wait for Completion
# ============================================================================

echo ""
echo "============================================================================"
echo "Waiting for all remaining processes to complete..."
echo "============================================================================"

for pid in "${PIDS[@]}"; do
    echo "Waiting for process ${pid}..."
    wait_for_pid ${pid}
done

echo "[$(date)] Batch processing completed" >> ${OUTPUT_DIR}/batch_progress.log

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================================================"
echo "Batch Processing Completed!"
echo "============================================================================"
echo ""
echo "Summary:"
echo "  Processed shards: ${processed_count}"
success_count=$(grep -c "SUCCESS" ${OUTPUT_DIR}/batch_progress.log 2>/dev/null || echo "0")
failed_count=$(grep -c "FAILED" ${OUTPUT_DIR}/batch_progress.log 2>/dev/null || echo "0")
echo "  Successful:       ${success_count}"
echo "  Failed:           ${failed_count}"
echo ""

if [ ${failed_count} -gt 0 ]; then
    echo "WARNING: Some shards failed to process. Check ${OUTPUT_DIR}/batch_progress.log"
    exit 1
fi

echo "All shards processed successfully!"

exit 0

