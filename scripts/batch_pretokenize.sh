#!/bin/bash

# Batch pretokenization script for quadtree data
# Usage: bash scripts/batch_pretokenize.sh

# ============================================================================
# Configuration
# ============================================================================
source /mnt/shared-storage-user/jianglihan/mc3/bin/activate
conda activate 1d
cd /mnt/shared-storage-user/jianglihan/myc/code/quadtok
# Shard configuration
INPUT_SHARD_DIR="/mnt/shared-storage-user/idc2-shared/dataset/preprocess/imagenet"
OUTPUT_SHARD_DIR="/mnt/shared-storage-user/idc2-shared/dataset/preprocess/imagenet-tokenized-vq"
SHARD_PREFIX="imagenet-train"
START_IDX=0
END_IDX=70  # Process shards 000000 to 000070

# GPU configuration
NUM_GPUS=8  # Number of GPUs to use

# Training configuration
CONFIG_DIR="/mnt/shared-storage-user/jianglihan/myc/code/quadtok/configs/training/policy_stage/quadtok_ss256_vq_pretokenization.yaml"
OUTPUT_DIR="/mnt/shared-storage-user/jianglihan/myc/code/quadtok/visualization/pretokenization-vq"

# Accelerate configuration
MIXED_PRECISION="bf16"
BASE_PORT=12389  # Base port number, will increment for each process

# ============================================================================
# Script
# ============================================================================

# Create output directory
mkdir -p "${OUTPUT_SHARD_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Calculate total number of shards
TOTAL_SHARDS=$((END_IDX - START_IDX + 1))

echo "============================================================================"
echo "Batch Pretokenization Configuration"
echo "============================================================================"
echo "Input directory:  ${INPUT_SHARD_DIR}"
echo "Output directory: ${OUTPUT_SHARD_DIR}"
echo "Shard prefix:     ${SHARD_PREFIX}"
echo "Shard range:      $(printf '%06d' ${START_IDX}) to $(printf '%06d' ${END_IDX})"
echo "Total shards:     ${TOTAL_SHARDS}"
echo "Number of GPUs:   ${NUM_GPUS}"
echo "Config file:      ${CONFIG_DIR}"
echo "Output dir:       ${OUTPUT_DIR}"
echo "============================================================================"
echo ""

# Array to store background process PIDs
declare -a PIDS

# Counter for current GPU
gpu_idx=0

# Function to wait for a specific PID
wait_for_pid() {
    local pid=$1
    wait ${pid}
    local exit_code=$?
    if [ ${exit_code} -ne 0 ]; then
        echo "WARNING: Process ${pid} exited with code ${exit_code}"
    fi
}

# Process each shard
for ((shard_idx=START_IDX; shard_idx<=END_IDX; shard_idx++)); do
    # Format shard index as 6-digit string
    shard_str=$(printf '%06d' ${shard_idx})
    
    # Construct input and output paths
    input_path="${INPUT_SHARD_DIR}/${SHARD_PREFIX}-${shard_str}.tar"
    output_path="${OUTPUT_SHARD_DIR}/${SHARD_PREFIX}-${shard_str}.tar"
    
    # Check if input file exists
    if [ ! -f "${input_path}" ]; then
        echo "WARNING: Input file not found: ${input_path}, skipping..."
        continue
    fi
    
    # # Check if output file already exists
    # if [ -f "${output_path}" ]; then
    #     echo "INFO: Output file already exists: ${output_path}, skipping..."
    #     continue
    # fi
    
    # Calculate which GPU to use
    current_gpu=$((gpu_idx % NUM_GPUS))
    
    # Calculate port number (increment for each process to avoid conflicts)
    port=$((BASE_PORT + gpu_idx))
    
    # Create log file name
    log_file="${OUTPUT_DIR}/pretokenize_${shard_str}_gpu${current_gpu}.log"
    
    echo "Launching job for shard ${shard_str} on GPU ${current_gpu} (port ${port})..."
    echo "  Input:  ${input_path}"
    echo "  Output: ${output_path}"
    echo "  Log:    ${log_file}"
    
    # Launch the pretokenization process in the background
    (
        CUDA_VISIBLE_DEVICES=${current_gpu} WANDB_MODE=disabled accelerate launch \
            --mixed_precision=${MIXED_PRECISION} \
            --num_machines=1 \
            --num_processes=1 \
            --machine_rank=0 \
            --main_process_ip=127.0.0.1 \
            --main_process_port=${port} \
            --same_network \
            scripts/pretokenize_quadtree.py \
                --config_dir ${CONFIG_DIR} \
                --output_dir ${OUTPUT_DIR} \
                --shards_path ${input_path} \
                --output_tar_path ${output_path} \
            > ${log_file} 2>&1
        
        exit_code=$?
        if [ ${exit_code} -eq 0 ]; then
            echo "[$(date)] SUCCESS: Shard ${shard_str} completed on GPU ${current_gpu}" >> ${OUTPUT_DIR}/batch_progress.log
        else
            echo "[$(date)] FAILED: Shard ${shard_str} failed on GPU ${current_gpu} with exit code ${exit_code}" >> ${OUTPUT_DIR}/batch_progress.log
        fi
    ) &
    
    # Store the PID
    PIDS+=($!)
    
    # Increment GPU index
    gpu_idx=$((gpu_idx + 1))
    
    # If we've launched NUM_GPUS processes, wait for one to finish before continuing
    if [ ${gpu_idx} -ge ${NUM_GPUS} ]; then
        # Wait for the oldest process to finish
        oldest_pid=${PIDS[0]}
        echo "Waiting for process ${oldest_pid} to complete..."
        wait_for_pid ${oldest_pid}
        echo "Process ${oldest_pid} completed."
        # Remove the oldest PID from the array
        PIDS=("${PIDS[@]:1}")
    fi
    
    # Small delay to avoid race conditions
    sleep 2
done

# Wait for all remaining background processes to complete
echo ""
echo "============================================================================"
echo "Waiting for all remaining processes to complete..."
echo "============================================================================"
for pid in "${PIDS[@]}"; do
    echo "Waiting for process ${pid}..."
    wait_for_pid ${pid}
done

echo ""
echo "============================================================================"
echo "Batch pretokenization completed!"
echo "============================================================================"
echo "Check ${OUTPUT_DIR}/batch_progress.log for detailed progress."
echo ""

# Print summary
echo "Summary:"
grep "SUCCESS" ${OUTPUT_DIR}/batch_progress.log | wc -l | xargs echo "  Successful:"
grep "FAILED" ${OUTPUT_DIR}/batch_progress.log | wc -l | xargs echo "  Failed:"
echo ""

