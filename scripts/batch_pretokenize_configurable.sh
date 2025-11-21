#!/bin/bash

# Configurable batch pretokenization script for quadtree data
# Usage: bash scripts/batch_pretokenize_configurable.sh [OPTIONS]
#
# Example:
#   bash scripts/batch_pretokenize_configurable.sh \
#       --input-dir /path/to/input \
#       --output-dir /path/to/output \
#       --start-idx 0 \
#       --end-idx 70 \
#       --num-gpus 8

# ============================================================================
# Default Configuration
# ============================================================================

INPUT_SHARD_DIR="/mnt/shared-storage-user/idc2-shared/dataset/preprocess/imagenet"
OUTPUT_SHARD_DIR="/mnt/shared-storage-user/idc2-shared/dataset/preprocess/imagenet-tokenized"
SHARD_PREFIX="imagenet-train"
START_IDX=0
END_IDX=70
NUM_GPUS=8
CONFIG_DIR="configs/training/policy_stage/quadtok_ss256_vae_opt.yaml"
OUTPUT_DIR="visualization/seach_32channel_adv_gd3"
MIXED_PRECISION="bf16"
BASE_PORT=12389
SKIP_EXISTING=true

# ============================================================================
# Parse Command Line Arguments
# ============================================================================

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --input-dir DIR         Input directory containing shards (default: ${INPUT_SHARD_DIR})
    --output-dir DIR        Output directory for tokenized shards (default: ${OUTPUT_SHARD_DIR})
    --shard-prefix PREFIX   Shard filename prefix (default: ${SHARD_PREFIX})
    --start-idx NUM         Starting shard index (default: ${START_IDX})
    --end-idx NUM           Ending shard index (default: ${END_IDX})
    --num-gpus NUM          Number of GPUs to use (default: ${NUM_GPUS})
    --config-dir PATH       Path to config file (default: ${CONFIG_DIR})
    --work-dir DIR          Working output directory (default: ${OUTPUT_DIR})
    --mixed-precision TYPE  Mixed precision type (default: ${MIXED_PRECISION})
    --base-port NUM         Base port number (default: ${BASE_PORT})
    --no-skip-existing      Process even if output exists
    -h, --help              Show this help message

Example:
    $0 --start-idx 0 --end-idx 10 --num-gpus 4
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_SHARD_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_SHARD_DIR="$2"
            shift 2
            ;;
        --shard-prefix)
            SHARD_PREFIX="$2"
            shift 2
            ;;
        --start-idx)
            START_IDX="$2"
            shift 2
            ;;
        --end-idx)
            END_IDX="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --work-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --mixed-precision)
            MIXED_PRECISION="$2"
            shift 2
            ;;
        --base-port)
            BASE_PORT="$2"
            shift 2
            ;;
        --no-skip-existing)
            SKIP_EXISTING=false
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# ============================================================================
# Validation
# ============================================================================

if [ ! -d "${INPUT_SHARD_DIR}" ]; then
    echo "ERROR: Input directory does not exist: ${INPUT_SHARD_DIR}"
    exit 1
fi

if [ ! -f "${CONFIG_DIR}" ]; then
    echo "ERROR: Config file does not exist: ${CONFIG_DIR}"
    exit 1
fi

if [ ${START_IDX} -gt ${END_IDX} ]; then
    echo "ERROR: START_IDX (${START_IDX}) must be less than or equal to END_IDX (${END_IDX})"
    exit 1
fi

if [ ${NUM_GPUS} -lt 1 ]; then
    echo "ERROR: NUM_GPUS must be at least 1"
    exit 1
fi

# ============================================================================
# Setup
# ============================================================================

mkdir -p "${OUTPUT_SHARD_DIR}"
mkdir -p "${OUTPUT_DIR}"

TOTAL_SHARDS=$((END_IDX - START_IDX + 1))

echo "============================================================================"
echo "Batch Pretokenization Configuration"
echo "============================================================================"
echo "Input directory:    ${INPUT_SHARD_DIR}"
echo "Output directory:   ${OUTPUT_SHARD_DIR}"
echo "Shard prefix:       ${SHARD_PREFIX}"
echo "Shard range:        $(printf '%06d' ${START_IDX}) to $(printf '%06d' ${END_IDX})"
echo "Total shards:       ${TOTAL_SHARDS}"
echo "Number of GPUs:     ${NUM_GPUS}"
echo "Config file:        ${CONFIG_DIR}"
echo "Working directory:  ${OUTPUT_DIR}"
echo "Mixed precision:    ${MIXED_PRECISION}"
echo "Base port:          ${BASE_PORT}"
echo "Skip existing:      ${SKIP_EXISTING}"
echo "============================================================================"
echo ""

# Initialize progress log
echo "[$(date)] Batch pretokenization started" > ${OUTPUT_DIR}/batch_progress.log

# ============================================================================
# Main Processing Loop
# ============================================================================

declare -a PIDS
gpu_idx=0
processed_count=0
skipped_count=0

wait_for_pid() {
    local pid=$1
    wait ${pid}
    local exit_code=$?
    if [ ${exit_code} -ne 0 ]; then
        echo "WARNING: Process ${pid} exited with code ${exit_code}"
    fi
    return ${exit_code}
}

for ((shard_idx=START_IDX; shard_idx<=END_IDX; shard_idx++)); do
    shard_str=$(printf '%06d' ${shard_idx})
    input_path="${INPUT_SHARD_DIR}/${SHARD_PREFIX}-${shard_str}.tar"
    output_path="${OUTPUT_SHARD_DIR}/${SHARD_PREFIX}-${shard_str}.tar"
    
    # Check if input file exists
    if [ ! -f "${input_path}" ]; then
        echo "WARNING: Input file not found: ${input_path}, skipping..."
        echo "[$(date)] SKIPPED: ${input_path} (not found)" >> ${OUTPUT_DIR}/batch_progress.log
        skipped_count=$((skipped_count + 1))
        continue
    fi
    
    # Check if output file already exists
    if [ -f "${output_path}" ] && [ "${SKIP_EXISTING}" = true ]; then
        echo "INFO: Output file already exists: ${output_path}, skipping..."
        echo "[$(date)] SKIPPED: ${output_path} (already exists)" >> ${OUTPUT_DIR}/batch_progress.log
        skipped_count=$((skipped_count + 1))
        continue
    fi
    
    # Calculate which GPU to use
    current_gpu=$((gpu_idx % NUM_GPUS))
    
    # Calculate port number
    port=$((BASE_PORT + gpu_idx))
    
    # Create log file name
    log_file="${OUTPUT_DIR}/pretokenize_${shard_str}_gpu${current_gpu}.log"
    
    echo "Launching job for shard ${shard_str} on GPU ${current_gpu} (port ${port})..."
    echo "  Input:  ${input_path}"
    echo "  Output: ${output_path}"
    echo "  Log:    ${log_file}"
    
    # Launch the pretokenization process
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
    
    sleep 2
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

echo "[$(date)] Batch pretokenization completed" >> ${OUTPUT_DIR}/batch_progress.log

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================================================"
echo "Batch Pretokenization Completed!"
echo "============================================================================"
echo ""
echo "Summary:"
echo "  Total shards in range: ${TOTAL_SHARDS}"
echo "  Processed:             ${processed_count}"
echo "  Skipped:               ${skipped_count}"

success_count=$(grep -c "SUCCESS" ${OUTPUT_DIR}/batch_progress.log || echo "0")
failed_count=$(grep -c "FAILED" ${OUTPUT_DIR}/batch_progress.log || echo "0")

echo "  Successful:            ${success_count}"
echo "  Failed:                ${failed_count}"
echo ""
echo "Log file: ${OUTPUT_DIR}/batch_progress.log"
echo "Output directory: ${OUTPUT_SHARD_DIR}"
echo ""

if [ ${failed_count} -gt 0 ]; then
    echo "WARNING: Some shards failed to process. Check individual log files in ${OUTPUT_DIR}"
    echo ""
    echo "Failed shards:"
    grep "FAILED" ${OUTPUT_DIR}/batch_progress.log
    exit 1
fi

echo "All shards processed successfully!"

