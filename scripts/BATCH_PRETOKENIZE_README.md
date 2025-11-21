# Batch Pretokenization Scripts

This directory contains two scripts for batch pretokenization of quadtree data across multiple GPUs.

## Scripts

### 1. `batch_pretokenize.sh` - Simple Fixed Configuration

A straightforward script with hardcoded configuration values. Good for quick runs with standard settings.

**Usage:**
```bash
cd /mnt/shared-storage-user/jianglihan/myc/code/quadtok
bash scripts/batch_pretokenize.sh
```

**Configuration:**
Edit the script directly to modify these variables:
- `INPUT_SHARD_DIR`: Input directory containing tar shards
- `OUTPUT_SHARD_DIR`: Output directory for tokenized shards
- `SHARD_PREFIX`: Shard filename prefix (e.g., "imagenet-train")
- `START_IDX`: Starting shard index (e.g., 0)
- `END_IDX`: Ending shard index (e.g., 70)
- `NUM_GPUS`: Number of GPUs to use (e.g., 8)
- `CONFIG_DIR`: Path to YAML config file
- `OUTPUT_DIR`: Working directory for logs and temporary files

### 2. `batch_pretokenize_configurable.sh` - Flexible Command-Line Configuration

A more flexible script that accepts command-line arguments.

**Usage:**
```bash
cd /mnt/shared-storage-user/jianglihan/myc/code/quadtok

# Basic usage with custom range and GPUs
bash scripts/batch_pretokenize_configurable.sh \
    --start-idx 0 \
    --end-idx 70 \
    --num-gpus 8

# Full configuration example
bash scripts/batch_pretokenize_configurable.sh \
    --input-dir /mnt/shared-storage-user/idc2-shared/dataset/preprocess/imagenet \
    --output-dir /mnt/shared-storage-user/idc2-shared/dataset/preprocess/imagenet-tokenized \
    --shard-prefix imagenet-train \
    --start-idx 0 \
    --end-idx 70 \
    --num-gpus 8 \
    --config-dir configs/training/policy_stage/quadtok_ss256_vae_opt.yaml \
    --work-dir visualization/seach_32channel_adv_gd3

# Process even if output files exist
bash scripts/batch_pretokenize_configurable.sh \
    --start-idx 0 \
    --end-idx 10 \
    --num-gpus 4 \
    --no-skip-existing
```

**Options:**
- `--input-dir DIR`: Input directory containing shards
- `--output-dir DIR`: Output directory for tokenized shards
- `--shard-prefix PREFIX`: Shard filename prefix
- `--start-idx NUM`: Starting shard index
- `--end-idx NUM`: Ending shard index
- `--num-gpus NUM`: Number of GPUs to use
- `--config-dir PATH`: Path to config file
- `--work-dir DIR`: Working output directory
- `--mixed-precision TYPE`: Mixed precision type (bf16, fp16, etc.)
- `--base-port NUM`: Base port number for accelerate
- `--no-skip-existing`: Process even if output exists
- `-h, --help`: Show help message

## How It Works

Both scripts work similarly:

1. **Shard Discovery**: Scans for input shards matching the pattern `{SHARD_PREFIX}-{INDEX}.tar`
2. **GPU Allocation**: Distributes shards across available GPUs in a round-robin fashion
3. **Parallel Processing**: Launches multiple processes (one per GPU) to process shards concurrently
4. **Queue Management**: Maintains a queue of NUM_GPUS processes, starting a new one when an old one finishes
5. **Progress Tracking**: Logs all activities to `batch_progress.log` in the working directory

### Process Flow

```
For shards 000000 to 000070 with 8 GPUs:

Time 0:  Launch shards 000000-000007 on GPUs 0-7
Time 1:  GPU 0 finishes → Launch shard 000008 on GPU 0
Time 2:  GPU 3 finishes → Launch shard 000009 on GPU 3
...
Time N:  All shards completed
```

### File Organization

**Input Shards:**
```
/mnt/shared-storage-user/idc2-shared/dataset/preprocess/imagenet/
├── imagenet-train-000000.tar
├── imagenet-train-000001.tar
├── imagenet-train-000002.tar
└── ...
```

**Output Shards (same indices):**
```
/mnt/shared-storage-user/idc2-shared/dataset/preprocess/imagenet-tokenized/
├── imagenet-train-000000.tar
├── imagenet-train-000001.tar
├── imagenet-train-000002.tar
└── ...
```

**Log Files:**
```
visualization/seach_32channel_adv_gd3/
├── batch_progress.log                     # Overall progress
├── pretokenize_000000_gpu0.log           # Detailed log for shard 000000
├── pretokenize_000001_gpu1.log           # Detailed log for shard 000001
└── ...
```

## Features

### Skip Existing Files
By default, both scripts skip shards that have already been processed (output file exists). Use `--no-skip-existing` with the configurable script to force reprocessing.

### Automatic Port Assignment
Each process gets a unique port number to avoid conflicts:
- Process 0: port 12389
- Process 1: port 12390
- Process 2: port 12391
- etc.

### Error Handling
- Missing input files are logged and skipped
- Failed processes are logged with exit codes
- Summary report shows success/failure counts
- Exit with error code if any shard fails

### Resume Capability
If the script is interrupted, you can simply re-run it. It will skip already-processed shards and continue with the remaining ones.

## Monitoring Progress

### Real-time Monitoring
```bash
# Watch overall progress
watch -n 10 "tail -20 visualization/seach_32channel_adv_gd3/batch_progress.log"

# Monitor a specific GPU's log
tail -f visualization/seach_32channel_adv_gd3/pretokenize_000000_gpu0.log

# Count completed shards
grep "SUCCESS" visualization/seach_32channel_adv_gd3/batch_progress.log | wc -l
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

## Troubleshooting

### Issue: "Port already in use"
**Solution**: Change the `BASE_PORT` parameter to an unused port range.

### Issue: Out of GPU memory
**Solution**: Reduce the batch size in the config file or use fewer GPUs.

### Issue: Shard not found
**Solution**: Verify the input directory path and shard naming pattern.

### Issue: Process stuck
**Solution**: Check individual shard logs in the working directory. Kill stuck processes and adjust configuration if needed.

## Examples

### Process first 10 shards for testing
```bash
bash scripts/batch_pretokenize_configurable.sh \
    --start-idx 0 \
    --end-idx 9 \
    --num-gpus 2
```

### Process specific range on single GPU
```bash
bash scripts/batch_pretokenize_configurable.sh \
    --start-idx 50 \
    --end-idx 60 \
    --num-gpus 1
```

### Process validation set
```bash
bash scripts/batch_pretokenize_configurable.sh \
    --input-dir /path/to/validation/shards \
    --output-dir /path/to/tokenized/validation \
    --shard-prefix imagenet-val \
    --start-idx 0 \
    --end-idx 49 \
    --num-gpus 4
```

## Performance Tips

1. **GPU Count**: Use one GPU per concurrent process for best performance
2. **Batch Size**: Adjust `per_gpu_batch_size` in the config for your GPU memory
3. **I/O**: Use fast storage (SSD/NVMe) for better throughput
4. **Monitoring**: Keep an eye on GPU utilization to ensure full usage
5. **Testing**: Always test with a small range first (e.g., 0-2) to verify configuration

## Notes

- The script removes `sudo` from the original command as it's generally not needed
- Each process is completely independent and writes to its own output file
- The script is idempotent - running it multiple times is safe due to skip logic
- Output tar indices always match input tar indices

