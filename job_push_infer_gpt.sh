export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline 
export NCCL_DEBUG=INFO

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

echo "Master IP: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"


CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --machine_rank=0 \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  --mixed_precision=bf16 \
  scripts/inference_generator.py \
  --config /mnt/ultracube/zec016/quadtok_myc/configs/training/generator/gpt_quadtree.yaml \
  --num_samples 50500 \
  --batch_size 32 \
  --checkpoint /mnt/ultracube/zec016/quadtok_myc/gpt_base_4kcs_12ts_20k.bin \