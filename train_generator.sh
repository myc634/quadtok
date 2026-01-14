CONFIG="configs/training/generator/gpt_quadtree.yaml"
echo "CONFIG: $CONFIG"


TORCH_DISTRIBUTED_DEBUG=DETAIL WANDB_MODE=offline accelerate launch \
    --mixed_precision=bf16 \
    --num_machines=1 \
    --num_processes=1 \
    --machine_rank=0 \
    --main_process_ip=127.0.0.1 \
    --main_process_port=29500 \
    scripts/train_generator.py \
    config=${CONFIG}

