export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

echo "Master IP: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --machine_rank=0 \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  --mixed_precision=bf16 \
  scripts/train_generator.py config=/mnt/ultracube/zec016/quadtok/configs/training/generator/gpt_quadtree_fixtree.yaml