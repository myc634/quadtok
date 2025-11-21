source /mnt/shared-storage-user/jianglihan/mc3/bin/activate
conda activate 1d

cd /mnt/shared-storage-user/jianglihan/myc/code/quadtok
# sudo find / -name "libnvrtc.so"
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

export MASTER_PORT=${MASTER_PORT:-29500}

if [ -z "$RANK" ]; then
    export RANK=$NODE_RANK
fi

if [ -z "$WORLD_SIZE" ]; then
    export WORLD_SIZE=$NODE_COUNT
fi


echo "Debug info: RANK=$RANK, WORLD_SIZE=$WORLD_SIZE, MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"


WANDB_MODE=offline accelerate launch \
    --config_file configs/accelerate_config/mnode8gpu.yaml \
    --main_process_ip=${MASTER_ADDR} \
    --main_process_port=${MASTER_PORT} \
    --machine_rank=${RANK} \
    --num_machines=${WORLD_SIZE} \
    --num_processes=16 \
    scripts/train_generator.py config=configs/training/generator/mar_quadtree.yaml