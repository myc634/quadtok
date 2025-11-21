source /mnt/shared-storage-user/jianglihan/mc3/bin/activate
conda activate 1d

cd /mnt/shared-storage-user/jianglihan/myc/code/quadtok
# sudo find / -name "libnvrtc.so"
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
TORCH_DISTRIBUTED_DEBUG=DETAIL WANDB_MODE=offline accelerate launch --mixed_precision=bf16 --num_machines=1 --num_processes=8 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=2344 --same_network scripts/train_generator.py config=configs/training/generator/gpt_quadtree.yaml