source /mnt/shared-storage-user/jianglihan/mc3/bin/activate
conda activate 1d

export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
cd /mnt/shared-storage-user/jianglihan/myc/code/quadtok2

export WANDB_MODE=offline 

accelerate launch \
  --num_processes=4 \
  --num_machines=1 \
  --machine_rank=0 \
  --main_process_ip=127.0.0.1 \
  --main_process_port=29501 \
  --mixed_precision=bf16 \
  scripts/inference_generator.py \
  --config configs/inference/gpt_4k.yaml \
  --num_samples 50000 \
  --batch_size 128 \
  --checkpoint checkpoints/generator/gpt_quadtree_base_4096codebook_2lod_fixpretokenization/checkpoint-60000/ema_model/pytorch_model.bin