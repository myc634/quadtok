python scripts/inference_generator.py \
    --config configs/inference/gpt_16k_base.yaml \
    --num_samples 1000 \
    --batch_size 32 \
    --checkpoint final_ckpt/gpt/pytorch_model.bin
