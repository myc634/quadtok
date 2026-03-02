# python scripts/inference_generator.py \
#     --config configs/inference/gpt_16k_base.yaml \
#     --num_samples 1000 \
#     --batch_size 32 \
#     --checkpoint final_ckpt/gpt/pytorch_model.bin

python scripts/inference_generator_from_plan_pkl.py \
  --config configs/inference/gpt_16k_base.yaml \
  --checkpoint final_ckpt/gpt/pytorch_model.bin \
  --plan_pkl tree_planning/plan.pkl \
  --output_dir ./gen_out \
  --output_name class_1.png

python scripts/inference_generator_from_plan_pkl.py --config configs/inference/gpt_16k_base.yaml --checkpoint final_ckpt/gpt/pytorch_model.bin --plans-dir tree_planning/plans --output_dir tree_planning/gen_out