CUDA_VISIBLE_DEVICES=5 python scripts/extract_code_randomquadtree.py \
        --config_dir "./tokenizer_config.yaml" \
        --tokenizer_weight "./tokenizer_v3_search_new.bin" \
        --output_dir "extract_token_log/vq-ts12-4kcodebook-2lods" \
        --shards_index 0 \
        --output_tar_path "./tmp_imagenet_files" \
        --guaranteed_depth 3 \
        --expansion_probs 0.3 0.2 \
        --num_workers 2 \
        --crop_range 1.1