CUDA_VISIBLE_DEVICES=1 python scripts/extract_code_randomquadtree.py \
        --config_dir "/mnt/ultracube/zec016/quadtok_myc/quadtok-vq-4096codebook-expandprob32-12ts-config.yaml" \
        --tokenizer_weight "/mnt/ultracube/zec016/quadtok_myc/quadtok-vq-4096codebook-expandprob32-12ts.bin" \
        --output_dir "extract_token_log" \
        --shards_index 0 \
        --output_tar_path "/mnt/ultracube/zec016/quadtok_codes_tar" \
        --guaranteed_depth 3 \
        --num_workers 2 \
        --crop_range 1.1