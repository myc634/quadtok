bash scripts/extract_codes_c2i.sh \
    --vq-ckpt /mnt/ultracube/zec016/quadtok/vq_ckpts/tokenizer_ckpt.bin \
    --data-path /mnt/ultracube/datasets/imagenet/ILSVRC/Data/CLS-LOC/train \
    --code-path /mnt/ultracube/zec016/quadtok_codes/quadtree_fixtree_105 \
    --ten-crop \
    --crop-range 1.05 \
    --image-size 256 \

# --data-path /mnt/localssd/imagenet/train \
# --data-path /mnt/localssd/imagenet/train/n01440764 \
