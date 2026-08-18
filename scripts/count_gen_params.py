import sys, os, gc
REPO = '/sensei-fs-3/users/yuchengm/code/quadtok/quadtok-base-update'
sys.path.insert(0, REPO)
import torch
from omegaconf import OmegaConf
import modeling.mar as marmod

# Skip the O(M*P) weight init (overwritten by load anyway) so counting is fast.
for cls in ['MAR', 'CausalMAR', 'QuadtreeMAR', 'QuadtreeGPT']:
    if hasattr(marmod, cls) and hasattr(getattr(marmod, cls), 'initialize_weights'):
        setattr(getattr(marmod, cls), 'initialize_weights', lambda self: None)

cfg = OmegaConf.load(os.path.join(REPO, 'configs/inference/gpt_16k_base.yaml'))


def count(sz):
    cfg.model.generator.model_size = sz
    last = None
    for dev in ['meta', 'cpu']:
        try:
            with torch.device(dev):
                m = marmod.QuadtreeGPT(cfg)
            n = sum(p.numel() for p in m.parameters())
            # FFN hidden dim of first block, to show the multiplier effect
            ffn_h = None
            try:
                ffn_h = m.layers[0].feed_forward.w1.weight.shape[0]
            except Exception:
                pass
            del m
            gc.collect()
            return f'{n/1e6:8.1f}M  ({n:>12d})  ffn_hidden(layer0.w1)={ffn_h}  [{dev}]'
        except Exception as e:
            last = f'ERR {type(e).__name__}: {e}'
    return last


print('QuadtreeGPT param counts (depth base=24 / large=32 / xlarge=40):')
for sz in ['base', 'large', 'xlarge']:
    print(f'  {sz:7s}: {count(sz)}')
