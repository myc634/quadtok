
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

@torch.no_grad()
def load_encoders(model_type, resolution=256):

    encoder_type, architecture, model_config = model_type.split('-')
    if 'dinov2' in encoder_type:
        import timm
        if 'reg' in encoder_type:
            encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
        else:
            encoder = torch.hub.load("/mnt/petrelfs/jianglihan/my_code/dinov2", f'dinov2_vit{model_config}14', source='local', pretrained=False)
            state_dict = torch.load("pretrained_weight/dinov2_vitb14_pretrain.pth", map_location='cpu')
            encoder.load_state_dict(state_dict, strict=True)
            # encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')
        del encoder.head
        patch_resolution = 16 * (resolution // 256)
        encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            encoder.pos_embed.data, [patch_resolution, patch_resolution],
        )
        encoder.head = torch.nn.Identity()
        encoder.eval()

    return encoder

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x



