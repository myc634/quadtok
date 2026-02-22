import torch
import torch.nn.functional as F

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, return_sum: bool = True) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    if return_sum:
        return psnr.sum()
    else:
        return psnr

def gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    _1d_window = g.unsqueeze(1)
    _2d_window = g[:, None] * g[None, :]
    window = _2d_window.unsqueeze(0).unsqueeze(0)  # shape (1,1,window,window)
    window = window.repeat(channels, 1, 1, 1)      # shape (channels,1,window,window)
    return window

def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5, max_val: float = 1.0, C1=None, C2=None, return_sum: bool = True) -> torch.Tensor:
    """
    Computes SSIM between two batches of images.
    Args:
        pred: (N, C, H, W) predicted images in [0,1].
        target: (N, C, H, W) ground truth images in [0,1].
        window_size: Size of the Gaussian kernel.
        sigma: Standard deviation of the Gaussian.
        max_val: Maximum value in images.
        C1, C2: Stability constants. If None, default to standard settings.
    Returns:
        SSIM values per image in batch. Shape (N,)
    """
    if C1 is None:
        C1 = (0.01 * max_val) ** 2
    if C2 is None:
        C2 = (0.03 * max_val) ** 2

    N, C, H, W = pred.shape
    window = gaussian_window(window_size, sigma, C).to(pred.device)
    
    # Use grouped convolution to apply the same kernel to each channel independently
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=C)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=C) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Mean over channels and spatial dims
    ssim_val = ssim_map.view(N, C, -1).mean(dim=[1,2])
    if return_sum:
        return ssim_val.sum()
    else:
        return ssim_val