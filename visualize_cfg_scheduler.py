import matplotlib
matplotlib.use('Agg')  # 适合服务器环境
import matplotlib.pyplot as plt
import numpy as np

def compute_cfg_sine_peak(step, max_seq_len, guidance_scale):
    """
    Sine-peak 方案：基于 sin^2 的对称钟形曲线
    特点：在起点和终点完全平滑归 1，逻辑最简洁。
    """
    progress = step / max_seq_len
    # sin^2(progress * pi) 在 0 和 1 时为 0，在 0.5 时为 1
    scale_step = np.sin(progress * np.pi) ** 2
    cfg_iter = 1 + (guidance_scale - 1) * scale_step
    return cfg_iter

def compute_cfg_gaussian(step, max_seq_len, guidance_scale, mu=0.5, sigma=0.2):
    """
    Gaussian 方案：基于高斯分布（正态分布）的曲线
    特点：可以通过 mu 调整峰值位置，通过 sigma 调整高峰持续的时间跨度。
    """
    progress = step / max_seq_len
    # 高斯公式：exp(-(x - mu)^2 / (2 * sigma^2))
    scale_step = np.exp(-(progress - mu)**2 / (2 * sigma**2))
    
    # 归一化处理：让最大值正好等于 guidance_scale
    # 注意：高斯曲线在起点和终点可能不完全为 1（取决于 sigma），通常在推理中是可接受的
    cfg_iter = 1 + (guidance_scale - 1) * scale_step
    return cfg_iter

# --- 适配可视化代码 ---

max_seq_len = 220
guidance_scale = 7.0
steps = np.arange(max_seq_len)

# 计算数值
cfg_sine = [compute_cfg_sine_peak(s, max_seq_len, guidance_scale) for s in steps]
# 这里 mu=0.6 是为了对齐你之前代码中 peak_position=0.6 的偏好
cfg_gaussian = [compute_cfg_gaussian(s, max_seq_len, guidance_scale, mu=0.6, sigma=0.2) for s in steps]

# 绘制对比图
plt.figure(figsize=(10, 6))

plt.plot(steps, cfg_sine, 'g-', linewidth=2.5, label='Sine-peak (Symmetric)')
plt.plot(steps, cfg_gaussian, 'm-', linewidth=2.5, label='Gaussian (mu=0.6, sigma=0.2)')

# 装饰图形
plt.xlabel('Step', fontsize=12)
plt.ylabel('CFG Scale', fontsize=12)
plt.title('Advanced CFG Schedulers: Sine vs Gaussian\n(Smooth Low → High → Low)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.axhline(y=guidance_scale, color='r', linestyle=':', alpha=0.5, label='Max Scale')
plt.axhline(y=1.0, color='k', linestyle='-', alpha=0.2)
plt.legend(fontsize=11)
plt.ylim(0.5, guidance_scale * 1.1)

plt.tight_layout()

# 这里的路径建议根据你的实际环境微调
output_path = '/mnt/petrelfs/jianglihan/my_code/quadtok/cfg_advanced_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"高级 Scheduler 图像已保存到: {output_path}")
plt.close()