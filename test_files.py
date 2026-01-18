import subprocess
import tarfile
import io
import os
import sys
import argparse
from tqdm import tqdm

# --- 配置区 ---
# 远程路径的前缀
REMOTE_PATH_PREFIX = "hoss:jianglihan/data/imagenet-pretokenized/vq-ts12-4096codebook-base/"
# 预期的组件后缀
REQUIRED_EXTENSIONS = ['code_indices.npy', 'patch_indices.npy', 'lod_indices.npy', 'cls']

def check_tar_integrity(file_idx, show_progress=True):
    file_name = f"imagenet-train-{file_idx:06d}.tar"
    full_path = os.path.join(REMOTE_PATH_PREFIX, file_name)
    print("Checking file: ", full_path)
    # 构建 rclone 命令
    cmd = ["rclone", "cat", full_path]
    
    try:
        # 使用管道流式读取 tar 
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 使用 tarfile 读取流 (注意: 如果文件非常大，此方法依然依赖网络稳定)
        with tarfile.open(fileobj=io.BytesIO(process.stdout.read()), mode="r") as tar:
            members = tar.getnames()
            
            # 获取所有 base name (去掉后缀)
            keys = set(m.split('.')[0] for m in members if '.' in m)
            
            if not keys:
                return False, "文件为空或格式错误", 0

            # 检查每个 key 是否都有所有必需的后缀文件
            keys_list = list(keys)
            total_samples = len(keys_list)
            failed_samples = []
            
            iterator = tqdm(keys_list, desc=f"检查 {file_name}", leave=False, disable=not show_progress) if show_progress else keys_list
            for k in iterator:
                for ext in REQUIRED_EXTENSIONS:
                    expected = f"{k}.{ext}"
                    if expected not in members:
                        failed_samples.append((k, ext))
                        return False, f"样本 {k} 缺少组件 {ext}", total_samples
            
        return True, "OK", total_samples
    
    except Exception as e:
        return False, str(e), 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查 tar 文件的完整性")
    parser.add_argument("--file_idx", type=int, default=None, 
                        help="要检查的文件索引（0-19）。如果不指定，则检查所有文件")
    args = parser.parse_args()
    
    if args.file_idx is not None:
        # 单文件模式（用于 SLURM 数组任务）
        file_idx = args.file_idx
        
        success, msg, num_samples = check_tar_integrity(file_idx, show_progress=True)
        if success:
            print(f"✅ 文件 [{file_idx:06d}]: 正常 (共 {num_samples} 个样本)")
            sys.exit(0)
        else:
            print(f"❌ 文件 [{file_idx:06d}]: 错误 - {msg}")
            sys.exit(1)
    else:
        # 批量模式（检查所有文件）
        FILES_TO_CHECK = DEFAULT_FILES_TO_CHECK
        results = {}
        print(f"开始检查 {len(FILES_TO_CHECK)} 个文件...\n")
        
        for idx in tqdm(FILES_TO_CHECK, desc="总体进度", unit="文件"):
            success, msg, num_samples = check_tar_integrity(idx, show_progress=True)
            results[idx] = (success, msg, num_samples)
            if not success:
                status = f"❌ 错误: {msg}"
                tqdm.write(f"结果 [{idx:06d}]: {status}")
            else:
                tqdm.write(f"结果 [{idx:06d}]: ✅ 正常 (共 {num_samples} 个样本)")

        print("\n--- 统计汇总 ---")
        bad_files = [idx for idx, (s, m, n) in results.items() if not s]
        total_samples_checked = sum(n for _, (_, _, n) in results.items())
        
        if not bad_files:
            print(f"✅ 所有 {len(FILES_TO_CHECK)} 个测试文件均完整 (共检查 {total_samples_checked} 个样本)")
            print("报错大概率由训练时的网络/存储并发瓶颈引起。")
        else:
            print(f"❌ 检测到以下 {len(bad_files)} 个文件存在问题: {bad_files}")
            print(f"共检查 {total_samples_checked} 个样本")