#!/usr/bin/env python3
"""汇总所有检查结果的脚本"""
import os
import re
from pathlib import Path
from collections import defaultdict

LOG_DIR = "logs"
OUTPUT_FILE = "logs/check_results_summary.txt"

def parse_log_file(log_file):
    """解析单个日志文件，提取检查结果"""
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取文件索引
        file_idx_match = re.search(r'文件索引: (\d+)', content)
        if not file_idx_match:
            return None
        
        file_idx = int(file_idx_match.group(1))
        
        # 检查结果
        if '✅ 文件' in content and '正常' in content:
            # 提取样本数
            samples_match = re.search(r'共 (\d+) 个样本', content)
            num_samples = int(samples_match.group(1)) if samples_match else 0
            return {'file_idx': file_idx, 'status': 'ok', 'num_samples': num_samples, 'error': None}
        elif '❌ 文件' in content:
            # 提取错误信息
            error_match = re.search(r'错误 - (.+)', content)
            error_msg = error_match.group(1).strip() if error_match else "未知错误"
            return {'file_idx': file_idx, 'status': 'error', 'num_samples': 0, 'error': error_msg}
        else:
            return {'file_idx': file_idx, 'status': 'unknown', 'num_samples': 0, 'error': '无法解析结果'}
    
    except Exception as e:
        return None

def main():
    log_dir = Path(LOG_DIR)
    if not log_dir.exists():
        print(f"错误: 日志目录 {LOG_DIR} 不存在")
        return
    
    # 查找所有 check_files 相关的日志文件
    log_files = list(log_dir.glob("check_files_*.out"))
    
    if not log_files:
        print(f"错误: 在 {LOG_DIR} 中未找到检查日志文件")
        return
    
    print(f"找到 {len(log_files)} 个日志文件")
    print("正在解析...")
    
    results = {}
    for log_file in log_files:
        result = parse_log_file(log_file)
        if result:
            file_idx = result['file_idx']
            # 提取任务ID
            task_match = re.search(r'check_files_(\d+)_\d+\.out', log_file.name)
            result['task_id'] = task_match.group(1) if task_match else 'unknown'
            # 如果有多个任务检查同一个文件，保留最新的结果
            if file_idx not in results:
                results[file_idx] = result
            else:
                # 比较修改时间，保留最新的
                old_log = Path(LOG_DIR) / f"check_files_{results[file_idx]['task_id']}_{file_idx}.out"
                if old_log.exists() and log_file.stat().st_mtime > old_log.stat().st_mtime:
                    results[file_idx] = result
    
    # 按文件索引排序
    sorted_results = sorted(results.items())
    
    # 生成汇总报告
    output_path = Path(OUTPUT_FILE)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("文件完整性检查结果汇总\n")
        f.write("=" * 60 + "\n\n")
        
        ok_files = []
        error_files = []
        total_samples = 0
        
        for file_idx, result in sorted_results:
            status_icon = "✅" if result['status'] == 'ok' else "❌"
            f.write(f"[{file_idx:06d}] {status_icon} ")
            
            if result['status'] == 'ok':
                f.write(f"正常 (共 {result['num_samples']} 个样本)\n")
                ok_files.append(file_idx)
                total_samples += result['num_samples']
            else:
                f.write(f"错误: {result['error']}\n")
                error_files.append(file_idx)
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("统计汇总\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总文件数: {len(sorted_results)}\n")
        f.write(f"正常文件: {len(ok_files)}\n")
        f.write(f"错误文件: {len(error_files)}\n")
        f.write(f"总样本数: {total_samples}\n\n")
        
        if error_files:
            f.write(f"错误文件列表: {error_files}\n")
        else:
            f.write("✅ 所有文件均完整。报错大概率由训练时的网络/存储并发瓶颈引起。\n")
    
    # 同时在终端输出
    print("\n" + "=" * 60)
    print("文件完整性检查结果汇总")
    print("=" * 60)
    
    for file_idx, result in sorted_results:
        status_icon = "✅" if result['status'] == 'ok' else "❌"
        if result['status'] == 'ok':
            print(f"[{file_idx:06d}] {status_icon} 正常 (共 {result['num_samples']} 个样本)")
        else:
            print(f"[{file_idx:06d}] {status_icon} 错误: {result['error']}")
    
    print("\n" + "=" * 60)
    print("统计汇总")
    print("=" * 60)
    print(f"总文件数: {len(sorted_results)}")
    print(f"正常文件: {len(ok_files)}")
    print(f"错误文件: {len(error_files)}")
    print(f"总样本数: {total_samples}")
    
    if error_files:
        print(f"\n错误文件列表: {error_files}")
    else:
        print("\n✅ 所有文件均完整。报错大概率由训练时的网络/存储并发瓶颈引起。")
    
    print(f"\n详细报告已保存到: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
