#!/usr/bin/env python3
"""
计算Austrian Crop数据集的统计值
包括：类别分布、数据均值、标准差、最小值、最大值
"""

import os
import numpy as np
from glob import glob
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

def process_band_file(file_path):
    """处理单个band文件，返回统计信息"""
    try:
        data = np.load(file_path)  # Shape: H, W, C, T
        # 重塑为 (N, C) 形式，其中N = H * W * T
        H, W, C, T = data.shape
        data_reshaped = data.transpose(0, 1, 3, 2).reshape(-1, C)  # (H*W*T, C)
        
        # 计算每个通道的统计值
        channel_sum = np.sum(data_reshaped, axis=0)
        channel_sum_sq = np.sum(data_reshaped ** 2, axis=0)
        channel_min = np.min(data_reshaped, axis=0)
        channel_max = np.max(data_reshaped, axis=0)
        count = data_reshaped.shape[0]
        
        return {
            'sum': channel_sum,
            'sum_sq': channel_sum_sq,
            'min': channel_min,
            'max': channel_max,
            'count': count
        }
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_label_file(file_path):
    """处理单个label文件，返回类别计数"""
    try:
        data = np.load(file_path)  # Shape: H, W
        # 计算每个类别的像素数
        unique, counts = np.unique(data, return_counts=True)
        class_counts = np.zeros(18, dtype=np.int64)  # 18个类别
        for cls, cnt in zip(unique, counts):
            if 0 <= cls <= 17:
                class_counts[cls] = cnt
        
        return class_counts
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def compute_statistics(root_path, num_processes=None):
    """计算数据集的所有统计值"""
    if num_processes is None:
        num_processes = min(cpu_count(), 96)  # 使用可用的CPU核心数，最多96个
    
    print(f"使用 {num_processes} 个进程进行计算...")
    
    # 获取所有文件路径
    band_path = os.path.join(root_path, 'band_patch')
    label_path = os.path.join(root_path, 'label_patch')
    
    band_files = sorted(glob(os.path.join(band_path, '*.npy')))
    label_files = sorted(glob(os.path.join(label_path, '*.npy')))
    
    print(f"找到 {len(band_files)} 个band文件")
    print(f"找到 {len(label_files)} 个label文件")
    
    # 1. 计算波段统计值
    print("\n计算波段统计值...")
    with Pool(num_processes) as pool:
        band_results = pool.map(process_band_file, band_files)
    
    # 过滤掉None结果
    band_results = [r for r in band_results if r is not None]
    
    # 获取通道数
    if band_results:
        num_channels = len(band_results[0]['sum'])
        print(f"检测到 {num_channels} 个通道")
    else:
        print("错误：没有成功处理任何band文件")
        return
    
    # 聚合所有文件的统计值
    total_sum = np.zeros(num_channels)
    total_sum_sq = np.zeros(num_channels)
    total_min = np.full(num_channels, np.inf)
    total_max = np.full(num_channels, -np.inf)
    total_count = 0
    
    for result in band_results:
        total_sum += result['sum']
        total_sum_sq += result['sum_sq']
        total_min = np.minimum(total_min, result['min'])
        total_max = np.maximum(total_max, result['max'])
        total_count += result['count']
    
    # 计算均值和标准差
    data_mean = total_sum / total_count
    data_var = (total_sum_sq / total_count) - (data_mean ** 2)
    data_std = np.sqrt(np.maximum(data_var, 0))  # 避免负方差
    
    # 2. 计算类别分布
    print("\n计算类别分布...")
    with Pool(num_processes) as pool:
        label_results = pool.map(process_label_file, label_files)
    
    # 过滤掉None结果
    label_results = [r for r in label_results if r is not None]
    
    # 聚合所有文件的类别计数
    total_class_counts = np.zeros(18, dtype=np.int64)
    for result in label_results:
        total_class_counts += result
    
    # 计算类别分布比例（排除背景类）
    total_pixels = np.sum(total_class_counts)
    
    # 创建分布数组，背景类设为0
    distribution = np.zeros(18)
    
    # 计算非背景类的总像素数
    non_bg_total = np.sum(total_class_counts[1:])  # 从类别1开始
    
    # 对非背景类进行归一化
    if non_bg_total > 0:
        distribution[1:] = total_class_counts[1:] / non_bg_total
    
    # 背景类设为0
    distribution[0] = 0.0
    
    # 打印结果
    print("\n" + "="*80)
    print("统计结果（可直接复制到YAML文件）:")
    print("="*80)
    
    # 类别名称
    class_names = [
        "Background", "Legume", "Soy", "Summer Grain", "Winter Grain",
        "Corn", "Sunflower", "Mustard", "Potato", "Beet",
        "Squash", "Grapes", "Tree Fruit", "Cover Crop", "Grass",
        "Fallow", "Other (Plants)", "Other (Non Plants)"
    ]
    
    print("\n# 类别分布（背景类排除，其余类归一化）")
    print("distribution:")
    for i, (name, dist) in enumerate(zip(class_names, distribution)):
        if i == 0:
            print(f"  - {dist:.5f}  # {name} (excluded, {total_class_counts[i]:,} pixels)")
        else:
            print(f"  - {dist:.5f}  # {name} ({total_class_counts[i]:,} pixels, {dist*100:.2f}% of non-bg)")
    
    print("\n# 数据均值")
    print("data_mean:")
    print("  optical:")
    for i, mean in enumerate(data_mean):
        print(f"    - {mean:.4f}  # Band {i+1}")
    
    print("\n# 数据标准差")
    print("data_std:")
    print("  optical:")
    for i, std in enumerate(data_std):
        print(f"    - {std:.4f}  # Band {i+1}")
    
    print("\n# 数据最小值")
    print("data_min:")
    print(f"  optical: [{', '.join([f'{v:.1f}' for v in total_min])}]")
    
    print("\n# 数据最大值")
    print("data_max:")
    print(f"  optical: [{', '.join([f'{v:.1f}' for v in total_max])}]")
    
    # 额外统计信息
    print("\n" + "="*80)
    print("额外统计信息:")
    print("="*80)
    print(f"总像素数: {total_pixels:,}")
    print(f"背景像素数: {total_class_counts[0]:,} ({total_class_counts[0]/total_pixels*100:.2f}%)")
    print(f"非背景像素数: {non_bg_total:,} ({non_bg_total/total_pixels*100:.2f}%)")
    print(f"总样本数 (H*W*T): {total_count:,}")
    print(f"波段数: {num_channels}")
    print("\n类别像素占比（相对于所有像素）:")
    for i, (name, cnt) in enumerate(zip(class_names, total_class_counts)):
        if cnt > 0:
            print(f"  {name:20s}: {cnt/total_pixels*100:6.2f}% ({cnt:10,} pixels)")
    
    print("\n类别像素占比（相对于非背景像素）:")
    for i, (name, dist) in enumerate(zip(class_names[1:], distribution[1:]), 1):
        if total_class_counts[i] > 0:
            print(f"  {name:20s}: {dist*100:6.2f}% ({total_class_counts[i]:10,} pixels)")

def main():
    root_path = "/shared/amdgpu/home/avsm2_f4q/code/pangaea-bench/data/austrian_crop_v1.0_pipeline_prithvi"
    
    # 检查路径是否存在
    if not os.path.exists(root_path):
        print(f"错误：路径不存在 - {root_path}")
        return
    
    # 检查子文件夹是否存在
    band_path = os.path.join(root_path, 'band_patch')
    label_path = os.path.join(root_path, 'label_patch')
    
    if not os.path.exists(band_path):
        print(f"错误：band_patch文件夹不存在 - {band_path}")
        return
    
    if not os.path.exists(label_path):
        print(f"错误：label_patch文件夹不存在 - {label_path}")
        return
    
    # 计算统计值
    compute_statistics(root_path, num_processes=256)

if __name__ == "__main__":
    main()