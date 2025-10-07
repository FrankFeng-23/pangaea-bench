#!/usr/bin/env python3
"""
计算TreeSatAI数据集的统计信息
"""
import h5py
import json
import os
import numpy as np
from collections import Counter
from tqdm import tqdm

def calculate_dataset_stats(data_root):
    """计算数据集的统计信息"""
    h5_dir = os.path.join(data_root, "sentinel-ts")
    labels_path = os.path.join(data_root, "TreeSatBA_v9_60m_multi_labels.json")
    
    # 加载标签
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    # 定义类别
    classes = [
        "Abies", "Acer", "Alnus", "Betula", "Cleared", "Fagus", 
        "Fraxinus", "Larix", "Picea", "Pinus", "Populus", 
        "Pseudotsuga", "Quercus", "Salix", "Tilia"
    ]
    
    # 统计类别分布
    class_counts = Counter()
    for label_list in labels.values():
        for class_name, confidence in label_list:
            if class_name in classes:
                class_counts[class_name] += 1
    
    distribution = [class_counts.get(cls, 0) for cls in classes]
    
    # 计算数据统计信息
    optical_data_all = []
    sar_data_all = []
    
    h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
    
    # 采样一部分文件来计算统计信息（避免内存不足）
    sample_size = min(1000, len(h5_files))
    sampled_files = np.random.choice(h5_files, sample_size, replace=False)
    
    print(f"计算统计信息，采样 {sample_size} 个文件...")
    
    for h5_file in tqdm(sampled_files):
        h5_path = os.path.join(h5_dir, h5_file)
        
        try:
            with h5py.File(h5_path, 'r') as h5file:
                # Sentinel-2 数据
                sen2_data = h5file['sen-2-data'][:]  # Shape: (T, 10, 6, 6)
                optical_data_all.append(sen2_data.reshape(-1, sen2_data.shape[1]))
                
                # Sentinel-1 数据
                sen1_asc_data = h5file['sen-1-asc-data'][:]  # Shape: (T, 2, 6, 6)
                sen1_des_data = h5file['sen-1-des-data'][:]  # Shape: (T, 2, 6, 6)
                
                sar_data = np.concatenate([sen1_asc_data, sen1_des_data], axis=0)
                # 重塑为 (N, 2) 而不是 (N, channels)
                sar_reshaped = sar_data.reshape(-1, 2)  # 将 (T, 2, 6, 6) -> (T*6*6, 2)
                
                # 过滤掉无效值
                valid_mask = np.isfinite(sar_reshaped).all(axis=1)
                if np.any(valid_mask):
                    sar_data_all.append(sar_reshaped[valid_mask])
                
        except Exception as e:
            print(f"Error processing {h5_file}: {e}")
            continue
    
    # 合并所有数据
    optical_data_all = np.concatenate(optical_data_all, axis=0)  # Shape: (N, 10)
    sar_data_all = np.concatenate(sar_data_all, axis=0)  # Shape: (N, 2)
    
    # 计算统计信息
    optical_mean = np.mean(optical_data_all, axis=0).tolist()
    optical_std = np.std(optical_data_all, axis=0).tolist()
    optical_min = np.min(optical_data_all, axis=0).tolist()
    optical_max = np.max(optical_data_all, axis=0).tolist()
    
    sar_mean = np.mean(sar_data_all, axis=0).tolist()
    sar_std = np.std(sar_data_all, axis=0).tolist()
    sar_min = np.min(sar_data_all, axis=0).tolist()
    sar_max = np.max(sar_data_all, axis=0).tolist()
    
    stats = {
        "classes": classes,
        "num_classes": len(classes),
        "distribution": distribution,
        "total_samples": len(labels),
        "data_mean": {
            "optical": optical_mean,
            "sar": sar_mean
        },
        "data_std": {
            "optical": optical_std,
            "sar": sar_std
        },
        "data_min": {
            "optical": optical_min,
            "sar": sar_min
        },
        "data_max": {
            "optical": optical_max,
            "sar": sar_max
        }
    }
    
    return stats

def main():
    data_root = "/scratch/zf281/pangaea-bench/data/treesatai"
    
    print("计算TreeSatAI数据集统计信息...")
    stats = calculate_dataset_stats(data_root)
    
    print("\n=== 数据集统计信息 ===")
    print(f"类别数: {stats['num_classes']}")
    print(f"总样本数: {stats['total_samples']}")
    print(f"类别: {stats['classes']}")
    print(f"类别分布: {stats['distribution']}")
    
    print(f"\n光学数据统计:")
    print(f"  均值: {[f'{x:.4f}' for x in stats['data_mean']['optical']]}")
    print(f"  标准差: {[f'{x:.4f}' for x in stats['data_std']['optical']]}")
    print(f"  最小值: {[f'{x:.4f}' for x in stats['data_min']['optical']]}")
    print(f"  最大值: {[f'{x:.4f}' for x in stats['data_max']['optical']]}")
    
    print(f"\nSAR数据统计:")
    print(f"  均值: {[f'{x:.4f}' for x in stats['data_mean']['sar']]}")
    print(f"  标准差: {[f'{x:.4f}' for x in stats['data_std']['sar']]}")
    print(f"  最小值: {[f'{x:.4f}' for x in stats['data_min']['sar']]}")
    print(f"  最大值: {[f'{x:.4f}' for x in stats['data_max']['sar']]}")
    
    # 保存统计信息
    output_path = "/scratch/zf281/pangaea-bench/tests/treesatai_stats.json"
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n统计信息已保存到: {output_path}")

if __name__ == "__main__":
    main()