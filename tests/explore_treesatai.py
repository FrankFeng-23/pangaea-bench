#!/usr/bin/env python3
"""
探索TreeSatAI数据集的结构
"""
import h5py
import json
import os
import numpy as np
from collections import defaultdict, Counter

def explore_h5_file(h5_path):
    """探索单个H5文件的结构"""
    print(f"\n=== 探索H5文件: {os.path.basename(h5_path)} ===")
    
    with h5py.File(h5_path, 'r') as h5file:
        print("H5文件中的数据集:")
        for key in h5file.keys():
            dataset = h5file[key]
            print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
            
            # 显示一些样本数据
            if len(dataset.shape) > 0 and dataset.shape[0] > 0:
                if 'products' in key:
                    # 产品名称，显示前几个
                    sample_data = dataset[:min(3, dataset.shape[0])]
                    print(f"    样本数据: {sample_data}")
                else:
                    # 数值数据，显示统计信息
                    data = dataset[:]
                    print(f"    数据范围: [{np.min(data):.4f}, {np.max(data):.4f}]")
                    print(f"    数据均值: {np.mean(data):.4f}, 标准差: {np.std(data):.4f}")

def explore_labels(json_path):
    """探索标签JSON文件"""
    print(f"\n=== 探索标签文件: {os.path.basename(json_path)} ===")
    
    with open(json_path, 'r') as f:
        labels = json.load(f)
    
    print(f"总样本数: {len(labels)}")
    
    # 统计类别
    all_classes = set()
    class_counts = Counter()
    multi_label_count = 0
    
    for filename, label_list in labels.items():
        if len(label_list) > 1:
            multi_label_count += 1
        
        for class_name, confidence in label_list:
            all_classes.add(class_name)
            class_counts[class_name] += 1
    
    print(f"多标签样本数: {multi_label_count}")
    print(f"总类别数: {len(all_classes)}")
    print(f"类别列表: {sorted(all_classes)}")
    print(f"类别分布 (前10个):")
    for class_name, count in class_counts.most_common(10):
        print(f"  {class_name}: {count}")
    
    # 显示一些样本
    print(f"\n样本标签示例:")
    for i, (filename, label_list) in enumerate(list(labels.items())[:5]):
        print(f"  {filename}: {label_list}")
    
    return all_classes, labels

def match_h5_with_labels(h5_dir, labels):
    """匹配H5文件与标签"""
    print(f"\n=== 匹配H5文件与标签 ===")
    
    h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
    print(f"H5文件数量: {len(h5_files)}")
    
    matched_count = 0
    unmatched_h5 = []
    unmatched_labels = []
    
    # 检查H5文件是否有对应标签
    for h5_file in h5_files:
        # 从H5文件名提取基础名称
        base_name = h5_file.replace('.h5', '')
        # 移除年份后缀
        parts = base_name.split('_')
        if len(parts) >= 2 and parts[-1].isdigit():
            base_name_no_year = '_'.join(parts[:-1])
        else:
            base_name_no_year = base_name
        
        # 查找对应的标签
        label_key = base_name_no_year + '.tif'
        if label_key in labels:
            matched_count += 1
        else:
            unmatched_h5.append(h5_file)
    
    print(f"匹配的H5文件数量: {matched_count}")
    print(f"未匹配的H5文件数量: {len(unmatched_h5)}")
    
    if unmatched_h5:
        print(f"未匹配的H5文件示例: {unmatched_h5[:5]}")
    
    return matched_count

def main():
    # 数据路径
    data_root = "/scratch/zf281/pangaea-bench/data/treesatai"
    h5_dir = os.path.join(data_root, "sentinel-ts")
    labels_path = os.path.join(data_root, "TreeSatBA_v9_60m_multi_labels.json")
    
    # 探索标签
    all_classes, labels = explore_labels(labels_path)
    
    # 探索几个H5文件
    h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')][:3]
    for h5_file in h5_files:
        h5_path = os.path.join(h5_dir, h5_file)
        explore_h5_file(h5_path)
    
    # 匹配H5文件与标签
    matched_count = match_h5_with_labels(h5_dir, labels)
    
    print(f"\n=== 总结 ===")
    print(f"类别数: {len(all_classes)}")
    print(f"样本数: {len(labels)}")
    print(f"匹配的H5文件数: {matched_count}")

if __name__ == "__main__":
    main()