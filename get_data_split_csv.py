import os
import numpy as np
import pandas as pd
from pathlib import Path
import random
from typing import Dict, List, Set, Tuple

def read_patch_and_get_labels(patch_path: str) -> Set[int]:
    """读取patch并返回其中包含的所有唯一标签（除了0）"""
    data = np.load(patch_path)
    unique_labels = set(np.unique(data))
    unique_labels.discard(0)  # 移除背景标签0
    return unique_labels

def get_all_patches_with_labels(label_patch_dir: str) -> Tuple[Dict[str, Set[int]], int]:
    """读取所有patch文件并记录每个patch包含的标签，同时返回patch大小"""
    patch_labels = {}
    patch_size = None
    
    # 获取所有.npy文件
    npy_files = sorted([f for f in os.listdir(label_patch_dir) if f.endswith('.npy')])
    
    print(f"找到 {len(npy_files)} 个patch文件")
    
    for npy_file in npy_files:
        patch_path = os.path.join(label_patch_dir, npy_file)
        patch_name = npy_file.replace('.npy', '')
        
        # 获取标签
        labels = read_patch_and_get_labels(patch_path)
        patch_labels[patch_name] = labels
        
        # 获取patch大小（只需要读取第一个文件）
        if patch_size is None:
            data = np.load(patch_path)
            patch_size = data.shape[0]  # H和W应该相同
            print(f"Patch大小: {patch_size}x{patch_size}")
    
    return patch_labels, patch_size

def ensure_all_labels_in_train(patch_labels: Dict[str, Set[int]], 
                              train_patches: List[str],
                              all_patches: List[str],
                              required_labels: Set[int]) -> List[str]:
    """确保训练集包含所有必需的标签（1-17）"""
    # 检查当前训练集包含的所有标签
    train_labels = set()
    for patch in train_patches:
        train_labels.update(patch_labels[patch])
    
    # 找出缺失的标签
    missing_labels = required_labels - train_labels
    
    if not missing_labels:
        return train_patches
    
    print(f"训练集缺少标签: {sorted(missing_labels)}")
    
    # 从剩余的patches中找包含缺失标签的
    remaining_patches = [p for p in all_patches if p not in train_patches]
    
    # 为每个缺失的标签找到包含它的patch
    for label in missing_labels:
        candidates = [p for p in remaining_patches 
                     if label in patch_labels[p] and p not in train_patches]
        
        if candidates:
            # 选择包含最多其他缺失标签的patch
            best_patch = max(candidates, 
                           key=lambda p: len(patch_labels[p].intersection(missing_labels)))
            train_patches.append(best_patch)
            remaining_patches.remove(best_patch)
            print(f"添加 {best_patch} 到训练集以包含标签 {label}")
    
    # 再次验证
    final_train_labels = set()
    for patch in train_patches:
        final_train_labels.update(patch_labels[patch])
    
    still_missing = required_labels - final_train_labels
    if still_missing:
        print(f"警告：即使调整后，训练集仍缺少标签: {sorted(still_missing)}")
    
    return train_patches

def split_dataset(patch_labels: Dict[str, Set[int]], 
                 train_ratio: float = 0.1) -> Dict[str, str]:
    """划分数据集为训练/验证/测试集"""
    all_patches = list(patch_labels.keys())
    random.shuffle(all_patches)
    
    total_count = len(all_patches)
    train_count = int(total_count * train_ratio)
    
    # 初始划分
    train_patches = all_patches[:train_count]
    remaining = all_patches[train_count:]
    
    # 确保训练集包含所有标签1-17
    required_labels = set(range(1, 18))
    train_patches = ensure_all_labels_in_train(patch_labels, train_patches, 
                                              all_patches, required_labels)
    
    # 更新剩余的patches
    remaining = [p for p in all_patches if p not in train_patches]
    
    # 将剩余的按1:7比例分为验证集和测试集
    val_count = len(remaining) // 8  # 1/(1+7) = 1/8
    val_patches = remaining[:val_count]
    test_patches = remaining[val_count:]
    
    # 创建结果字典
    split_dict = {}
    for patch in train_patches:
        split_dict[patch] = 'train'
    for patch in val_patches:
        split_dict[patch] = 'val'
    for patch in test_patches:
        split_dict[patch] = 'test'
    
    # 打印统计信息
    print(f"\n数据集划分统计:")
    print(f"总数: {total_count}")
    print(f"训练集: {len(train_patches)} ({len(train_patches)/total_count*100:.1f}%)")
    print(f"验证集: {len(val_patches)} ({len(val_patches)/total_count*100:.1f}%)")
    print(f"测试集: {len(test_patches)} ({len(test_patches)/total_count*100:.1f}%)")
    
    # 验证训练集标签覆盖
    train_labels = set()
    for patch in train_patches:
        train_labels.update(patch_labels[patch])
    print(f"\n训练集包含的标签: {sorted(train_labels)}")
    
    return split_dict

def save_split_csv(split_dict: Dict[str, str], 
                  output_path: str,
                  patch_size: int,
                  train_ratio: float):
    """保存划分结果到CSV文件"""
    # 创建DataFrame
    df_data = []
    for patch_name, split_type in sorted(split_dict.items()):
        df_data.append([patch_name, split_type])
    
    df = pd.DataFrame(df_data, columns=['patch_name', 'split'])
    
    # 生成文件名
    filename = f"patchsize_{patch_size}_train_ratio_{train_ratio}.csv"
    full_path = os.path.join(output_path, filename)
    
    # 保存CSV
    df.to_csv(full_path, index=False, header=False)
    print(f"\nCSV文件已保存到: {full_path}")

def main():
    # 设置路径
    label_patch_dir = "/shared/amdgpu/home/avsm2_f4q/code/pangaea-bench/data/austrian_crop_v1.0_pipeline_prithvi/label_patch"
    output_dir = "/shared/amdgpu/home/avsm2_f4q/code/pangaea-bench/data/austrian_crop_v1.0_pipeline_prithvi"
    
    # 设置训练集比例
    train_ratio = 0.01
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)
    
    print("开始处理数据集划分...")
    
    # 读取所有patches和标签
    patch_labels, patch_size = get_all_patches_with_labels(label_patch_dir)
    
    # 划分数据集
    split_dict = split_dataset(patch_labels, train_ratio)
    
    # 保存到CSV
    save_split_csv(split_dict, output_dir, patch_size, train_ratio)
    
    print("\n完成!")

if __name__ == "__main__":
    main()