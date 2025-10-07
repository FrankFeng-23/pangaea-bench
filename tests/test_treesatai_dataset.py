#!/usr/bin/env python3
"""
测试TreeSatAI数据集实现 - 快速版本
"""
import sys
import os
sys.path.append('/scratch/zf281/pangaea-bench')

import torch
import numpy as np
from pangaea.datasets.treesatai import TreeSatAI

def test_dataset_loading_fast():
    """快速测试数据集加载 - 只加载少量样本"""
    print("=== 快速测试数据集加载 ===")
    
    try:
        # 创建一个修改版的TreeSatAI类，只加载前100个样本进行测试
        class FastTreeSatAI(TreeSatAI):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # 限制样本数量以加快测试
                max_samples = 100
                if len(self.split_indices) > max_samples:
                    print(f"限制测试样本数量为 {max_samples} (原始: {len(self.split_indices)})")
                    self.split_indices = self.split_indices[:max_samples]
        
        # 测试训练集
        print("加载训练集...")
        train_dataset = FastTreeSatAI(split="train", multi_temporal=6)
        print(f"训练集大小: {len(train_dataset)}")
        
        # 测试验证集
        print("加载验证集...")
        val_dataset = FastTreeSatAI(split="val", multi_temporal=6)
        print(f"验证集大小: {len(val_dataset)}")
        
        # 测试测试集
        print("加载测试集...")
        test_dataset = FastTreeSatAI(split="test", multi_temporal=6)
        print(f"测试集大小: {len(test_dataset)}")
        
        total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
        print(f"总样本数: {total_samples}")
        
        return train_dataset, val_dataset, test_dataset
        
    except Exception as e:
        print(f"数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_sample_loading_fast(dataset):
    """快速测试样本加载 - 只测试前几个样本"""
    print("\n=== 快速测试样本加载 ===")
    
    if dataset is None:
        print("数据集为空，跳过测试")
        return
    
    try:
        # 只加载前3个样本进行测试
        num_samples_to_test = min(3, len(dataset))
        print(f"测试前 {num_samples_to_test} 个样本...")
        
        for i in range(num_samples_to_test):
            print(f"\n--- 样本 {i+1} ---")
            sample = dataset[i]
            
            print(f"样本键: {list(sample.keys())}")
            
            # 检查光学数据
            optical = sample["optical"]
            print(f"光学数据形状: {optical.shape}")
            print(f"光学数据类型: {optical.dtype}")
            print(f"光学数据范围: [{optical.min():.4f}, {optical.max():.4f}]")
            
            # 检查SAR数据
            sar = sample["sar"]
            print(f"SAR数据形状: {sar.shape}")
            print(f"SAR数据类型: {sar.dtype}")
            print(f"SAR数据范围: [{sar.min():.4f}, {sar.max():.4f}]")
            
            # 检查标签
            label = sample["label"]
            print(f"标签形状: {label.shape}")
            print(f"标签类型: {label.dtype}")
            print(f"标签值: {label}")
            print(f"激活的类别数: {(label > 0).sum().item()}")
            
            # 检查文件名
            filename = sample["filename"]
            print(f"文件名: {filename}")
        
        return True
        
    except Exception as e:
        print(f"样本加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_loading_fast(dataset):
    """快速测试批量加载 - 小批次"""
    print("\n=== 快速测试批量加载 ===")
    
    if dataset is None:
        print("数据集为空，跳过测试")
        return
    
    try:
        from torch.utils.data import DataLoader
        
        # 创建数据加载器 - 使用小批次
        batch_size = min(2, len(dataset))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # 加载一个批次
        print(f"加载批次大小为 {batch_size} 的数据...")
        batch = next(iter(dataloader))
        
        print(f"批次键: {list(batch.keys())}")
        
        # 检查批次形状
        optical_batch = batch["optical"]
        sar_batch = batch["sar"]
        label_batch = batch["label"]
        
        print(f"光学批次形状: {optical_batch.shape}")
        print(f"SAR批次形状: {sar_batch.shape}")
        print(f"标签批次形状: {label_batch.shape}")
        
        # 检查数据类型
        print(f"光学数据类型: {optical_batch.dtype}")
        print(f"SAR数据类型: {sar_batch.dtype}")
        print(f"标签数据类型: {label_batch.dtype}")
        
        return True
        
    except Exception as e:
        print(f"批量加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_class_distribution_fast(dataset):
    """快速测试类别分布 - 只采样少量数据"""
    print("\n=== 快速测试类别分布 ===")
    
    if dataset is None:
        print("数据集为空，跳过测试")
        return
    
    try:
        # 统计类别分布 - 只采样10个样本
        class_counts = np.zeros(dataset.num_classes)
        
        sample_size = min(10, len(dataset))
        print(f"采样 {sample_size} 个样本进行类别分布统计...")
        
        for idx in range(sample_size):
            sample = dataset[idx]
            label = sample["label"].numpy()
            class_counts += (label > 0).astype(int)
        
        print(f"类别分布 (采样 {sample_size} 个样本):")
        for i, (cls, count) in enumerate(zip(dataset.classes, class_counts)):
            print(f"  {cls}: {int(count)}")
        
        return True
        
    except Exception as e:
        print(f"类别分布统计失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_temporal_settings_fast():
    """快速测试不同的多时相设置"""
    print("\n=== 快速测试多时相设置 ===")
    
    try:
        # 创建快速测试类
        class FastTreeSatAI(TreeSatAI):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # 只保留前5个样本
                self.split_indices = self.split_indices[:5]
        
        for multi_temporal in [1, 6, 12]:  # 减少测试的时相数量
            print(f"\n测试 multi_temporal={multi_temporal}...")
            dataset = FastTreeSatAI(split="train", multi_temporal=multi_temporal)
            sample = dataset[0]
            
            optical = sample["optical"]
            sar = sample["sar"]
            
            print(f"  光学数据形状: {optical.shape}")
            print(f"  SAR数据形状: {sar.shape}")
            
            # 检查时间维度
            assert optical.shape[1] == multi_temporal, f"光学数据时间维度不匹配: {optical.shape[1]} != {multi_temporal}"
            assert sar.shape[1] == multi_temporal, f"SAR数据时间维度不匹配: {sar.shape[1]} != {multi_temporal}"
        
        print("多时相设置测试通过")
        return True
        
    except Exception as e:
        print(f"多时相设置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("开始快速测试TreeSatAI数据集实现...")
    
    # 测试数据集加载
    train_dataset, val_dataset, test_dataset = test_dataset_loading_fast()
    
    if train_dataset is not None:
        # 测试样本加载
        test_sample_loading_fast(train_dataset)
        
        # 测试批量加载
        test_batch_loading_fast(train_dataset)
        
        # 测试类别分布
        test_class_distribution_fast(train_dataset)
        
        # 测试多时相设置
        test_multi_temporal_settings_fast()
        
        print("\n=== 快速测试总结 ===")
        print("TreeSatAI数据集实现快速测试完成！")
        print(f"训练集: {len(train_dataset)} 样本")
        print(f"验证集: {len(val_dataset)} 样本")
        print(f"测试集: {len(test_dataset)} 样本")
        print(f"类别数: {train_dataset.num_classes}")
        print(f"图像大小: {train_dataset.img_size}x{train_dataset.img_size}")
        print(f"多模态: {train_dataset.multi_modal}")
        print("注意: 这是快速测试版本，使用了有限的样本数量")
        
    else:
        print("数据集加载失败，请检查数据路径和实现")

if __name__ == "__main__":
    main()