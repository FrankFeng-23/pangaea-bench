import os
import shutil
from pathlib import Path
import re

def extract_patch_id_from_label(filename):
    """从标签文件名中提取patch_id"""
    # 标签文件格式: {patch_id}_agbm.npy
    match = re.match(r'(.+?)_agbm\.npy$', filename)
    if match:
        return match.group(1)
    return None

def extract_patch_id_from_data(filename):
    """从数据文件名中提取patch_id"""
    # 数据文件格式: {year}_{patch_id}_agbm.npy 或 {year}_{patch_id}_agbm_scales.npy
    match = re.match(r'\d{4}_(.+?)_agbm(?:_scales)?\.npy$', filename)
    if match:
        return match.group(1)
    return None

def find_data_file(patch_id, data_dir, is_scales=False):
    """在数据目录中查找对应patch_id的文件"""
    for filename in os.listdir(data_dir):
        if extract_patch_id_from_data(filename) == patch_id:
            if is_scales and '_scales' in filename:
                return filename
            elif not is_scales and '_scales' not in filename:
                return filename
    return None

def main():
    # 定义路径
    base_path = Path('/shared/amdgpu/home/avsm2_f4q/code/biomassters_data')
    labels_path = base_path / 'labels'
    data_path = base_path / 'data'
    
    # 原始数据路径
    test_repr_path = base_path / 'biomassters_test_representation'
    test_scales_path = base_path / 'biomassters_test_scales'
    train_repr_path = base_path / 'biomassters_train_representation'
    train_scales_path = base_path / 'biomassters_train_scales'
    
    # 创建目标目录结构
    splits = ['test_agbm', 'train_agbm', 'val_agbm']
    subdirs = ['representation', 'scales']
    
    # 创建所有必要的目录
    for split in splits:
        for subdir in subdirs:
            target_dir = data_path / split / subdir
            target_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个split
    for split in splits:
        label_dir = labels_path / split
        if not label_dir.exists():
            print(f"警告: 标签目录 {label_dir} 不存在")
            continue
            
        print(f"\n处理 {split}...")
        
        # 根据split确定源数据目录
        if split == 'test_agbm':
            repr_source = test_repr_path
            scales_source = test_scales_path
        else:  # train_agbm 或 val_agbm
            repr_source = train_repr_path
            scales_source = train_scales_path
        
        # 统计信息
        success_count = 0
        failed_count = 0
        
        # 遍历标签文件
        for label_file in os.listdir(label_dir):
            if not label_file.endswith('.npy'):
                continue
                
            patch_id = extract_patch_id_from_label(label_file)
            if not patch_id:
                print(f"  无法解析标签文件: {label_file}")
                failed_count += 1
                continue
            
            # 查找并复制representation文件
            repr_file = find_data_file(patch_id, repr_source, is_scales=False)
            if repr_file:
                src_repr = repr_source / repr_file
                dst_repr = data_path / split / 'representation' / f'{patch_id}_agbm.npy'
                shutil.copy2(src_repr, dst_repr)
            else:
                print(f"  找不到representation文件: patch_id={patch_id}")
                failed_count += 1
                continue
            
            # 查找并复制scales文件
            scales_file = find_data_file(patch_id, scales_source, is_scales=True)
            if scales_file:
                src_scales = scales_source / scales_file
                # 注意：scales文件也重命名为{patch_id}_agbm.npy
                dst_scales = data_path / split / 'scales' / f'{patch_id}_agbm.npy'
                shutil.copy2(src_scales, dst_scales)
                success_count += 1
            else:
                print(f"  找不到scales文件: patch_id={patch_id}")
                failed_count += 1
        
        print(f"  完成: 成功 {success_count} 个，失败 {failed_count} 个")
    
    # 验证结果
    print("\n验证结果:")
    for split in splits:
        label_count = len(list((labels_path / split).glob('*.npy')))
        repr_count = len(list((data_path / split / 'representation').glob('*.npy')))
        scales_count = len(list((data_path / split / 'scales').glob('*.npy')))
        
        print(f"{split}:")
        print(f"  标签文件数: {label_count}")
        print(f"  representation文件数: {repr_count}")
        print(f"  scales文件数: {scales_count}")
        
        if label_count == repr_count == scales_count:
            print(f"  ✓ 数据完整性验证通过")
        else:
            print(f"  ✗ 数据不完整，请检查")

if __name__ == "__main__":
    main()