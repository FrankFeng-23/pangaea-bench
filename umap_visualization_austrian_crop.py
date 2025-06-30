#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AMD GPU优化版本 - 使用PyTorch进行GPU加速

import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

# 标准CPU库
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from umap import UMAP
from scipy import ndimage
import torch.nn.functional as F

# 设置线程数
os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["OMP_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"

# 检查GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
else:
    device = torch.device('cpu')
    print("GPU not available, using CPU")

# matplotlib设置
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

def resize_feature_data_torch(data, target_shape):
    """使用PyTorch在GPU上进行resize"""
    H, W, C = data.shape
    target_H, target_W = target_shape
    
    if (H, W) == (target_H, target_W):
        return data
    
    print(f"  Resizing feature data from ({H}, {W}) to ({target_H}, {target_W})...")
    
    # 转换为PyTorch张量并移至GPU
    data_tensor = torch.from_numpy(data).float().to(device)
    
    # 调整形状为 (1, C, H, W) 以使用PyTorch的插值
    data_tensor = data_tensor.permute(2, 0, 1).unsqueeze(0)
    
    # 使用双线性插值
    resized_tensor = F.interpolate(
        data_tensor, 
        size=(target_H, target_W), 
        mode='bilinear', 
        align_corners=False
    )
    
    # 调整回原始形状 (H, W, C) 并转回CPU
    resized_data = resized_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # 清理GPU内存
    del data_tensor, resized_tensor
    torch.cuda.empty_cache()
    
    return resized_data

def gpu_accelerated_pca(data, n_components=50):
    """使用PyTorch在GPU上加速PCA计算"""
    print(f"Applying GPU-accelerated PCA (n_components={n_components})...")
    t0 = time.time()
    
    # 转换为PyTorch张量
    data_tensor = torch.from_numpy(data).float().to(device)
    
    # 中心化数据
    mean = data_tensor.mean(dim=0)
    data_centered = data_tensor - mean
    
    # 计算协方差矩阵
    cov_matrix = torch.mm(data_centered.t(), data_centered) / (data_tensor.shape[0] - 1)
    
    # 特征值分解
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    
    # 按特征值降序排序
    idx = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 选择前n_components个主成分
    components = eigenvectors[:, :n_components]
    
    # 变换数据
    data_pca = torch.mm(data_centered, components)
    
    # 计算解释方差比
    explained_variance_ratio = eigenvalues[:n_components] / eigenvalues.sum()
    ev = explained_variance_ratio.sum().item()
    
    # 转回NumPy
    data_pca_np = data_pca.cpu().numpy()
    
    # 清理GPU内存
    del data_tensor, data_centered, cov_matrix, eigenvalues, eigenvectors, components, data_pca
    torch.cuda.empty_cache()
    
    print(f"  GPU PCA done in {time.time()-t0:.2f}s; explained variance={ev:.4f}")
    return data_pca_np, ev

def batch_process_umap(data, labels, batch_size=100000, n_neighbors=30, min_dist=0.1):
    """分批处理UMAP以处理大规模数据"""
    print(f"Running UMAP in batches (batch_size={batch_size})...")
    t0 = time.time()
    
    n_samples = data.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    all_embeddings = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch_data = data[start_idx:end_idx]
        
        print(f"  Processing batch {i+1}/{n_batches} ({end_idx-start_idx} samples)...")
        
        # 使用多核CPU进行UMAP
        reducer = UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, batch_data.shape[0] - 1),
            min_dist=min_dist,
            n_epochs=500,
            metric='euclidean',
            n_jobs=256,
            random_state=42,
            verbose=False,
            low_memory=False
        )
        
        batch_embedding = reducer.fit_transform(batch_data)
        all_embeddings.append(batch_embedding)
    
    # 合并所有批次的结果
    full_embedding = np.vstack(all_embeddings)
    
    print(f"  Batch UMAP done in {time.time()-t0:.2f}s")
    return full_embedding

def parallel_stratified_sampling(data, labels, max_samples_per_class=50000, total_max_samples=500000):
    """使用PyTorch进行并行化的分层采样"""
    print("Performing parallel stratified sampling...")
    t0 = time.time()
    
    # 转换为PyTorch张量以利用GPU
    labels_tensor = torch.from_numpy(labels).to(device)
    
    unique_labels = torch.unique(labels_tensor[labels_tensor > 0])
    unique_labels_np = unique_labels.cpu().numpy()
    
    # 计算每个类别的样本数
    class_counts = {}
    for label in unique_labels_np:
        mask = (labels_tensor == label)
        class_counts[label] = mask.sum().item()
    
    # 动态分配采样数量
    total_available = sum(class_counts.values())
    if total_available > total_max_samples:
        scale_factor = total_max_samples / total_available
        samples_per_class = {
            label: min(max(int(count * scale_factor), 1000), count)
            for label, count in class_counts.items()
        }
    else:
        samples_per_class = {
            label: min(count, max_samples_per_class)
            for label, count in class_counts.items()
        }
    
    # 执行采样
    sampled_indices = []
    sampling_info = {}
    
    for label in unique_labels_np:
        # 使用GPU找到标签索引
        label_mask = (labels_tensor == label)
        label_indices = torch.where(label_mask)[0].cpu().numpy()
        
        n_available = len(label_indices)
        n_samples = samples_per_class[label]
        
        if n_samples < n_available:
            sampled_label_indices = np.random.choice(label_indices, size=n_samples, replace=False)
        else:
            sampled_label_indices = label_indices
        
        sampled_indices.extend(sampled_label_indices)
        sampling_info[int(label)] = {'original': n_available, 'sampled': n_samples}
        
        print(f"  Class {label:2d}: {n_available:8,} → {n_samples:6,} samples")
    
    # 清理GPU内存
    del labels_tensor
    torch.cuda.empty_cache()
    
    sampled_indices = np.array(sampled_indices)
    np.random.shuffle(sampled_indices)
    
    sampled_data = data[sampled_indices]
    sampled_labels = labels[sampled_indices]
    
    total_sampled = len(sampled_indices)
    print(f"  Parallel sampling completed in {time.time()-t0:.2f}s")
    print(f"  Total: {total_available:,} → {total_sampled:,} samples")
    
    return sampled_data, sampled_labels, sampling_info

# 主函数和其他辅助函数保持基本相同，但使用PyTorch优化的版本
def main():
    # 配置
    # input_file = '/shared/amdgpu/home/avsm2_f4q/code/pangaea-bench/data/complete_map/asutria_Tessera_representation.npy'
    input_file = '/shared/amdgpu/home/avsm2_f4q/code/pangaea-bench/data/complete_map/austria_Presto_embeddings.npy'
    label_file = '/shared/amdgpu/home/avsm2_f4q/code/pangaea-bench/data/complete_map/fieldtype_17classes.npy'
    output_dir = '/shared/amdgpu/home/avsm2_f4q/code/pangaea-bench/'
    
    # 参数设置
    MAX_SAMPLES_PER_CLASS = 50000
    TOTAL_MAX_SAMPLES = 500000
    USE_GPU_PCA = True  # 是否使用GPU加速的PCA
    BATCH_SIZE_UMAP = 100000  # UMAP批处理大小
    
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()
    
    print("="*60)
    print("UMAP VISUALIZATION - AMD GPU OPTIMIZED VERSION")
    print("="*60)
    print(f"Device: {device}")
    print(f"CPU Cores: {os.cpu_count()}")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*60)

    # 1. 加载标签
    print("\n[Step 1] Loading labels...")
    labels = np.load(label_file)
    label_shape = labels.shape[:2]
    labels_flat = labels.reshape(-1)
    print(f"  Label shape: {labels.shape}")
    
    # 2. 加载特征数据
    print("\n[Step 2] Loading features...")
    t0 = time.time()
    data = np.load(input_file, mmap_mode='r' if os.path.getsize(input_file) > 1e9 else None)
    print(f"  Feature shape: {data.shape} loaded in {time.time()-t0:.2f}s")
    
    # 检查是否需要resize
    if data.shape[:2] != label_shape:
        print("  Resizing features to match label dimensions...")
        if isinstance(data, np.memmap):
            data = np.array(data)  # 加载到内存
        data = resize_feature_data_torch(data, label_shape)
    
    # 转换为float32并展平
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    H, W, C = data.shape
    data_flat = data.reshape(-1, C)
    data_flat = np.nan_to_num(data_flat, nan=0.0)
    
    # 3. 过滤背景
    print("\n[Step 3] Filtering background...")
    mask = labels_flat > 0
    data_sel = data_flat[mask]
    labels_sel = labels_flat[mask]
    print(f"  Filtered: {data_sel.shape[0]:,} samples remain")
    
    # 清理内存
    del data_flat, labels_flat, mask
    
    # 4. 分层采样
    print("\n[Step 4] Stratified sampling...")
    data_sampled, labels_sampled, sampling_info = parallel_stratified_sampling(
        data_sel, labels_sel,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS,
        total_max_samples=TOTAL_MAX_SAMPLES
    )
    
    del data_sel, labels_sel
    
    # 5. PCA降维
    print("\n[Step 5] Dimensionality reduction...")
    if data_sampled.shape[1] > 50:
        if USE_GPU_PCA and torch.cuda.is_available():
            data_for_umap, ev = gpu_accelerated_pca(data_sampled, n_components=50)
        else:
            pca = PCA(n_components=50, random_state=42)
            data_for_umap = pca.fit_transform(data_sampled)
            ev = pca.explained_variance_ratio_.sum()
            print(f"  CPU PCA: explained variance={ev:.4f}")
    else:
        data_for_umap = data_sampled
    
    # 6. UMAP（批处理版本）
    print("\n[Step 6] UMAP embedding...")
    if data_for_umap.shape[0] > BATCH_SIZE_UMAP:
        umap_Y = batch_process_umap(data_for_umap, labels_sampled, batch_size=BATCH_SIZE_UMAP)
    else:
        reducer = UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            n_epochs=500,
            metric='euclidean',
            n_jobs=64,
            random_state=42,
            verbose=True
        )
        umap_Y = reducer.fit_transform(data_for_umap)
    
    # 7. 计算指标
    print("\n[Step 7] Computing metrics...")
    metrics = compute_clustering_metrics(umap_Y, labels_sampled)
    
    # 8. 可视化
    print("\n[Step 8] Creating visualization...")
    png_prefix = input_file.split('/')[-1].replace('.npy', '')
    out_png = os.path.join(output_dir, f"{png_prefix}_umap.png")
    create_nature_visualization(
        umap_Y,
        labels_sampled,
        out_png,
        sampling_info=sampling_info,
        metrics=metrics
    )
    
    # 最终统计
    total_time = time.time() - t_start
    print("\n" + "="*60)
    print("FINAL SUMMARY:")
    print("="*60)
    print(f"Processing completed in {total_time:.2f} seconds")
    print(f"Output saved to: {out_png}")
    print("="*60)

# 复用原有的辅助函数
def compute_clustering_metrics(umap_Y, labels, sample_size=50000):
    """计算聚类指标"""
    print("  Computing clustering metrics...")
    t0 = time.time()
    
    n_samples = umap_Y.shape[0]
    if n_samples > sample_size:
        indices = np.random.choice(n_samples, size=sample_size, replace=False)
        Y_sample = umap_Y[indices]
        labels_sample = labels[indices]
    else:
        Y_sample = umap_Y
        labels_sample = labels
    
    try:
        silhouette = silhouette_score(Y_sample, labels_sample, metric='euclidean')
        db_index = davies_bouldin_score(Y_sample, labels_sample)
        print(f"    Silhouette Score: {silhouette:.4f}")
        print(f"    Davies-Bouldin Index: {db_index:.4f}")
    except Exception as e:
        print(f"    Warning: Failed to compute metrics: {e}")
        silhouette = np.nan
        db_index = np.nan
    
    print(f"  Metrics computed in {time.time()-t0:.2f}s")
    return {'silhouette': silhouette, 'davies_bouldin': db_index}

def create_nature_visualization(umap_Y, labels, output_path, sampling_info=None, metrics=None):
    """创建Nature风格的可视化"""
    print("  Creating UMAP plot...")
    
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
    
    n_pts = umap_Y.shape[0]
    if n_pts > 200000:
        sz, alpha = 0.1, 0.4
    elif n_pts > 100000:
        sz, alpha = 0.2, 0.5
    elif n_pts > 50000:
        sz, alpha = 0.3, 0.6
    else:
        sz, alpha = 0.5, 0.7
    
    # 颜色映射
    colors = cm.get_cmap('tab20')(np.linspace(0, 1, 20))
    selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18]
    custom_colors = colors[selected_indices]
    custom_cmap = ListedColormap(custom_colors)
    
    scatter = ax.scatter(
        umap_Y[:, 0], umap_Y[:, 1],
        c=labels - 1,
        cmap=custom_cmap,
        s=sz,
        alpha=alpha,
        rasterized=True,
        edgecolors='none'
    )
    
    ax.set_xlabel('UMAP 1', fontsize=8)
    ax.set_ylabel('UMAP 2', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=7)
    
    # 添加指标
    if metrics is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        text_x = xlim[1] - 0.05 * (xlim[1] - xlim[0])
        text_y = ylim[1] - 0.05 * (ylim[1] - ylim[0])
        
        metric_text = []
        if not np.isnan(metrics['silhouette']):
            metric_text.append(f"Silhouette: {metrics['silhouette']:.3f}")
        if not np.isnan(metrics['davies_bouldin']):
            metric_text.append(f"Davies-Bouldin: {metrics['davies_bouldin']:.3f}")
        
        if metric_text:
            bbox_props = dict(boxstyle="round,pad=0.3", 
                            facecolor='white', 
                            edgecolor='gray',
                            alpha=0.8,
                            linewidth=0.5)
            ax.text(text_x, text_y, '\n'.join(metric_text),
                   transform=ax.transData,
                   fontsize=7,
                   ha='right',
                   va='top',
                   bbox=bbox_props,
                   zorder=100)
    
    # 颜色条
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.arange(17))
    cbar.set_ticklabels(np.arange(1, 18))
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Class', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight',
                format='png', facecolor='white', edgecolor='none')
    
    # 保存信息
    if sampling_info:
        info_path = output_path.replace('.png', '_info.txt')
        with open(info_path, 'w') as f:
            f.write("Sampling Information:\n")
            f.write("="*50 + "\n")
            for label, info in sampling_info.items():
                f.write(f"Class {label:2d}: {info['original']:8,} → {info['sampled']:6,} samples\n")
            if metrics:
                f.write("\nMetrics:\n")
                f.write(f"Silhouette Score: {metrics['silhouette']:.4f}\n")
                f.write(f"Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}\n")

if __name__ == "__main__":
    main()