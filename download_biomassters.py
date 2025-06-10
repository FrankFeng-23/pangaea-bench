#!/usr/bin/env python
import os
from huggingface_hub import snapshot_download

def download_dataset(repo_id, output_dir, revision="main"):
    """
    下载指定 Hugging Face 数据集到 output_dir 目录。
    
    参数:
      repo_id: 数据集的仓库名称（例如 "nascetti-a/BioMassters"）
      output_dir: 下载到的本地目录路径
      revision: 分支或标签（默认为 "main"）
    """
    # 如果目标目录不存在，则创建目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 下载数据集快照到指定目录
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=output_dir,
        repo_type="dataset",
    )
    print(f"数据集已下载到：{output_dir}")

if __name__ == "__main__":
    # 数据集仓库名称
    repo_id = "nascetti-a/BioMassters"
    # 指定你的下载目录
    output_dir = "data/Biomassters"  # 请将此路径替换为你自己的目标目录
    download_dataset(repo_id, output_dir)