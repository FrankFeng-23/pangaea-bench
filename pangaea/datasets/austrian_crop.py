import os
import glob
import numpy as np
import torch
from einops import rearrange
from typing import Dict, List, Tuple
from pangaea.datasets.base import RawGeoFMDataset
import csv

class AustrianCrop(RawGeoFMDataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
    ):
        """Initialize the Austrian Crop dataset.

        Args:
            split (str): split of the dataset (train, val, test).
            dataset_name (str): dataset name.
            multi_modal (bool): if the dataset is multi-modal.
            multi_temporal (int): number of temporal frames.
            root_path (str): root path of the dataset.
            classes (list): classes of the dataset.
            num_classes (int): number of classes.
            ignore_index (int): index to ignore for metrics and loss.
            img_size (int): size of the image.
            bands (dict[str, list[str]]): bands of the dataset.
            distribution (list[int]): class distribution.
            data_mean (dict[str, list[str]]): mean for each band for each modality.
            data_std (dict[str, list[str]]): std for each band for each modality.
            data_min (dict[str, list[str]]): min for each band for each modality.
            data_max (dict[str, list[str]]): max for each band for each modality.
            download_url (str): url to download the dataset.
            auto_download (bool): whether to download the dataset automatically.
        """
        super(AustrianCrop, self).__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            root_path=root_path,
            classes=classes,
            num_classes=num_classes,
            ignore_index=ignore_index,
            img_size=img_size,
            bands=bands,
            distribution=distribution,
            data_mean=data_mean,
            data_std=data_std,
            data_min=data_min,
            data_max=data_max,
            download_url=download_url,
            auto_download=auto_download,
        )

        assert split in ["train", "val", "test"], "Split must be train, val or test"
        
        # 获取所有band patch文件
        self.band_patch_dir = os.path.join(self.root_path, "band_patch")
        self.label_patch_dir = os.path.join(self.root_path, "label_patch")
        self.sar_patch_dir = os.path.join(self.root_path, "sar_band_patch")
        
        # 读取CSV文件来获取数据集划分信息
        csv_path = os.path.join(self.root_path, "patchsize_32_train_ratio_0.05.csv")
        
        # 存储各个split的patch文件名
        train_patches = []
        test_patches = []
        
        # 读取CSV文件
        with open(csv_path, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if len(row) >= 2:
                    patch_name = row[0].strip()
                    split_label = row[1].strip()
                    
                    if split_label == 'train':
                        train_patches.append(patch_name)
                    elif split_label == 'test':
                        test_patches.append(patch_name)
        
        # 根据split获取对应的patch文件
        if split == "train":
            selected_patches = train_patches
        elif split == "val":
            # 如果CSV中没有val split，从test中分出一部分作为val
            # 这里我们将test的前50%作为val，后50%作为test
            n_test = len(test_patches)
            selected_patches = test_patches[:n_test//2]
        else:  # test
            # test使用test_patches的后50%
            n_test = len(test_patches)
            selected_patches = test_patches[n_test//2:]
        
        # 构建完整的文件路径列表
        self.patch_files = []
        for patch_name in selected_patches:
            patch_path = os.path.join(self.band_patch_dir, f"{patch_name}.npy")
            if os.path.exists(patch_path):
                self.patch_files.append(patch_path)
            else:
                print(f"Warning: Patch file not found: {patch_path}")
        
        # 对文件列表进行排序以保证一致性
        self.patch_files = sorted(self.patch_files)
        
        self.num_classes = 18
        
        print(f"Loaded {len(self.patch_files)} patches for {split} split")
        
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Get the item at index idx.

        Args:
            idx (int): index of the item.

        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary following the format
            {"image":
                {"optical": torch.Tensor,
                 "sar": torch.Tensor},
            "target": torch.Tensor,
             "metadata": dict}.
        """
        # 获取文件路径
        band_patch_path = self.patch_files[idx]
        patch_name = os.path.basename(band_patch_path)
        label_patch_path = os.path.join(self.label_patch_dir, patch_name)
        
        # 读取光学数据
        optical_data = np.load(band_patch_path).astype(np.float32)  # 形状: (H, W, C, T)
        # 转换维度从 (H, W, C, T) -> (T, C, H, W)
        optical_data = rearrange(optical_data, 'h w c t -> t c h w')  # 形状: (T, C, H, W)
        
        # 读取标签数据
        label_data = np.load(label_patch_path).astype(np.int32)  # 形状: (H, W)
        
        # 转换为torch张量
        optical_ts = torch.from_numpy(optical_data).float()  # 形状: (T, C, H, W)
        label = torch.from_numpy(label_data).long()  # 形状: (H, W)
        
        # SAR数据placeholder（当SAR数据准备好后，取消注释下面的代码）
        # sar_patch_path = os.path.join(self.sar_patch_dir, patch_name)
        # if os.path.exists(sar_patch_path):
        #     sar_data = np.load(sar_patch_path).astype(np.float32)  # 形状: (H, W, C, T)
        #     sar_data = rearrange(sar_data, 'h w c t -> t c h w')  # 形状: (T, C, H, W)
        #     sar_ts = torch.from_numpy(sar_data).float()  # 形状: (T, C, H, W)
        # else:
        #     # 如果SAR数据不存在，创建零张量
        #     T, _, H, W = optical_ts.shape
        #     sar_ts = torch.zeros((T, 3, H, W), dtype=torch.float32)  # 假设SAR有3个通道
        
        # 临时创建SAR数据的零张量placeholder
        T, _, H, W = optical_ts.shape
        sar_ts = torch.zeros((T, 3, H, W), dtype=torch.float32)  # 形状: (T, 3, H, W)
        
        # 处理多时相选择
        # 从 (T, C, H, W) -> (C, T, H, W) 以便选择时间步
        optical_ts = rearrange(optical_ts, 't c h w -> c t h w')  # 形状: (C, T, H, W)
        sar_ts = rearrange(sar_ts, 't c h w -> c t h w')  # 形状: (3, T, H, W)
        
        if self.multi_temporal == 1:
            # 只取最后一帧
            optical_ts = optical_ts[:, -1]  # 形状: (C, H, W)
            sar_ts = sar_ts[:, -1]  # 形状: (3, H, W)
        else:
            # 选择均匀分布的时间步
            T_optical = optical_ts.shape[1]
            T_sar = sar_ts.shape[1]
            
            optical_indexes = torch.linspace(
                0, T_optical - 1, self.multi_temporal, dtype=torch.long
            )
            sar_indexes = torch.linspace(
                0, T_sar - 1, self.multi_temporal, dtype=torch.long
            )
            
            optical_ts = optical_ts[:, optical_indexes]  # 形状: (C, multi_temporal, H, W)
            sar_ts = sar_ts[:, sar_indexes]  # 形状: (3, multi_temporal, H, W)
        
        # 裁剪到指定的img_size（如果需要）
        if self.img_size is not None and (H != self.img_size or W != self.img_size):
            # 中心裁剪
            h_start = (H - self.img_size) // 2
            w_start = (W - self.img_size) // 2
            h_end = h_start + self.img_size
            w_end = w_start + self.img_size
            
            if self.multi_temporal == 1:
                optical_ts = optical_ts[:, h_start:h_end, w_start:w_end]  # 形状: (C, img_size, img_size)
                sar_ts = sar_ts[:, h_start:h_end, w_start:w_end]  # 形状: (3, img_size, img_size)
            else:
                optical_ts = optical_ts[:, :, h_start:h_end, w_start:w_end]  # 形状: (C, T, img_size, img_size)
                sar_ts = sar_ts[:, :, h_start:h_end, w_start:w_end]  # 形状: (3, T, img_size, img_size)
            
            label = label[h_start:h_end, w_start:w_end]  # 形状: (img_size, img_size)
        
        # 返回数据
        return {
            "image": {
                "optical": optical_ts.to(torch.float32),  # 形状: (C, T, H, W) 或 (C, H, W)
                "sar": sar_ts.to(torch.float32),  # 形状: (3, T, H, W) 或 (3, H, W)
            },
            "target": label.to(torch.int64),  # 形状: (H, W)
            "metadata": {
                "patch_name": patch_name,
            },
        }
    
    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: length of the dataset.
        """
        return len(self.patch_files)
    
    @staticmethod
    def download():
        """Download function placeholder."""
        pass