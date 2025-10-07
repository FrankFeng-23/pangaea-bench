import numpy as np
import os


filepath = "/scratch/zf281/pangaea-bench/data/austrian_crop_v1.0_pipeline_prithvi/label_patch/patch_000003.npy"
data = np.load(filepath)
print(data.shape)  # 输出数据的形状
print(data.dtype)  # 输出数据的数据类型