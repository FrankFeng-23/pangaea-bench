import numpy as np


file_path = "/scratch/zf281/pangaea-bench/data/austrian_crop_v1.0_pipeline_prithvi/band_patch/patch_000003.npy"
data = np.load(file_path, allow_pickle=True)
print(data.shape)