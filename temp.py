import numpy as np


file_path = "/shared/amdgpu/home/avsm2_f4q/code/biomassters_data/preprocessed_data/test_agbm/processed/0a2d4cec_agbm.npy"
data = np.load(file_path, allow_pickle=True)
print(data.shape) #（H, W, C, T）
print(data[:,:,0])

# import matplotlib.pyplot as plt
# # 随机选择一个时间步
# time_step = np.random.randint(0, data.shape[3])
# print(f"Selected time step: {time_step}")
# single_image = data[:, :, :3, time_step]
# # 转为float
# single_image = single_image.astype(np.float32)
# # 归一化
# for i in range(single_image.shape[2]):
#     single_image[:, :, i] = (single_image[:, :, i] - np.min(single_image[:, :, i])) / (np.max(single_image[:, :, i]) - np.min(single_image[:, :, i]))
# plt.imshow(single_image)
# plt.axis('off')  # 不显示坐标轴
# plt.savefig('rgb.png'.format(time_step), bbox_inches='tight', pad_inches=0)
# plt.close()
# print(data[32, 32, :, time_step])