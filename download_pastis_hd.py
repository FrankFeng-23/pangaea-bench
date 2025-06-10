from datasets import load_dataset
import os

# 设置要下载的数据集名称和目标文件夹
dataset_name = "IGNF/PASTIS-HD"
data_dir = "data/PASTIS-HD"

# 创建目标文件夹（如果不存在）
os.makedirs(data_dir, exist_ok=True)

# 下载数据集
dataset = load_dataset(dataset_name)

# 保存数据集中的 DATA_SPOT 文件夹
for split in dataset.keys():
    split_data = dataset[split]
    for idx in range(len(split_data)):
        # 假设 DATA_SPOT 中的每个条目都是一个文件
        file_name = f"{data_dir}/DATA_SPOT_{split}_{idx}.json"  # 或者其他格式
        with open(file_name, 'w') as f:
            f.write(str(split_data[idx]))

print(f"Dataset downloaded and saved to {data_dir}")