import pandas as pd

# File path to the original CSV file
file_path = '/scratch/zf281/pangaea-bench/data/Biomassters/The_BioMassters_-_features_metadata.csv.csv'

# Read the CSV file
df = pd.read_csv(file_path)

# Process for each split type (train and test)
for split in ['train', 'test']:
    # Filter rows for the current split
    split_df = df[df['split'] == split]
    # Get the unique chip_id values
    unique_chip_ids = split_df['chip_id'].drop_duplicates()
    # Save the unique chip_id values to a new CSV file with a header
    unique_chip_ids.to_csv(f"{split}_Data_list.csv", index=False, header=['chip_id'])