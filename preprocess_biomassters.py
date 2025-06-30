import os
import glob
import numpy as np
from scipy.ndimage import zoom
from multiprocessing import Pool
from tqdm import tqdm
import argparse

def dequantize_and_resize(rep_path, scale_path):
    """Dequantize and resize representation data"""
    int8_array = np.load(rep_path)
    scale = np.load(scale_path)
    float_array = int8_array.astype(np.float32) * scale[..., np.newaxis]
    zoom_factors = (256 / float_array.shape[0], 256 / float_array.shape[1], 1)
    return zoom(float_array, zoom=zoom_factors, order=1)

def process_single_file(args):
    """Process a single file (for multiprocessing)"""
    rep_file, scale_file, output_file = args
    
    try:
        # Skip if output already exists
        if os.path.exists(output_file):
            return f"Skipped (exists): {output_file}"
        
        # Process the data
        processed_data = dequantize_and_resize(rep_file, scale_file)
        
        # Save as float16 to save space (optional, use float32 if precision is critical)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, processed_data.astype(np.float16))
        
        return f"Processed: {output_file}"
    except Exception as e:
        return f"Error processing {rep_file}: {str(e)}"

def preprocess_dataset(data_root, output_root, num_workers=None):
    """Preprocess all datasets"""
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 64)  # Cap at 64 to avoid overwhelming the system
    
    splits = ['train_agbm', 'val_agbm', 'test_agbm']
    
    for split in splits:
        print(f"\nProcessing {split}...")
        
        # Get all representation files
        rep_dir = os.path.join(data_root, split, 'representation')
        rep_files = sorted(glob.glob(os.path.join(rep_dir, '*.npy')))
        
        # Prepare arguments for multiprocessing
        process_args = []
        for rep_file in rep_files:
            basename = os.path.basename(rep_file)
            scale_file = os.path.join(data_root, split, 'scales', basename)
            output_file = os.path.join(output_root, split, 'processed', basename)
            
            if os.path.exists(scale_file):
                process_args.append((rep_file, scale_file, output_file))
        
        print(f"Found {len(process_args)} files to process")
        
        # Process files in parallel
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, process_args),
                total=len(process_args),
                desc=f"Processing {split}",
                ncols=100
            ))
        
        # Print summary
        processed = sum(1 for r in results if r.startswith("Processed:"))
        skipped = sum(1 for r in results if r.startswith("Skipped"))
        errors = sum(1 for r in results if r.startswith("Error"))
        
        print(f"{split} summary: {processed} processed, {skipped} skipped, {errors} errors")

def main():
    parser = argparse.ArgumentParser(description="Preprocess Borneo dataset")
    parser.add_argument("--data_root", type=str, 
                      default="/shared/amdgpu/home/avsm2_f4q/code/biomassters_data/data",
                      help="Root directory of the original data")
    parser.add_argument("--output_root", type=str,
                      default="/shared/amdgpu/home/avsm2_f4q/code/biomassters_data/preprocessed_data",
                      help="Root directory for preprocessed data")
    parser.add_argument("--num_workers", type=int, default=256,
                      help="Number of parallel workers (default: 64)")
    
    args = parser.parse_args()
    
    print(f"Data root: {args.data_root}")
    print(f"Output root: {args.output_root}")
    print(f"Number of workers: {args.num_workers}")
    
    preprocess_dataset(args.data_root, args.output_root, args.num_workers)
    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()