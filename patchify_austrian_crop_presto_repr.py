import os
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from scipy.ndimage import zoom

# Ignore potential warnings from libraries
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---

# Input Data Paths
REPR_PATH = "/shared/amdgpu/home/avsm2_f4q/code/pangaea-bench/data/complete_map/austria_Presto_embeddings.npy"
LABEL_PATH = "/shared/amdgpu/home/avsm2_f4q/code/pangaea-bench/data/complete_map/fieldtype_17classes.npy"

# Output Directory
OUTPUT_DIR = "/shared/amdgpu/home/avsm2_f4q/code/pangaea-bench/data/austrian_crop_v1.0_pipeline_presto_repr"
REPR_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "representation")

# Reference directory for final verification
REFERENCE_BAND_PATCH_DIR = "/shared/amdgpu/home/avsm2_f4q/code/pangaea-bench/data/austrian_crop_v1.0_pipeline_prithvi/band_patch"

# Patch settings
PATCH_SIZE = 32
INVALID_LABEL_VALUE = 0  # The value in the label file that indicates a pixel to be ignored

# System settings
NUM_PROCESSES = 256 # Utilize all 96 cores

# Resize method: 'crop' or 'interpolate'
RESIZE_METHOD = 'interpolate'  # Change to 'interpolate' if you prefer interpolation

# --- HELPER FUNCTIONS ---

def resize_representation_to_labels(representation: np.ndarray, target_shape: tuple, method: str = 'crop') -> np.ndarray:
    """
    Resize representation array to match label dimensions.
    
    Args:
        representation: Input representation array of shape (H, W, C)
        target_shape: Target spatial shape (target_H, target_W)
        method: 'crop' to crop from center, 'interpolate' to use interpolation
    
    Returns:
        Resized representation array of shape (target_H, target_W, C)
    """
    current_h, current_w, channels = representation.shape
    target_h, target_w = target_shape
    
    print(f"Resizing representation from ({current_h}, {current_w}) to ({target_h}, {target_w}) using '{method}' method...")
    
    if method == 'crop':
        # Crop from center
        start_h = (current_h - target_h) // 2
        start_w = (current_w - target_w) // 2
        end_h = start_h + target_h
        end_w = start_w + target_w
        
        resized_repr = representation[start_h:end_h, start_w:end_w, :]
        
    elif method == 'interpolate':
        # Use scipy zoom for interpolation
        zoom_factors = (target_h / current_h, target_w / current_w, 1.0)  # Don't zoom the channel dimension
        resized_repr = zoom(representation, zoom_factors, order=1)  # Linear interpolation
        
        # Ensure exact target shape (zoom might have small rounding differences)
        if resized_repr.shape[:2] != (target_h, target_w):
            resized_repr = resized_repr[:target_h, :target_w, :]
    
    else:
        raise ValueError(f"Unknown resize method: {method}. Use 'crop' or 'interpolate'.")
    
    print(f"Resize completed. New shape: {resized_repr.shape}")
    return resized_repr

def compute_patch_locations(label_shape: tuple, patch_size: int) -> list:
    """
    Pre-computes all patch location information based on the full data shape.
    This function is identical to the one in your original script to ensure consistency.
    """
    H, W = label_shape
    n_patches_h = H // patch_size
    n_patches_w = W // patch_size
    
    patches = []
    patch_idx = 0
    
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            patch_idx += 1
            row_start = i * patch_size
            row_end = row_start + patch_size
            col_start = j * patch_size
            col_end = col_start + patch_size
            
            patches.append({
                'idx': patch_idx,
                'row_start': row_start,
                'row_end': row_end,
                'col_start': col_start,
                'col_end': col_end
            })
    
    return patches

def process_and_save_patch(args: tuple):
    """
    Worker function to process a single patch: check its validity based on the
    label, and if valid, save the corresponding representation patch.
    """
    patch_info, label_patch, repr_patch, repr_dir = args
    
    # Check if the corresponding label patch is entirely composed of the invalid value.
    if np.all(label_patch == INVALID_LABEL_VALUE):
        return f"Patch {patch_info['idx']:06d} skipped (all background label)"

    try:
        # Define the output filename, consistent with the original script
        patch_name = f"patch_{patch_info['idx']:06d}.npy"
        
        # Save the representation patch (maintaining its dtype)
        repr_save_path = os.path.join(repr_dir, patch_name)
        np.save(repr_save_path, repr_patch)
        
        return f"Patch {patch_info['idx']:06d} completed"
    except Exception as e:
        return f"Patch {patch_info['idx']:06d} error: {str(e)}"

# --- MAIN EXECUTION ---

def patchify_representation():
    """
    Main function to orchestrate the patchification process.
    """
    start_time = time.time()
    
    print("🚀 Starting patchification process for Presto representations...")

    # 1. Setup Output Directories
    os.makedirs(REPR_OUTPUT_DIR, exist_ok=True)
    print(f"Output directory created at: {OUTPUT_DIR}")

    # 2. Load Data using Memory-Mapping for efficiency
    print("Loading source data (using memory-mapping)...")
    try:
        labels = np.load(LABEL_PATH)
        # Use mmap_mode='r' to avoid loading the entire large file into RAM at once
        representation = np.load(REPR_PATH, mmap_mode='r')
    except FileNotFoundError as e:
        print(f"❌ Error: Input file not found. {e}")
        return

    print(f"  - Label shape: {labels.shape}")
    print(f"  - Representation shape: {representation.shape}, dtype: {representation.dtype}")

    # 3. Check and resize representation if needed
    if labels.shape[:2] != representation.shape[:2]:
        print(f"⚠️ Dimension mismatch detected!")
        print(f"  - Labels: {labels.shape[:2]}")
        print(f"  - Representation: {representation.shape[:2]}")
        
        # Load the full representation into memory for resizing
        print("Loading full representation into memory for resizing...")
        representation_full = np.array(representation)  # Convert from memmap to regular array
        
        # Resize representation to match labels
        representation = resize_representation_to_labels(
            representation_full, 
            labels.shape[:2], 
            method=RESIZE_METHOD
        )
        
        print(f"✅ Representation resized to: {representation.shape}")
        
        # Clear the full representation from memory to save RAM
        del representation_full
    
    # Final dimension check
    assert labels.shape[:2] == representation.shape[:2], "Dimension mismatch between label and representation arrays after resize!"

    # 4. Compute Patch Grid
    patches = compute_patch_locations(labels.shape, PATCH_SIZE)
    total_patches = len(patches)
    print(f"\nGrid computed. Total potential patches: {total_patches}")
    print(f"Using {NUM_PROCESSES} processes for parallel execution.")

    # 5. Process Patches in Parallel
    print("\nProcessing patches...")
    
    completed_count = 0
    skipped_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        futures = []
        for p_info in patches:
            # Efficiently slice the required data from the arrays
            r_start, r_end = p_info['row_start'], p_info['row_end']
            c_start, c_end = p_info['col_start'], p_info['col_end']
            
            label_patch = labels[r_start:r_end, c_start:c_end]
            repr_patch = representation[r_start:r_end, c_start:c_end, :]
            
            # Submit the task to the pool
            args = (p_info, label_patch, repr_patch, REPR_OUTPUT_DIR)
            futures.append(executor.submit(process_and_save_patch, args))
            
        # Process results as they complete with a progress bar
        for future in tqdm(as_completed(futures), total=total_patches, desc="Creating Patches"):
            result = future.result()
            if "completed" in result:
                completed_count += 1
            elif "skipped" in result:
                skipped_count += 1
            else:
                error_count += 1
                tqdm.write(result) # Print errors to console

    # 6. Final Report and Verification
    elapsed_time = time.time() - start_time
    print("\n--- Processing Summary ---")
    print(f"✅ Successfully created: {completed_count} patches")
    print(f"⏭️ Skipped (all background): {skipped_count} patches")
    print(f"❌ Errors encountered: {error_count} patches")
    print(f"⏱️ Total execution time: {elapsed_time:.2f} seconds.")
    
    # 7. Verification Step
    print("\n--- Verification ---")
    try:
        new_repr_patches = os.listdir(REPR_OUTPUT_DIR)
        ref_band_patches = os.listdir(REFERENCE_BAND_PATCH_DIR)
        
        print(f"Number of patches in new 'representation' dir: {len(new_repr_patches)}")
        print(f"Number of patches in reference 'band_patch' dir: {len(ref_band_patches)}")
        
        if len(new_repr_patches) == len(ref_band_patches):
            print("\n🎉 SUCCESS: The number of generated representation patches matches the reference band_patch count.")
        else:
            print("\n⚠️ WARNING: Mismatch in patch count between generated representation and reference band_patch.")
            
    except FileNotFoundError:
        print(f"\nCould not perform verification. Reference directory not found at: {REFERENCE_BAND_PATCH_DIR}")
    except Exception as e:
        print(f"\nAn error occurred during verification: {e}")

if __name__ == "__main__":
    patchify_representation()