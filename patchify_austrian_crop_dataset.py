import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import gc
from typing import List, Tuple

warnings.filterwarnings('ignore')

# Configuration
DATA_RAW_DIR = "/shared/amdgpu/home/avsm2_f4q/code/pangaea-bench/data/austrian_crop_v1.0_raw_data/data_raw"
LABEL_PATH = "/shared/amdgpu/home/avsm2_f4q/code/pangaea-bench/data/austrian_crop_v1.0_raw_data/fieldtype_17classes.npy"
OUTPUT_DIR = "/shared/amdgpu/home/avsm2_f4q/code/pangaea-bench/data/austrian_crop_v1.0_pipeline_prithvi"

ALL_BANDS = ["red", "blue", "green", "nir", "nir08", "rededge1", "rededge2", "rededge3", "swir16", "swir22"]
NUM_BANDS = len(ALL_BANDS)

PATCH_SIZE = 32
MIN_VALID_PIXELS_RATIO = 0.9  # 90% valid pixels required
MIN_VALID_PIXELS_COUNT = int(PATCH_SIZE * PATCH_SIZE * MIN_VALID_PIXELS_RATIO)

# SCL classification values for invalid pixels
INVALID_SCL_VALUES = {0, 1, 2, 3, 8, 9}

# Number of processes to use, leveraging the powerful server
NUM_PROCESSES = os.cpu_count()

def get_reference_info(data_dir: Path) -> Tuple[any, tuple, any, tuple]:
    """Get reference CRS, bounds, transform, and shape from a 10m resolution band."""
    blue_dir = data_dir / "blue"
    first_file = sorted([f for f in blue_dir.iterdir() if f.suffix == '.tiff'])[0]
    with rasterio.open(first_file) as src:
        return src.crs, src.bounds, src.transform, src.shape

def get_all_dates(data_dir: Path) -> List[Tuple[str, int, int, int, int]]:
    """Get all available dates from the blue band directory."""
    blue_dir = data_dir / "blue"
    dates = []
    for f in sorted(blue_dir.glob('*.tiff')):
        date_str = f.name.split('_')[0]
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        dates.append((date_str, date_obj.year, date_obj.month, date_obj.day, date_obj.timetuple().tm_yday))
    return dates

def _load_reproject_worker(args: Tuple[int, Path, tuple, any, any, int]) -> Tuple[int, np.ndarray]:
    """Worker function to load and reproject a single GeoTIFF."""
    idx, file_path, ref_shape, ref_transform, ref_crs, resampling_method = args
    try:
        with rasterio.open(file_path) as src:
            resampled_data = np.zeros(ref_shape, dtype=src.profile['dtype'])
            reproject(
                source=rasterio.band(src, 1),
                destination=resampled_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=resampling_method
            )
        return idx, resampled_data
    except Exception as e:
        print(f"Error processing file {file_path.name}: {e}")
        return idx, np.zeros(ref_shape, dtype=np.uint8) # Return empty array on error


def load_and_reproject_data_parallel(
    files_to_process: List[Tuple[int, Path]],
    ref_shape: tuple,
    ref_transform: any,
    ref_crs: any,
    resampling_method: Resampling,
    desc: str
) -> np.ndarray:
    """Loads and reprojects a list of GeoTIFF files in parallel."""
    
    num_files = len(files_to_process)
    # Determine dtype from first file to pre-allocate array
    with rasterio.open(files_to_process[0][1]) as src:
        dtype = src.profile['dtype']

    # Pre-allocate memory for all results
    results_array = np.zeros((num_files, *ref_shape), dtype=dtype)
    
    args_list = [
        (idx, path, ref_shape, ref_transform, ref_crs, resampling_method)
        for idx, path in files_to_process
    ]

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        futures = [executor.submit(_load_reproject_worker, arg) for arg in args_list]
        
        for future in tqdm(as_completed(futures), total=num_files, desc=desc):
            idx, data = future.result()
            results_array[idx] = data

    return results_array

def save_patch_worker(args):
    """Worker function to extract data from preloaded arrays and save a single patch."""
    patch_idx, patch_info, valid_timesteps_indices, output_dirs = args
    
    band_patch_dir, label_patch_dir, geo_patch_dir, time_patch_dir = output_dirs
    
    row_start, row_end = patch_info['row_start'], patch_info['row_end']
    col_start, col_end = patch_info['col_start'], patch_info['col_end']

    try:
        # Extract band data using the indices of valid timesteps for this patch
        # Slicing is extremely fast on numpy arrays in memory
        band_patch_data = ALL_BANDS_DATA[valid_timesteps_indices, :, row_start:row_end, col_start:col_end]
        # Transpose to (H, W, C, T) -> (64, 64, 10, num_valid_timesteps)
        band_patch_data = np.transpose(band_patch_data, (2, 3, 1, 0))

        # Extract label patch
        label_patch = LABEL_DATA[row_start:row_end, col_start:col_end]
        
        # Extract geo and time data
        geo_patch = GEO_DATA[row_start:row_end, col_start:col_end, :]
        time_patch = TIME_DATA[valid_timesteps_indices, :]

        # Save all data
        patch_name = f"patch_{(patch_idx + 1):06d}"
        np.save(band_patch_dir / f"{patch_name}.npy", band_patch_data.astype(np.float32))
        np.save(label_patch_dir / f"{patch_name}.npy", label_patch)
        np.save(geo_patch_dir / f"{patch_name}.npy", geo_patch)
        np.save(time_patch_dir / f"{patch_name}.npy", time_patch.astype(np.float32))
        
        return patch_idx, len(valid_timesteps_indices), None
    except Exception as e:
        return patch_idx, 0, str(e)

def create_patches_optimized():
    start_time = time.time()
    
    # Use Pathlib for robust path management
    data_dir = Path(DATA_RAW_DIR)
    output_dir = Path(OUTPUT_DIR)

    # Create output directories
    band_patch_dir = output_dir / "band_patch"
    label_patch_dir = output_dir / "label_patch"
    geo_patch_dir = output_dir / "geo_patch"
    time_patch_dir = output_dir / "time_patch"
    output_dirs = (band_patch_dir, label_patch_dir, geo_patch_dir, time_patch_dir)
    for dir_path in output_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 1. Get Reference Info and Date List
    print("Initializing...")
    ref_crs, ref_bounds, ref_transform, ref_shape = get_reference_info(data_dir)
    all_dates = get_all_dates(data_dir)
    H, W = ref_shape
    print(f"Reference shape: {H}x{W}. Found {len(all_dates)} dates.")
    print(f"Using {NUM_PROCESSES} processes.")

    # 2. Load and Reproject all SCL data into a single numpy array
    scl_files = [(i, data_dir / "scl" / f"{date_info[0]}_mosaic.tiff") for i, date_info in enumerate(all_dates)]
    scl_data_stack = load_and_reproject_data_parallel(
        scl_files, ref_shape, ref_transform, ref_crs, Resampling.nearest, "Loading SCL data"
    )

    # 3. Vectorized Validity Calculation
    print("Calculating pixel validity for all timesteps...")
    # Create a boolean mask where True means the pixel is valid
    valid_mask = np.ones_like(scl_data_stack, dtype=bool)
    for val in INVALID_SCL_VALUES:
        valid_mask &= (scl_data_stack != val)
    del scl_data_stack; gc.collect() # Free memory

    # Reshape for vectorized patch processing
    n_patches_h = H // PATCH_SIZE
    n_patches_w = W // PATCH_SIZE
    total_patches = n_patches_h * n_patches_w
    
    # Crop the valid_mask to be divisible by PATCH_SIZE
    valid_mask_cropped = valid_mask[:, :n_patches_h * PATCH_SIZE, :n_patches_w * PATCH_SIZE]
    
    # Reshape and sum to get valid pixel counts for every patch at every timestep
    # Shape becomes (num_dates, num_patches, patch_area)
    reshaped = valid_mask_cropped.reshape(
        len(all_dates), n_patches_h, PATCH_SIZE, n_patches_w, PATCH_SIZE
    ).transpose(0, 1, 3, 2, 4).reshape(len(all_dates), total_patches, -1)
    
    # Result is a (total_patches, num_dates) array
    patch_valid_pixel_counts = np.sum(reshaped, axis=2).T
    del valid_mask, valid_mask_cropped, reshaped; gc.collect()

    # 4. Load Global Data into memory (to be shared with worker processes)
    print("Loading label data...")
    global LABEL_DATA
    LABEL_DATA = np.load(LABEL_PATH)

    # 5. Pre-calculate Geo-coordinate data
    print("Pre-calculating geographic coordinates...")
    global GEO_DATA
    cols, rows = np.meshgrid(np.arange(W), np.arange(H), indexing='xy') # Use 'xy' indexing for clarity
    # Apply affine transformation directly in a vectorized way
    # T = (a, b, c, d, e, f) where x' = a*x + b*y + c and y' = d*x + e*y + f
    # In rasterio, x corresponds to column and y to row.
    T = ref_transform
    lon = T.a * cols + T.b * rows + T.c
    lat = T.d * cols + T.e * rows + T.f
    GEO_DATA = np.stack((lon, lat), axis=-1).astype(np.float32)

    # 6. Pre-calculate Time data
    global TIME_DATA
    TIME_DATA = np.array([[d[1], d[2], d[3]] for d in all_dates])
    
    # 7. Load and Reproject ALL band data into a single massive array
    # This is the most memory-intensive step.
    # Estimated size: 94 dates * 10 bands * 4587 * 5174 * 4 bytes/float32 ≈ 84 GB
    print(f"Memory check: The next step will allocate a ~{len(all_dates) * NUM_BANDS * H * W * 4 / (1024**3):.2f} GB array for band data.")
    
    band_files = [
        (d_idx * NUM_BANDS + b_idx, data_dir / band_name / f"{date_info[0]}_mosaic.tiff")
        for d_idx, date_info in enumerate(all_dates)
        for b_idx, band_name in enumerate(ALL_BANDS)
    ]
    # This is a flat list of (index, path) tuples
    
    # Load all band data in parallel
    flat_band_data = load_and_reproject_data_parallel(
        band_files, ref_shape, ref_transform, ref_crs, Resampling.bilinear, "Loading All Band Data"
    )
    
    # Reshape the flat array into the desired (D, B, H, W) structure
    global ALL_BANDS_DATA
    ALL_BANDS_DATA = flat_band_data.reshape(len(all_dates), NUM_BANDS, H, W)
    del flat_band_data; gc.collect()
    
    # 8. Prepare and execute patch saving in parallel
    print(f"Preparing to save {total_patches} patches...")
    patch_args = []
    skipped_background = 0
    patches_to_process = 0
    
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            patch_idx = i * n_patches_w + j
            
            row_start, row_end = i * PATCH_SIZE, (i + 1) * PATCH_SIZE
            col_start, col_end = j * PATCH_SIZE, (j + 1) * PATCH_SIZE
            
            # Check if label patch is all background
            if np.all(LABEL_DATA[row_start:row_end, col_start:col_end] == 0):
                skipped_background += 1
                continue
            
            # Find timesteps that meet the validity threshold for this patch
            valid_timesteps_for_patch = np.where(patch_valid_pixel_counts[patch_idx] >= MIN_VALID_PIXELS_COUNT)[0]

            if len(valid_timesteps_for_patch) > 0:
                patch_info = {
                    'row_start': row_start, 'row_end': row_end,
                    'col_start': col_start, 'col_end': col_end
                }
                patch_args.append((patch_idx, patch_info, valid_timesteps_for_patch, output_dirs))
                patches_to_process += 1

    # Clean up before final processing step
    del patch_valid_pixel_counts; gc.collect()

    # 9. Final Parallel Saving Step
    print(f"Saving {patches_to_process} valid patches...")
    completed = 0
    errors = 0
    timestep_counts = []

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        futures = [executor.submit(save_patch_worker, arg) for arg in patch_args]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Saving Patches"):
            patch_idx, num_timesteps, error_msg = future.result()
            if error_msg:
                errors += 1
                tqdm.write(f"Error processing patch {patch_idx}: {error_msg}")
            else:
                completed += 1
                timestep_counts.append(num_timesteps)

    # Final Report
    elapsed_time = time.time() - start_time
    print(f"\n✅ Processing complete in {elapsed_time:.2f} seconds!")
    print(f"   Successfully processed: {completed} patches")
    print(f"   Skipped (all background): {skipped_background} patches")
    print(f"   Skipped (no valid timesteps): {total_patches - skipped_background - completed - errors} patches")
    print(f"   Errors: {errors} patches")

    if timestep_counts:
        counts = np.array(timestep_counts)
        print("\nTimestep Statistics for Saved Patches:")
        print(f"   Min: {np.min(counts)} | Max: {np.max(counts)} | Mean: {np.mean(counts):.2f} | Median: {np.median(counts):.0f}")

    print(f"\nCreated patches in {output_dir}")

if __name__ == "__main__":
    create_patches_optimized()