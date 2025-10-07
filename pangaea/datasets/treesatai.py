"""
TreeSatAI Dataset Implementation

TreeSatAI is a multi-label classification dataset for tree species identification
using Sentinel-1 and Sentinel-2 time series data.
"""

import os
import json
import h5py
import torch
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import tqdm

from pangaea.datasets.base import RawGeoFMDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_h5_file_batch(args):
    """
    Process a batch of H5 files in parallel.
    
    Args:
        args: Tuple containing (h5_files_batch, h5_dir, all_labels, class_to_idx, num_classes, batch_id)
    
    Returns:
        Tuple of (samples, processed_labels) for this batch
    """
    h5_files_batch, h5_dir, all_labels, class_to_idx, num_classes, batch_id = args
    
    batch_samples = []
    batch_labels = []
    
    logger.info(f"Processing batch {batch_id} with {len(h5_files_batch)} files")
    
    for i, h5_file in enumerate(h5_files_batch):
        try:
            # Extract base name from H5 filename
            h5_base = h5_file.replace('.h5', '')
            h5_parts = h5_base.split('_')
            if len(h5_parts) >= 2 and h5_parts[-1].isdigit():
                h5_base_no_year = '_'.join(h5_parts[:-1])
            else:
                h5_base_no_year = h5_base
            
            # Find corresponding label
            label_filename = h5_base_no_year + '.tif'
            if label_filename in all_labels:
                h5_path = os.path.join(h5_dir, h5_file)
                if os.path.exists(h5_path):
                    # Convert multi-label to binary vector
                    label_list = all_labels[label_filename]
                    label_vector = np.zeros(num_classes, dtype=np.float32)
                    for class_name, confidence in label_list:
                        if class_name in class_to_idx:
                            idx = class_to_idx[class_name]
                            # Convert confidence to binary (1 if confidence > 0.5, else 0)
                            label_vector[idx] = 1.0 if confidence > 0.5 else 0.0
                    
                    batch_samples.append(h5_path)
                    batch_labels.append(label_vector)
        
        except Exception as e:
            logger.warning(f"Error processing file {h5_file}: {e}")
            continue
        
        # Log progress every 100 files
        if (i + 1) % 100 == 0:
            logger.info(f"Batch {batch_id}: Processed {i + 1}/{len(h5_files_batch)} files")
    
    logger.info(f"Batch {batch_id} completed: {len(batch_samples)} valid samples")
    return batch_samples, batch_labels


class TreeSatAI(RawGeoFMDataset):
    """
    TreeSatAI Dataset for multi-label tree species classification.
    
    The dataset contains Sentinel-1 and Sentinel-2 time series data with
    multi-label annotations for tree species identification.
    """
    
    def __init__(
        self,
        split: str = "train",
        dataset_name: str = "treesatai",
        multi_modal: bool = True,
        multi_temporal: int = 12,
        root_path: str = "/scratch/zf281/pangaea-bench/data/treesatai",
        classes: List[str] = None,
        num_classes: int = 15,
        ignore_index: int = -1,
        img_size: int = 6,  # TreeSatAI patches are 6x6
        bands: Dict[str, List[str]] = None,
        distribution: List[int] = None,
        data_mean: Dict[str, List[float]] = None,
        data_std: Dict[str, List[float]] = None,
        data_min: Dict[str, List[float]] = None,
        data_max: Dict[str, List[float]] = None,
        download_url: str = "",
        auto_download: bool = False,
        num_workers: int = 64,  # Number of parallel processes
        batch_size: int = 1000,  # Files per batch for parallel processing
        **kwargs
    ):
        # Default classes for TreeSatAI
        if classes is None:
            classes = [
                "Abies", "Acer", "Alnus", "Betula", "Cleared", "Fagus", 
                "Fraxinus", "Larix", "Picea", "Pinus", "Populus", 
                "Pseudotsuga", "Quercus", "Salix", "Tilia"
            ]
        
        # Default bands configuration
        if bands is None:
            bands = {
                "optical": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
                "sar": ["VV", "VH"]
            }
        
        # Default distribution (will be calculated from actual data)
        if distribution is None:
            distribution = [0] * num_classes
        
        # Default data statistics (will be calculated from actual data)
        if data_mean is None:
            data_mean = {
                "optical": [0.0] * 10,
                "sar": [0.0] * 2
            }
        
        if data_std is None:
            data_std = {
                "optical": [1.0] * 10,
                "sar": [1.0] * 2
            }
        
        if data_min is None:
            data_min = {
                "optical": [0.0] * 10,
                "sar": [0.0] * 2
            }
        
        if data_max is None:
            data_max = {
                "optical": [1.0] * 10,
                "sar": [1.0] * 2
            }
        
        # Store parallel processing parameters
        self.num_workers = min(num_workers, cpu_count())
        self.batch_size = batch_size
        
        super().__init__(
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
            **kwargs
        )
        
        # Set up paths
        self.h5_dir = os.path.join(root_path, "sentinel-ts")
        self.labels_path = os.path.join(root_path, "TreeSatBA_v9_60m_multi_labels.json")
        
        # Load data with parallel processing
        logger.info(f"Loading TreeSatAI dataset with {self.num_workers} workers")
        start_time = time.time()
        self._load_labels_parallel()
        self._create_splits()
        load_time = time.time() - start_time
        logger.info(f"Dataset loading completed in {load_time:.2f} seconds")
        
    def _load_labels_parallel(self):
        """Load labels from JSON file using parallel processing."""
        logger.info("Loading labels from JSON file...")
        with open(self.labels_path, 'r') as f:
            self.all_labels = json.load(f)
        
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all H5 files
        logger.info("Scanning H5 files...")
        h5_files = [f for f in os.listdir(self.h5_dir) if f.endswith('.h5')]
        logger.info(f"Found {len(h5_files)} H5 files")
        
        # Create batches for parallel processing
        batches = []
        for i in range(0, len(h5_files), self.batch_size):
            batch = h5_files[i:i + self.batch_size]
            batch_id = i // self.batch_size + 1
            batches.append((batch, self.h5_dir, self.all_labels, self.class_to_idx, self.num_classes, batch_id))
        
        logger.info(f"Created {len(batches)} batches for parallel processing")
        
        # Process batches in parallel
        self.samples = []
        self.processed_labels = []
        
        if self.num_workers > 1:
            logger.info(f"Starting parallel processing with {self.num_workers} workers...")
            with Pool(processes=self.num_workers) as pool:
                results = pool.map(process_h5_file_batch, batches)
            
            # Collect results
            logger.info("Collecting results from parallel processing...")
            for batch_samples, batch_labels in results:
                self.samples.extend(batch_samples)
                self.processed_labels.extend(batch_labels)
        else:
            # Fallback to serial processing
            logger.info("Using serial processing...")
            for batch_args in batches:
                batch_samples, batch_labels = process_h5_file_batch(batch_args)
                self.samples.extend(batch_samples)
                self.processed_labels.extend(batch_labels)
        
        logger.info(f"Loaded {len(self.samples)} samples with labels")
    
    def _load_labels(self):
        """Load labels from JSON file."""
        with open(self.labels_path, 'r') as f:
            self.all_labels = json.load(f)
        
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Process labels and find corresponding H5 files
        self.samples = []
        self.processed_labels = []
        
        for label_filename, label_list in self.all_labels.items():
            # Extract base name from label filename (remove .tif extension)
            base_name = label_filename.replace('.tif', '')
            
            # Find corresponding H5 files (there might be multiple years)
            h5_files = []
            for h5_file in os.listdir(self.h5_dir):
                if h5_file.endswith('.h5'):
                    h5_base = h5_file.replace('.h5', '')
                    # Remove year suffix from H5 filename
                    h5_parts = h5_base.split('_')
                    if len(h5_parts) >= 2 and h5_parts[-1].isdigit():
                        h5_base_no_year = '_'.join(h5_parts[:-1])
                    else:
                        h5_base_no_year = h5_base
                    
                    if h5_base_no_year == base_name:
                        h5_files.append(h5_file)
            
            # Process each H5 file
            for h5_file in h5_files:
                h5_path = os.path.join(self.h5_dir, h5_file)
                if os.path.exists(h5_path):
                    # Convert multi-label to binary vector
                    label_vector = np.zeros(self.num_classes, dtype=np.float32)
                    for class_name, confidence in label_list:
                        if class_name in self.class_to_idx:
                            idx = self.class_to_idx[class_name]
                            # Convert confidence to binary (1 if confidence > 0.5, else 0)
                            label_vector[idx] = 1.0 if confidence > 0.5 else 0.0
                    
                    self.samples.append(h5_path)
                    self.processed_labels.append(label_vector)
        
        print(f"Loaded {len(self.samples)} samples with labels")
    
    def _create_splits(self):
        """Create train/val/test splits with logging."""
        logger.info("Creating train/val/test splits...")
        
        # Use stratified split based on dominant class
        dominant_classes = []
        for label_vector in self.processed_labels:
            dominant_class = np.argmax(label_vector)
            dominant_classes.append(dominant_class)
        
        # Create splits: 70% train, 15% val, 15% test
        indices = np.arange(len(self.samples))
        
        train_indices, temp_indices = train_test_split(
            indices, test_size=0.3, random_state=42, 
            stratify=dominant_classes
        )
        
        temp_dominant = [dominant_classes[i] for i in temp_indices]
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=0.5, random_state=42,
            stratify=temp_dominant
        )
        
        # Store split indices
        if self.split == "train":
            self.split_indices = train_indices
        elif self.split == "val":
            self.split_indices = val_indices
        elif self.split == "test":
            self.split_indices = test_indices
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        logger.info(f"Split {self.split}: {len(self.split_indices)} samples")
        logger.info(f"Total samples - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    def __len__(self) -> int:
        """Return the number of samples in the current split."""
        return len(self.split_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Get a sample from the dataset."""
        # Get the actual sample index
        sample_idx = self.split_indices[idx]
        h5_path = self.samples[sample_idx]
        label_vector = self.processed_labels[sample_idx]
        
        # Load data from H5 file
        with h5py.File(h5_path, 'r') as h5file:
            # Load Sentinel-2 optical data
            sen2_data = h5file['sen-2-data'][:]  # Shape: (T, 10, 6, 6)
            
            # Load Sentinel-1 SAR data (combine ascending and descending)
            sen1_asc_data = h5file['sen-1-asc-data'][:]  # Shape: (T_asc, 2, 6, 6)
            sen1_des_data = h5file['sen-1-des-data'][:]  # Shape: (T_des, 2, 6, 6)
            
            # Combine SAR data (concatenate along time dimension)
            sar_data = np.concatenate([sen1_asc_data, sen1_des_data], axis=0)  # Shape: (T_sar, 2, 6, 6)
        
        # Process temporal dimension
        optical_data = self._process_temporal_data(sen2_data)  # Shape: (multi_temporal, 10, 6, 6)
        sar_data = self._process_temporal_data(sar_data)      # Shape: (multi_temporal, 2, 6, 6)
        
        # Convert to torch tensors and rearrange dimensions
        # From (T, C, H, W) to (C, T, H, W)
        optical_tensor = torch.from_numpy(optical_data).permute(1, 0, 2, 3).float()
        sar_tensor = torch.from_numpy(sar_data).permute(1, 0, 2, 3).float()
        
        # Create sample dictionary following the standard format
        sample = {
            "image": {
                "optical": optical_tensor,
                "sar": sar_tensor,
            },
            "target": torch.from_numpy(label_vector).float(),
            "metadata": {
                "filename": os.path.basename(h5_path)
            }
        }
        
        return sample
    
    def _process_temporal_data(self, data: np.ndarray) -> np.ndarray:
        """Process temporal dimension of the data."""
        T, C, H, W = data.shape
        
        if T >= self.multi_temporal:
            # If we have enough time steps, sample uniformly
            indices = np.linspace(0, T - 1, self.multi_temporal, dtype=int)
            return data[indices]
        else:
            # If we don't have enough time steps, repeat the last frame
            repeated_data = np.zeros((self.multi_temporal, C, H, W), dtype=data.dtype)
            repeated_data[:T] = data
            repeated_data[T:] = data[-1:].repeat(self.multi_temporal - T, axis=0)
            return repeated_data
    
    def download(self):
        """Download dataset if needed."""
        # TreeSatAI dataset should be manually downloaded
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(
                f"TreeSatAI dataset not found at {self.root_path}. "
                "Please download the dataset manually."
            )
        
        if not os.path.exists(self.h5_dir):
            raise FileNotFoundError(
                f"TreeSatAI H5 files not found at {self.h5_dir}. "
                "Please download the dataset manually."
            )