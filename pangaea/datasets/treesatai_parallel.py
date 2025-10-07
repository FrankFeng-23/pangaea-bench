"""
TreeSatAI Dataset with Parallel Loading and Detailed Logging
Optimized version for faster data loading using multiprocessing.
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
                            label_vector[idx] = confidence
                    
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


class TreeSatAIParallel(RawGeoFMDataset):
    """
    TreeSatAI Dataset with parallel loading and detailed logging.
    
    This dataset handles multi-label classification of tree species using
    multi-modal (optical + SAR) and multi-temporal satellite data.
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
        use_predefined_splits: bool = True,  # Use predefined split files
        train_split_file: str = "train_filenames.lst",
        val_split_file: str = "val_filenames.lst", 
        test_split_file: str = "test_filenames.lst",
        **kwargs
    ):
        logger.info(f"Initializing TreeSatAI dataset with {num_workers} workers")
        
        # Default classes
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
        
        # Store split file parameters
        self.use_predefined_splits = use_predefined_splits
        self.train_split_file = train_split_file
        self.val_split_file = val_split_file
        self.test_split_file = test_split_file
        
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
        
        # Load labels
        self.labels_path = os.path.join(self.root_path, "TreeSatBA_v9_60m_multi_labels.json")
        self.h5_dir = os.path.join(self.root_path, "sentinel-ts")
        
        logger.info(f"Labels path: {self.labels_path}")
        logger.info(f"H5 directory: {self.h5_dir}")
        
        # Load and process data
        self._load_labels_parallel()
        if self.use_predefined_splits:
            self._load_predefined_splits()
        else:
            self._create_splits()
        
    def _load_labels_parallel(self):
        """Load labels from JSON file using parallel processing."""
        start_time = time.time()
        logger.info("Starting parallel label loading...")
        
        # Load labels JSON
        logger.info("Loading labels JSON file...")
        with open(self.labels_path, 'r') as f:
            self.all_labels = json.load(f)
        logger.info(f"Loaded {len(self.all_labels)} labels from JSON")
        
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all H5 files
        logger.info("Scanning H5 directory...")
        all_h5_files = [f for f in os.listdir(self.h5_dir) if f.endswith('.h5')]
        logger.info(f"Found {len(all_h5_files)} H5 files")
        
        # Split files into batches for parallel processing
        batches = []
        for i in range(0, len(all_h5_files), self.batch_size):
            batch = all_h5_files[i:i + self.batch_size]
            batches.append((batch, self.h5_dir, self.all_labels, self.class_to_idx, self.num_classes, i // self.batch_size))
        
        logger.info(f"Created {len(batches)} batches for parallel processing")
        logger.info(f"Using {self.num_workers} worker processes")
        
        # Process batches in parallel
        self.samples = []
        self.processed_labels = []
        
        with Pool(processes=self.num_workers) as pool:
            logger.info("Starting parallel processing...")
            results = []
            
            # Submit all batches
            for batch_args in batches:
                result = pool.apply_async(process_h5_file_batch, (batch_args,))
                results.append(result)
            
            # Collect results with progress bar
            logger.info("Collecting results from worker processes...")
            for i, result in enumerate(results):
                try:
                    batch_samples, batch_labels = result.get(timeout=300)  # 5 minute timeout per batch
                    self.samples.extend(batch_samples)
                    self.processed_labels.extend(batch_labels)
                    logger.info(f"Collected batch {i+1}/{len(results)}: {len(batch_samples)} samples")
                except Exception as e:
                    logger.error(f"Error collecting batch {i}: {e}")
        
        end_time = time.time()
        logger.info(f"Parallel loading completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Total samples loaded: {len(self.samples)}")
        
    def _load_predefined_splits(self):
        """Load predefined train/val/test splits from files."""
        logger.info("Loading predefined data splits...")
        
        # Determine which split file to use
        if self.split == "train":
            split_file = self.train_split_file
        elif self.split == "val":
            split_file = self.val_split_file
        elif self.split == "test":
            split_file = self.test_split_file
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        split_path = os.path.join(self.root_path, split_file)
        
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")
        
        logger.info(f"Loading split file: {split_path}")
        
        # Load filenames from split file
        with open(split_path, 'r') as f:
            split_filenames = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Found {len(split_filenames)} files in {self.split} split")
        
        # Create mapping from filename to index
        filename_to_idx = {}
        for idx, sample_path in enumerate(self.samples):
            # Extract filename from path
            filename = os.path.basename(sample_path)
            # Remove .h5 extension and year suffix to match split file format
            if filename.endswith('.h5'):
                # Remove .h5 extension
                base_name = filename.replace('.h5', '')
                # Remove year suffix (e.g., _2017, _2018, etc.)
                parts = base_name.split('_')
                if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 4:
                    base_name_no_year = '_'.join(parts[:-1])
                else:
                    base_name_no_year = base_name
                # Add .tif extension to match split file format
                tif_filename = base_name_no_year + '.tif'
                filename_to_idx[tif_filename] = idx
        
        # Find indices for files in the split
        split_indices = []
        missing_files = []
        
        for filename in split_filenames:
            if filename in filename_to_idx:
                split_indices.append(filename_to_idx[filename])
            else:
                missing_files.append(filename)
        
        if missing_files:
            logger.warning(f"Could not find {len(missing_files)} files from split file in dataset")
            if len(missing_files) <= 10:
                logger.warning(f"Missing files: {missing_files}")
        
        self.split_indices = np.array(split_indices)
        logger.info(f"Split '{self.split}': {len(self.split_indices)} samples loaded from predefined split")

    def _create_splits(self):
        """Create train/val/test splits."""
        logger.info("Creating data splits...")
        
        # Use stratified split based on dominant class
        dominant_classes = []
        for label_vector in self.processed_labels:
            dominant_class = np.argmax(label_vector)
            dominant_classes.append(dominant_class)
        
        logger.info("Computing class distribution...")
        unique_classes, counts = np.unique(dominant_classes, return_counts=True)
        for cls, count in zip(unique_classes, counts):
            logger.info(f"Class {self.classes[cls]}: {count} samples")
        
        # Create splits: 70% train, 15% val, 15% test
        indices = np.arange(len(self.samples))
        
        logger.info("Creating train/temp split (70%/30%)...")
        train_indices, temp_indices = train_test_split(
            indices, test_size=0.3, random_state=42, 
            stratify=dominant_classes
        )
        
        logger.info("Creating val/test split (15%/15%)...")
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
        
        logger.info(f"Split '{self.split}': {len(self.split_indices)} samples")
        logger.info(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        
    def __len__(self) -> int:
        """Return the number of samples in the current split."""
        return len(self.split_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
            - 'optical': Optical data tensor [T, C, H, W]
            - 'sar': SAR data tensor [T, C, H, W] (if multi_modal)
            - 'label': Multi-label tensor [num_classes]
            - 'filename': H5 filename
        """
        # Get actual sample index
        sample_idx = self.split_indices[idx]
        h5_path = self.samples[sample_idx]
        label_vector = self.processed_labels[sample_idx]
        
        # Load H5 data
        with h5py.File(h5_path, 'r') as f:
            # Load Sentinel-2 optical data (using correct key name)
            s2_data = f['sen-2-data'][:]  # Shape: [T, C, H, W]
            
            # Process temporal dimension
            s2_data = self._process_temporal_data(s2_data)
            
            result = {
                'optical': torch.from_numpy(s2_data).float(),
                'label': torch.from_numpy(label_vector).float(),
                'filename': os.path.basename(h5_path)
            }
            
            # Load SAR data if multi-modal
            if self.multi_modal:
                # Load ascending and descending SAR data
                s1_asc = f['sen-1-asc-data'][:]  # Shape: [T, C, H, W]
                s1_desc = f['sen-1-des-data'][:]  # Shape: [T, C, H, W]
                
                # Handle different temporal lengths by taking minimum
                min_time = min(s1_asc.shape[0], s1_desc.shape[0])
                s1_asc = s1_asc[:min_time]
                s1_desc = s1_desc[:min_time]
                
                # Combine ascending and descending SAR data
                s1_data = np.concatenate([s1_asc, s1_desc], axis=1)  # [T, 2*C, H, W]
                
                # Process temporal dimension
                s1_data = self._process_temporal_data(s1_data)
                
                result['sar'] = torch.from_numpy(s1_data).float()
        
        return {
            "image": {
                "optical": result['optical'],
                "sar": result.get('sar', torch.zeros_like(result['optical'][:, :2]))  # Fallback for SAR
            },
            "target": result['label'],
            "metadata": {
                "filename": result['filename']
            }
        }
    
    def _process_temporal_data(self, data: np.ndarray) -> np.ndarray:
        """
        Process temporal dimension to match multi_temporal setting.
        
        Args:
            data: Input data with shape [T, C, H, W]
            
        Returns:
            Processed data with shape [multi_temporal, C, H, W]
        """
        T = data.shape[0]
        
        if T >= self.multi_temporal:
            # Sample uniformly from available time steps
            indices = np.linspace(0, T-1, self.multi_temporal, dtype=int)
            return data[indices]
        else:
            # Repeat data to reach desired temporal length
            repeat_factor = (self.multi_temporal + T - 1) // T
            repeated_data = np.tile(data, (repeat_factor, 1, 1, 1))
            return repeated_data[:self.multi_temporal]
    
    def download(self):
        """Download the dataset (placeholder)."""
        raise NotImplementedError(
            "TreeSatAI dataset download is not implemented. "
            "Please download the dataset manually from the official source."
        )