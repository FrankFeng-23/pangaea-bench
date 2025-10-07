#!/usr/bin/env python3
"""
Test script for TreeSatAI dataset with parallel loading.
This script tests the optimized TreeSatAI dataset implementation with detailed logging.
"""

import sys
import os
import time
import logging
import traceback
import torch
from torch.utils.data import DataLoader

# Add the project root to Python path
sys.path.insert(0, '/scratch/zf281/pangaea-bench')

from pangaea.datasets.treesatai_parallel import TreeSatAIParallel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_dataset_loading():
    """Test basic dataset loading with parallel processing."""
    logger.info("=" * 60)
    logger.info("Testing TreeSatAI dataset loading with parallel processing")
    logger.info("=" * 60)
    
    try:
        start_time = time.time()
        
        # Test train split
        logger.info("Loading train dataset...")
        train_dataset = TreeSatAIParallel(
            split="train",
            num_workers=64,  # Use 64 parallel processes
            batch_size=1000  # Process 1000 files per batch
        )
        
        train_time = time.time() - start_time
        logger.info(f"Train dataset loaded in {train_time:.2f} seconds")
        logger.info(f"Train dataset size: {len(train_dataset)}")
        
        # Test validation split
        logger.info("Loading validation dataset...")
        val_start = time.time()
        val_dataset = TreeSatAIParallel(
            split="val",
            num_workers=32,  # Use fewer workers for val/test
            batch_size=500
        )
        
        val_time = time.time() - val_start
        logger.info(f"Validation dataset loaded in {val_time:.2f} seconds")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        # Test test split
        logger.info("Loading test dataset...")
        test_start = time.time()
        test_dataset = TreeSatAIParallel(
            split="test",
            num_workers=32,
            batch_size=500
        )
        
        test_time = time.time() - test_start
        logger.info(f"Test dataset loaded in {test_time:.2f} seconds")
        logger.info(f"Test dataset size: {len(test_dataset)}")
        
        total_time = time.time() - start_time
        logger.info(f"Total loading time: {total_time:.2f} seconds")
        
        return train_dataset, val_dataset, test_dataset
        
    except Exception as e:
        logger.error(f"Error in dataset loading: {e}")
        logger.error(traceback.format_exc())
        return None, None, None


def test_sample_loading(dataset, num_samples=5):
    """Test loading individual samples from the dataset."""
    logger.info("=" * 60)
    logger.info(f"Testing sample loading (first {num_samples} samples)")
    logger.info("=" * 60)
    
    if dataset is None:
        logger.error("Dataset is None, skipping sample loading test")
        return
    
    try:
        for i in range(min(num_samples, len(dataset))):
            logger.info(f"Loading sample {i+1}/{num_samples}...")
            start_time = time.time()
            
            sample = dataset[i]
            
            load_time = time.time() - start_time
            logger.info(f"Sample {i} loaded in {load_time:.3f} seconds")
            
            # Check sample structure
            logger.info(f"Sample keys: {list(sample.keys())}")
            
            if 'optical' in sample:
                optical_shape = sample['optical'].shape
                optical_dtype = sample['optical'].dtype
                optical_range = (sample['optical'].min().item(), sample['optical'].max().item())
                logger.info(f"Optical data - Shape: {optical_shape}, Dtype: {optical_dtype}, Range: {optical_range}")
            
            if 'sar' in sample:
                sar_shape = sample['sar'].shape
                sar_dtype = sample['sar'].dtype
                sar_range = (sample['sar'].min().item(), sample['sar'].max().item())
                logger.info(f"SAR data - Shape: {sar_shape}, Dtype: {sar_dtype}, Range: {sar_range}")
            
            if 'label' in sample:
                label_shape = sample['label'].shape
                label_dtype = sample['label'].dtype
                num_active_labels = (sample['label'] > 0).sum().item()
                logger.info(f"Label - Shape: {label_shape}, Dtype: {label_dtype}, Active labels: {num_active_labels}")
            
            if 'filename' in sample:
                logger.info(f"Filename: {sample['filename']}")
            
            logger.info("-" * 40)
            
    except Exception as e:
        logger.error(f"Error in sample loading: {e}")
        logger.error(traceback.format_exc())


def test_batch_loading(dataset, batch_size=8, num_batches=3):
    """Test batch loading using DataLoader."""
    logger.info("=" * 60)
    logger.info(f"Testing batch loading (batch_size={batch_size}, {num_batches} batches)")
    logger.info("=" * 60)
    
    if dataset is None:
        logger.error("Dataset is None, skipping batch loading test")
        return
    
    try:
        # Use fewer workers for DataLoader to avoid conflicts
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,  # Reduced number of workers
            pin_memory=True
        )
        
        logger.info(f"DataLoader created with {len(dataloader)} batches total")
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            logger.info(f"Processing batch {batch_idx + 1}/{num_batches}...")
            start_time = time.time()
            
            # Process batch
            if 'optical' in batch:
                optical_shape = batch['optical'].shape
                logger.info(f"Batch optical shape: {optical_shape}")
            
            if 'sar' in batch:
                sar_shape = batch['sar'].shape
                logger.info(f"Batch SAR shape: {sar_shape}")
            
            if 'label' in batch:
                label_shape = batch['label'].shape
                logger.info(f"Batch label shape: {label_shape}")
            
            batch_time = time.time() - start_time
            logger.info(f"Batch {batch_idx} processed in {batch_time:.3f} seconds")
            logger.info("-" * 40)
            
    except Exception as e:
        logger.error(f"Error in batch loading: {e}")
        logger.error(traceback.format_exc())


def test_class_distribution(dataset):
    """Test class distribution analysis."""
    logger.info("=" * 60)
    logger.info("Testing class distribution analysis")
    logger.info("=" * 60)
    
    if dataset is None:
        logger.error("Dataset is None, skipping class distribution test")
        return
    
    try:
        logger.info("Analyzing class distribution (sampling 100 samples)...")
        
        # Sample a subset for faster analysis
        sample_size = min(100, len(dataset))
        class_counts = torch.zeros(dataset.num_classes)
        
        for i in range(sample_size):
            if i % 20 == 0:
                logger.info(f"Processed {i}/{sample_size} samples for class analysis")
            
            sample = dataset[i]
            label = sample['label']
            
            # Count active labels (multi-label)
            active_labels = (label > 0).nonzero(as_tuple=True)[0]
            for class_idx in active_labels:
                class_counts[class_idx] += 1
        
        logger.info("Class distribution (from sampled data):")
        for i, (class_name, count) in enumerate(zip(dataset.classes, class_counts)):
            logger.info(f"  {class_name}: {count.item()}")
            
    except Exception as e:
        logger.error(f"Error in class distribution analysis: {e}")
        logger.error(traceback.format_exc())


def test_multi_temporal_settings(root_path):
    """Test different multi-temporal settings."""
    logger.info("=" * 60)
    logger.info("Testing different multi-temporal settings")
    logger.info("=" * 60)
    
    temporal_settings = [1, 6, 12]  # Reduced number of settings for faster testing
    
    for multi_temporal in temporal_settings:
        try:
            logger.info(f"Testing multi_temporal={multi_temporal}...")
            start_time = time.time()
            
            dataset = TreeSatAIParallel(
                split="train",
                multi_temporal=multi_temporal,
                root_path=root_path,
                num_workers=16,  # Use fewer workers for these tests
                batch_size=200
            )
            
            load_time = time.time() - start_time
            logger.info(f"Dataset with multi_temporal={multi_temporal} loaded in {load_time:.2f} seconds")
            
            # Test a sample
            if len(dataset) > 0:
                sample = dataset[0]
                if 'optical' in sample:
                    optical_shape = sample['optical'].shape
                    logger.info(f"Optical temporal dimension: {optical_shape[0]} (expected: {multi_temporal})")
                
                if 'sar' in sample:
                    sar_shape = sample['sar'].shape
                    logger.info(f"SAR temporal dimension: {sar_shape[0]} (expected: {multi_temporal})")
            
            logger.info("-" * 40)
            
        except Exception as e:
            logger.error(f"Error testing multi_temporal={multi_temporal}: {e}")
            logger.error(traceback.format_exc())


def main():
    """Main test function."""
    logger.info("Starting TreeSatAI parallel dataset tests")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Test dataset loading
    train_dataset, val_dataset, test_dataset = test_dataset_loading()
    
    if train_dataset is not None:
        # Test sample loading
        test_sample_loading(train_dataset, num_samples=3)
        
        # Test batch loading
        test_batch_loading(train_dataset, batch_size=4, num_batches=2)
        
        # Test class distribution
        test_class_distribution(train_dataset)
        
        # Test multi-temporal settings
        test_multi_temporal_settings(train_dataset.root_path)
    
    logger.info("All tests completed!")


if __name__ == "__main__":
    main()