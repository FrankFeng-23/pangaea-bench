#!/usr/bin/env python3
"""
Simple test script to verify TreeSatAI dataset structure.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
import logging

# Add pangaea to path
sys.path.insert(0, '/scratch/zf281/pangaea-bench')

from pangaea.datasets.treesatai import TreeSatAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset_structure():
    """Test TreeSatAI dataset structure."""
    try:
        # Create dataset
        logger.info("Creating TreeSatAI dataset...")
        dataset = TreeSatAI(
            split="train",
            root_path="/scratch/zf281/pangaea-bench/data/treesatai",
            num_workers=4,
            batch_size=100
        )
        logger.info(f"Dataset created with {len(dataset)} samples")
        
        # Test a single sample
        logger.info("Testing single sample...")
        sample = dataset[0]
        logger.info(f"Sample keys: {sample.keys()}")
        
        if 'image' in sample:
            logger.info(f"Image keys: {sample['image'].keys()}")
            logger.info(f"Optical shape: {sample['image']['optical'].shape}")
            logger.info(f"SAR shape: {sample['image']['sar'].shape}")
        else:
            logger.error("'image' key not found in sample!")
            return False
            
        if 'target' in sample:
            logger.info(f"Target shape: {sample['target'].shape}")
        else:
            logger.error("'target' key not found in sample!")
            return False
            
        if 'metadata' in sample:
            logger.info(f"Metadata keys: {sample['metadata'].keys()}")
        else:
            logger.error("'metadata' key not found in sample!")
            return False
        
        # Test DataLoader with a few samples
        logger.info("Testing DataLoader...")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        
        for i, batch in enumerate(dataloader):
            logger.info(f"Batch {i} keys: {batch.keys()}")
            logger.info(f"Batch {i} image keys: {batch['image'].keys()}")
            logger.info(f"Batch {i} optical shape: {batch['image']['optical'].shape}")
            logger.info(f"Batch {i} SAR shape: {batch['image']['sar'].shape}")
            logger.info(f"Batch {i} target shape: {batch['target'].shape}")
            if i >= 2:  # Test only first 3 batches
                break
        
        logger.info("All tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset_structure()
    sys.exit(0 if success else 1)