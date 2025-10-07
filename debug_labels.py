#!/usr/bin/env python3
"""
Debug script to check TreeSatAI label format and data types.
"""

import os
import sys
import torch
import numpy as np
import logging

# Add pangaea to path
sys.path.insert(0, '/scratch/zf281/pangaea-bench')

from pangaea.datasets.treesatai import TreeSatAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_labels():
    """Debug TreeSatAI label format."""
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
        
        # Check several samples
        for i in range(5):
            sample = dataset[i]
            target = sample['target']
            
            logger.info(f"Sample {i}:")
            logger.info(f"  Target type: {type(target)}")
            logger.info(f"  Target dtype: {target.dtype}")
            logger.info(f"  Target shape: {target.shape}")
            logger.info(f"  Target values: {target}")
            logger.info(f"  Target min: {target.min()}, max: {target.max()}")
            logger.info(f"  Target sum: {target.sum()}")
            logger.info(f"  Non-zero elements: {torch.nonzero(target).shape[0]}")
            
            # Check if it's binary (0 or 1)
            unique_values = torch.unique(target)
            logger.info(f"  Unique values: {unique_values}")
            
            # Check if it's continuous (between 0 and 1)
            is_binary = torch.all((target == 0) | (target == 1))
            is_continuous = torch.all((target >= 0) & (target <= 1))
            logger.info(f"  Is binary (0/1): {is_binary}")
            logger.info(f"  Is continuous [0,1]: {is_continuous}")
            logger.info("")
        
        return True
        
    except Exception as e:
        logger.error(f"Debug failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_labels()
    sys.exit(0 if success else 1)