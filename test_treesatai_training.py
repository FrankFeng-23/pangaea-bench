#!/usr/bin/env python3
"""
Test script to debug TreeSatAI training issues step by step.
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import logging

# Add pangaea to path
sys.path.insert(0, '/scratch/zf281/pangaea-bench')

from pangaea.datasets.treesatai import TreeSatAI
from pangaea.engine.data_preprocessor import Preprocessor
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset_with_preprocessor():
    """Test TreeSatAI dataset with preprocessor."""
    try:
        # Initialize Hydra
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        with initialize(version_base=None, config_path="configs"):
            # Load configurations
            cfg = compose(config_name="train", overrides=[
                "dataset=treesatai",
                "encoder=remoteclip", 
                "decoder=cls_linear",
                "preprocessing=cls_resize",
                "criterion=binary_cross_entropy",
                "task=linear_classification_multi_label"
            ])
            
            logger.info("Configuration loaded successfully")
            
            # Create dataset
            logger.info("Creating TreeSatAI dataset...")
            dataset = TreeSatAI(
                split="train",
                root_path="/scratch/zf281/pangaea-bench/data/treesatai",
                num_workers=4,
                batch_size=100
            )
            logger.info(f"Dataset created with {len(dataset)} samples")
            
            # Create preprocessor
            logger.info("Creating preprocessor...")
            preprocessor = Preprocessor(
                preprocessor_cfg=cfg.preprocessing,
                dataset_cfg=cfg.dataset,
                encoder_cfg=cfg.encoder
            )
            logger.info("Preprocessor created successfully")
            
            # Test a single sample
            logger.info("Testing single sample...")
            sample = dataset[0]
            logger.info(f"Sample keys: {sample.keys()}")
            logger.info(f"Image keys: {sample['image'].keys()}")
            logger.info(f"Optical shape: {sample['image']['optical'].shape}")
            logger.info(f"SAR shape: {sample['image']['sar'].shape}")
            logger.info(f"Target shape: {sample['target'].shape}")
            
            # Test preprocessor
            logger.info("Testing preprocessor...")
            processed_sample = preprocessor(sample)
            logger.info(f"Processed sample keys: {processed_sample.keys()}")
            logger.info(f"Processed image keys: {processed_sample['image'].keys()}")
            logger.info(f"Processed optical shape: {processed_sample['image']['optical'].shape}")
            logger.info(f"Processed SAR shape: {processed_sample['image']['sar'].shape}")
            logger.info(f"Processed target shape: {processed_sample['target'].shape}")
            
            # Test DataLoader
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
    success = test_dataset_with_preprocessor()
    sys.exit(0 if success else 1)