#!/usr/bin/env python3
"""
Debug script to check encoder initialization and weight loading during actual training
"""

import os
import sys
import torch
from pathlib import Path

# Add pangaea to path
sys.path.insert(0, '/scratch/zf281/pangaea-bench')

from hydra.utils import instantiate
from omegaconf import OmegaConf
from pangaea.utils.logger import init_logger

def debug_training_encoder():
    """Debug the encoder initialization and weight loading process during training"""
    
    # Load the actual configuration used in training
    config_path = "/scratch/zf281/pangaea-bench/configs/encoder/prithvi.yaml"
    encoder_cfg = OmegaConf.load(config_path)
    
    # Load dataset config to get the correct input_bands
    dataset_config_path = "/scratch/zf281/pangaea-bench/configs/dataset/treesatai.yaml"
    dataset_cfg = OmegaConf.load(dataset_config_path)
    
    print("=== Encoder Configuration ===")
    print(OmegaConf.to_yaml(encoder_cfg))
    
    print("\n=== Dataset Configuration (bands) ===")
    print(f"Bands: {dataset_cfg.bands}")
    
    # Update encoder config with dataset bands
    encoder_cfg.input_bands = dataset_cfg.bands
    
    # Also need to resolve the num_frames interpolation
    encoder_cfg.num_frames = dataset_cfg.multi_temporal
    
    # Update in_chans to match the actual number of optical bands
    num_optical_bands = len(encoder_cfg.input_bands['optical'])
    encoder_cfg.in_chans = num_optical_bands
    
    print(f"\n=== Creating encoder with input_bands: {encoder_cfg.input_bands} ===")
    print(f"num_frames: {encoder_cfg.num_frames}")
    print(f"in_chans: {encoder_cfg.in_chans} (updated to match optical bands)")
    
    # Create logger
    logger = init_logger("/tmp/debug_encoder.log", rank=0)
    
    # Create encoder instance
    print("Creating encoder instance...")
    encoder = instantiate(encoder_cfg)
    
    print(f"Encoder created: {encoder.model_name}")
    print(f"Encoder num_frames: {encoder.num_frames}")
    print(f"Encoder patch_embed grid_size: {encoder.patch_embed.grid_size}")
    print(f"Encoder patch_embed num_patches: {encoder.patch_embed.num_patches}")
    print(f"Initial pos_embed shape: {encoder.pos_embed.shape}")
    
    # Load encoder weights
    print("\n=== Loading encoder weights ===")
    encoder.load_encoder_weights(logger)
    
    print(f"After loading weights, pos_embed shape: {encoder.pos_embed.shape}")
    
    # Test forward pass with correct input format
    print("\n=== Testing forward pass ===")
    try:
        # Create dummy input matching TreeSatAI format
        batch_size = 2
        num_frames = encoder.num_frames
        height, width = 224, 224  # input_size from config
        
        # TreeSatAI has optical bands
        num_optical_bands = len(encoder_cfg.input_bands['optical'])
        
        dummy_input = {
            'optical': torch.randn(batch_size, num_optical_bands, num_frames, height, width)
        }
        
        print(f"Input shape: {dummy_input['optical'].shape}")
        
        with torch.no_grad():
            output = encoder(dummy_input)
            
        print(f"Forward pass successful!")
        print(f"Output shapes: {[out.shape for out in output]}")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_training_encoder()