#!/usr/bin/env python3

import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# Add pangaea to path
sys.path.append(str(Path(__file__).parent / "pangaea"))

from pangaea.utils.logger import init_logger
from pangaea.datasets import build_dataset
from pangaea.encoders import build_encoder
from pangaea.decoders import build_decoder

def test_classification_training():
    # Set up basic environment
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
    # Initialize logger
    logger = init_logger(rank=0)
    
    # Load configurations
    encoder_cfg = OmegaConf.load("configs/encoder/prithvi.yaml")
    dataset_cfg = OmegaConf.load("configs/dataset/treesatai.yaml")
    decoder_cfg = OmegaConf.load("configs/decoder/cls_linear.yaml")
    
    print(f"Dataset config: {dataset_cfg}")
    print(f"Encoder config: {encoder_cfg}")
    print(f"Decoder config: {decoder_cfg}")
    
    # Set up encoder configuration for TreeSatAI
    encoder_cfg.num_frames = dataset_cfg.multi_temporal
    encoder_cfg.in_chans = len(dataset_cfg.bands['optical'])
    
    print(f"Encoder input_bands: {encoder_cfg.input_bands}")
    print(f"Encoder num_frames: {encoder_cfg.num_frames}")
    print(f"Encoder in_chans: {encoder_cfg.in_chans}")
    
    # Build encoder
    print("\n=== Building encoder ===")
    encoder = build_encoder(encoder_cfg)
    print(f"Encoder created: {encoder.__class__.__name__}")
    
    # Load encoder weights
    print("\n=== Loading encoder weights ===")
    encoder.load_encoder_weights(logger)
    
    # Build decoder
    print("\n=== Building decoder ===")
    decoder = build_decoder(decoder_cfg)
    print(f"Decoder created: {decoder.__class__.__name__}")
    
    # Test forward pass
    print("\n=== Testing forward pass ===")
    batch_size = 2
    channels = encoder_cfg.in_chans
    frames = encoder_cfg.num_frames
    height, width = 224, 224
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, channels, frames, height, width)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass through encoder
    with torch.no_grad():
        encoder_output = encoder(dummy_input)
        print(f"Encoder output shape: {encoder_output.shape}")
        
        # Forward pass through decoder
        decoder_output = decoder(encoder_output)
        print(f"Decoder output shape: {decoder_output.shape}")
    
    print("\nClassification training setup successful!")
    return True

if __name__ == "__main__":
    try:
        test_classification_training()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)