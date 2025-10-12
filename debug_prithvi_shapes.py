#!/usr/bin/env python3
import sys
sys.path.append('/scratch/zf281/pangaea-bench')

import torch
import numpy as np
from pangaea.datasets.treesatai_parallel import TreeSatAIParallel
from pangaea.engine.data_preprocessor import Preprocessor
from pangaea.encoders.prithvi_encoder import Prithvi_Encoder
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

def debug_shapes():
    # Initialize Hydra
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="train", overrides=[
            "dataset=treesatai",
            "encoder=prithvi", 
            "decoder=cls_linear",
            "preprocessing=cls_resize",
            "criterion=binary_cross_entropy",
            "task=linear_classification_multi_label"
        ])
        
        # Create dataset
        dataset = TreeSatAIParallel(
            split="train",
            root_path="/scratch/zf281/pangaea-bench/data/treesatai",
            num_workers=4,
            batch_size=100
        )
        
        # Create preprocessor
        preprocessor = Preprocessor(cfg.preprocessing.train.preprocessor_cfg)
        preprocessor.set_encoder_info(cfg.encoder)
        
        # Get a sample
        sample = dataset[0]
        print(f"Original sample optical shape: {sample['image']['optical'].shape}")
        
        # Apply preprocessing
        processed_sample = preprocessor(sample)
        print(f"Processed sample optical shape: {processed_sample['image']['optical'].shape}")
        
        # Create encoder
        encoder = Prithvi_Encoder(**cfg.encoder)
        print(f"Encoder input_size: {encoder.input_size}")
        print(f"Encoder patch_size: {encoder.patch_size}")
        print(f"Encoder num_frames: {encoder.num_frames}")
        print(f"Encoder pos_embed shape: {encoder.pos_embed.shape}")
        
        # Test patch embedding
        x = processed_sample['image']['optical'].unsqueeze(0)  # Add batch dimension
        print(f"Input to patch_embed shape: {x.shape}")
        
        x_patches = encoder.patch_embed(x)
        print(f"After patch_embed shape: {x_patches.shape}")
        
        # Add cls token
        cls_tokens = encoder.cls_token.expand(x_patches.shape[0], -1, -1)
        x_with_cls = torch.cat((cls_tokens, x_patches), dim=1)
        print(f"After adding cls token shape: {x_with_cls.shape}")
        
        print(f"Expected pos_embed shape: {encoder.pos_embed.shape}")
        print(f"Actual tensor shape: {x_with_cls.shape}")
        
        # Calculate grid sizes
        current_num_patches = x_with_cls.shape[1] - 1
        patches_per_frame = current_num_patches // encoder.num_frames
        current_spatial_grid_size = int(np.sqrt(patches_per_frame))
        
        print(f"Current num patches (without cls): {current_num_patches}")
        print(f"Patches per frame: {patches_per_frame}")
        print(f"Current spatial grid size: {current_spatial_grid_size}")
        
        # Check original grid size
        original_num_patches = encoder.pos_embed.shape[1] - 1
        original_patches_per_frame = original_num_patches // encoder.num_frames
        original_spatial_grid_size = int(np.sqrt(original_patches_per_frame))
        
        print(f"Original num patches (without cls): {original_num_patches}")
        print(f"Original patches per frame: {original_patches_per_frame}")
        print(f"Original spatial grid size: {original_spatial_grid_size}")

if __name__ == "__main__":
    debug_shapes()
