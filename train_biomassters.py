import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import torch.nn.functional as F
import csv
import random
import argparse
import time
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.logging import RichHandler
import math

# Set up rich console for beautiful output
console = Console()

def setup_logger(rank: int, log_dir: str) -> logging.Logger:
    """Setup logger with Rich formatting"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(f"trainer_rank_{rank}")
    logger.setLevel(logging.INFO)
    
    # Only rank 0 logs to console
    if rank == 0:
        # Rich console handler
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True
        )
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
    
    # File handler for all ranks
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"train_rank_{rank}.log")
    )
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://')
    return True, rank, world_size, gpu

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def compute_target_stats(target_files):
    """Compute mean and std of target values"""
    all_values = []
    for f in target_files:
        arr = np.load(f)
        valid = arr[~np.isnan(arr)]
        all_values.append(valid)
    all_values = np.concatenate(all_values)
    return np.mean(all_values), np.std(all_values)

class BorneoDatasetOptimized(Dataset):
    """Optimized dataset that loads preprocessed data"""
    def __init__(self, preprocessed_root, label_root, split, target_mean, target_std):
        self.preprocessed_root = preprocessed_root
        self.label_root = label_root
        self.split = split
        self.target_mean = target_mean
        self.target_std = target_std
        
        # Get preprocessed files
        processed_dir = os.path.join(preprocessed_root, split, 'processed')
        self.processed_files = sorted(glob.glob(os.path.join(processed_dir, '*.npy')))
        
        # Get corresponding label files
        self.label_files = []
        valid_indices = []
        
        for i, processed_file in enumerate(self.processed_files):
            basename = os.path.basename(processed_file)
            label_file = os.path.join(label_root, split, basename)
            
            if os.path.exists(label_file):
                # Check if label is not all NaN
                target = np.load(label_file)
                if not np.all(np.isnan(target)):
                    self.label_files.append(label_file)
                    valid_indices.append(i)
        
        # Keep only valid files
        self.processed_files = [self.processed_files[i] for i in valid_indices]
        
        print(f"[Dataset] Found {len(self.processed_files)} valid {split} samples")
        
    def __len__(self):
        return len(self.processed_files)
    
    def __getitem__(self, idx):
        # Load preprocessed representation (already resized)
        rep = np.load(self.processed_files[idx]).astype(np.float32)
        target = np.load(self.label_files[idx])
        
        # Convert to tensors
        rep = torch.from_numpy(rep).float().permute(2, 0, 1)  # (256,256,128) -> (128,256,256)
        target = torch.from_numpy(target).float().unsqueeze(0)  # (256,256) -> (1,256,256)
        
        # Normalize target
        target = (target - self.target_mean) / self.target_std
        
        filename = os.path.basename(self.label_files[idx]).replace(".npy", "")
        
        return rep, target, filename

class DoubleConv(nn.Module):
    """Two 3x3 convolutions + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=128, out_channels=1, features=[128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        curr_in_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in_channels, feature))
            curr_in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
        
        # Final conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
            
        return self.final_conv(x)

# ====================== ViT Components ======================

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(self, img_size=256, patch_size=16, in_channels=128, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Use Conv2d for patch embedding - preserves the 128 channels
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # (B, 128, 256, 256) -> (B, embed_dim, n_patches_h, n_patches_w)
        x = self.proj(x)
        # (B, embed_dim, n_patches_h, n_patches_w) -> (B, embed_dim, n_patches)
        x = x.flatten(2)
        # (B, embed_dim, n_patches) -> (B, n_patches, embed_dim)
        x = x.transpose(1, 2)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""
    def __init__(self, dim, n_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """MLP module"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP"""
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads=n_heads, qkv_bias=qkv_bias, 
                                     attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ConvDecoder(nn.Module):
    """Convolutional decoder for dense prediction"""
    def __init__(self, embed_dim, decoder_embed_dim, decoder_depth, patch_size, img_size, out_channels=1):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # Calculate the number of upsampling layers needed
        # Starting from patches: img_size // patch_size
        initial_size = img_size // patch_size  # 256 // 16 = 16
        scale_factor = img_size // initial_size  # 256 // 16 = 16
        num_upsample = int(math.log2(scale_factor))  # log2(16) = 4
        
        # Ensure we have enough decoder blocks
        decoder_depth = max(decoder_depth, num_upsample)
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(decoder_depth):
            in_channels = decoder_embed_dim if i == 0 else decoder_embed_dim // (2**i)
            out_channels_block = decoder_embed_dim // (2**(i+1)) if i < decoder_depth - 1 else out_channels
            
            if i < num_upsample:
                # Upsampling block
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels_block, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels_block),
                    nn.ReLU(inplace=True) if i < decoder_depth - 1 else nn.Identity(),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
            else:
                # Final conv blocks without upsampling
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels_block, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels_block) if i < decoder_depth - 1 else nn.Identity(),
                    nn.ReLU(inplace=True) if i < decoder_depth - 1 else nn.Identity()
                )
            
            self.decoder_blocks.append(block)
        
        self.patch_size = patch_size
        self.img_size = img_size
        
    def forward(self, x, H, W):
        # x: (B, N, embed_dim)
        x = self.decoder_embed(x)  # (B, N, decoder_embed_dim)
        
        # Reshape to 2D
        B, N, C = x.shape
        h = H // self.patch_size
        w = W // self.patch_size
        x = x.transpose(1, 2).reshape(B, C, h, w)  # (B, C, h, w)
        
        # Progressive upsampling through decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        
        # Ensure output size matches target
        if x.shape[-1] != H or x.shape[-2] != W:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            
        return x

class VisionTransformerDense(nn.Module):
    """Vision Transformer for dense prediction"""
    def __init__(self, 
                 img_size=256, 
                 patch_size=16, 
                 in_channels=128, 
                 out_channels=1,
                 embed_dim=768, 
                 depth=12, 
                 n_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 decoder_embed_dim=512,
                 decoder_depth=4):
        super().__init__()
        
        self.img_size = img_size
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Decoder with img_size parameter
        self.decoder = ConvDecoder(embed_dim, decoder_embed_dim, decoder_depth, patch_size, img_size, out_channels)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize position embedding
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        
        # Initialize other layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        
        # Decode to dense prediction
        x = self.decoder(x, H, W)
        
        return x

def get_model(model_type='unet', **kwargs):
    """Factory function to get model based on type"""
    if model_type == 'unet':
        return UNet(
            in_channels=kwargs.get('in_channels', 128),
            out_channels=kwargs.get('out_channels', 1),
            features=kwargs.get('features', [128, 256, 512])
        )
    elif model_type == 'vit':
        return VisionTransformerDense(
            img_size=kwargs.get('img_size', 256),
            patch_size=kwargs.get('patch_size', 16),
            in_channels=kwargs.get('in_channels', 128),
            out_channels=kwargs.get('out_channels', 1),
            embed_dim=kwargs.get('embed_dim', 768),
            depth=kwargs.get('depth', 12),
            n_heads=kwargs.get('n_heads', 12),
            mlp_ratio=kwargs.get('mlp_ratio', 4.),
            qkv_bias=kwargs.get('qkv_bias', True),
            drop_rate=kwargs.get('drop_rate', 0.),
            attn_drop_rate=kwargs.get('attn_drop_rate', 0.),
            decoder_embed_dim=kwargs.get('decoder_embed_dim', 512),
            decoder_depth=kwargs.get('decoder_depth', 4)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_limited_labels(data_root, label_root, split, train_fraction, seed=42):
    """Get limited training labels"""
    rep_dir = os.path.join(data_root, split, 'representation')
    rep_files = sorted(glob.glob(os.path.join(rep_dir, '*.npy')))
    
    random.seed(seed)
    selected_rep = random.sample(rep_files, int(train_fraction * len(rep_files)))
    
    os.makedirs("splits", exist_ok=True)
    csvname = f"limited_labels_fraction_{train_fraction}.csv"
    
    with open(os.path.join("splits", csvname), mode="w", newline="") as f:
        writer = csv.writer(f)
        for file_path in selected_rep:
            writer.writerow([os.path.basename(file_path)])
    
    return train_fraction

def masked_mse_loss(pred, target):
    """MSE loss only on non-NaN values"""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return (pred * 0).sum()
    loss = torch.mean((pred[mask] - target[mask]) ** 2)
    return loss

def compute_metrics(pred, target, target_mean, target_std):
    """Compute MAE, RMSE, R2 metrics"""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Denormalize
    pred_denorm = pred * target_std + target_mean
    target_denorm = target * target_std + target_mean

    # Valid pixels only
    pred_valid = pred_denorm[mask]
    target_valid = target_denorm[mask]
    
    mae = torch.mean(torch.abs(pred_valid - target_valid))
    bias = torch.mean(pred_valid - target_valid)
    rmse = torch.sqrt(torch.mean((pred_valid - target_valid) ** 2))
    
    # R2
    pred_flat = pred_valid.view(-1)
    target_flat = target_valid.view(-1)
    ss_res = torch.sum((target_flat - pred_flat) ** 2)
    ss_tot = torch.sum((target_flat - torch.mean(target_flat)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else torch.tensor(0.0)
    
    return mae.item(), bias.item(), rmse.item(), r2.item()

class MetricsTracker:
    """Track and display training metrics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_r2': []
        }
    
    def update(self, metric_name: str, value: float):
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
    
    def get_average(self, metric_name: str) -> float:
        if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
            return np.mean(self.metrics[metric_name])
        return 0.0
    
    def display_epoch_summary(self, epoch: int, total_epochs: int):
        """Display beautiful epoch summary using Rich"""
        table = Table(title=f"Epoch {epoch}/{total_epochs} Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        table.add_row("Train Loss", f"{self.get_average('train_loss'):.4f}")
        table.add_row("Val Loss", f"{self.get_average('val_loss'):.4f}")
        table.add_row("Val MAE", f"{self.get_average('val_mae'):.4f}")
        table.add_row("Val RMSE", f"{self.get_average('val_rmse'):.4f}")
        table.add_row("Val R²", f"{self.get_average('val_r2'):.4f}")
        
        console.print(table)

def train_model(model, train_loader, val_loader, device, target_mean, target_std, 
                epochs, lr, run_id, rank, world_size, logger):
    """Enhanced training function with better logging"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    metrics_tracker = MetricsTracker()
    
    if rank == 0:
        checkpoint_dir = os.path.join("checkpoints", "downstream", run_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"best_borneo_patch_ckpt_{run_id}.pth")
        
        # Log training configuration
        logger.info("="*60)
        logger.info(f"Training Configuration:")
        logger.info(f"  - Model: {model.__class__.__name__}")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Learning Rate: {lr}")
        logger.info(f"  - Batch Size per GPU: {train_loader.batch_size}")
        logger.info(f"  - Total Batch Size: {train_loader.batch_size * world_size}")
        logger.info(f"  - Number of training samples: {len(train_loader.dataset)}")
        logger.info(f"  - Number of validation samples: {len(val_loader.dataset)}")
        logger.info("="*60)
    
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        metrics_tracker.reset()
        
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
            
        # Training phase
        model.train()
        
        if rank == 0:
            progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                TimeRemainingColumn(),
                console=console
            )
            train_task = progress.add_task(f"[cyan]Epoch {epoch}/{epochs} - Training", total=len(train_loader))
            progress.start()
        
        for batch_idx, (rep, target, _) in enumerate(train_loader):
            rep = rep.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(rep)
            loss = masked_mse_loss(output, target)
            loss.backward()
            optimizer.step()
            
            metrics_tracker.update('train_loss', loss.item())
            
            if rank == 0:
                progress.update(train_task, advance=1)
                if batch_idx % 50 == 0:
                    mae, _, rmse, r2 = compute_metrics(output, target, target_mean, target_std)
                    progress.console.print(
                        f"[dim]Batch {batch_idx}/{len(train_loader)} - "
                        f"Loss: {loss.item():.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}[/dim]"
                    )
        
        if rank == 0:
            progress.stop()
        
        # Validation phase
        model.eval()
        val_metrics = {'loss': 0.0, 'mae': 0.0, 'rmse': 0.0, 'r2': 0.0}
        
        with torch.no_grad():
            for rep, target, _ in val_loader:
                rep = rep.to(device)
                target = target.to(device)
                output = model(rep)
                loss = masked_mse_loss(output, target)
                mae, bias, rmse, r2 = compute_metrics(output, target, target_mean, target_std)
                
                val_metrics['loss'] += loss.item()
                val_metrics['mae'] += mae
                val_metrics['rmse'] += rmse
                val_metrics['r2'] += r2
        
        # Average validation metrics
        num_val = len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= num_val
            
        # Update metrics tracker
        metrics_tracker.update('val_loss', val_metrics['loss'])
        metrics_tracker.update('val_mae', val_metrics['mae'])
        metrics_tracker.update('val_rmse', val_metrics['rmse'])
        metrics_tracker.update('val_r2', val_metrics['r2'])
        
        # Synchronize validation loss across GPUs
        if world_size > 1:
            val_loss_tensor = torch.tensor(val_metrics['loss']).to(device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_metrics['loss'] = val_loss_tensor.item()
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Logging and checkpointing
        if rank == 0:
            epoch_time = time.time() - epoch_start_time
            metrics_tracker.display_epoch_summary(epoch, epochs)
            
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best checkpoint
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_metrics': val_metrics
                }
                torch.save(checkpoint, checkpoint_path)
                console.print(f"[bold green]✓ Saved new best checkpoint (Val Loss: {val_metrics['loss']:.4f})[/bold green]")

def main():
    parser = argparse.ArgumentParser(description="Optimized Borneo training script with ViT support")
    parser.add_argument("--train_fraction", type=float, default=0.05, help="Fraction of training data to use")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of data loading workers per GPU")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="Number of batches to prefetch per worker")
    parser.add_argument("--preprocessed_root", type=str, 
                      default="/shared/amdgpu/home/avsm2_f4q/code/biomassters_data/preprocessed_data",
                      help="Root directory of preprocessed data")
    
    # Model selection
    parser.add_argument("--model_type", type=str, default="vit", choices=["unet", "vit"],
                      help="Model type to use")
    
    # UNet specific args
    parser.add_argument("--unet_features", nargs='+', type=int, default=[128, 256, 512],
                      help="Feature channels for UNet encoder")
    
    # ViT specific args
    parser.add_argument("--vit_patch_size", type=int, default=16, help="Patch size for ViT")
    parser.add_argument("--vit_embed_dim", type=int, default=768, help="Embedding dimension for ViT")
    parser.add_argument("--vit_depth", type=int, default=12, help="Number of transformer blocks")
    parser.add_argument("--vit_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--vit_mlp_ratio", type=float, default=4., help="MLP hidden dim ratio")
    parser.add_argument("--vit_drop_rate", type=float, default=0., help="Dropout rate")
    parser.add_argument("--vit_attn_drop_rate", type=float, default=0., help="Attention dropout rate")
    parser.add_argument("--vit_decoder_embed_dim", type=int, default=512, help="Decoder embedding dimension")
    parser.add_argument("--vit_decoder_depth", type=int, default=4, help="Decoder depth")
    
    args = parser.parse_args()
    
    # Setup distributed training
    is_distributed, rank, world_size, gpu = setup_distributed()
    
    if is_distributed:
        device = torch.device(f'cuda:{gpu}')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup logging
    run_id = f"{args.model_type}_fraction_{args.train_fraction:.5f}_optimized"
    log_dir = os.path.join("logs", run_id)
    logger = setup_logger(rank, log_dir)
    
    # Data paths
    data_root = "/shared/amdgpu/home/avsm2_f4q/code/biomassters_data/data"
    label_root = "/shared/amdgpu/home/avsm2_f4q/code/biomassters_data/labels"
    results_path = "/shared/amdgpu/home/avsm2_f4q/code/biomassters_data/results"
    pred_path = "/shared/amdgpu/home/avsm2_f4q/code/biomassters_data/predictions"
    
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(pred_path, exist_ok=True)
    
    if rank == 0:
        console.print(f"[bold cyan]Starting training with configuration:[/bold cyan]")
        console.print(f"  - Model type: {args.model_type}")
        console.print(f"  - Train fraction: {args.train_fraction}")
        console.print(f"  - Batch size per GPU: {args.batch_size}")
        console.print(f"  - Total batch size: {args.batch_size * world_size}")
        console.print(f"  - Workers per GPU: {args.num_workers}")
        console.print(f"  - Total workers: {args.num_workers * world_size}")
        console.print(f"  - Prefetch factor: {args.prefetch_factor}")
        
        if args.model_type == "vit":
            console.print(f"\n[bold yellow]ViT Configuration:[/bold yellow]")
            console.print(f"  - Patch size: {args.vit_patch_size}")
            console.print(f"  - Embedding dim: {args.vit_embed_dim}")
            console.print(f"  - Depth: {args.vit_depth}")
            console.print(f"  - Heads: {args.vit_heads}")
            console.print(f"  - MLP ratio: {args.vit_mlp_ratio}")
    
    # Get limited labels if needed
    if args.train_fraction < 1.0:
        get_limited_labels(data_root, label_root, 'train_agbm', args.train_fraction)
    
    # Compute target statistics
    if rank == 0:
        train_label_files = sorted(glob.glob(os.path.join(label_root, 'train_agbm', '*.npy')))
        if args.train_fraction < 1.0:
            csv_file = os.path.join("splits", f"limited_labels_fraction_{args.train_fraction}.csv")
            if os.path.exists(csv_file):
                selected_files = set()
                with open(csv_file, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        selected_files.add(row[0])
                train_label_files = [f for f in train_label_files if os.path.basename(f) in selected_files]
        
        target_mean, target_std = compute_target_stats(train_label_files)
        logger.info(f"Target statistics - Mean: {target_mean:.4f}, Std: {target_std:.4f}")
    else:
        target_mean, target_std = 0.0, 1.0
    
    # Broadcast statistics
    if is_distributed:
        stats_tensor = torch.tensor([target_mean, target_std]).to(device)
        dist.broadcast(stats_tensor, src=0)
        target_mean, target_std = stats_tensor[0].item(), stats_tensor[1].item()
    
    # Create datasets
    train_dataset = BorneoDatasetOptimized(args.preprocessed_root, label_root, 'train_agbm', target_mean, target_std)
    val_dataset = BorneoDatasetOptimized(args.preprocessed_root, label_root, 'val_agbm', target_mean, target_std)
    test_dataset = BorneoDatasetOptimized(args.preprocessed_root, label_root, 'test_agbm', target_mean, target_std)
    
    # Create data loaders with optimized settings
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
    
    # Optimized DataLoader settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True
    )
    
    # Create model
    model_kwargs = {
        'in_channels': 128,
        'out_channels': 1,
    }
    
    if args.model_type == 'unet':
        model_kwargs['features'] = args.unet_features
    elif args.model_type == 'vit':
        model_kwargs.update({
            'img_size': 256,
            'patch_size': args.vit_patch_size,
            'embed_dim': args.vit_embed_dim,
            'depth': args.vit_depth,
            'n_heads': args.vit_heads,
            'mlp_ratio': args.vit_mlp_ratio,
            'qkv_bias': True,
            'drop_rate': args.vit_drop_rate,
            'attn_drop_rate': args.vit_attn_drop_rate,
            'decoder_embed_dim': args.vit_decoder_embed_dim,
            'decoder_depth': args.vit_decoder_depth,
        })
    
    model = get_model(args.model_type, **model_kwargs).to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[gpu])
    
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        console.print(f"[bold green]Model initialized: {args.model_type.upper()} with {num_params:,} parameters[/bold green]")
    
    # Train model
    train_model(
        model, train_loader, val_loader, device, 
        target_mean, target_std, args.epochs, 
        args.learning_rate, run_id, rank, world_size, logger
    )
    
    # Test evaluation (only on rank 0)
    if rank == 0:
        console.rule("[bold cyan]Test Evaluation[/bold cyan]")
        
        # Load best checkpoint
        checkpoint_path = os.path.join("checkpoints", "downstream", run_id, f"best_borneo_patch_ckpt_{run_id}.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint["model_state_dict"])
            console.print(f"[green]Loaded best checkpoint from epoch {checkpoint['epoch']}[/green]")
        
        # Test evaluation
        model.eval()
        test_metrics = {'loss': 0.0, 'mae': 0.0, 'bias': 0.0, 'rmse': 0.0, 'r2': 0.0}
        
        console.print(f"Evaluating on {len(test_loader.dataset)} test samples...")
        
        with torch.no_grad():
            for i, (rep, target, filenames) in enumerate(tqdm(test_loader, desc="Testing")):
                rep = rep.to(device)
                target = target.to(device)
                output = model(rep)
                
                # Save predictions if needed
                if args.train_fraction == 0.1:
                    preds = output * target_std + target_mean
                    preds_np = preds.cpu().numpy()
                    
                    for j in range(preds_np.shape[0]):
                        base_name = filenames[j]
                        out_path = os.path.join(pred_path, run_id, f"{base_name}.npy")
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        np.save(out_path, preds_np[j, 0])
                
                loss = masked_mse_loss(output, target)
                mae, bias, rmse, r2 = compute_metrics(output, target, target_mean, target_std)
                
                test_metrics['loss'] += loss.item()
                test_metrics['mae'] += mae
                test_metrics['bias'] += bias
                test_metrics['rmse'] += rmse
                test_metrics['r2'] += r2
        
        # Average test metrics
        num_test = len(test_loader)
        for key in test_metrics:
            test_metrics[key] /= num_test
        
        # Display test results
        test_table = Table(title="Test Results", show_header=True, header_style="bold magenta")
        test_table.add_column("Metric", style="cyan", no_wrap=True)
        test_table.add_column("Value", style="green")
        
        test_table.add_row("Loss", f"{test_metrics['loss']:.4f}")
        test_table.add_row("MAE", f"{test_metrics['mae']:.4f}")
        test_table.add_row("Mean Bias", f"{test_metrics['bias']:.4f}")
        test_table.add_row("RMSE", f"{test_metrics['rmse']:.4f}")
        test_table.add_row("R²", f"{test_metrics['r2']:.4f}")
        
        console.print(test_table)
        
        # Save test results
        csv_file = os.path.join(results_path, f"test_metrics_{args.model_type}_fraction_{args.train_fraction}_optimized.csv")
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for metric, value in test_metrics.items():
                writer.writerow([metric.upper(), value])
        
        console.print(f"[green]Test results saved to {csv_file}[/green]")
    
    # Cleanup
    cleanup_distributed()
    if rank == 0:
        console.print("[bold green]Training completed successfully![/bold green]")

if __name__ == "__main__":
    main()