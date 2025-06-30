import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import logging
from datetime import datetime
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import gc

# torchrun --nnodes=1 --nproc_per_node=8 train_presto_repr.py

# =============================================================================
# 1. Environment and Seed Setup
# =============================================================================

# Set environment variable for matplotlib
os.environ["QT_QPA_PLATFORM"] = "offscreen"

def set_seed(seed=42):
    """Sets random seeds for CPU and GPU to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clear_memory(device):
    """Clear GPU memory cache"""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(rank)
        return rank, world_size, True
    else:
        return 0, 1, False

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

# =============================================================================
# 2. Data Handling and Loading
# =============================================================================

def read_dataset_split_from_csv(csv_path: str):
    """
    Reads dataset split information from a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing patch names and their split (train/val/test).
        
    Returns:
        A tuple containing lists of train, validation, and test patch names.
    """
    train_files, val_files, test_files = [], [], []
    logging.info(f"Reading dataset split from: {csv_path}")
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            # Skip header if your CSV has one
            # next(reader, None)  
            for row in reader:
                if len(row) >= 2:
                    patch_name = row[0]
                    split = row[1].strip().lower()
                    
                    if split == 'train':
                        train_files.append(patch_name)
                    elif split == 'val':
                        val_files.append(patch_name)
                    elif split == 'test':
                        test_files.append(patch_name)
    except FileNotFoundError:
        logging.error(f"Split CSV file not found at: {csv_path}")
        raise
        
    logging.info(f"Loaded split from CSV: Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    return train_files, val_files, test_files

class PatchSegmentationDataset(Dataset):
    """
    Dataset class for pixel-wise segmentation with f32 representation files.
    - Reads file list from CSV.
    - Directly loads f32 representation (no dequantization needed).
    - Handles NaN values by replacing them with 0.
    """
    def __init__(self, patch_names, rep_dir, label_dir, label_map, ignore_label=0):
        self.patch_names = patch_names
        self.rep_dir = rep_dir
        self.label_dir = label_dir
        self.label_map = label_map
        self.ignore_label = ignore_label
        self.ignore_index = -100  # Standard PyTorch ignore index for loss functions

        self.valid_patch_names = []
        logging.info(f"Checking {len(self.patch_names)} patch files for valid labels...")
        
        # Pre-filter patches that do not contain any valid labels
        for name in tqdm(self.patch_names, desc="Filtering patches"):
            try:
                label_path = os.path.join(self.label_dir, f"{name}.npy")
                label_patch_original = np.load(label_path)
                
                # Keep the patch if it contains any label other than the ignore_label
                if np.any(label_patch_original != self.ignore_label):
                    self.valid_patch_names.append(name)
            except Exception as e:
                logging.error(f"Error checking patch {name}: {e}")
                continue
                
        logging.info(f"Kept {len(self.valid_patch_names)} patches with valid labels for segmentation.")

    def __len__(self):
        return len(self.valid_patch_names)

    def __getitem__(self, idx):
        patch_name = self.valid_patch_names[idx]
        
        # Construct file paths
        rep_file_path = os.path.join(self.rep_dir, f"{patch_name}.npy")
        label_file_path = os.path.join(self.label_dir, f"{patch_name}.npy")
        
        # Load representation data (already in f32 format)
        rep_patch = np.load(rep_file_path)  # Shape (H, W, C)
        label_patch = np.load(label_file_path)  # Shape (H, W)
        
        # Handle NaN values in representation by replacing with 0
        if np.any(np.isnan(rep_patch)):
            # logging.warning(f"Found NaN values in representation patch {patch_name}, replacing with 0")
            rep_patch = np.nan_to_num(rep_patch, nan=0.0)
        
        # Handle NaN values in labels by replacing with ignore_label
        if np.any(np.isnan(label_patch)):
            # logging.warning(f"Found NaN values in label patch {patch_name}, replacing with ignore_label")
            label_patch = np.nan_to_num(label_patch, nan=self.ignore_label)
        
        # Check for infinite values as well
        if np.any(np.isinf(rep_patch)):
            # logging.warning(f"Found infinite values in representation patch {patch_name}, clipping to finite range")
            rep_patch = np.clip(rep_patch, -1e6, 1e6)
        
        # Convert to (C, H, W) to match PyTorch CNN input format
        rep_patch_tensor = torch.tensor(rep_patch, dtype=torch.float32).permute(2, 0, 1)
        
        # Additional check for tensor validity
        if torch.isnan(rep_patch_tensor).any() or torch.isinf(rep_patch_tensor).any():
            # logging.error(f"NaN or Inf detected in tensor for patch {patch_name} after conversion")
            rep_patch_tensor = torch.nan_to_num(rep_patch_tensor, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Map labels and handle the ignore label
        label_patch_mapped = np.full_like(label_patch, self.ignore_index, dtype=np.int64)
        for original_label, mapped_label in self.label_map.items():
            mask = (label_patch == original_label)
            label_patch_mapped[mask] = mapped_label
            
        label_patch_tensor = torch.tensor(label_patch_mapped, dtype=torch.long)
        
        return rep_patch_tensor, label_patch_tensor

# =============================================================================
# 3. Model Definitions (UNet, DepthwiseUNet, etc.)
# =============================================================================

class DoubleConv(nn.Module):
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
        
        curr_in_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in_channels, feature))
            curr_in_channels = feature
            
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
            
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
            
        return self.final_conv(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        reduction = max(1, channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x

class DepthwiseUNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_channel_attention=False, use_spatial_attention=False, channel_reduction=16, use_residual=False):
        super(DepthwiseUNetBlock, self).__init__()
        self.use_residual = use_residual
        self.residual_conv = None
        if use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.depthwise_conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.depthwise_conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.channel_attention = ChannelAttention(out_channels, channel_reduction) if use_channel_attention else None
        self.spatial_attention = SpatialAttention() if use_spatial_attention else None
    def forward(self, x):
        identity = x
        x = self.relu1(self.bn1(self.depthwise_conv1(x)))
        x = self.bn2(self.depthwise_conv2(x))
        if self.channel_attention is not None:
            x = self.channel_attention(x)
        if self.spatial_attention is not None:
            x = self.spatial_attention(x)
        if self.use_residual:
            if self.residual_conv is not None:
                identity = self.residual_conv(identity)
            x = x + identity
        x = self.relu2(x)
        return x

class DepthwiseUNet(nn.Module):
    def __init__(self, in_channels=128, out_channels=1, features=[64, 128, 256, 512], use_channel_attention=True, use_spatial_attention=False, channel_reduction=16, use_residual=True, dropout_rate=0.1, use_deep_supervision=False, use_bilinear_upsample=False):
        super(DepthwiseUNet, self).__init__()
        self.use_deep_supervision = use_deep_supervision
        self.use_bilinear_upsample = use_bilinear_upsample
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None
        
        in_channels_temp = in_channels
        for feature in features:
            self.downs.append(DepthwiseUNetBlock(in_channels_temp, feature, use_channel_attention, use_spatial_attention, channel_reduction, use_residual))
            in_channels_temp = feature
        
        self.bottleneck = DepthwiseUNetBlock(features[-1], features[-1]*2, use_channel_attention, use_spatial_attention, channel_reduction, use_residual)
        
        self.deep_outputs = nn.ModuleList() if use_deep_supervision else None
        
        for idx, feature in enumerate(reversed(features)):
            if use_bilinear_upsample:
                self.ups.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(features[-idx-1]*2 if idx == 0 else features[-idx], feature, kernel_size=1, bias=False),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True)
                ))
            else:
                self.ups.append(nn.ConvTranspose2d(features[-idx-1]*2 if idx == 0 else features[-idx], feature, kernel_size=2, stride=2))
            self.ups.append(DepthwiseUNetBlock(feature*2, feature, use_channel_attention, use_spatial_attention, channel_reduction, use_residual))
            if use_deep_supervision and idx > 0:
                self.deep_outputs.append(nn.Conv2d(feature, out_channels, kernel_size=1))
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            if self.dropout is not None:
                x = self.dropout(x)
        
        x = self.bottleneck(x)
        deep_outputs = [] if self.use_deep_supervision else None
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)
            
            if self.use_deep_supervision and idx < len(self.ups) - 2:
                deep_out = self.deep_outputs[idx // 2](x)
                deep_out = F.interpolate(deep_out, size=skip_connections[-1].shape[2:], mode='bilinear', align_corners=True)
                deep_outputs.append(deep_out)
            
            if self.dropout is not None and idx < len(self.ups) - 2:
                x = self.dropout(x)
        
        final_output = self.final_conv(x)
        
        if self.use_deep_supervision:
            return final_output, deep_outputs
        return final_output

class PatchSegmenter(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(PatchSegmenter, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.relu1(self.conv1(x))
        x2 = self.pool1(x1)
        x3 = self.relu2(self.conv2(x2))
        x4 = self.pool2(x3)
        x5 = self.relu3(self.conv3(x4))
        x6 = self.upconv2(x5)
        x7 = self.relu4(self.conv4(x6))
        x8 = self.upconv1(x7)
        x9 = self.relu5(self.conv5(x8))
        output = self.final_conv(x9)
        return output

# =============================================================================
# 4. Metrics and Main Training Function
# =============================================================================

def compute_pixel_metrics_for_batch(pred, target, ignore_index=-100):
    """Computes pixel-wise accuracy and F1 score for a single batch (for quick feedback during training)."""
    pred_labels = pred.argmax(dim=1)
    valid_mask = (target != ignore_index)
    
    if valid_mask.sum() == 0:
        return 0.0, 0.0
    
    pred_valid = pred_labels[valid_mask]
    target_valid = target[valid_mask]
    
    pred_np = pred_valid.cpu().numpy()
    target_np = target_valid.cpu().numpy()
    
    accuracy = accuracy_score(target_np, pred_np) * 100  # Convert to percentage
    f1 = f1_score(target_np, pred_np, average='weighted', zero_division=0) * 100  # Convert to percentage
    
    return accuracy, f1

def calculate_metrics(confusion_matrix):
    """Calculates accuracy, F1, and mIoU from a confusion matrix, safely handling division by zero."""
    num_classes = confusion_matrix.shape[0]
    total_samples = confusion_matrix.sum()
    if total_samples == 0:
        return {"accuracy": 0, "f1_weighted": 0, "f1_macro": 0, "mIoU": 0, "iou_per_class": np.zeros(num_classes)}

    # Overall accuracy
    accuracy = np.diag(confusion_matrix).sum() / total_samples
    
    # Per-class metrics
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    
    # Safely calculate Precision, Recall, F1-score
    precision = np.divide(tp, (tp + fp), out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, (tp + fn), out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    f1_per_class = np.divide(2 * (precision * recall), (precision + recall), out=np.zeros_like(tp, dtype=float), where=(precision + recall) != 0)
    
    # Weighted F1
    class_totals = confusion_matrix.sum(axis=1)
    f1_weighted = (f1_per_class * class_totals).sum() / total_samples
    
    # Macro F1 (unweighted average)
    f1_macro = np.mean(f1_per_class)

    # IoU and mIoU
    intersection = np.diag(confusion_matrix)
    union = confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)
    mIoU = np.mean(iou)
    
    # Convert all metrics to percentage (0-100)
    return {
        "accuracy": accuracy * 100,
        "f1_weighted": f1_weighted * 100,
        "f1_macro": f1_macro * 100,
        "mIoU": mIoU * 100,
        "iou_per_class": iou * 100,
        "precision_per_class": precision * 100,
        "recall_per_class": recall * 100,
        "f1_per_class": f1_per_class * 100
    }

def print_detailed_metrics(metrics, label_map, prefix=""):
    """Print detailed metrics including per-class information"""
    idx_to_original_label = {v: k for k, v in label_map.items()}
    
    print(f"\n{prefix} Detailed Metrics:")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Weighted F1-Score: {metrics['f1_weighted']:.2f}%")
    print(f"Macro F1-Score: {metrics['f1_macro']:.2f}%")
    print(f"Mean IoU (mIoU): {metrics['mIoU']:.2f}%")
    print(f"{'='*60}")
    
    print(f"{'Class':<15} {'IoU':<8} {'Precision':<10} {'Recall':<8} {'F1':<8}")
    print(f"{'-'*60}")
    for i, (iou, prec, rec, f1) in enumerate(zip(
        metrics['iou_per_class'], 
        metrics['precision_per_class'], 
        metrics['recall_per_class'], 
        metrics['f1_per_class']
    )):
        class_name = str(idx_to_original_label.get(i, f"Class_{i}"))
        print(f"{class_name:<15} {iou:<8.2f}% {prec:<10.2f}% {rec:<8.2f}% {f1:<8.2f}%")

def visualize_and_save_predictions(model, dataset, num_samples, device, label_map, save_dir=".", rank=0):
    """
    Randomly selects test samples, visualizes prediction results against ground truth, and saves them.
    """
    if rank != 0:
        return  # Only rank 0 saves visualizations
        
    logging.info(f"Visualizing {num_samples} random samples...")
    model.eval()
    
    num_classes = len(label_map)
    # Create a consistent colormap for visualization
    cmap = plt.get_cmap('viridis', num_classes) 
    colors = (cmap(np.linspace(0, 1, num_classes))[:, :3] * 255).astype(np.uint8)
    ignore_color = np.array([0, 0, 0], dtype=np.uint8) # black for ignored pixels

    def map_labels_to_colors(label_patch, ignore_index=-100):
        h, w = label_patch.shape
        color_image = np.zeros((h, w, 3), dtype=np.uint8)
        for class_idx, color in enumerate(colors):
            color_image[label_patch == class_idx] = color
        color_image[label_patch == ignore_index] = ignore_color
        return color_image

    # Randomly select sample indices from the dataset
    num_available_samples = len(dataset)
    if num_available_samples == 0:
        logging.warning("Test dataset is empty, cannot generate visualizations.")
        return
        
    sample_indices = random.sample(range(num_available_samples), min(num_samples, num_available_samples))

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            rep_tensor, label_tensor = dataset[idx]
            
            # Prediction
            rep_tensor = rep_tensor.unsqueeze(0).to(device) # Add batch dimension and move to device
            output = model(rep_tensor)
            if isinstance(output, tuple): # Handle deep supervision output
                output = output[0]
            
            pred_labels = torch.argmax(output, dim=1).squeeze(0).cpu().numpy() # (H, W)
            true_labels = label_tensor.cpu().numpy() # (H, W)

            # Convert to visualizable color images
            pred_color = map_labels_to_colors(pred_labels)
            true_color = map_labels_to_colors(true_labels)

            # Create and save the comparison image
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(pred_color)
            axes[0].set_title("Prediction")
            axes[0].axis('off')

            axes[1].imshow(true_color)
            axes[1].set_title("Ground Truth")
            axes[1].axis('off')

            plt.tight_layout()
            save_path = os.path.join(save_dir, f"segmentation_comparison_{i}.png")
            plt.savefig(save_path)
            plt.close(fig)
            logging.info(f"Saved comparison image to {save_path}")
            
            # Clear memory
            del rep_tensor, output, pred_labels, true_labels, pred_color, true_color

def evaluate_model(model, dataloader, criterion, device, num_classes, label_map, phase_name="Validation"):
    """Evaluate model and return detailed metrics"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        for rep_patches, label_patches in tqdm(dataloader, desc=f"{phase_name} Evaluation", leave=False):
            rep_patches, label_patches = rep_patches.to(device), label_patches.to(device)
            outputs = model(rep_patches)

            if isinstance(outputs, tuple):
                final_output, _ = outputs
                loss = criterion(final_output, label_patches)
                outputs_for_metrics = final_output
            else:
                loss = criterion(outputs, label_patches)
                outputs_for_metrics = outputs

            total_loss += loss.item()
            num_batches += 1

            # Move predictions and labels to CPU immediately to save GPU memory
            preds = outputs_for_metrics.argmax(dim=1).cpu().numpy().flatten()
            labels = label_patches.cpu().numpy().flatten()
            
            # Clear GPU tensors
            del outputs, outputs_for_metrics, rep_patches, label_patches, loss
            
            valid_mask = labels != -100
            if np.any(valid_mask):
                valid_preds = preds[valid_mask]
                valid_labels = labels[valid_mask]
                total_cm += confusion_matrix(valid_labels, valid_preds, labels=list(range(num_classes)))
            
            # Clear CPU arrays
            del preds, labels, valid_mask
    
    # Clear memory after evaluation
    clear_memory(device)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    metrics = calculate_metrics(total_cm)
    
    # Print detailed metrics
    print_detailed_metrics(metrics, label_map, prefix=phase_name)
    
    return avg_loss, metrics

def main_patch_segmentation():
    # -------------------------
    # 0. Initialization and Setup
    # -------------------------
    rank, world_size, is_distributed = setup_distributed()
    
    set_seed(42 + rank)  # Different seed for each process
    
    # Setup logging
    log_level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
        logging.info(f"Using GPU {rank} of {world_size}")
    else:
        device = torch.device('cpu')
        logging.warning("No GPU available, using CPU")
    
    # -------------------------
    # 1. Model Selection and Configuration
    # -------------------------
    model_type = "unet"  # Options: "unet", "depthwise_unet", "simple_cnn"
    
    unet_config = {"features": [128, 256, 512]}
    depthwise_unet_config = {
        "features": [64, 128, 256, 512], "use_channel_attention": True,
        "use_spatial_attention": False, "channel_reduction": 16, "use_residual": True,
        "dropout_rate": 0.1, "use_deep_supervision": False, "use_bilinear_upsample": False
    }

    # -------------------------
    # 2. Define Data Paths and Split
    # -------------------------
    REP_DIR = "/shared/amdgpu/home/avsm2_f4q/code/pangaea-bench/data/austrian_crop_v1.0_pipeline_presto_repr/representation"
    LABEL_DIR = "/shared/amdgpu/home/avsm2_f4q/code/pangaea-bench/data/austrian_crop_v1.0_pipeline_prithvi/label_patch"
    SPLIT_CSV_PATH = "/shared/amdgpu/home/avsm2_f4q/code/pangaea-bench/data/austrian_crop_v1.0_pipeline_prithvi/patchsize_32_train_ratio_0.01.csv"
    
    train_patch_names, val_patch_names, test_patch_names = read_dataset_split_from_csv(SPLIT_CSV_PATH)
    
    if not train_patch_names:
        raise ValueError("No training patch names found in CSV. Check the CSV path and content.")

    # -------------------------
    # 3. Determine Number of Classes and Label Mapping
    # -------------------------
    all_original_labels = set()
    sample_patch_names = train_patch_names[:min(100, len(train_patch_names))]
    
    if rank == 0:
        logging.info(f"Scanning {len(sample_patch_names)} training patches to determine classes...")
        
    for name in tqdm(sample_patch_names, desc="Scanning labels", disable=(rank != 0)):
        label_path = os.path.join(LABEL_DIR, f"{name}.npy")
        try:
            patch_l = np.load(label_path)
            all_original_labels.update(np.unique(patch_l))
        except FileNotFoundError:
            if rank == 0:
                logging.warning(f"Label file not found for patch {name}, skipping.")
            continue

    unique_original_nonzero_labels = sorted([l for l in all_original_labels if l != 0])
    if not unique_original_nonzero_labels:
        raise ValueError("No non-zero labels found in sample patches. Cannot determine classes.")

    label_map_for_dataset = {label: i for i, label in enumerate(unique_original_nonzero_labels)}
    num_classes = len(unique_original_nonzero_labels)

    if rank == 0:
        logging.info(f"Original non-zero labels found: {unique_original_nonzero_labels}")
        logging.info(f"Label map for segmentation: {label_map_for_dataset}")
        logging.info(f"Number of classes for pixel-wise segmentation: {num_classes}")

    # -------------------------
    # 4. Create Datasets and DataLoaders
    # -------------------------
    sample_rep_path = os.path.join(REP_DIR, f"{train_patch_names[0]}.npy")
    sample_rep_patch = np.load(sample_rep_path)  # Direct loading, already f32
    patch_h, patch_w, patch_c = sample_rep_patch.shape
    
    if rank == 0:
        logging.info(f"Detected patch properties: Size HxW=({patch_h}x{patch_w}), Channels C={patch_c}")

    train_dataset = PatchSegmentationDataset(train_patch_names, REP_DIR, LABEL_DIR, label_map_for_dataset, ignore_label=0)
    val_dataset = PatchSegmentationDataset(val_patch_names, REP_DIR, LABEL_DIR, label_map_for_dataset, ignore_label=0)
    test_dataset = PatchSegmentationDataset(test_patch_names, REP_DIR, LABEL_DIR, label_map_for_dataset, ignore_label=0)

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after filtering. Check patch files and labels.")

    # Adjust batch size for distributed training
    batch_size_per_gpu = 8
    
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size_per_gpu, sampler=train_sampler, num_workers=2, pin_memory=False,drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_per_gpu, sampler=val_sampler, num_workers=2, pin_memory=False,drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_per_gpu, sampler=test_sampler, num_workers=2, pin_memory=False,drop_last=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size_per_gpu, shuffle=True, num_workers=2, pin_memory=False,drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_per_gpu, shuffle=False, num_workers=2, pin_memory=False,drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_per_gpu, shuffle=False, num_workers=2, pin_memory=False,drop_last=False)

    if rank == 0:
        logging.info(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # -------------------------
    # 5. Model Initialization
    # -------------------------
    if model_type == "unet":
        model = UNet(in_channels=patch_c, out_channels=num_classes, **unet_config)
    elif model_type == "depthwise_unet":
        model = DepthwiseUNet(in_channels=patch_c, out_channels=num_classes, **depthwise_unet_config)
    elif model_type == "simple_cnn":
        model = PatchSegmenter(input_channels=patch_c, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model = model.to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        logging.info(f"Using {model_type} model with {param_count} parameters.")

    criterion = nn.CrossEntropyLoss(ignore_index=-100).to(device)
    lr = 0.0001 if model_type in ["unet", "depthwise_unet"] else 0.001
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # -------------------------
    # 6. Training and Validation Loop
    # -------------------------
    num_epochs = 40
    best_val_f1 = 0.0
    best_epoch = 0
    checkpoint_save_folder = f"checkpoints/patch_segmenter_{model_type}_{timestamp}/"
    
    if rank == 0:
        os.makedirs(checkpoint_save_folder, exist_ok=True)
        best_checkpoint_path = os.path.join(checkpoint_save_folder, "best_model.pt")
        logging.info(f"Starting pixel-wise segmentation training for {num_epochs} epochs.")

    for epoch in range(num_epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)  # Shuffle data differently for each epoch
            
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False, disable=(rank != 0))
        
        for rep_patches, label_patches in progress_bar:
            rep_patches, label_patches = rep_patches.to(device), label_patches.to(device)
            
            # Check for NaN in inputs
            if torch.isnan(rep_patches).any():
                logging.error(f"NaN detected in input batch at epoch {epoch+1}")
                continue
            
            optimizer.zero_grad()
            outputs = model(rep_patches)
            
            # Store whether outputs is a tuple before deletion
            is_deep_supervision = isinstance(outputs, tuple)
            
            if is_deep_supervision: # Handle deep supervision
                final_output, deep_outputs = outputs
                main_loss = criterion(final_output, label_patches)
                deep_loss = sum(criterion(deep_out, label_patches) for deep_out in deep_outputs)
                loss = main_loss + 0.5 * (deep_loss / len(deep_outputs))
                outputs_for_metrics = final_output
            else:
                loss = criterion(outputs, label_patches)
                outputs_for_metrics = outputs
            
            # Check for NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                logging.error(f"NaN or Inf loss detected at epoch {epoch+1}, batch {num_batches}")
                logging.error(f"Loss value: {loss.item()}")
                logging.error(f"Output stats - min: {outputs_for_metrics.min().item()}, max: {outputs_for_metrics.max().item()}, mean: {outputs_for_metrics.mean().item()}")
                logging.error(f"Label stats - unique values: {torch.unique(label_patches).cpu().numpy()}")
                continue
            
            loss.backward()
            
            # More aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Check for NaN in gradients
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    logging.error(f"NaN or Inf gradient detected in {name}")
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                logging.error("Skipping batch due to NaN gradients")
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1  # Move this outside the rank == 0 condition
            
            if rank == 0:
                acc, f1 = compute_pixel_metrics_for_batch(outputs_for_metrics, label_patches)
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%', 'f1': f'{f1:.2f}%'})
            
            # Clear tensors to free memory
            del rep_patches, label_patches, outputs, loss
            if is_deep_supervision:
                del final_output, deep_outputs, main_loss, deep_loss
            del outputs_for_metrics

        # Handle case where all batches were skipped
        if num_batches > 0:
            avg_train_loss = running_loss / num_batches
        else:
            avg_train_loss = float('inf')
            logging.warning(f"Rank {rank}: No valid batches in epoch {epoch+1}")
        
        # Clear memory after training
        clear_memory(device)
        
        if rank == 0:
            logging.info(f"Epoch {epoch+1} Train: Loss={avg_train_loss:.4f}")

        # Validation with detailed metrics (only on rank 0 for simplicity)
        if rank == 0:
            # Get the model without DDP wrapper for evaluation
            eval_model = model.module if is_distributed else model
            avg_val_loss, val_metrics = evaluate_model(eval_model, val_loader, criterion, device, num_classes, label_map_for_dataset, "Validation")
            
            logging.info(f"Epoch {epoch+1} VAL: Loss={avg_val_loss:.4f}, Pixel_Acc={val_metrics['accuracy']:.2f}%, Pixel_F1(w)={val_metrics['f1_weighted']:.2f}%, mIoU={val_metrics['mIoU']:.2f}%")
            
            current_val_f1 = val_metrics['f1_weighted']
            if current_val_f1 > best_val_f1:
                best_val_f1 = current_val_f1
                best_epoch = epoch + 1
                
                # Save model state without DDP wrapper
                model_state = eval_model.state_dict()
                
                torch.save({
                    'epoch': best_epoch, 'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(), 'val_f1': best_val_f1,
                    'val_metrics': val_metrics, 'label_map': label_map_for_dataset,
                    'patch_size_h': patch_h, 'patch_size_w': patch_w,
                    'input_channels_c': patch_c, 'num_classes': num_classes,
                    'model_type': model_type,
                    'model_config': unet_config if model_type == "unet" else depthwise_unet_config if model_type == "depthwise_unet" else {}
                }, best_checkpoint_path)
                logging.info(f"🚀 Best Checkpoint Saved! Epoch {best_epoch}, Val F1: {best_val_f1:.2f}%, Val mIoU: {val_metrics['mIoU']:.2f}%")        
        # Clear memory after validation
        clear_memory(device)
        
        # Synchronize all processes
        if is_distributed:
            dist.barrier()

    if rank == 0:
        logging.info(f"Training completed. Best Val F1: {best_val_f1:.2f}% at Epoch {best_epoch}")

    # -------------------------
    # 7. Final Test Evaluation
    # -------------------------
    if rank == 0 and 'best_checkpoint_path' in locals() and os.path.exists(best_checkpoint_path):
        logging.info(f"Loading best checkpoint from: {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        
        eval_model = model.module if is_distributed else model
        eval_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final test evaluation with detailed metrics
        avg_test_loss, test_metrics = evaluate_model(eval_model, test_loader, criterion, device, num_classes, label_map_for_dataset, "Final Test")
        
        logging.info(f"\n--- Final Test Summary ---")
        logging.info(f"Test Loss: {avg_test_loss:.4f}")
        logging.info(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        logging.info(f"Test Weighted F1: {test_metrics['f1_weighted']:.2f}%")
        logging.info(f"Test Macro F1: {test_metrics['f1_macro']:.2f}%")
        logging.info(f"Test mIoU: {test_metrics['mIoU']:.2f}%")

        # -----------------------------------
        # 8. Visualize Test Set Results
        # -----------------------------------
        visualize_and_save_predictions(
            model=eval_model,
            dataset=test_dataset,
            num_samples=5,
            device=device,
            label_map=label_map_for_dataset,
            save_dir=".",
            rank=rank
        )
    
    # Cleanup distributed training
    cleanup_distributed()

if __name__ == "__main__":
    main_patch_segmentation()