#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import jittor as jt
from jittor import nn
import jittor.optim as optim
from jittor.dataset import Dataset

from models.builder import build_model
from utils.metrics import SegmentationMetric
from utils.logger import setup_logger


class RGBDDataset(Dataset):
    """RGBD Dataset for semantic segmentation."""
    
    def __init__(self, data_dir, split='train', transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load data paths
        self.rgb_dir = os.path.join(data_dir, 'RGB')
        self.depth_dir = os.path.join(data_dir, 'Depth')
        self.label_dir = os.path.join(data_dir, 'Label')
        
        # Load file list
        split_file = os.path.join(data_dir, f'{split}.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.file_list = [line.strip() for line in f.readlines()]
        else:
            # Use dummy data if split file doesn't exist
            self.file_list = [f'dummy_{i}' for i in range(100)]
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        
        # TODO: Load actual RGB, depth and label images
        # For now using dummy data
        rgb = jt.array(np.random.randn(3, 480, 640)).float32()
        depth = jt.array(np.random.randn(1, 480, 640)).float32()
        label = jt.array(np.random.randint(0, 40, (480, 640))).int32()
        
        return rgb, depth, label


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DFormer model')
    parser.add_argument('--data_dir', type=str, default='datasets/NYUDepthv2')
    parser.add_argument('--backbone', type=str, default='DFormer-Base')
    parser.add_argument('--decoder', type=str, default='ham')
    parser.add_argument('--num_classes', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--val_interval', type=int, default=25)
    
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, criterion, epoch, logger, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (rgb, depth, label) in enumerate(pbar):
        # Forward pass
        loss = model(rgb, depth, label)
        
        # Backward pass
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Log
        if batch_idx % args.log_interval == 0:
            logger.info(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {avg_loss:.4f}')
    
    return avg_loss


def validate(model, dataloader, criterion, logger, args):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    metric = SegmentationMetric(num_classes=args.num_classes)
    
    with jt.no_grad():
        for rgb, depth, label in tqdm(dataloader, desc='Validation'):
            # Forward pass
            pred = model(rgb, depth)
            loss = criterion(pred, label)
            
            # Update metrics
            total_loss += loss.item()
            pred_label = pred.argmax(dim=1)
            metric.update(pred_label, label)
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    miou = metric.get_miou()
    acc = metric.get_pixel_acc()
    
    logger.info(f'Validation Loss: {avg_loss:.4f}, mIoU: {miou:.4f}, Acc: {acc:.4f}')
    
    return avg_loss, miou, acc


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logger
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger('train', os.path.join(args.save_dir, 'train.log'))
    logger.info(f'Training with args: {args}')
    
    # Build model
    model = build_model(
        backbone=args.backbone,
        decoder=args.decoder,
        num_classes=args.num_classes,
        decoder_embed_dim=512
    )
    
    print(f'Model built: {args.backbone} + {args.decoder}')
    
    # Build datasets
    train_dataset = RGBDDataset(args.data_dir, split='train')
    val_dataset = RGBDDataset(args.data_dir, split='test')
    
    train_loader = train_dataset.set_attrs(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = val_dataset.set_attrs(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Build optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # Training loop
    best_miou = 0.0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch, logger, args)
        
        # Validate
        if epoch % args.val_interval == 0:
            val_loss, miou, acc = validate(model, val_loader, criterion, logger, args)
            
            # Save best model
            if miou > best_miou:
                best_miou = miou
                model_path = os.path.join(args.save_dir, 'best_model.pkl')
                model.save(model_path)
                logger.info(f'Best model saved with mIoU: {best_miou:.4f}')
        
        # Save checkpoint
        if epoch % 50 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'epoch_{epoch}.pkl')
            model.save(checkpoint_path)
    
    # Training finished
    total_time = time.time() - start_time
    logger.info(f'Training finished in {total_time:.2f}s, best mIoU: {best_miou:.4f}')


if __name__ == '__main__':
    main()