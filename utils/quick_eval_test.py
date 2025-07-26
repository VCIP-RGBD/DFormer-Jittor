#!/usr/bin/env python3
"""
Quick evaluation test to verify the fix
"""

import os
import sys
import time
import numpy as np
import jittor as jt
from jittor import nn

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from importlib import import_module
from utils.dataloader.dataloader import get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.metric import SegmentationMetric
from utils.val_mm import evaluate
from utils.jt_utils import load_model
from utils.engine.engine import Engine
from models import build_model

def quick_test():
    """Quick test with a few samples."""
    print("Starting quick evaluation test...")
    
    # Load config
    config = getattr(import_module('local_configs.NYUDepthv2.DFormer_Large'), "C")
    
    # Set device
    jt.flags.use_cuda = 1
    
    # Create engine (simplified for evaluation)
    engine = Engine()
    
    # Create data loader with smaller batch size for quick test
    val_loader, val_sampler = get_val_loader(
        engine=engine,
        dataset_cls=RGBXDataset,
        config=config,
        val_batch_size=1
    )
    
    print(f"Dataset: {config.dataset_name}")
    print(f"Number of classes: {val_loader.dataset.num_classes}")
    print(f"Total samples: {len(val_loader.dataset)}")
    
    # Create model
    model = build_model(config)
    
    # Load checkpoint
    checkpoint_path = "checkpoints/trained/NYUv2_DFormer_Large.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model = load_model(model, checkpoint_path)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return
    
    model.eval()
    
    # Test on first 10 samples
    print("Testing on first 10 samples...")
    start_time = time.time()
    
    metric = SegmentationMetric(val_loader.dataset.num_classes)
    
    with jt.no_grad():
        for i, minibatch in enumerate(val_loader):
            if i >= 10:  # Only test first 10 samples
                break
                
            try:
                images = minibatch['data']
                labels = minibatch['label']
                modal_xs = minibatch['modal_x']
                
                print(f"Sample {i+1}: RGB shape: {images.shape}, Depth shape: {modal_xs.shape}, Label shape: {labels.shape}")
                
                # Forward pass
                outputs = model(images, modal_xs)
                
                # Handle model output format
                if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
                    outputs = outputs[0]
                    if isinstance(outputs, list):
                        outputs = outputs[0]
                elif isinstance(outputs, dict):
                    outputs = outputs['out']
                
                predictions = jt.argmax(outputs, dim=1)

                # Handle Jittor argmax which returns tuple
                if isinstance(predictions, tuple):
                    predictions = predictions[0]

                # Update metrics
                pred_numpy = predictions.numpy()
                label_numpy = labels.numpy()
                metric.update(pred_numpy, label_numpy)
                
                print(f"  Prediction shape: {predictions.shape}")
                print(f"  Unique predictions: {np.unique(pred_numpy)[:10]}...")  # Show first 10 unique values
                print(f"  Unique labels: {np.unique(label_numpy)[:10]}...")
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    
    end_time = time.time()
    
    # Get results
    results = metric.get_results()
    
    # Print results
    print("\nQuick Test Results (10 samples):")
    print("-" * 50)
    print(f"mIoU: {results['mIoU']:.4f}")
    print(f"mAcc: {results['mAcc']:.4f}")
    print(f"Overall Acc: {results['Overall_Acc']:.4f}")
    print(f"FWIoU: {results['FWIoU']:.4f}")
    print(f"Test time: {end_time - start_time:.2f}s")
    
    # Print some per-class IoU for debugging
    print("\nFirst 10 class IoUs:")
    for i in range(min(10, len(results['IoU_per_class']))):
        print(f"Class {i:2d}: {results['IoU_per_class'][i]:.4f}")
    
    return results

if __name__ == '__main__':
    quick_test()
