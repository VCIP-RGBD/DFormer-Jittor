#!/usr/bin/env python3
"""
GPU Usage Check Script for DFormer-Jittor
"""

import os
import sys
import time
import jittor as jt
from importlib import import_module

# Set environment variables
os.environ.setdefault("use_cutlass", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set Jittor optimization flags
jt.flags.use_cuda = 1
jt.flags.lazy_execution = 1
jt.flags.use_stat_allocator = 1

def check_gpu_memory():
    """Check GPU memory usage"""
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        for i, line in enumerate(lines):
            used, total = map(int, line.split(','))
            print(f"GPU {i}: {used}MB / {total}MB ({used/total*100:.1f}%)")
    except Exception as e:
        print(f"Error checking GPU memory: {e}")

def test_model_memory():
    """Test model memory usage with different batch sizes"""
    print("Testing model memory usage...")
    
    # Add current directory to path
    _CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    _ROOT_DIR = os.path.dirname(_CUR_DIR)
    sys.path.insert(0, _ROOT_DIR)
    
    from models.builder import EncoderDecoder as segmodel
    from jittor import nn
    
    # Load config
    config = getattr(import_module('local_configs.NYUDepthv2.DFormerv2_S'), "C")
    
    # Create dummy model
    criterion = nn.CrossEntropyLoss(ignore_index=config.background)
    BatchNorm2d = nn.BatchNorm2d
    
    model = segmodel(
        cfg=config,
        criterion=criterion,
        norm_layer=BatchNorm2d,
        syncbn=False,
    )
    
    print("Model created successfully")
    check_gpu_memory()
    
    # Test with different batch sizes
    for batch_size in [4, 8, 16, 32]:
        try:
            print(f"\nTesting batch size {batch_size}:")
            
            # Create dummy data
            imgs = jt.random((batch_size, 3, config.image_height, config.image_width))
            modal_xs = jt.random((batch_size, 1, config.image_height, config.image_width))
            gts = jt.randint(0, config.num_classes, (batch_size, config.image_height, config.image_width))
            
            # Forward pass
            output = model(imgs, modal_xs, gts)
            
            # Handle output format
            if isinstance(output, tuple):
                if len(output) == 2:
                    predictions, loss = output
                else:
                    loss = output[-1]
            else:
                loss = output
            
            print(f"  Forward pass successful, loss: {loss.item():.4f}")
            check_gpu_memory()
            
            # Simulate backward pass (in actual training, optimizer.step(loss) is used)
            print(f"  Backward pass would be handled by optimizer.step(loss)")
            check_gpu_memory()
            
            # Clear memory
            jt.clean()
            
        except Exception as e:
            print(f"  Error with batch size {batch_size}: {e}")
            break

if __name__ == "__main__":
    print("=== GPU Usage Check ===")
    check_gpu_memory()
    print("\n=== Model Memory Test ===")
    test_model_memory()
