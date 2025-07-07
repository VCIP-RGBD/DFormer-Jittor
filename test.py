#!/usr/bin/env python3
import os
import argparse
import numpy as np

import jittor as jt
from jittor import nn

from models.builder import build_model


def test_model():
    """Test if the DFormer model can be built and run forward pass."""
    print("Testing DFormer Jittor implementation...")
    
    # Test different model configurations
    configs = [
        ('DFormer-Tiny', 'ham', 40),
        ('DFormer-Base', 'ham', 40),
        ('DFormerv2_S', 'MLPDecoder', 40),
    ]
    
    for backbone, decoder, num_classes in configs:
        print(f"\nTesting {backbone} + {decoder}...")
        
        try:
            # Build model
            model = build_model(
                backbone=backbone,
                decoder=decoder,
                num_classes=num_classes,
                decoder_embed_dim=512
            )
            
            # Create dummy inputs
            rgb = jt.random((2, 3, 480, 640))  # Batch size 2
            depth = jt.random((2, 1, 480, 640))
            
            # Forward pass
            with jt.no_grad():
                output = model(rgb, depth)
                print(f"Output shape: {output.shape}")
                assert output.shape == (2, num_classes, 480, 640), f"Expected shape (2, {num_classes}, 480, 640), got {output.shape}"
            
            print(f"✓ {backbone} + {decoder} test passed!")
            
        except Exception as e:
            import traceback
            print(f"✗ {backbone} + {decoder} test failed: {e}")
            print("Traceback:")
            traceback.print_exc()
    
    print("\nModel testing completed!")


def test_training_step():
    """Test if training step works."""
    print("\nTesting training step...")
    
    try:
        # Build model
        model = build_model(
            backbone='DFormer-Base',
            decoder='ham',
            num_classes=40,
            decoder_embed_dim=512
        )
        
        # Create dummy inputs with labels
        rgb = jt.random((2, 3, 480, 640))
        depth = jt.random((2, 1, 480, 640))
        labels = jt.randint(0, 40, (2, 480, 640))
        
        # Training mode
        model.train()
        
        # Forward pass with loss computation
        loss = model(rgb, depth, labels)
        print(f"Training loss: {loss.item()}")
        
        print("✓ Training step test passed!")
        
    except Exception as e:
        import traceback
        print(f"✗ Training step test failed: {e}")
        print("Traceback:")
        traceback.print_exc()


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test DFormer Jittor implementation')
    parser.add_argument('--test_model', action='store_true', help='Test model building and forward pass')
    parser.add_argument('--test_training', action='store_true', help='Test training step')
    
    args = parser.parse_args()
    
    if args.test_model or (not args.test_model and not args.test_training):
        test_model()
    
    if args.test_training or (not args.test_model and not args.test_training):
        test_training_step()


if __name__ == '__main__':
    main() 