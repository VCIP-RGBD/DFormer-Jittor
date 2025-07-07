#!/usr/bin/env python3
"""
Test script to verify that the nn.Parameter warnings are fixed
and that the loss functions work correctly.
"""

import os
import sys
import jittor as jt
from jittor import nn

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from models.losses import FocalLoss, DiceLoss, LovaszLoss, TverskyLoss
from models.builder import build_model


def test_parameter_warnings():
    """Test if nn.Parameter warnings are fixed."""
    print("Testing Parameter warnings fix...")
    
    # Test DFormer model creation
    try:
        model = build_model(
            'DFormer-Base',
            decoder='ham',
            num_classes=40,
            decoder_embed_dim=512
        )
        print("✓ DFormer model created without errors")
        
        # Create dummy input
        rgb = jt.randn(2, 3, 240, 320)
        depth = jt.randn(2, 1, 240, 320)
        
        # Forward pass
        with jt.no_grad():
            output = model(rgb, depth)
        
        print(f"✓ Model forward pass successful, output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def test_loss_functions():
    """Test all loss functions."""
    print("\nTesting loss functions...")
    
    # Create dummy data
    batch_size, num_classes, height, width = 2, 40, 120, 160
    pred = jt.randn(batch_size, num_classes, height, width)
    target = jt.randint(0, num_classes, (batch_size, height, width))
    
    losses = {
        'FocalLoss': FocalLoss(use_sigmoid=False),
        'DiceLoss': DiceLoss(),
        'LovaszLoss': LovaszLoss(),
        'TverskyLoss': TverskyLoss()
    }
    
    all_passed = True
    
    for loss_name, loss_fn in losses.items():
        try:
            loss_value = loss_fn(pred, target)
            print(f"✓ {loss_name}: {loss_value.item():.4f}")
        except Exception as e:
            print(f"✗ {loss_name} failed: {e}")
            all_passed = False
    
    return all_passed


def test_encoders():
    """Test encoder models for parameter warnings."""
    print("\nTesting encoder models...")
    
    from models.encoders.DFormer import DFormer_Base
    from models.encoders.DFormerv2 import DFormerv2_S
    
    rgb = jt.randn(2, 3, 240, 320)
    depth = jt.randn(2, 1, 240, 320)
    
    models = {
        'DFormer_Base': DFormer_Base(),
        'DFormerv2_S': DFormerv2_S()
    }
    
    all_passed = True
    
    for model_name, model in models.items():
        try:
            with jt.no_grad():
                if 'DFormerv2' in model_name:
                    outputs = model(rgb, depth)
                else:
                    outputs, _ = model(rgb, depth)
                print(f"✓ {model_name} forward pass successful")
        except Exception as e:
            print(f"✗ {model_name} failed: {e}")
            all_passed = False
    
    return all_passed


def main():
    """Main test function."""
    print("=" * 60)
    print("Testing DFormer Jittor Implementation")
    print("Checking if nn.Parameter warnings are resolved")
    print("=" * 60)
    
    # Capture warnings to see if nn.Parameter warnings are gone
    import warnings
    warnings.filterwarnings('error', message='.*Parameter.*')
    
    results = []
    
    try:
        # Test 1: Parameter warnings
        results.append(test_parameter_warnings())
        
        # Test 2: Loss functions
        results.append(test_loss_functions())
        
        # Test 3: Encoder models
        results.append(test_encoders())
        
    except Warning as w:
        print(f"⚠️  Warning detected: {w}")
        return False
    
    # Summary
    print("\n" + "=" * 60)
    if all(results):
        print("🎉 All tests passed! nn.Parameter warnings should be resolved.")
        print("✓ Model creation successful")
        print("✓ Loss functions working")
        print("✓ Encoder models working")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 