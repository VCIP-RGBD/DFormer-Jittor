#!/usr/bin/env python3
"""
Basic test script for DFormer Jittor implementation
"""

import os
import sys
import jittor as jt
import numpy as np

# Set Jittor to use CPU for testing
jt.flags.use_cuda = 0

def test_model_creation():
    """Test basic model creation."""
    print("Testing model creation...")
    
    try:
        from models import build_model
        
        # Create a simple config
        class Config:
            backbone = 'DFormer-Base'
            decoder = 'ham'
            decoder_embed_dim = 512
            num_classes = 40
            drop_path_rate = 0.1
            aux_rate = 0.0
        
        config = Config()
        model = build_model(config)
        
        print("✓ Model creation successful")
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_model_forward():
    """Test model forward pass."""
    print("Testing model forward pass...")
    
    try:
        from models import build_model
        
        # Create config
        class Config:
            backbone = 'DFormer-Base'
            decoder = 'ham'
            decoder_embed_dim = 512
            num_classes = 40
            drop_path_rate = 0.1
            aux_rate = 0.0
        
        config = Config()
        model = build_model(config)
        model.eval()
        
        # Create synthetic input
        batch_size = 2
        height = 480
        width = 640
        
        rgb_input = jt.randn(batch_size, 3, height, width)
        depth_input = jt.randn(batch_size, 3, height, width)
        
        # Forward pass
        with jt.no_grad():
            output = model(rgb_input, depth_input)
        
        # Check output shape
        expected_shape = (batch_size, 40, height, width)
        if output.shape == expected_shape:
            print(f"✓ Forward pass successful, output shape: {output.shape}")
            return True
        else:
            print(f"✗ Forward pass failed, expected shape {expected_shape}, got {output.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation():
    """Test loss computation."""
    print("Testing loss computation...")
    
    try:
        from models import build_model
        
        # Create config
        class Config:
            backbone = 'DFormer-Base'
            decoder = 'ham'
            decoder_embed_dim = 512
            num_classes = 40
            drop_path_rate = 0.1
            aux_rate = 0.0
        
        config = Config()
        model = build_model(config)
        model.train()
        
        # Create synthetic input and target
        batch_size = 2
        height = 480
        width = 640
        
        rgb_input = jt.randn(batch_size, 3, height, width)
        depth_input = jt.randn(batch_size, 3, height, width)
        target = jt.randint(0, 40, (batch_size, height, width))
        
        # Forward pass with loss
        loss = model(rgb_input, depth_input, target)
        
        if isinstance(loss, jt.Var) and loss.ndim == 0:
            print(f"✓ Loss computation successful, loss value: {loss.item():.4f}")
            return True
        else:
            print(f"✗ Loss computation failed, unexpected loss type or shape")
            return False
            
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_backbones():
    """Test different backbone configurations."""
    print("Testing different backbones...")
    
    backbones = ['DFormer-Tiny', 'DFormer-Small', 'DFormer-Base']
    
    for backbone in backbones:
        try:
            from models import build_model
            
            class Config:
                backbone = backbone
                decoder = 'ham'
                decoder_embed_dim = 512
                num_classes = 40
                drop_path_rate = 0.1
                aux_rate = 0.0
            
            config = Config()
            model = build_model(config)
            
            # Test forward pass
            rgb_input = jt.randn(1, 3, 480, 640)
            depth_input = jt.randn(1, 3, 480, 640)
            
            with jt.no_grad():
                output = model(rgb_input, depth_input)
            
            print(f"✓ {backbone} test passed, output shape: {output.shape}")
            
        except Exception as e:
            print(f"✗ {backbone} test failed: {e}")
            return False
    
    return True


def test_data_transforms():
    """Test data transforms."""
    print("Testing data transforms...")
    
    try:
        from utils.transforms import Compose, ToTensor, Normalize
        
        # Create transforms
        transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Apply transform
        transformed = transform(dummy_image)
        
        if isinstance(transformed, jt.Var) and transformed.shape == (3, 480, 640):
            print("✓ Data transforms test passed")
            return True
        else:
            print(f"✗ Data transforms test failed, unexpected shape: {transformed.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Data transforms test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("DFormer Jittor Basic Tests")
    print("=" * 60)
    
    tests = [
        test_model_creation,
        test_model_forward,
        test_loss_computation,
        test_different_backbones,
        test_data_transforms,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! DFormer Jittor is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
