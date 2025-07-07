"""
Validation and evaluation utilities for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import os
import cv2
import numpy as np
import jittor as jt
from jittor import nn

from utils.metric import SegmentationMetric
from utils.transforms import pad_image_size_to_multiples_of


def evaluate(model, data_loader, device=None, verbose=False):
    """Evaluate model on validation dataset."""
    model.eval()
    
    metric = SegmentationMetric(data_loader.dataset.num_classes)
    
    with jt.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            if isinstance(images, (list, tuple)):
                # Multi-modal input (RGB + depth/modal)
                rgb, modal = images
                outputs = model(rgb, modal)
            else:
                # Single modal input
                outputs = model(images)
            
            if isinstance(outputs, dict):
                outputs = outputs['out']
            
            # Get predictions
            predictions = jt.argmax(outputs, dim=1)
            
            # Update metrics
            metric.update(predictions.numpy(), targets.numpy())
            
            if verbose and i % 100 == 0:
                print(f"Processed {i}/{len(data_loader)} batches")
    
    # Calculate metrics
    results = metric.get_results()
    
    return results


def evaluate_msf(model, data_loader, scales=[1.0], flip=False, device=None, verbose=False):
    """Evaluate model with multi-scale and flip augmentation."""
    model.eval()
    
    metric = SegmentationMetric(data_loader.dataset.num_classes)
    
    with jt.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            if isinstance(images, (list, tuple)):
                rgb, modal = images
                batch_size, _, h, w = rgb.shape
            else:
                rgb = images
                modal = None
                batch_size, _, h, w = rgb.shape
            
            # Initialize prediction accumulator
            pred_logits = jt.zeros((batch_size, data_loader.dataset.num_classes, h, w))
            
            # Multi-scale evaluation
            for scale in scales:
                if scale != 1.0:
                    # Resize images
                    new_h, new_w = int(h * scale), int(w * scale)
                    rgb_scaled = jt.nn.interpolate(rgb, size=(new_h, new_w), mode='bilinear', align_corners=True)
                    if modal is not None:
                        modal_scaled = jt.nn.interpolate(modal, size=(new_h, new_w), mode='bilinear', align_corners=True)
                else:
                    rgb_scaled = rgb
                    modal_scaled = modal
                
                # Forward pass
                if modal_scaled is not None:
                    outputs = model(rgb_scaled, modal_scaled)
                else:
                    outputs = model(rgb_scaled)
                
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                
                # Resize back to original size
                if scale != 1.0:
                    outputs = jt.nn.interpolate(outputs, size=(h, w), mode='bilinear', align_corners=True)
                
                pred_logits += outputs
                
                # Flip augmentation
                if flip:
                    # Horizontal flip
                    rgb_flipped = jt.flip(rgb_scaled, dims=[3])
                    if modal_scaled is not None:
                        modal_flipped = jt.flip(modal_scaled, dims=[3])
                        outputs_flipped = model(rgb_flipped, modal_flipped)
                    else:
                        outputs_flipped = model(rgb_flipped)
                    
                    if isinstance(outputs_flipped, dict):
                        outputs_flipped = outputs_flipped['out']
                    
                    # Flip back and resize
                    outputs_flipped = jt.flip(outputs_flipped, dims=[3])
                    if scale != 1.0:
                        outputs_flipped = jt.nn.interpolate(outputs_flipped, size=(h, w), mode='bilinear', align_corners=True)
                    
                    pred_logits += outputs_flipped
            
            # Average predictions
            num_augs = len(scales) * (2 if flip else 1)
            pred_logits /= num_augs
            
            # Get final predictions
            predictions = jt.argmax(pred_logits, dim=1)
            
            # Update metrics
            metric.update(predictions.numpy(), targets.numpy())
            
            if verbose and i % 100 == 0:
                print(f"Processed {i}/{len(data_loader)} batches")
    
    # Calculate metrics
    results = metric.get_results()
    
    return results


def sliding_window_inference(model, image, modal=None, window_size=(512, 512), stride=(256, 256), num_classes=40):
    """Perform sliding window inference for large images."""
    batch_size, channels, height, width = image.shape
    
    # Pad image to ensure it can be evenly divided
    pad_h = (window_size[0] - height % window_size[0]) % window_size[0]
    pad_w = (window_size[1] - width % window_size[1]) % window_size[1]
    
    if pad_h > 0 or pad_w > 0:
        image = jt.nn.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
        if modal is not None:
            modal = jt.nn.pad(modal, (0, pad_w, 0, pad_h), mode='reflect')
    
    new_height, new_width = image.shape[2], image.shape[3]
    
    # Initialize prediction map
    pred_map = jt.zeros((batch_size, num_classes, new_height, new_width))
    count_map = jt.zeros((batch_size, 1, new_height, new_width))
    
    # Sliding window
    for y in range(0, new_height - window_size[0] + 1, stride[0]):
        for x in range(0, new_width - window_size[1] + 1, stride[1]):
            # Extract window
            img_window = image[:, :, y:y+window_size[0], x:x+window_size[1]]
            if modal is not None:
                modal_window = modal[:, :, y:y+window_size[0], x:x+window_size[1]]
                pred_window = model(img_window, modal_window)
            else:
                pred_window = model(img_window)
            
            if isinstance(pred_window, dict):
                pred_window = pred_window['out']
            
            # Add to prediction map
            pred_map[:, :, y:y+window_size[0], x:x+window_size[1]] += pred_window
            count_map[:, :, y:y+window_size[0], x:x+window_size[1]] += 1
    
    # Average overlapping predictions
    pred_map = pred_map / count_map
    
    # Remove padding
    if pad_h > 0 or pad_w > 0:
        pred_map = pred_map[:, :, :height, :width]
    
    return pred_map


def test_single_image(model, image_path, modal_path=None, output_path=None, config=None):
    """Test model on a single image."""
    model.eval()
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load modal image if provided
    modal = None
    if modal_path:
        modal = cv2.imread(modal_path)
        if modal is not None:
            modal = cv2.cvtColor(modal, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    if config:
        # Apply normalization
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(config.norm_mean)) / np.array(config.norm_std)
        
        if modal is not None:
            modal = modal.astype(np.float32) / 255.0
            modal = (modal - np.array(config.norm_mean)) / np.array(config.norm_std)
    
    # Convert to tensor and add batch dimension
    image = jt.array(image.transpose(2, 0, 1)).unsqueeze(0).float32()
    if modal is not None:
        modal = jt.array(modal.transpose(2, 0, 1)).unsqueeze(0).float32()
    
    with jt.no_grad():
        if modal is not None:
            output = model(image, modal)
        else:
            output = model(image)
        
        if isinstance(output, dict):
            output = output['out']
        
        prediction = jt.argmax(output, dim=1).squeeze(0).numpy()
    
    # Save result if output path provided
    if output_path:
        # Convert prediction to color map if needed
        # This would require a color palette mapping
        cv2.imwrite(output_path, prediction.astype(np.uint8))
    
    return prediction
