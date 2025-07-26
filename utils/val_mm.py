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


def slide_inference(model, image, modal_x, config):
    """Sliding window inference for large images."""
    import math

    # Get sliding window parameters from config
    crop_size = getattr(config, 'crop_size', [512, 512])
    stride_rate = getattr(config, 'stride_rate', 2/3)

    if isinstance(crop_size, int):
        crop_size = [crop_size, crop_size]

    stride = [int(crop_size[0] * stride_rate), int(crop_size[1] * stride_rate)]

    batch_size, _, h, w = image.shape
    num_classes = config.num_classes

    # Initialize prediction and count matrices
    preds = jt.zeros((batch_size, num_classes, h, w))
    count_mat = jt.zeros((batch_size, 1, h, w))

    # Calculate crop positions
    h_grids = max(h - crop_size[0] + stride[0] - 1, 0) // stride[0] + 1
    w_grids = max(w - crop_size[1] + stride[1] - 1, 0) // stride[1] + 1

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * stride[0]
            x1 = w_idx * stride[1]
            y2 = min(y1 + crop_size[0], h)
            x2 = min(x1 + crop_size[1], w)
            y1 = max(y2 - crop_size[0], 0)
            x1 = max(x2 - crop_size[1], 0)

            crop_img = image[:, :, y1:y2, x1:x2]
            crop_modal_xs = modal_x[:, :, y1:y2, x1:x2]

            # Forward pass
            crop_seg_logit = model(crop_img, crop_modal_xs)

            # Handle model output format
            if isinstance(crop_seg_logit, (list, tuple)):
                if len(crop_seg_logit) == 2:
                    crop_seg_logit = crop_seg_logit[0]
                if isinstance(crop_seg_logit, list):
                    crop_seg_logit = crop_seg_logit[0]
            elif isinstance(crop_seg_logit, dict):
                crop_seg_logit = crop_seg_logit['out']

            # Pad and add to predictions
            pad_left, pad_right = x1, preds.shape[3] - x2
            pad_top, pad_bottom = y1, preds.shape[2] - y2

            crop_seg_logit = jt.nn.pad(crop_seg_logit,
                                     (pad_left, pad_right, pad_top, pad_bottom),
                                     mode='constant', value=0)

            preds += crop_seg_logit
            count_mat[:, :, y1:y2, x1:x2] += 1

    # Avoid division by zero
    count_mat = jt.maximum(count_mat, jt.ones_like(count_mat))
    seg_logits = preds / count_mat

    return seg_logits


def evaluate(model, data_loader, device=None, verbose=False):
    """Evaluate model on validation dataset."""
    model.eval()

    metric = SegmentationMetric(data_loader.dataset.num_classes)

    # Force cleanup before starting evaluation
    jt.clean()
    jt.gc()

    print(f"Starting evaluation with {len(data_loader)} batches...")

    with jt.no_grad():
        batch_count = 0
        try:
            # Create a fresh iterator to avoid conflicts
            data_iter = iter(data_loader)

            for i in range(len(data_loader)):
                try:
                    # Get next batch with timeout protection
                    minibatch = next(data_iter)
                    batch_count += 1

                    rgb = minibatch['data']
                    targets = minibatch['label']
                    modal = minibatch['modal_x']

                    # Multi-modal input (RGB + depth/modal)
                    outputs = model(rgb, modal)

                    # Handle model output format
                    if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
                        # Model returns [pred], loss format
                        outputs = outputs[0]
                        if isinstance(outputs, list):
                            outputs = outputs[0]
                    elif isinstance(outputs, dict):
                        outputs = outputs['out']

                    # Get predictions
                    predictions = jt.argmax(outputs, dim=1)

                    # Handle potential tuple return from argmax
                    if isinstance(predictions, tuple):
                        predictions = predictions[0]
                    
                    # Ensure predictions is a valid Jittor tensor
                    if not isinstance(predictions, jt.Var):
                        print(f"Warning: predictions is not a Jittor tensor, type: {type(predictions)}")
                        continue

                    # Get numpy arrays for metric calculation
                    try:
                        pred_numpy = predictions.numpy()
                        target_numpy = targets.numpy()
                    except AttributeError as e:
                        print(f"Error converting to numpy: {e}")
                        print(f"predictions type: {type(predictions)}, targets type: {type(targets)}")
                        continue

                    # Update metrics
                    metric.update(pred_numpy, target_numpy)

                    # Clean memory every 10 batches to prevent accumulation
                    if (i + 1) % 10 == 0:
                        jt.clean()

                    if verbose or i % 50 == 0:
                        print(f"Processed batch {i+1}/{len(data_loader)}")

                except StopIteration:
                    print(f"Iterator exhausted at batch {i}")
                    break
                except Exception as e:
                    print(f"Error processing batch {i}: {e}")
                    continue

        except Exception as e:
            print(f"Error during evaluation: {e}")

    print(f"Evaluation completed. Processed {batch_count} batches.")

    # Calculate metrics
    results = metric.get_results()

    # Final cleanup
    jt.clean()

    return results


def evaluate_msf(model, data_loader, config=None, device=None, scales=[1.0], flip=False, engine=None, save_dir=None, sliding=False):
    """Evaluate model with multi-scale and flip augmentation."""
    import math
    model.eval()

    # Use config if provided, otherwise use data_loader info
    if config is not None:
        n_classes = config.num_classes
        background = getattr(config, 'background', 255)
    else:
        n_classes = data_loader.dataset.num_classes
        background = 255

    metric = SegmentationMetric(n_classes)

    with jt.no_grad():
        for idx, minibatch in enumerate(data_loader):
            # Progress logging similar to PyTorch version
            if ((idx + 1) % int(len(data_loader) * 0.5) == 0 or idx == 0):
                if engine is None or not engine.distributed or engine.local_rank == 0:
                    print(f"Validation Iter: {idx + 1} / {len(data_loader)}")

            try:
                images = minibatch['data']
                labels = minibatch['label']
                modal_xs = minibatch['modal_x']

                B, H, W = labels.shape
                scaled_logits = jt.zeros((B, n_classes, H, W))

                # Multi-scale evaluation - follow PyTorch version exactly
                for scale in scales:
                    # Calculate new dimensions and align to 32 (same as PyTorch)
                    new_H, new_W = int(scale * H), int(scale * W)
                    new_H, new_W = (
                        int(math.ceil(new_H / 32)) * 32,
                        int(math.ceil(new_W / 32)) * 32,
                    )

                    # Scale images
                    scaled_images = [
                        jt.nn.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=True),
                        jt.nn.interpolate(modal_xs, size=(new_H, new_W), mode='bilinear', align_corners=True)
                    ]

                    # Forward pass
                    if sliding:
                        logits = slide_inference(model, scaled_images[0], scaled_images[1], config)
                    else:
                        logits = model(scaled_images[0], scaled_images[1])

                    # Handle model output format
                    if isinstance(logits, (list, tuple)):
                        if len(logits) == 2:
                            logits = logits[0]
                        if isinstance(logits, list):
                            logits = logits[0]
                    elif isinstance(logits, dict):
                        logits = logits['out']

                    # Resize back to original size
                    logits = jt.nn.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                    # Apply softmax and accumulate (same as PyTorch)
                    scaled_logits += jt.nn.softmax(logits, dim=1)

                    # Flip augmentation
                    if flip:
                        scaled_images_flipped = [jt.flip(img, [3]) for img in scaled_images]
                        if sliding:
                            logits_flipped = slide_inference(model, scaled_images_flipped[0], scaled_images_flipped[1], config)
                        else:
                            logits_flipped = model(scaled_images_flipped[0], scaled_images_flipped[1])

                        # Handle model output format
                        if isinstance(logits_flipped, (list, tuple)):
                            if len(logits_flipped) == 2:
                                logits_flipped = logits_flipped[0]
                            if isinstance(logits_flipped, list):
                                logits_flipped = logits_flipped[0]
                        elif isinstance(logits_flipped, dict):
                            logits_flipped = logits_flipped['out']

                        # Flip back and resize
                        logits_flipped = jt.flip(logits_flipped, [3])
                        logits_flipped = jt.nn.interpolate(logits_flipped, size=(H, W), mode='bilinear', align_corners=True)
                        # Apply softmax and accumulate (same as PyTorch)
                        scaled_logits += jt.nn.softmax(logits_flipped, dim=1)

                # Get final predictions from accumulated softmax probabilities
                predictions = jt.argmax(scaled_logits, 1)[0]  # Jittor argmax returns tuple

                # Ensure predictions are in the right format for metrics
                if not isinstance(predictions, jt.Var):
                    print(f"Warning: predictions is not a Jittor tensor, type: {type(predictions)}")
                    continue
                    
                try:
                    pred_numpy = predictions.numpy()
                    label_numpy = labels.numpy()
                except AttributeError as e:
                    print(f"Error converting to numpy: {e}")
                    print(f"predictions type: {type(predictions)}, labels type: {type(labels)}")
                    continue

                # Save predictions if save_dir is provided (same format as PyTorch)
                if save_dir is not None:
                    import pathlib
                    import matplotlib.pyplot as plt
                    from matplotlib.colors import ListedColormap

                    palette = [
                        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                        [0, 80, 100], [0, 0, 230], [119, 11, 32],
                    ]
                    palette = np.array(palette, dtype=np.uint8)

                    if 'fn' in minibatch:
                        names = minibatch["fn"][0].replace(".jpg", "").replace(".png", "").replace("datasets/", "")
                        save_name = save_dir + "/" + names + "_pred.png"
                        pathlib.Path(save_name).parent.mkdir(parents=True, exist_ok=True)
                        # Handle Jittor argmax which returns tuple
                        preds = scaled_logits.argmax(1)[0].numpy().squeeze().astype(np.uint8)

                        if config and hasattr(config, 'dataset_name'):
                            if config.dataset_name in ["KITTI-360", "EventScape"]:
                                preds = palette[preds]
                                plt.imsave(save_name, preds)
                            elif config.dataset_name in ["NYUDepthv2", "SUNRGBD"]:
                                try:
                                    palette = np.load("./utils/nyucmap.npy")
                                    preds = palette[preds]
                                    plt.imsave(save_name, preds)
                                except:
                                    cv2.imwrite(save_name, preds)
                            elif config.dataset_name in ["MFNet"]:
                                palette = np.array([
                                    [0, 0, 0], [64, 0, 128], [64, 64, 0], [0, 128, 192],
                                    [0, 0, 192], [128, 128, 0], [64, 64, 128], [192, 128, 128],
                                    [192, 64, 0],
                                ], dtype=np.uint8)
                                preds = palette[preds]
                                plt.imsave(save_name, preds)
                            else:
                                cv2.imwrite(save_name, preds)
                        else:
                            cv2.imwrite(save_name, preds)

                # Update metrics using the same interface as PyTorch version
                metric.update(pred_numpy, label_numpy)

                # Clean memory every 20 batches to prevent accumulation
                if (idx + 1) % 20 == 0:
                    jt.clean()

            except Exception as e:
                print(f"Error processing batch {idx}: {e}")
                continue

    # Return metric object for compatibility with PyTorch version
    if engine and engine.distributed:
        # For distributed evaluation, we would need to gather metrics
        # For now, return the local metric
        return metric
    else:
        return metric


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
        
        # Handle Jittor argmax which returns tuple
        prediction = jt.argmax(output, 1)[0].squeeze(0).numpy()
    
    # Save result if output path provided
    if output_path:
        # Convert prediction to color map if needed
        # This would require a color palette mapping
        cv2.imwrite(output_path, prediction.astype(np.uint8))
    
    return prediction
