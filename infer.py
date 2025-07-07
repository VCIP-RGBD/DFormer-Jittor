"""
Inference script for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import os
import argparse
import time
import jittor as jt

from utils.inference import InferenceEngine, load_model_for_inference, batch_inference
from utils.visualization import visualize_results, save_prediction_image
from utils.jt_utils import ensure_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DFormer Inference')
    
    # Model arguments
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file path')
    
    # Input arguments
    parser.add_argument('--input', required=True, help='input image or directory')
    parser.add_argument('--modal', help='modal image or directory (for RGBD)')
    
    # Output arguments
    parser.add_argument('--output', required=True, help='output directory')
    parser.add_argument('--save-vis', action='store_true', help='save visualization')
    parser.add_argument('--save-pred', action='store_true', help='save prediction masks')
    
    # Inference arguments
    parser.add_argument('--dataset', default='nyudepthv2', 
                       choices=['nyudepthv2', 'sunrgbd'], help='dataset type')
    parser.add_argument('--device', default='cuda', help='device to use')
    
    # Visualization arguments
    parser.add_argument('--alpha', type=float, default=0.5, help='overlay alpha')
    parser.add_argument('--show', action='store_true', help='show results')
    
    return parser.parse_args()


def infer_single_image(args, model, config):
    """Infer on a single image."""
    print(f"Processing single image: {args.input}")
    
    # Create inference engine
    engine = InferenceEngine(model, config)
    
    # Run inference
    pred_mask = engine.infer_single_image(args.input, args.modal)
    
    # Save prediction
    if args.save_pred:
        pred_path = os.path.join(args.output, 'prediction.png')
        save_prediction_image(pred_mask, pred_path, dataset=args.dataset)
        print(f"Prediction saved to: {pred_path}")
    
    # Save visualization
    if args.save_vis:
        import cv2
        from utils.visualization import get_palette, overlay_mask
        
        # Load original image
        image = cv2.imread(args.input)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        palette = get_palette(args.dataset)
        vis_image = overlay_mask(image, pred_mask, palette, args.alpha)
        
        # Save visualization
        vis_path = os.path.join(args.output, 'visualization.png')
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(vis_path, vis_image_bgr)
        print(f"Visualization saved to: {vis_path}")
        
        # Show if requested
        if args.show:
            visualize_results(image, pred_mask, palette=palette, 
                            dataset=args.dataset, show=True)


def infer_directory(args, model, config):
    """Infer on a directory of images."""
    print(f"Processing directory: {args.input}")
    
    # Setup output directories
    pred_dir = os.path.join(args.output, 'predictions') if args.save_pred else None
    vis_dir = os.path.join(args.output, 'visualizations') if args.save_vis else None
    
    if pred_dir:
        ensure_dir(pred_dir)
    if vis_dir:
        ensure_dir(vis_dir)
    
    # Get image files
    image_files = [f for f in os.listdir(args.input) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No image files found in input directory")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Create inference engine
    engine = InferenceEngine(model, config)
    
    # Process each image
    for i, image_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_file}")
        
        image_path = os.path.join(args.input, image_file)
        
        # Find corresponding modal image
        modal_path = None
        if args.modal and os.path.isdir(args.modal):
            modal_path = os.path.join(args.modal, image_file)
            if not os.path.exists(modal_path):
                modal_path = None
        
        # Run inference
        pred_mask = engine.infer_single_image(image_path, modal_path)
        
        # Save prediction
        if args.save_pred:
            pred_name = os.path.splitext(image_file)[0] + '_pred.png'
            pred_path = os.path.join(pred_dir, pred_name)
            save_prediction_image(pred_mask, pred_path, dataset=args.dataset)
        
        # Save visualization
        if args.save_vis:
            import cv2
            from utils.visualization import get_palette, overlay_mask
            
            # Load original image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create visualization
            palette = get_palette(args.dataset)
            vis_image = overlay_mask(image, pred_mask, palette, args.alpha)
            
            # Save visualization
            vis_name = os.path.splitext(image_file)[0] + '_vis.png'
            vis_path = os.path.join(vis_dir, vis_name)
            vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(vis_path, vis_image_bgr)
    
    print(f"Processing complete. Results saved to: {args.output}")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Set device
    if args.device == 'cuda':
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0
    
    print(f"Using device: {'CUDA' if jt.flags.use_cuda else 'CPU'}")
    
    # Load model
    print("Loading model...")
    model, config = load_model_for_inference(args.config, args.checkpoint)
    print("Model loaded successfully")
    
    # Create output directory
    ensure_dir(args.output)
    
    # Run inference
    start_time = time.time()
    
    if os.path.isfile(args.input):
        # Single image inference
        infer_single_image(args, model, config)
    elif os.path.isdir(args.input):
        # Directory inference
        infer_directory(args, model, config)
    else:
        raise ValueError(f"Input path does not exist: {args.input}")
    
    end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
