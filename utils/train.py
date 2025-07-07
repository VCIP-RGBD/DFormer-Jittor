#!/usr/bin/env python3
"""
DFormer Jittor Training Script
Adapted from PyTorch version for Jittor framework
"""

import argparse
import datetime
import os
import pprint
import random
import time
from importlib import import_module

import numpy as np
import jittor as jt
from jittor import nn

from models.builder import EncoderDecoder as segmodel
from utils.dataloader.dataloader import get_train_loader, get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from utils.init_func import configure_optimizers, group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.val_mm import evaluate, evaluate_msf


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="train config file path")
    parser.add_argument("--gpus", default=2, type=int, help="used gpu number")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    parser.add_argument("--epochs", default=0)
    parser.add_argument("--show_image", "-s", default=False, action="store_true")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--continue_fpath")
    parser.add_argument("--sliding", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--syncbn", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--mst", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--val_amp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--pad_SUNRGBD", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_seed", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--local-rank", default=0)
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Import config
    config = getattr(import_module(args.config), "C")
    
    # Setup logger
    logger = get_logger(config.log_dir, config.log_file, rank=0)
    
    # Validate arguments
    if args.pad_SUNRGBD and config.dataset_name != "SUNRGBD":
        args.pad_SUNRGBD = False
        logger.warning("pad_SUNRGBD is only used for SUNRGBD dataset")
    
    if (args.pad_SUNRGBD) and (not config.backbone.startswith("DFormerv2")):
        raise ValueError("DFormerv1 is not recommended with pad_SUNRGBD")
    
    if (not args.pad_SUNRGBD) and config.backbone.startswith("DFormerv2") and config.dataset_name == "SUNRGBD":
        raise ValueError("DFormerv2 is not recommended without pad_SUNRGBD")
    
    config.pad = args.pad_SUNRGBD
    
    # Set seed for reproducibility
    if args.use_seed:
        set_seed(config.seed)
        logger.info(f"set seed {config.seed}")
    else:
        logger.info("use random seed")
    
    # Create data loaders
    train_loader, train_sampler = get_train_loader(None, RGBXDataset, config)
    
    # Determine validation batch size factor
    if args.gpus == 2:
        val_dl_factor = 1.3 if args.mst and args.val_amp else 2
    elif args.gpus == 4:
        val_dl_factor = 0.6 if args.mst and args.val_amp else 2
    else:
        val_dl_factor = 1.5
    
    val_loader, val_sampler = get_val_loader(
        None,
        RGBXDataset,
        config,
        val_batch_size=int(config.batch_size * val_dl_factor) if config.dataset_name != "SUNRGBD" else int(args.gpus),
    )
    logger.info(f"val dataset len:{len(val_loader)}")
    
    # Setup tensorboard logging
    tb_dir = config.tb_dir + "/{}".format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
    generate_tb_dir = config.tb_dir + "/tb"
    
    pp = pprint.PrettyPrinter(indent=4)
    logger.info("config: \n" + pp.pformat(config))
    
    logger.info("args parsed:")
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))
    
    # Define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=config.background)
    
    # Build model
    if args.syncbn:
        BatchNorm2d = nn.BatchNorm2d  # TODO: Implement SyncBatchNorm for Jittor
        logger.info("using syncbn")
    else:
        BatchNorm2d = nn.BatchNorm2d
        logger.info("using regular bn")
    
    model = segmodel(
        cfg=config,
        criterion=criterion,
        norm_layer=BatchNorm2d,
        syncbn=args.syncbn,
    )
    
    logger.info("Model created successfully")
    
    # TODO: Continue with training loop implementation
    # This will be completed in the next iteration
    
    logger.info("Training script setup completed")


if __name__ == "__main__":
    main()
