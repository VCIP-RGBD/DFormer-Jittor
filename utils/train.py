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
import os
import sys

os.environ.setdefault("use_cutlass", "0")

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR in sys.path:
    sys.path.remove(_CUR_DIR)

import jittor as jt
from jittor import nn

if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from models.builder import EncoderDecoder as segmodel
from utils.dataloader.dataloader import get_train_loader, get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from utils.init_func import configure_optimizers, group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.val_mm import evaluate, evaluate_msf


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="train config file path")
    parser.add_argument("--gpus", default=2, type=int, help="used gpu number")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    parser.add_argument("--epochs", default=0)
    parser.add_argument("--show_image", "-s", default=False, action="store_true")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--continue_fpath")
    parser.add_argument("--sliding", default=False, action="store_true")
    parser.add_argument("--no-sliding", dest="sliding", action="store_false")
    parser.add_argument("--syncbn", default=True, action="store_true")
    parser.add_argument("--no-syncbn", dest="syncbn", action="store_false")
    parser.add_argument("--mst", default=True, action="store_true")
    parser.add_argument("--no-mst", dest="mst", action="store_false")
    parser.add_argument("--amp", default=True, action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--val_amp", default=True, action="store_true")
    parser.add_argument("--no-val_amp", dest="val_amp", action="store_false")
    parser.add_argument("--pad_SUNRGBD", default=False, action="store_true")
    parser.add_argument("--no-pad_SUNRGBD", dest="pad_SUNRGBD", action="store_false")
    parser.add_argument("--use_seed", default=True, action="store_true")
    parser.add_argument("--no-use_seed", dest="use_seed", action="store_false")
    parser.add_argument("--local-rank", default=0)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = getattr(import_module(args.config), "C")
    
    engine = Engine()
    engine.distributed = False
    engine.local_rank = 0
    engine.world_size = 1
    
    logger = get_logger(config.log_dir, config.log_file, rank=0)
    
    if args.pad_SUNRGBD and config.dataset_name != "SUNRGBD":
        args.pad_SUNRGBD = False
        logger.warning("pad_SUNRGBD is only used for SUNRGBD dataset")
    
    if (args.pad_SUNRGBD) and (not config.backbone.startswith("DFormerv2")):
        raise ValueError("DFormerv1 is not recommended with pad_SUNRGBD")
    
    if (not args.pad_SUNRGBD) and config.backbone.startswith("DFormerv2") and config.dataset_name == "SUNRGBD":
        raise ValueError("DFormerv2 is not recommended without pad_SUNRGBD")
    
    config.pad = args.pad_SUNRGBD
    
    if args.use_seed:
        set_seed(config.seed)
        logger.info("set seed %d", config.seed)
    else:
        logger.info("use random seed")
    
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset, config)
    
    if args.gpus == 2:
        val_dl_factor = 1.3 if args.mst and args.val_amp else 2
    elif args.gpus == 4:
        val_dl_factor = 0.6 if args.mst and args.val_amp else 2
    else:
        val_dl_factor = 1.5
    
    val_loader, val_sampler = get_val_loader(
        engine,
        RGBXDataset,
        config,
        val_batch_size=int(config.batch_size * val_dl_factor) if config.dataset_name != "SUNRGBD" else int(args.gpus),
    )
    logger.info("val dataset len:%d", len(val_loader))
    
    tb_dir = config.tb_dir + "/{}".format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
    generate_tb_dir = config.tb_dir + "/tb"
    
    pp = pprint.PrettyPrinter(indent=4)
    logger.info("config: \n%s", pp.pformat(config))
    
    logger.info("args parsed:")
    for k in args.__dict__:
        logger.info("%s: %s", k, str(args.__dict__[k]))
    
    criterion = nn.CrossEntropyLoss(ignore_index=config.background)
    
    if args.syncbn:
        BatchNorm2d = nn.BatchNorm2d
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
    
    base_lr = config.lr
    if hasattr(config, 'lr_power'):
        lr_power = config.lr_power
    else:
        lr_power = 0.9
    
    if config.optimizer == 'SGD':
        optimizer = jt.optim.SGD(model.parameters(), lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer == 'AdamW':
        optimizer = jt.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=config.weight_decay)
    else:
        optimizer = jt.optim.Adam(model.parameters(), lr=base_lr, weight_decay=config.weight_decay)
    
    if hasattr(config, 'use_warmup') and config.use_warmup:
        total_iteration = config.nepochs * config.niters_per_epoch
        warmup_iteration = config.warmup_iters if hasattr(config, 'warmup_iters') else config.warm_up_epoch * config.niters_per_epoch
        lr_scheduler = None
        logger.info("Learning rate scheduler temporarily disabled")
    else:
        lr_scheduler = None
    
    max_epochs = int(args.epochs) if args.epochs != 0 else config.nepochs
    logger.info("Starting progressive training approach")
    
    progressive_stages = [
        {"batch_size": 1, "iterations": 3, "epochs": 1, "description": "Stage 1: Single sample test"},
        {"batch_size": 2, "iterations": 5, "epochs": 1, "description": "Stage 2: Small batch test"},
        {"batch_size": 4, "iterations": 10, "epochs": 2, "description": "Stage 3: Medium batch test"},
    ]
    
    for stage_idx, stage in enumerate(progressive_stages):
        logger.info(f"\n=== {stage['description']} ===")
        logger.info(f"Batch size: {stage['batch_size']}, Iterations: {stage['iterations']}, Epochs: {stage['epochs']}")
            
        for epoch in range(stage['epochs']):
            logger.info("Stage %d, Epoch %d/%d", stage_idx + 1, epoch + 1, stage['epochs'])
            
            model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch_data in enumerate(train_loader):
                if batch_idx >= stage['iterations']:
                    break
                
                rgb = batch_data['data']
                modal_x = batch_data['modal_x']
                target = batch_data['label']
                
                if rgb.shape[0] > stage['batch_size']:
                    rgb = rgb[:stage['batch_size']]
                    modal_x = modal_x[:stage['batch_size']]
                    target = target[:stage['batch_size']]
                    
                if model.training:
                    pred, loss = model(rgb, modal_x, target)
                else:
                    pred, _ = model(rgb, modal_x)
                    loss = criterion(pred[0], target)
                
                optimizer.step(loss)
                
                if lr_scheduler is not None:
                    lr_scheduler.step()
                
                train_loss += loss.item()
                num_batches += 1
                
                log_freq = 1 if stage_idx == 0 else 2
                if batch_idx % log_freq == 0:
                    current_lr = optimizer.lr if hasattr(optimizer, 'lr') else base_lr
                    logger.info("Stage %d, Batch %d/%d, Loss: %.4f, LR: %.6f", 
                              stage_idx + 1, batch_idx, stage['iterations'], loss.item(), current_lr)
            
            avg_train_loss = train_loss / num_batches if num_batches > 0 else 0
            logger.info("Stage %d, Epoch %d completed. Average training loss: %.4f", 
                       stage_idx + 1, epoch + 1, avg_train_loss)
            
            logger.info("Running quick validation...")
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with jt.no_grad():
                for batch_idx, batch_data in enumerate(val_loader):
                    if batch_idx >= 2:
                        break
                    
                    rgb = batch_data['data']
                    modal_x = batch_data['modal_x']
                    target = batch_data['label']
                    
                    if rgb.shape[0] > stage['batch_size']:
                        rgb = rgb[:stage['batch_size']]
                        modal_x = modal_x[:stage['batch_size']]
                        target = target[:stage['batch_size']]
                    
                    pred, _ = model(rgb, modal_x)
                    loss = criterion(pred[0], target)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            logger.info("Stage %d validation loss: %.4f", stage_idx + 1, avg_val_loss)
        
        logger.info(f"=== {stage['description']} completed successfully ===\n")

if __name__ == "__main__":
    main()
