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
from utils.jt_utils import all_reduce_tensor


class GpuTimer:
    def __init__(self, beta=0.6):
        self.start_time = None
        self.stop_time = None
        self.mean_time = None
        self.beta = beta
        self.first_call = True

    def start(self):
        jt.sync_all(True)
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is None:
            print("Use start() before stop(). ")
            return
        jt.sync_all(True)
        self.stop_time = time.perf_counter()
        elapsed = self.stop_time - self.start_time
        self.start_time = None
        if self.first_call:
            self.mean_time = elapsed
            self.first_call = False
        else:
            self.mean_time = self.beta * self.mean_time + (1 - self.beta) * elapsed


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def is_eval(epoch, config):
    return epoch > int(config.checkpoint_start_epoch) or epoch == 1 or epoch % 10 == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="train config file path")
    parser.add_argument("--gpus", default=1, type=int, help="used gpu number")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    parser.add_argument("--epochs", default=0, type=int)
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
    parser.add_argument("--amp", default=False, action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--val_amp", default=False, action="store_true")
    parser.add_argument("--no-val_amp", dest="val_amp", action="store_false")
    parser.add_argument("--pad_SUNRGBD", default=False, action="store_true")
    parser.add_argument("--no-pad_SUNRGBD", dest="pad_SUNRGBD", action="store_false")
    parser.add_argument("--use_seed", default=True, action="store_true")
    parser.add_argument("--no-use_seed", dest="use_seed", action="store_false")
    parser.add_argument("--local-rank", default=0, type=int)

    args = parser.parse_args()
    
    config = getattr(import_module(args.config), "C")
    
    engine = Engine(custom_parser=parser)
    engine.distributed = True if jt.world_size > 1 else False
    engine.local_rank = jt.rank
    engine.world_size = jt.world_size
    
    logger = get_logger(config.log_dir, config.log_file, rank=engine.local_rank)
    
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
        logger.info(f"set seed {config.seed}")
    else:
        logger.info("use random seed")
    
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset, config)
    
    val_dl_factor = 1.0
    val_loader, val_sampler = get_val_loader(
        engine,
        RGBXDataset,
        config,
        val_batch_size=int(config.batch_size * val_dl_factor) if config.dataset_name != "SUNRGBD" else int(args.gpus),
    )
    logger.info(f"val dataset len:{len(val_loader) * int(args.gpus)}")
    
    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + "/{}".format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + "/tb"
        pp = pprint.PrettyPrinter(indent=4)
        logger.info("config: \n%s", pp.pformat(config))
    
    logger.info("args parsed:")
    for k, v in args.__dict__.items():
        logger.info("%s: %s", k, str(v))
    
    criterion = nn.CrossEntropyLoss(ignore_index=config.background)
    
    if args.syncbn:
        try:
            BatchNorm2d = nn.SyncBatchNorm
            logger.info("using syncbn")
        except AttributeError:
            logger.warning("SyncBatchNorm not available in Jittor, using regular BatchNorm2d")
            BatchNorm2d = nn.BatchNorm2d
    else:
        BatchNorm2d = nn.BatchNorm2d
        logger.info("using regular bn")
    
    model = segmodel(
        cfg=config,
        criterion=criterion,
        norm_layer=BatchNorm2d,
        syncbn=args.syncbn,
    )
    
    base_lr = config.lr
    params_list = group_weight(model)

    if config.optimizer == 'AdamW':
        optimizer = jt.optim.AdamW(params_list, lr=base_lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = jt.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError
    
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(
        optimizer,
        power=config.lr_power,
        max_iters=total_iteration,
        warmup_iters=config.niters_per_epoch * config.warm_up_epoch,
    )
    
    if engine.distributed:
        logger.info(".............distributed training.............")
        model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    logger.info("begin trainning:")

    miou, best_miou = 0.0, 0.0
    train_timer = GpuTimer()
    eval_timer = GpuTimer()
    
    for epoch in range(engine.state.epoch, config.nepochs + 1):
        model.train()
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        
        dataloader = iter(train_loader)
        sum_loss = 0
        
        train_timer.start()
        for idx in range(config.niters_per_epoch):
            engine.update_iteration(epoch, idx)

            minibatch = next(dataloader)
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']

            loss = model(imgs, modal_xs, gts)
            
            if isinstance(loss, tuple):
                if len(loss) == 2:
                    predictions, loss = loss
                else:
                    loss = loss[-1]

            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
            
            optimizer.step(loss)
            
            current_idx = (epoch - 1) * config.niters_per_epoch + idx
            lr_policy.step(current_idx)
            
            if engine.distributed:
                sum_loss += reduce_loss.item()
                current_lr = optimizer.lr if hasattr(optimizer, 'lr') else lr_policy.get_lr()[0]
                print_str = (
                    f"Epoch {epoch}/{config.nepochs} "
                    f"Iter {idx + 1}/{config.niters_per_epoch}: "
                    f"lr={current_lr:.4e} "
                    f"loss={reduce_loss.item():.4f} total_loss={(sum_loss / (idx + 1)):.4f}"
                )
            else:
                sum_loss += loss.item()
                current_lr = optimizer.lr if hasattr(optimizer, 'lr') else lr_policy.get_lr()[0]
                print_str = (
                    f"Epoch {epoch}/{config.nepochs} "
                    f"Iter {idx + 1}/{config.niters_per_epoch}: "
                    f"lr={current_lr:.4e} loss={loss.item():.4f} total_loss={(sum_loss / (idx + 1)):.4f}"
                )

            if ((idx + 1) % int(config.niters_per_epoch * 0.1) == 0 or idx == 0) and \
               ((engine.distributed and engine.local_rank == 0) or not engine.distributed):
                logger.info(print_str)

        train_timer.stop()
        
        if is_eval(epoch, config):
            eval_timer.start()
            model.eval()
            with jt.no_grad():
                if args.mst:
                    all_metrics = evaluate_msf(model, val_loader, config, [0.5, 0.75, 1.0, 1.25, 1.5], True, engine, sliding=args.sliding)
                else:
                    all_metrics = evaluate(model, val_loader, config, engine, sliding=args.sliding)

                if engine.distributed:
                    if engine.local_rank == 0:
                        metric = all_metrics[0]
                        for other_metric in all_metrics[1:]:
                            metric.update_hist(other_metric.hist)
                        ious, miou = metric.compute_iou()
                        acc, macc = metric.compute_pixel_acc()
                        f1, mf1 = metric.compute_f1()
                else:
                    ious, miou = all_metrics.compute_iou()
                    acc, macc = all_metrics.compute_pixel_acc()
                    f1, mf1 = all_metrics.compute_f1()

                if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                    if miou > best_miou:
                        best_miou = miou
                        engine.save_and_link_checkpoint(
                            config.log_dir,
                            config.log_dir,
                            config.log_dir_link,
                            infor=f"_miou_{miou}",
                            metric=miou
                        )
                    logger.info(f"Epoch {epoch} validation result: mIoU {miou:.4f}, best mIoU {best_miou:.4f}")
            eval_timer.stop()

        if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
            eval_count = sum(1 for i in range(epoch + 1, config.nepochs + 1) if is_eval(i, config))
            left_time = train_timer.mean_time * (config.nepochs - epoch) + eval_timer.mean_time * eval_count
            eta = (datetime.datetime.now() + datetime.timedelta(seconds=left_time)).strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Avg train time: {train_timer.mean_time:.2f}s, avg eval time: {eval_timer.mean_time:.2f}s, left eval count: {eval_count}, ETA: {eta}")


if __name__ == "__main__":
    main()
