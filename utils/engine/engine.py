"""
Training engine for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import os
import os.path as osp
import time
import argparse

import jittor as jt

from .logger import get_logger
from utils.jt_utils import (
    load_model,
    parse_devices,
    extant_file,
    link_file,
    ensure_dir,
)

logger = get_logger()


class State(object):
    """Training state management."""
    
    def __init__(self):
        self.epoch = 1
        self.iteration = 0
        self.dataloader = None
        self.model = None
        self.optimizer = None

    def register(self, **kwargs):
        """Register state variables."""
        for k, v in kwargs.items():
            assert k in ["epoch", "iteration", "dataloader", "model", "optimizer"]
            setattr(self, k, v)


class Engine(object):
    """Training engine for managing training process."""
    
    def __init__(self, custom_parser=None):
        logger.info("Jittor Version {}".format(jt.__version__))
        self.state = State()
        self.devices = None
        self.distributed = False

        if custom_parser is None:
            self.parser = argparse.ArgumentParser()
        else:
            assert isinstance(custom_parser, argparse.ArgumentParser)
            self.parser = custom_parser

        self.inject_default_parser()
        
        # For Jittor, we don't parse args here as it's done in main script
        self.continue_state_object = None
        self.local_rank = 0
        self.world_size = 1
        self.devices = [0]  # Default to single GPU
        
        self.checkpoint_state = []

    def inject_default_parser(self):
        """Inject default parser arguments."""
        p = self.parser
        p.add_argument("-d", "--devices", default="", help="set data parallel training")
        p.add_argument(
            "-c",
            "--continue",
            type=str,  # Changed from extant_file for simplicity
            metavar="FILE",
            dest="continue_fpath",
            help="continue from one certain checkpoint",
        )
        p.add_argument("--local_rank", default=0, type=int, help="process rank on node")
        p.add_argument(
            "-p",
            "--port",
            type=str,
            default="16005",
            dest="port",
            help="port for init_process_group",
        )

    def register_state(self, **kwargs):
        """Register state variables."""
        self.state.register(**kwargs)

    def update_iteration(self, epoch, iteration):
        """Update training iteration."""
        self.state.epoch = epoch
        self.state.iteration = iteration

    def save_checkpoint(self, path):
        """Save training checkpoint."""
        logger.info("Saving checkpoint to {}".format(path))
        
        checkpoint = {
            'epoch': self.state.epoch,
            'iteration': self.state.iteration,
            'model': self.state.model.state_dict() if self.state.model else None,
            'optimizer': self.state.optimizer.state_dict() if self.state.optimizer else None,
        }
        
        # Ensure directory exists
        ensure_dir(osp.dirname(path))
        
        # Save using Jittor's save function
        jt.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load training checkpoint."""
        if not osp.exists(path):
            logger.warning("Checkpoint file {} not found".format(path))
            return
            
        logger.info("Loading checkpoint from {}".format(path))
        
        checkpoint = jt.load(path)
        
        if 'epoch' in checkpoint:
            self.state.epoch = checkpoint['epoch']
        if 'iteration' in checkpoint:
            self.state.iteration = checkpoint['iteration']
        if 'model' in checkpoint and self.state.model:
            self.state.model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and self.state.optimizer:
            self.state.optimizer.load_state_dict(checkpoint['optimizer'])

    def link_tb(self, source, target):
        """Link tensorboard directory."""
        link_file(source, target)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        # Cleanup if needed
        pass


def get_world_size():
    """Get world size for distributed training."""
    return 1  # Single GPU for now


def get_rank():
    """Get current rank for distributed training."""
    return 0  # Single GPU for now


def is_main_process():
    """Check if current process is main process."""
    return get_rank() == 0


def synchronize():
    """Synchronize all processes."""
    # No-op for single GPU
    pass
