import torch

from utils.config import extract_config
from train.train import train
from train.step import supervised_step_fn

def train_predictor(config, verbose=True):
    cfg = extract_config(
        config,
        required={
            'device',
            'train_loader', 'val_loader',
            'optimizer', 'scheduler',
            'train_config', 'criterion'
        },
        optional={
            'step_fn': supervised_step_fn
        },
        verbose=verbose
    )

    # Compose config to pass to train()
    train_config = {
        **cfg.train_config,
        'device': cfg.device,
        'train_loader': cfg.train_loader,
        'val_loader': cfg.val_loader,
        'criterion': cfg.criterion,
        'optimizer': cfg.optimizer,
        'scheduler': cfg.scheduler,
        'step_fn': cfg.step_fn
    }

    # Delegate to generic training loop
    train(train_config, verbose=verbose)
