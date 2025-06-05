import torch

from loss.rnc_loss import RnCLoss
from utils.config import extract_config
from train.train import train
from train.step import supervised_two_view_step_fn

def train_encoder(config, verbose=True):
    cfg = extract_config(
        config,
        required={
            'device', 'temperature',
            'train_loader', 'val_loader',
            'optimizer', 'scheduler',
            'train_config'   
        },
        optional={
            'label_difference': 'l1',
            'feature_similarity': 'l2',
            'step_fn': supervised_two_view_step_fn
        },
        verbose=verbose
    )

    # Build loss specific to encoder training
    criterion = RnCLoss(
        temperature=cfg.temperature,
        label_diff=cfg.label_difference,
        feature_sim=cfg.feature_similarity
    ).to(cfg.device)

    # Compose config to pass to train()
    train_config = {
        **cfg.train_config,
        'device': cfg.device,
        'train_loader': cfg.train_loader,
        'val_loader': cfg.val_loader,
        'criterion': criterion,
        'optimizer': cfg.optimizer,
        'scheduler': cfg.scheduler,
        'step_fn': cfg.step_fn
    }

    # Delegate to generic training loop
    train(train_config, verbose=verbose)
