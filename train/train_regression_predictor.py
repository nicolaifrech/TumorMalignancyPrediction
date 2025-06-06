import torch

from utils.config import extract_config
from loss.composite_loss import CompositeLoss
from loss.soft_mcc_loss import SoftMCCLossMulti
from loss.scalar_to_classification_loss import gaussian_kernel_logits, ScalarToClassificationLoss
from train.train import train
from train.step import supervised_step_fn

def train_regression_predictor(config, verbose=True):
    cfg = extract_config(
        config,
        required={
            'device',
            'train_loader', 'val_loader', 'optimizer', 'scheduler',
            'train_config', 'ordinal_map', 'loss_regularization'
        },    
        verbose=verbose
    )
  
    criterion = CompositeLoss([
        (torch.nn.L1Loss(), 1.0, 'l1'),
        (
            ScalarToClassificationLoss(
                classification_loss=SoftMCCLossMulti(),
                kernel_fn=gaussian_kernel_logits,
                ordinal_map=cfg.ordinal_map,
                one_hot_encoding=True
            ),
            cfg.loss_regularization,
            'mcc'
        )
    ])
    #criterion = torch.nn.L1Loss()

    # Compose config to pass to train()
    train_config = {
        **cfg.train_config,
        'device': cfg.device,
        'train_loader': cfg.train_loader,
        'val_loader': cfg.val_loader,
        'criterion': criterion,
        'optimizer': cfg.optimizer,
        'scheduler': cfg.scheduler,
        'step_fn': supervised_step_fn
    }

    # Delegate to generic training loop 
    train(train_config, verbose=verbose)
