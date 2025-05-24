import torch
import torch.optim as optim

from loss import RnCLoss
from analysis.monitor import Monitor
from utils.config import extract_config
from utils.printing import print_verbose
from datasets.loading import get_data_loaders
from train.train import train

def train_encoder(config, verbose=True):
    cfg = extract_config(
        config,
        required={
            'device', 'data_folder', 'batch_size', 'model',
            'num_epochs', 'learning_rate', 'temperature', 'augmentations',
            'train_size', 'monitor',
        },
        optional={
            'metrics': ["val_loss", "embedding_norm", "embedding_variance", "lr"]
        },
        verbose=verbose
    )

    # Build data loaders
    train_loader, val_loader = get_data_loaders(
        cfg.data_folder, cfg.augmentations,
        batch_size=cfg.batch_size, train_size=cfg.train_size
    )

    # Build loss specific to encoder training
    criterion = RnCLoss(
        temperature=cfg.train_size,
        label_diff='l1',
        feature_sim='l2'
    ).to(cfg.device)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(cfg.model.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=verbose
    )

    # Compose config to pass to train()
    run_config = {
        **config,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'criterion': criterion,
        'optimizer': optimizer,
        'scheduler': scheduler,
    }

    # Delegate to generic training loop
    train(run_config, verbose=verbose)
