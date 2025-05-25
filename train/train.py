import torch

from utils.config import extract_config
from utils.printing import print_verbose
from train.train_one_epoch import train_one_epoch
from metrics import compute_metrics

def train(config, verbose=True):
  
    cfg = extract_config(
        config,
        required={
            'device', 'model',
            'num_epochs', 'train_loader', 'val_loader',
            'criterion', 'optimizer', 'scheduler', 'monitor',
        },
        optional={
            'metrics': ['val_loss', 'embedding_norm', 'embedding_variance', 'lr']
        },
        verbose=verbose
    ) 

    print_verbose(f"Training on device: {cfg.device}", verbose) 
    
    # Training loop
    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(epoch, cfg.model, cfg.train_loader, cfg.criterion, cfg.optimizer, cfg.num_epochs, cfg.device, verbose)
        cfg.scheduler.step(train_loss)
 
        other_metrics = compute_metrics(cfg.model, cfg.val_loader, cfg.device, cfg.metrics, optimizer=cfg.optimizer)
        metrics = { 
            'train_loss': train_loss,
            **other_metrics,
        }   
        if hasattr(cfg, 'monitor') and cfg.monitor:
            training_status = cfg.monitor.update(cfg.model, metrics, epoch=epoch)
            if training_status.get('early_stopping') == 'early_stop':
                print_verbose('Stopping early.', verbose)
                break 
 
    print_verbose('Training complete!', verbose)
