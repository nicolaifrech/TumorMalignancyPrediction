import torch
import torch.optim as optim

from loss import RnCLoss
from analysis.monitor import Monitor
from metrics import compute_metrics
from train.train_one_epoch import train_one_epoch

from utils.config import extract_config
from utils.printing import print_verbose
from datasets.loading import get_data_loaders

def train_encoder(config, verbose=True):

    cfg = extract_config(
        config,
        required={
            'device', 'data_folder', 'batch_size', 'model',
            'num_epochs', 'learning_rate', 'temperature', 'augmentations',
            'train_size', 'monitor' 
        },
        optional={
            'metrics': ["val_loss", "embedding_norm", "embedding_variance", "lr"] 
        },
        verbose=verbose
    )

    train_loader, val_loader = get_data_loaders(cfg.data_folder, cfg.augmentations, batch_size=cfg.batch_size, train_size=cfg.train_size)
 
    # Initialize Rank-N-Contrast loss
    criterion = RnCLoss(temperature=cfg.train_size, label_diff='l1', feature_sim='l2').to(cfg.device)
    optimizer = torch.optim.Adam(cfg.model.parameters(), lr=cfg.learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=verbose
    ) 
    
    print_verbose(f"Training on device: {cfg.device}", verbose) 
    
    # Training loop
    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(epoch, cfg.model, train_loader, criterion, optimizer, cfg.num_epochs, cfg.device, verbose)
        scheduler.step(train_loss)
 
        other_metrics = compute_metrics(cfg.model, val_loader, cfg.device, cfg.metrics, optimizer=optimizer)
        metrics = {
            'train_loss': train_loss,
            **other_metrics,
        }
        if hasattr(cfg, "monitor") and cfg.monitor:
            training_status = cfg.monitor.update(cfg.model, metrics, epoch=epoch)
            if training_status.get("early_stopping") == "early_stop":
                print_verbose("Stopping early.", verbose)
                break 
 
    print_verbose("Training complete!", verbose)  
