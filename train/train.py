import torch
import torch.nn.functional as F
from tqdm import tqdm

from metrics.metrics import compute_metrics
from utils.config import extract_config
from utils.printing import print_verbose

def train_one_epoch(epoch, model, loader, criterion, optimizer, num_epochs,
        step_fn, device, verbose=True):
    model.train()
    total_loss = 0

    with tqdm(loader, unit='batch', ncols=80,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]',
              leave=False) as tepoch:
        tepoch.set_description(f"Training Epoch {epoch}/{num_epochs}")
        for batch in tepoch:
            # step_fn handles batch unpacking, forward, loss, and backward
            loss = step_fn(model, batch, criterion, optimizer, device)
            total_loss += loss
            tepoch.set_postfix(loss=loss)

    avg_loss = total_loss / len(loader)
    return avg_loss

def train(config, verbose=True):
  
    cfg = extract_config(
        config,
        required={
            'device', 'model',
            'num_epochs', 'train_loader', 'val_loader',
            'criterion', 'optimizer', 'scheduler', 'monitor',
            'step_fn'
        },
        optional={
            'metrics': {'metric_names': ['val_loss', 'embedding_norm', 'embedding_variance', 'lr']}
        },
        verbose=verbose
    ) 

    print_verbose(f"Training on device: {cfg.device}", verbose) 
    
    # Training loop
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_one_epoch(epoch, cfg.model, cfg.train_loader, cfg.criterion, 
            cfg.optimizer, cfg.num_epochs, cfg.step_fn, cfg.device, verbose
        )    

        cfg.scheduler.step(train_loss)

        cfg.model.eval()
        other_metrics=compute_metrics(
            model=cfg.model,
            train_loader=cfg.train_loader,
            val_loader=cfg.val_loader,
            optimizer=cfg.optimizer,
            device=cfg.device, 
            config=cfg.metrics
        )
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
