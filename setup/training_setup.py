import os
from torch.utils.tensorboard import SummaryWriter

from device import setup_device
from metrics.knn_analyzer import KNNAnalyzer
from metrics.knr_analyzer import KNRAnalyzer
from setup.optimizer_setup import setup_optimizer
from setup.scheduler_setup import setup_scheduler
from setup.monitor_setup import setup_monitor
from setup.model_setup import setup_encoder

def setup_training_environment(config):
    momentum = config.get('momentum', 0.9)
    device, data_parallel = setup_device(use_gpu_1_only=config['use_gpu_1_only'], verbose=config['verbose'])
    model = setup_encoder(config['backbone_model'], data_parallel, config['pretrained'], device)
    optimizer = setup_optimizer(config['optimizer'], model, config['learning_rate'], momentum)
    scheduler = setup_scheduler(config['scheduler'], optimizer, step_size=10, gamma=0.1, epochs=config['num_epochs'])

    monitor = setup_monitor({
        'key_metric': config['key_metric'],
        'mode': config['mode'],
        'save_best_model': config['save_best_model'],
        'best_model_file': os.path.join(config['test_dir'], config['best_model_file_name']),
        'save_intermediate_models': config['save_intermediate_models'],
        'analysis_dir': os.path.join(config['test_dir'], 'analysis'),
        'use_early_stopping': config['use_early_stopping'],
        'patience': config.get('patience', 10),
        'delta': config.get('delta', 1e-4),
        'write_to_tensorboard': config['write_to_tensorboard'],
        'writer': SummaryWriter(log_dir=config['test_dir'])
    }, verbose=config['verbose'])

    return device, model, optimizer, scheduler, monitor

def setup_train_config(device, model, optimizer, scheduler, monitor, train_loader, val_loader, train_loader_clean, config):
    metric_names = config.get("training_metrics", [])
    nearest_neighbors = config.get("nearest_neighbors", 5)

    # Optional metrics
    knn_analyzer = KNNAnalyzer(train_loader_clean, k=nearest_neighbors) if "knn_accuracy" in metric_names else None
    knr_analyzer = KNRAnalyzer(train_loader_clean, k=nearest_neighbors, metric='mae') if "knr_error" in metric_names else None

    # Only include analyzers if they're used
    metrics_config = {
        "metric_names": metric_names,
    }
    if knn_analyzer:
        metrics_config["knn_analyzer"] = knn_analyzer
    if knr_analyzer:
        metrics_config["knr_analyzer"] = knr_analyzer

    # Create the final training config object
    train_config = {
        "device": device,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "temperature": config['temperature'],
        "train_config": {
            "model": model,
            "num_epochs": config["num_epochs"],
            "monitor": monitor,
            "metrics": metrics_config
        }
    }

    return train_config
