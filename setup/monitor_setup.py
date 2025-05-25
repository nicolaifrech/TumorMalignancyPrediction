from utils.config import extract_config
from utils.printing import print_verbose
from analysis.monitor import (
    LoggingMonitor,
    OptimumMonitor,
    EarlyStoppingMonitor,
    SaveModelMonitor,
    TensorboardMonitor,
    CompositeMonitor
)

def setup_optimum_monitor(key_metric, mode, config, verbose=True):
    cfg = extract_config(
        config,
        required={
            'best_model_file'
        },
        verbose=False
    ) 
    return OptimumMonitor(key_metric=key_metric, mode=mode, save_path=cfg.best_model_file, verbose=verbose)

def setup_intermediate_model_monitor(config, verbose=True):
    cfg = extract_config(
        config,
        required={
            'analysis_dir'
        },
        verbose=False
    ) 
    return SaveModelMonitor(save_dir=cfg.analysis_dir)

def setup_early_stopping_monitor(key_metric, mode, config, verbose=True): 
    patience = config.get('patience', 10)
    delta = config.get('delta', 1e-4)
    return EarlyStoppingMonitor(key_metric=key_metric, patience=patience, delta=delta, mode=mode, verbose=verbose) 

def setup_tensorboard_monitor(config, verbose=True):
    cfg = extract_config(
        config,
        required={
            'writer'
        },
        verbose=False
    ) 
    return TensorboardMonitor(cfg.writer) 

def setup_monitor(config, verbose=True):
    cfg = extract_config(
        config,
        required={
            'key_metric', 'mode'
        },
        optional={
            'save_best_model': False,
            'save_intermediate_models': False,
            'use_early_stopping': False,
            'write_to_tensorboard': False
        },
        verbose=False
    )

    monitors = []
    monitors.append(LoggingMonitor(key_metric=cfg.key_metric, mode=cfg.mode, verbose=verbose))
    
    if cfg.save_best_model:
        monitors.append(setup_optimum_monitor(cfg.key_metric, cfg.mode, config, verbose))
    if cfg.save_intermediate_models:
        monitors.append(setup_intermediate_model_monitor(config, verbose))
    if cfg.use_early_stopping:
        monitors.append(setup_early_stopping_monitor(cfg.key_metric, cfg.mode, config, verbose))
    if cfg.write_to_tensorboard:
        monitors.append(setup_tensorboard_monitor(config, verbose))

    print_verbose(f"Enabled monitors: {[type(m).__name__ for m in monitors]}", verbose)
    return CompositeMonitor(monitors)
