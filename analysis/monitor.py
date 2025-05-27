import torch
import os
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod

from utils.checkpoint import save_checkpoint
from utils.printing import print_verbose

class Monitor(ABC):
    @abstractmethod
    def update(self, model, metrics, epoch):
        pass

class CompositeMonitor(Monitor):
    def __init__(self, monitors):
        self.monitors = monitors

    def update(self, model, metrics, epoch):
        combined_results = {}
        for monitor in self.monitors:
            result = monitor.update(model, metrics, epoch)
            if result:  # skip None or empty dicts
                combined_results.update(result)
        return combined_results

class LoggingMonitor(Monitor):
    def __init__(self, key_metric, mode='min', verbose=True):
        self.key_metric = key_metric
        self.mode = mode
        self.verbose = verbose

    def update(self, model, metrics, epoch): 
        value = metrics[self.key_metric]
        msg = f"[Epoch {epoch}] {self.key_metric}: {value:.4f} ({self.mode})"
        print_verbose(msg, self.verbose) 
        return {"log_message": msg}

class OptimumMonitor(Monitor):
    def __init__(self, key_metric, mode='min', save_path='best_model.pth', verbose=True):
        self.key_metric = key_metric
        self.mode = mode  # 'min' or 'max'
        self.save_path = save_path
        self.verbose = verbose
        self.best_value = float('inf') if mode == 'min' else -float('inf')

    def update(self, model, metrics, epoch):
        new_value = metrics[self.key_metric]
        if self._is_improvement(new_value):
            self.best_value = new_value  
            save_checkpoint(model, self.save_path, epoch, metrics)
            #self.save(self.save_path, model, epoch, metrics)
            print_verbose(f"✅ New best {self.key_metric}: {new_value:.4f} ({self.mode})", self.verbose) 
            result = {
                'result': 'improvement',
                'best_value': new_value,
                'epoch': epoch,
                'save_path': self.save_path
            }
        else: 
            result = {
                'result': 'no_improvement',
                'best_value': self.best_value,
                'epoch': epoch
            }
        return {'optimization': result}

    def _is_improvement(self, value):
        if self.mode == 'min':
            return value < self.best_value
        else:
            return value > self.best_value

class EarlyStoppingMonitor(Monitor):
    def __init__(self, key_metric, patience=10, delta=0.0, mode='min', verbose=True):
        self.key_metric = key_metric
        self.patience = patience
        self.delta = delta
        self.mode = mode  # 'min' or 'max'
        self.verbose = verbose
        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.epochs_since_improvement = 0 

    def update(self, model, metrics, epoch):
        new_value = metrics[self.key_metric] 
        if self._is_improvement(new_value):
            self.best_value = new_value
            self.epochs_since_improvement = 0 
            result = 'improvement'
        else: 
            self.epochs_since_improvement += 1
            if self.epochs_since_improvement >= self.patience: 
                print_verbose(f"⏹️ Early stopping triggered after {self.patience} epochs without improvement.", self.verbose)
                result = 'early_stop' 
            else:
                result = 'no_improvement'
        return {
            'early_stopping': result
        }
    
    def _is_improvement(self, value):
        if self.mode == 'min':
            return value < self.best_value - self.delta
        else:
            return value > self.best_value + self.delta

class SaveModelMonitor(Monitor):
    def __init__(self, save_dir, filename_template="model_epoch_{epoch:03d}.pth"):
        self.save_dir = save_dir
        self.filename_template = filename_template
        os.makedirs(self.save_dir, exist_ok=True)

    def update(self, model, metrics, epoch):
        filename = self.filename_template.format(epoch=epoch)
        full_path = os.path.join(self.save_dir, filename)
        #torch.save(model.state_dict(), full_path)
        save_checkpoint(model, full_path, epoch, metrics)
        return {"model_saved_to": full_path}

class TensorboardMonitor(Monitor):
    def __init__(self, writer, prefix=""):
        self.writer = writer
        self.prefix = prefix

    def update(self, model, metrics, epoch):
        for key, value in metrics.items():
            self.writer.add_scalar(f"{self.prefix}{key}", float(value), epoch)
        return {"logged_to_tensorboard": list(metrics.keys())}
