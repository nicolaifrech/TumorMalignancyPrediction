import torch
import os
from torch.utils.tensorboard import SummaryWriter

from utils.config import extract_config
from utils.printing import print_verbose

# TODO: Single responsibility principle!
class Monitor:
    def __init__(self, config, verbose=True):
        """
        Args:
            key_metric (str): Name of the metric to monitor.
            save_analysis (bool): Whether to save result after every epoch to see progress.
            mode (str): 'min' or 'max'.
            base_dir (str): Base folder where everything will be saved.
            writer (SummaryWriter): Writer object to access tensor board.
            verbose (bool): Whether to print updates.
        """ 
        self.verbose = verbose

        self.cfg = extract_config(
            config,
            required={
                'key_metric'
            },  
            optional={
                'save_analysis': False,
                'mode': 'min',
                'base_dir': 'outputs/current',
                'best_model_file': 'outputs/current/best_model.pth',
                'analysis_dir': 'outputs/current/analysis',
                'writer': None,
                'early_stopping': False,
                'patience': 10,
                'delta': 0.0
            },  
            verbose=False
        )
    
        self.cfg.writer = self.cfg.writer or SummaryWriter(log_dir=self.cfg.base_dir)
        self._epochs_since_improvement = 0 

        self.best_value = float('inf') if self.cfg.mode == 'min' else -float('inf')
        self.compare = min if self.cfg.mode == 'min' else max 

        os.makedirs(self.cfg.analysis_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.cfg.best_model_file), exist_ok=True)

    def save(self, path, model, epoch, metrics, extra_data):
        data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics
        }
        if extra_data:
            data.update(extra_data)
        torch.save(data, path)

    def update(self, model, metrics, epoch=None, extra_data=None):
        """
        Checks for improvement and optionally saves model and analysis data.

        Returns:
            improved (bool): Whether a new best value was found.
        """
        # Save analysis data if wanted
        if self.cfg.save_analysis and model is not None and epoch is not None:
            analysis_path = os.path.join(self.cfg.analysis_dir, f"epoch_{epoch:03d}.pt")
            self.save(analysis_path, model, epoch, metrics, extra_data)

        if self.cfg.writer and epoch is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.cfg.writer.add_scalar(name, value, epoch)

        # Save best model if improved
        new_value = metrics[self.cfg.key_metric]
        if self.compare(new_value, self.best_value) != self.best_value:
            self.best_value = new_value
            self._epochs_since_improvement = 0
            if model is not None:
                self.save(self.cfg.best_model_file, model, epoch, metrics, extra_data)
            print_verbose(f"✅ New best {self.cfg.key_metric}: {new_value:.4f} ({self.cfg.mode})", self.verbose) 
            return 'improvement'
        else:
            if self.cfg.early_stopping:
                self._epochs_since_improvement += 1
                if self._epochs_since_improvement >= self.cfg.patience: 
                    print_verbose(f"⏹️_ Early stopping triggered after {self.cfg.patience} epochs without improvement.", self.verbose) 
                    return 'early_stop'
            return 'no_improvement'

    def close(self):
        """
        Gracefully close resources (e.g. TensorBoard writer).
        """
        if self.cfg.writer is not None:
            self.cfg.writer.close()
            print_verbose("📝 TensorBoard writer closed.", self.verbose)
