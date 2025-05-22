import torch
import os
from torch.utils.tensorboard import SummaryWriter

from utils.config import check_config
from utils.printing import print_verbose

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

        check_config(config, {'key_metric'}, verbose=False)
        
        self.key_metric = config['key_metric']
        self.save_analysis = config.get('save_analysis', False)
        self.mode = config.get('mode', 'min')
        self.base_dir = config.get('base_dir', 'outputs/current')
        self.best_model_file = config.get('best_model_file', os.path.join(self.base_dir, 'best_model.pth'))
        self.analysis_sub_dir = config.get('analysis_dir', 'analysis')
        self.writer = config.get('writer', SummaryWriter(log_dir=self.base_dir)) 

        # Parameters for early stopping
        self.early_stopping = config.get('early_stopping', False)
        self.patience = config.get('patience', 10)
        self.delta = config.get('delta', 0.0)  # Minimum change to qualify as improvement
        self._epochs_since_improvement = 0 
 
        self.analysis_dir = os.path.join(self.base_dir, self.analysis_sub_dir) 

        self.best_value = float('inf') if self.mode == 'min' else -float('inf')
        self.compare = min if self.mode == 'min' else max 

        os.makedirs(self.analysis_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.best_model_file), exist_ok=True)

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
        if self.save_analysis and model is not None and epoch is not None:
            analysis_path = os.path.join(self.analysis_dir, f"epoch_{epoch:03d}.pt")
            self.save(analysis_path, model, epoch, metrics, extra_data)

        if self.writer and epoch is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(name, value, epoch)

        # Save best model if improved
        new_value = metrics[self.key_metric]
        if self.compare(new_value, self.best_value) != self.best_value:
            self.best_value = new_value
            self._epochs_since_improvement = 0
            if model is not None:
                self.save(self.best_model_file, model, epoch, metrics, extra_data)
            print_verbose(f"✅ New best {self.key_metric}: {new_value:.4f} ({self.mode})", self.verbose) 
            return 'improvement'
        else:
            if self.early_stopping:
                self._epochs_since_improvement += 1
                if self._epochs_since_improvement >= self.patience: 
                    print_verbose(f"⏹️_ Early stopping triggered after {self.patience} epochs without improvement.", self.verbose) 
                    return 'early_stop'
            return 'no_improvement'

    def close(self):
        """
        Gracefully close resources (e.g. TensorBoard writer).
        """
        if self.writer is not None:
            self.writer.close()
            print_verbose("📝 TensorBoard writer closed.", self.verbose)
