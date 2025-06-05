import torch
import time
import warnings
import os
import sys
from numba.core.errors import NumbaWarning

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from device import setup_device
from datasets.loading import get_data_loaders
from model.loading import load_model
from train.train_encoder import train_encoder
from utils.checkpoint import save_checkpoint
from utils.config_io import save_configuration
from utils.cleanup import clear_gpu_cache, run_gc
from utils.printing import print_verbose
from utils.seed import set_seed
from utils.cli import load_config
from setup.model_setup import setup_encoder
from setup.test_dir_setup import setup_testing_directory
from setup.training_setup import setup_training_environment, setup_train_config

warnings.filterwarnings("ignore", message=".*force_all_finite.*")
warnings.filterwarnings("ignore", category=NumbaWarning)
warnings.filterwarnings("ignore", message=".*verbose parameter is deprecated.*")
warnings.filterwarnings("ignore", message=".*epoch parameter in `scheduler.step.*")

# SETUP

def prepare_test(config):
    clear_gpu_cache()
    run_gc()
    set_seed(config['seed'])

def setup_test_environment(config):
    setup_testing_directory(config, create_unique_dir=True)
    save_configuration(config)

def setup_test(config):
    device, model, optimizer, scheduler, monitor = setup_training_environment(
        {**config, 'model_builder': setup_encoder}
    )   

    train_loader, val_loader, test_loader, train_loader_clean = get_data_loaders(config)

    train_config = setup_train_config(
        device, model, optimizer, scheduler, monitor,
        train_loader, val_loader, train_loader_clean,
        config
    )
    return model, train_config

# TRAINING

def run_test(config, verbose=True):
    start_time = time.time()
    train_encoder(config, verbose=verbose)
    end_time = time.time()
    print_verbose(f"Training completed in: {end_time - start_time:.2f} seconds", verbose)

def main():
    config = load_config()
    prepare_test(config)
    setup_test_environment(config)
    model, train_config = setup_test(config)
    save_checkpoint(model, config['initial_model_file'], epoch=0)
    run_test(train_config, config['verbose'])

if __name__ == "__main__":
    main()
