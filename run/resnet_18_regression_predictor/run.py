import torch
import time
import warnings
import os
import sys
from numba.core.errors import NumbaWarning

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from device import setup_device
from model.loading import load_model
from train.train_regression_predictor import train_regression_predictor
from utils.checkpoint import save_checkpoint
from utils.config_io import save_configuration
from utils.cleanup import clear_gpu_cache, run_gc
from utils.printing import print_verbose
from utils.seed import set_seed
from utils.cli import load_config
from setup.model_setup import setup_predictor
from setup.utk_data_loading import get_data_loaders_for_predictor_training
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
        {**config, 'model_builder': setup_predictor}
    )   

    train_loader, val_loader, test_loader, ordinal_map = get_data_loaders_for_predictor_training(config)

    train_config = setup_train_config(
        device, model, optimizer, scheduler, monitor,
        train_loader, val_loader, train_loader, ordinal_map,
        config
    )
    return train_config

# TRAINING

def run_test(config, verbose=True):
    start_time = time.time()
    train_regression_predictor(config, verbose=verbose)
    end_time = time.time()
    print_verbose(f"Training completed in: {end_time - start_time:.2f} seconds", verbose)

def main():
    config = load_config()
    prepare_test(config)
    setup_test_environment(config)
    train_config = setup_test(config) 
    run_test(train_config, config['verbose'])

if __name__ == "__main__":
    main()
