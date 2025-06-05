import torch
import time
import warnings
import os
import sys
from datetime import datetime
from numba.core.errors import NumbaWarning
import argparse

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from device import setup_device
from datasets.loading import get_data_loaders
from model.loading import load_model
from train.train_encoder import train_encoder
from analysis.visualization import visualize
from analysis.extraction import (
    extract_embeddings_from_model,
    extract_all_embeddings_from_model
)
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

def prepare_test(config):
    clear_gpu_cache()
    run_gc()
    set_seed(config['seed']) 

def setup_test_environment(config, create_unique_dir=True):
    setup_testing_directory(config, create_unique_dir=create_unique_dir)
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
    return device, model, optimizer, scheduler, monitor, train_config

def run_test(train_config, config):
    start_time = time.time()
    train_encoder(train_config, verbose=config['verbose'])
    end_time = time.time()
    print_verbose(f"Training completed in: {end_time - start_time:.2f} seconds", config['verbose'])

def run_test_group(config):
    prepare_test(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    testgroup_dir_name = f"{config['test_group_name']}_{timestamp}"

    for temperature in config['temperatures']:
        config['temperature'] = temperature
        t_str = f"lr{str(temperature).replace('.', 'p')}"

        subtest_name = f"{t_str}"
        subtest_dir = os.path.join(testgroup_dir_name, subtest_name)
        config['test_name'] = subtest_dir

        setup_test_environment(config, False)
        device, model, optimizer, scheduler, monitor, train_config = setup_test(config)
        run_test(train_config, config)

def main():
    config = load_config()
    run_test_group(config)

if __name__ == "__main__":
    main()
