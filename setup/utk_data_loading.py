import torch
from torch.utils.data import DataLoader, random_split
import json
from os.path import join, exists
import os
import io

from datasets.transforms import get_transforms, TwoCropTransform
from datasets.splits import load_or_create_split
from datasets.utk_dataset import UTKFaceDataset
from mappings.ordinal_map import OrdinalMap
from datasets.discretized_dataset import DiscretizedDataset
from utils.config import extract_config

def train_collate_fn(batch):
    images1 = torch.stack([item[0][0] for item in batch], dim=0)
    images2 = torch.stack([item[0][1] for item in batch], dim=0)
    ages = torch.tensor([item[1] for item in batch])
    return (images1, images2), ages

def val_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    ages = torch.tensor([item[1] for item in batch])
    return images, ages

def describe_and_save_ordinal_map(ordinal_map, loader, output_path, verbose):
    values = []
    for _, targets in loader:
        if isinstance(targets, (tuple, list)):
            targets = targets[0]
        values.append(targets)
    scalar_values = torch.cat(values, dim=0)
    ordinal_map.describe(scalar_values, file=output_path, verbose=verbose)

def get_data_loaders_for_predictor_training(config):
    cfg = extract_config(config,
        required={
            'data_folder', 'split', 'val_size', 'test_size',
            'augmentations', 'base_dir'
        },
        optional={
            'seed': 0,
            'verbose': True,
            'batch_size': 64,
            'num_workers': 4,
            'discretize_labels': False,
            'num_classes': 10,
            'dataset_description_file_name': None
        },
        verbose=False
    )
   
    # Load or create split
    split_dict = load_or_create_split(cfg.data_folder, cfg.split, cfg.val_size, cfg.test_size, seed=cfg.seed, verbose=cfg.verbose)
    
    # Transforms 
    transform = get_transforms('val', '')

    # Datasets
    train_ds = UTKFaceDataset(cfg.data_folder, split_files=split_dict, split='train', transform=transform)
    val_ds   = UTKFaceDataset(cfg.data_folder, split_files=split_dict, split='val',   transform=transform)
    test_ds  = UTKFaceDataset(cfg.data_folder, split_files=split_dict, split='test', transform=transform)

    # Discretization
    if cfg.discretize_labels:
        ordinal_map = OrdinalMap(domain=(0, 116), num_classes=cfg.num_classes)
        train_ds = DiscretizedDataset(train_ds, ordinal_map)
        val_ds = DiscretizedDataset(val_ds, ordinal_map) 
        test_ds = DiscretizedDataset(test_ds, ordinal_map) 
    else:
        ordinal_map = None 

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=cfg.num_workers, pin_memory=True) 
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=cfg.num_workers, pin_memory=True)

    describe_and_save_ordinal_map(ordinal_map, train_loader, cfg.dataset_description_file_name, cfg.verbose)

    return train_loader, val_loader, test_loader, ordinal_map 

def get_data_loaders_for_encoder_training(config):
    cfg = extract_config(config,
        required={
            'data_folder', 'split', 'val_size', 'test_size',
            'augmentations', 'train_collate_fn', 'val_collate_fn'
        },
        optional={
            'seed': 0,
            'verbose': True,
            'batch_size': 64,
            'num_workers': 4,
        },
        verbose=False
    )
   
    # Load or create split
    split_dict = load_or_create_split(cfg.data_folder, cfg.split, cfg.val_size, cfg.test_size, seed=cfg.seed, verbose=cfg.verbose)
    
    # Transforms
    train_transform = TwoCropTransform(get_transforms('train', cfg.augmentations))
    val_transform = get_transforms('val', '')

    # Datasets
    train_ds = UTKFaceDataset(cfg.data_folder, split_files=split_dict, split='train', transform=train_transform)
    val_ds   = UTKFaceDataset(cfg.data_folder, split_files=split_dict, split='val',   transform=val_transform)
    train_clean_ds  = UTKFaceDataset(cfg.data_folder, split_files=split_dict, split='train', transform=val_transform)
  
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=cfg.train_collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=cfg.num_workers, pin_memory=True) 
    train_clean_loader = DataLoader(train_clean_ds, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=cfg.num_workers, pin_memory=True)

    return train_loader, val_loader_train_clean_loader

def get_data_loaders(config):
    data_folder = config['data_folder']
    split_name = config['split']
    val_size = config['val_size']
    test_size = config['test_size'] 
    discretize_labels = config.get('discretize_labels', False)
    num_classes = config.get('num_classes', 10)
    seed = config.get('seed', 0) 
    verbose = config.get('verbose', True)

    # Load or create split
    split_dict = load_or_create_split(data_folder, split_name, val_size, test_size, seed=seed, verbose=verbose)

    # Transforms
    train_transform = TwoCropTransform(get_transforms('train', config['augmentations']))
    val_transform = get_transforms('val', '')

    # Datasets
    train_ds = UTKFaceDataset(data_folder, split_files=split_dict, split='train', transform=train_transform)
    val_ds   = UTKFaceDataset(data_folder, split_files=split_dict, split='val',   transform=val_transform) 
    test_ds         = UTKFaceDataset(data_folder, split_files=split_dict, split='test', transform=val_transform)
    train_clean_ds  = UTKFaceDataset(data_folder, split_files=split_dict, split='train', transform=val_transform)

    # Optional Discretization
    if discretize_labels:
        ordinal_map = OrdinalMap(domain=(0, 116), num_classes=num_classes)
        train_ds = DiscretizedDataset(train_ds, ordinal_map)
        val_ds = DiscretizedDataset(val_ds, ordinal_map)
        test_ds = DiscretizedDataset(test_ds, ordinal_map)
        train_clean_ds = DiscretizedDataset(train_clean_ds, ordinal_map) 
    else:
        ordinal_map = None

    # Loader params
    batch_size = config.get("batch_size", 64)
    num_workers = config.get("num_workers", 4)
    train_collate_fn = config.get("train_collate_fn")
    val_collate_fn = config.get("val_collate_fn")

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=train_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    train_clean_loader = DataLoader(train_clean_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    if ordinal_map is not None:
        describe_and_save_ordinal_map(ordinal_map, train_clean_loader, 
            os.path.join(config['base_dir'], config['dataset_description_file_name']),
            verbose=verbose
        )

    return train_loader, val_loader, test_loader, train_clean_loader, ordinal_map

#def get_full_loader(data_folder, batch_size=64, num_workers=4):
#    transform = get_transforms('val', '')  # Minimal transform for visualization
#    dataset = UTKFaceDataset(data_folder=data_folder, transform=transform)
#    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn)
#    return loader
