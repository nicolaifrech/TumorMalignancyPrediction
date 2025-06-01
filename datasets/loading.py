import torch
from torch.utils.data import DataLoader, random_split
import json
from os.path import join, exists


from datasets.transforms import get_transforms, TwoCropTransform
from datasets.splits import load_or_create_split
from datasets.utk_dataset import UTKFaceDataset

def train_collate_fn(batch):
    images1 = torch.stack([item[0][0] for item in batch], dim=0)
    images2 = torch.stack([item[0][1] for item in batch], dim=0)
    ages = torch.tensor([item[1] for item in batch])
    return (images1, images2), ages

def val_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    ages = torch.tensor([item[1] for item in batch])
    return images, ages

def get_data_loaders(config):
    data_folder = config['data_folder']
    split_name = config['split']
    val_size = config['val_size']
    test_size = config['test_size'] 
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

    return train_loader, val_loader, test_loader, train_clean_loader

def get_full_loader(data_folder, batch_size=64, num_workers=4):
    transform = get_transforms('val', '')  # Minimal transform for visualization
    dataset = UTKFaceDataset(data_folder=data_folder, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn)
    return loader
