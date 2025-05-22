import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model import Encoder
from loss import RnCLoss
from analysis import Monitor
from metrics import compute_metrics
from train.train_one_epoch import train_one_epoch
from utk_dataset import UTKFaceDataset

from utils.config import check_config
from utils.transforms import get_transforms, TwoCropTransform
from utils.printing import print_verbose

# Used in get_data_loaders.
# Replaces the lambda to allow multiprocessing on macOS and Linux
def collate_fn(batch):
    images1 = torch.stack([item[0][0] for item in batch], dim=0)
    images2 = torch.stack([item[0][1] for item in batch], dim=0)
    ages = torch.tensor([item[1] for item in batch])
    return (images1, images2), ages

def train_collate_fn(batch):
    """
    Collate function for the training data loader.
    Combines pairs of augmented images and their corresponding labels.
    """
    images1 = torch.stack([item[0][0] for item in batch], dim=0)
    images2 = torch.stack([item[0][1] for item in batch], dim=0)
    ages = torch.tensor([item[1] for item in batch])
    return (images1, images2), ages

def val_collate_fn(batch):
    """
    Collate function for the validation data loader.
    Combines single images and their corresponding labels.
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    ages = torch.tensor([item[1] for item in batch])
    return images, ages

def get_data_loaders(data_folder, aug, batch_size=64, train_size=0.8):
    # Initialize transforms
    train_transform = TwoCropTransform(get_transforms('train', aug))
    val_transform = get_transforms('val', '')

    # Load the full dataset with respective transforms
    full_dataset = UTKFaceDataset(data_folder=data_folder)

    # Get dataset lengths for splitting
    train_len = int(train_size * len(full_dataset))
    val_len = len(full_dataset) - train_len

    # Random split for indices (to ensure reproducibility)
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [train_len, val_len]
    )

    # Create train and validation datasets using the precomputed indices
    train_ds = UTKFaceDataset(data_folder=data_folder, transform=train_transform)
    val_ds = UTKFaceDataset(data_folder=data_folder, transform=val_transform)

    # Subset the datasets using the indices
    train_ds = torch.utils.data.Subset(train_ds, train_indices)
    val_ds = torch.utils.data.Subset(val_ds, val_indices)

    num_workers = 4
    # Train DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate_fn
    )

    # Validation DataLoader
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, val_loader
 
def train_encoder(config, verbose=True):

    # Required onfig parameters
    required_keys = {
        'device', 'data_folder', 'batch_size', 'model',
	    'num_epochs', 'learning_rate', 'temperature', 'augmentations', 'train_size',
        'monitor_config'
    }
    check_config(config, required_keys, verbose=verbose)
    
    device = config['device']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate'] 
    aug = config['augmentations']
    data_folder = config['data_folder']
    train_size = config['train_size']
    model = config['model']
    monitor_config = config['monitor_config']

    # Optional config parameters
    default_metrics = [
        "val_loss", "embedding_norm",
        "embedding_variance", "lr"
    ]
    metric_names = config.get('metrics', default_metrics)

    print_verbose(f"Training on device: {device}", verbose) 

    train_loader, val_loader = get_data_loaders(data_folder, aug, batch_size=batch_size, train_size=train_size)
 
    # Initialize Rank-N-Contrast loss
    criterion = RnCLoss(temperature=config['temperature'], label_diff='l1', feature_sim='l2').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=False
    )

    monitor = Monitor(monitor_config, verbose=verbose)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(epoch, model, train_loader, criterion, optimizer, num_epochs, device, verbose)
        scheduler.step(train_loss)
 
        other_metrics = compute_metrics(model, val_loader, device, metric_names, optimizer=optimizer)
        metrics = {
            'train_loss': train_loss,
            **other_metrics,
        }
        training_status = monitor.update(model, metrics, epoch=epoch) 
        if training_status == 'early_stop':
            print_verbose("Stopping early.", verbose)
            break
    
    monitor.close()
    print_verbose("Training complete!", verbose)  
