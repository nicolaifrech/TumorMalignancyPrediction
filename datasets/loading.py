import torch
from torch.utils.data import DataLoader, random_split

from datasets.transforms import get_transforms, TwoCropTransform
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

def get_data_loaders(data_folder, aug, batch_size=64, train_size=0.8, num_workers=4):
    train_transform = TwoCropTransform(get_transforms('train', aug))
    val_transform = get_transforms('val', '')

    full_dataset = UTKFaceDataset(data_folder=data_folder)

    train_len = int(train_size * len(full_dataset))
    val_len = len(full_dataset) - train_len

    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_len, val_len]
    )

    train_ds = UTKFaceDataset(data_folder=data_folder, transform=train_transform)
    val_ds = UTKFaceDataset(data_folder=data_folder, transform=val_transform)

    train_ds = torch.utils.data.Subset(train_ds, train_indices)
    val_ds = torch.utils.data.Subset(val_ds, val_indices)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=train_collate_fn
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=val_collate_fn
    )

    return train_loader, val_loader

def get_full_loader(data_folder, batch_size=64, num_workers=4):
    transform = get_transforms('val', '')  # Minimal transform for visualization
    dataset = UTKFaceDataset(data_folder=data_folder, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn)
    return loader
