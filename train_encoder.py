import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from tqdm import tqdm

from utk_dataset import UTKFaceDataset
from model import Encoder
from loss import RnCLoss  # Your provided loss implementation
from utils import *

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

    # Train DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=train_collate_fn
    )

    # Validation DataLoader
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=val_collate_fn
    )

    return train_loader, val_loader
 
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    # Use tqdm to create a progress bar
    with tqdm(loader, unit="batch") as tepoch:
        for (images1, images2), ages in tepoch:
            tepoch.set_description("Training")

            # Move images and labels to the device
            images1, images2 = images1.to(device), images2.to(device)
            ages = ages.to(device).float().unsqueeze(1)
            bsz = ages.size(0)  # Batch size

            optimizer.zero_grad()

            # Efficient forward pass: concatenate the image pairs
            #images = torch.cat((images1, images2), dim=0)  # Shape: [2*bsz, C, H, W] 
            images = torch.cat((images1, images2), dim=0)  # Concatenate along batch dimension
            embeddings = model(images)  # Single forward pass
            embeddings1, embeddings2 = torch.split(embeddings, [bsz, bsz], dim=0)  # Split back
            
            # Stack the embeddings to match the required format [bsz, 2, embedding_dim]
            features = torch.stack([embeddings1, embeddings2], dim=1)

            # Calculate loss
            loss = criterion(features, ages)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            tepoch.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    return avg_loss

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0 
    with torch.no_grad():
        for images, ages in loader:
            images = images.to(device)
            ages = ages.to(device).float().unsqueeze(1)
            embeddings = model(images)
            preds = embeddings.mean(dim=1)
            mae = torch.abs(preds - ages).mean().item()
            total_loss += mae 
    return total_loss / len(loader) 

def save_checkpoint(model, epoch, val_mae, file_name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_mae': val_mae
    }, file_name)
    print(f"Model saved at epoch {epoch} with validation MAE: {val_mae:.4f}")

def train_encoder(config):

    required_keys = {
        'device', 'data_folder', 'model_file', 'backbone_model',
        'batch_size', 'num_epochs', 'learning_rate', 'temperature',
        'augmentations', 'train_size'
    }
    check_config(config, required_keys)
    
    device = config['device']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate'] 
    aug = config['augmentations']
    data_folder = config['data_folder']
    train_size = config['train_size']

    print(f"Training on device: {device}")

    train_loader, val_loader = get_data_loaders(data_folder, aug, batch_size=batch_size, train_size=train_size)

    model = Encoder(backbone_name=config['backbone_model']).to(device)   
    # Initialize Rank-N-Contrast loss
    criterion = RnCLoss(temperature=config['temperature'], label_diff='l1', feature_sim='l2').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #best_val_mae = float('inf')
    best_train_loss = float('inf')

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        #val_mae = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")
        #print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val MAE = {val_mae:.2f}")

        #if val_mae < best_val_mae:
        #    best_val_mae = val_mae
        #    save_checkpoint(model, epoch, best_val_mae, file_name="model_checkpoint.pth")
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            save_checkpoint(model, epoch, best_train_loss, file_name=config['model_file'])
    
    print("Training complete!")
