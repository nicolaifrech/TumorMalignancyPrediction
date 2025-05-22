import torch
from tqdm import tqdm

def train_one_epoch(epoch, model, loader, criterion, optimizer, num_epochs, device, verbose=True):
    model.train()
    total_loss = 0 

    # Use tqdm to create a progress bar   
    with tqdm(loader, unit='batch', ncols=80, 
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]',
        leave=verbose) as tepoch:
        tepoch.set_description(f"Training Epoch {epoch+1}/{num_epochs}")
        for (images1, images2), ages in tepoch: 
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
