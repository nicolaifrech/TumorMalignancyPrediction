import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from utk_dataset import UTKFaceDataset
from model import Encoder
from train_encoder import *
from utils import *

def load_model(filepath, backbone_name='resnet18', output_dim=128):
    model = Encoder(backbone_name=backbone_name, output_dim=output_dim)
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {filepath}, epoch {checkpoint['epoch']}, best loss: {checkpoint['val_mae']:.4f}")
    return model

def get_full_loader(data_folder, batch_size=64):
    transform = get_transforms('val', '')  # Minimal transformation for visualization
    dataset = UTKFaceDataset(data_folder=data_folder, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader

def extract_embeddings(model, loader, device):
    model.to(device)
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        with tqdm(loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                tepoch.set_description("Extracting Embeddings")

                images = images.to(device)
                embeddings = model(images)
                all_embeddings.append(embeddings.cpu())
                all_labels.extend(labels.cpu().numpy())

                # Update progress bar with the number of extracted embeddings
                tepoch.set_postfix(embeddings=len(all_embeddings))

    all_embeddings = torch.cat(all_embeddings).numpy()
    all_labels = np.array(all_labels)
    return all_embeddings, all_labels

def plot_umap(embeddings, labels, fig_size=8, title="UMAP Projection"):
    # Create a UMAP reducer
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', n_jobs=1)
    embedding_2d = reducer.fit_transform(embeddings)

    # Plot the UMAP projection
    plt.figure(figsize=(fig_size, fig_size))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='Spectral', alpha=0.7)
    plt.title(title)
    plt.colorbar(scatter, label="Age")
    plt.show()

def visualize(config):

    required_keys = {
        'device', 'data_folder', 'model_file', 'backbone_model', 'fig_size'
    }
    check_config(config, required_keys)

    model = load_model(config['model_file'], backbone_name=config['backbone_model'])
    full_loader = get_full_loader(config['data_folder']) 
    
    # Disable the projection layer
    if hasattr(model, 'projector'):
        model.projector = nn.Identity()
        
    embeddings, labels = extract_embeddings(model, full_loader, device=config['device'])
    print(f"Shape of embeddings: {embeddings.shape}")
    print(f"Memory usage of embeddings: {embeddings.nbytes / (1024 ** 2):.2f} MB")
    plot_umap(embeddings, labels, fig_size=config['fig_size'])

