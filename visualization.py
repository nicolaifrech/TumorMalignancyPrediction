import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os

from model import Encoder
from datasets.utk_dataset import UTKFaceDataset
from utils.config import extract_config
from utils.extraction import extract_embeddings
from utils.transforms import get_transforms
from utils.printing import print_verbose

def load_model(model, filepath, output_dim=128, verbose=True): 
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print_verbose(f"Loaded model from {filepath}, epoch {checkpoint['epoch']}, best loss: {checkpoint['metrics']['train_loss']:.4f}", verbose)
    return model

def get_full_loader(data_folder, batch_size=64):
    transform = get_transforms('val', '')  # Minimal transformation for visualization
    dataset = UTKFaceDataset(data_folder=data_folder, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader

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

def visualize_similarity_matrix(model, dataloader, device, n=2000, title='Feature Similarity Matrix'):
    """
    Randomly samples `n` examples from `dataloader`, computes cosine similarity
    of their embeddings, and plots a similarity matrix sorted by labels.

    Args:
        model: Trained model.
        dataloader: A DataLoader yielding (inputs, labels).
        device: CUDA or CPU device.
        n: Number of samples to use (default: 2000).
        title: Title for the plot.
    """
    model.eval()
    embeddings = []
    labels = []

    # Collect all embeddings and labels
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            emb = model(x)
            embeddings.append(emb.cpu())
            labels.append(y.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    # Sample n examples randomly
    if len(embeddings) > n:
        indices = torch.randperm(len(embeddings))[:n]
        embeddings = embeddings[indices]
        labels = labels[indices]

    # Sort by label
    sorted_indices = torch.argsort(labels)
    sorted_embeddings = embeddings[sorted_indices]
    sorted_labels = labels[sorted_indices] 

    # Compute cosine similarity matrix
    sim_matrix = F.cosine_similarity(
        sorted_embeddings.unsqueeze(1),
        sorted_embeddings.unsqueeze(0),
        dim=-1
    ).numpy()

    # Plot
    plt.figure(figsize=(5, 5))
    plt.imshow(sim_matrix, cmap='plasma', interpolation='nearest')
    plt.title(title)
    plt.colorbar(label='Cosine similarity')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize(config, verbose=True):

    cfg = extract_config(
        config,
        required={
            'device', 'data_folder', 'best_model_path', 'model' 
        },  
        optional={
            'fig_size': 5
        },  
        verbose=verbose
    )   

    model = load_model(cfg.model, cfg.best_model_path, verbose=verbose)
    full_loader = get_full_loader(cfg.data_folder)
    
    # Disable the projection layer
    if hasattr(cfg.model, 'projector'):
        model.projector = nn.Identity()
        
    embeddings, labels = extract_embeddings(cfg.model, full_loader, device=cfg.device, verbose=verbose)
    print_verbose(f"Shape of embeddings: {embeddings.shape}", verbose)
    print_verbose(f"Memory usage of embeddings: {embeddings.nbytes / (1024 ** 2):.2f} MB", verbose) 
    plot_umap(embeddings, labels, fig_size=cfg.fig_size)

    visualize_similarity_matrix(cfg.model, full_loader, cfg.device)

