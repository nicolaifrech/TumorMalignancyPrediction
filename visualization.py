import torch
import torch.nn.functional as F
import math
import umap
import matplotlib.pyplot as plt
import numpy as np

from utils.config import check_config
from utils.printing import print_verbose

def plot_umap(embeddings, labels, figsize=(5, 5), title="UMAP Projection", verbose=True, **kwargs):
    # Create a UMAP reducer
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', n_jobs=1)
    embedding_2d = reducer.fit_transform(embeddings)

    print_verbose(f"[plot_umap] embeddings: {tuple(embeddings.shape)}, labels: {tuple(labels.shape)}", verbose)

    # Plot the UMAP projection
    plt.figure(figsize=figsize)
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='Spectral', alpha=0.7)
    plt.title(title)
    plt.colorbar(scatter, label="Age")
    plt.show()

def plot_umap_all_epochs(embeddings_list, labels_list, max_cols=4, figssize=(5, 5), verbose=True, **kwargs):
    """
    Plot UMAP projections from multiple epochs in a grid layout.

    Args:
        embeddings_list (list of Tensors or np.ndarrays): Embeddings from each epoch.
        labels_list (list of Tensors or np.ndarrays): Corresponding labels.
        max_cols (int): Max number of columns in the subplot grid.
        figsize (tuple): Size of each subplot (width, height).
        verbose (bool): Whether to print debug info.
    """
    num_epochs = len(embeddings_list)
    cols = min(num_epochs, max_cols)
    rows = math.ceil(num_epochs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(figssize[0] * cols, figssize[1] * rows))
    axes = axes.flatten() if num_epochs > 1 else [axes]

    for i, (embeddings, labels) in enumerate(zip(embeddings_list, labels_list)):
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', n_jobs=1)
        embedding_2d = reducer.fit_transform(embeddings)

        if verbose:
            print_verbose(f"[plot_umap_all_epochs_grid] Epoch {i}: {embeddings.shape}, {labels.shape}", verbose)

        ax = axes[i]
        scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='Spectral', alpha=0.7)
        ax.set_title(f"Epoch {i}")
        ax.axis('off')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.colorbar(scatter, ax=axes, location='right', label='Label')
    plt.show()

def plot_similarity_matrix(embeddings, labels, n=2000, figsize=(5, 5), title='Feature Similarity Matrix', verbose=True, **kwargs):
    """
    Plots a cosine similarity matrix for a subset of the given embeddings,
    sorted by their labels.

    Args:
        embeddings (Tensor or np.ndarray): Shape (N, D)
        labels (Tensor or np.ndarray): Shape (N,)
        n (int): Number of samples to visualize (default: 2000)
        title (str): Title for the plot
    """
   
    # Subsample if needed
    if n is not None and len(embeddings) > n:
        indices = torch.randperm(len(embeddings))[:n]
        embeddings = embeddings[indices]
        labels = labels[indices]

    # Sort by label
    sorted_indices = torch.argsort(labels)
    sorted_embeddings = embeddings[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Compute cosine similarity matrix
    sim_matrix = F.cosine_similarity(
        sorted_embeddings.unsqueeze(1),  # (N, 1, D)
        sorted_embeddings.unsqueeze(0),  # (1, N, D)
        dim=-1
    ).numpy()

    print_verbose(f"[plot_similarity_matrix] embeddings: {embeddings.shape}, labels: {labels.shape}", verbose)

    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(sim_matrix, cmap='plasma', interpolation='nearest')
    plt.title(title)
    plt.colorbar(label='Cosine similarity')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

VISUALIZATION_FUNCS = {
    "umap": {
        "func": plot_umap,
        "required": {"embeddings", "labels"}
    },
    "similarity_matrix": {
        "func": plot_similarity_matrix,
        "required": {"embeddings", "labels"}
    },
    "umap_all": {
        "func": plot_umap_all_epochs,
        "required": {"embeddings_list", "labels_list"}
    }
}

def visualize(selected: list[str], args: dict, verbose=True):
    for name in selected:
        if name not in VISUALIZATION_FUNCS:
            raise ValueError(f"Unknown visualization: '{name}'")

        entry = VISUALIZATION_FUNCS[name]
        func = entry["func"]
        required_args = entry.get("required", set())

        # Validate required arguments
        check_config(args, required_args, verbose=False)

        args.setdefault("verbose", verbose)

        print_verbose(f"[visualize] Running: {name}", verbose)
        func(**args)
