import torch
import torch.nn.functional as F
import math
import umap
import matplotlib.pyplot as plt
import numpy as np

from utils.config import extract_config, check_config
from utils.printing import print_verbose

def plot_umap(embeddings, labels, figsize, title, n_neighbors, min_dist, metric, n_jobs, cmap, alpha, label, verbose=True):

    # Create a UMAP reducer
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, n_jobs=n_jobs)
    embedding_2d = reducer.fit_transform(embeddings)

    print_verbose(f"[plot_umap] embeddings: {tuple(embeddings.shape)}, labels: {tuple(labels.shape)}", verbose)

    # Plot the UMAP projection
    plt.figure(figsize=figsize)
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap=cmap, alpha=alpha)
    plt.title(title)
    plt.colorbar(scatter, label=label)
    plt.show()

def plot_umaps(umap_data, verbose=True, **kwargs):
    for config in umap_data:
        cfg = extract_config(
            config,
            required={
                'embeddings', 'labels'
            },
            optional={
                'figsize': (5, 5),
                'title': "UMAP Projection",
                'num_neighbors': 15,
                'min_dist': 0.1,
                'metric': 'euclidean',
                'num_jobs': 1,
                'cmap': 'Spectral',
                'alpha': 0.7,
                'label': 'Age'
            },
            verbose=False
        )
        plot_umap(
            cfg.embeddings, cfg.labels, cfg.figsize, cfg.title, 
            cfg.num_neighbors, cfg.min_dist, cfg.metric, cfg.num_jobs, 
            cfg.cmap, cfg.alpha, cfg.label, verbose
        )

def plot_umap_all_epochs(umap_all_data, verbose=True, **kwargs):
    """
    Plot UMAP projections from multiple epochs in a grid layout.

    Args:
        embeddings_list (list of Tensors or np.ndarrays): Embeddings from each epoch.
        labels_list (list of Tensors or np.ndarrays): Corresponding labels.
        max_cols (int): Max number of columns in the subplot grid.
        figsize (tuple): Size of each subplot (width, height).
        verbose (bool): Whether to print debug info.
    """

    cfg = extract_config(
        umap_all_data,
        required={
            'embeddings_list', 'labels_list'
        },
        optional={
            'max_cols': 4,
            'figsize': (5, 5),
            'num_neighbors': 15,
            'min_dist': 0.1,
            'metric': 'euclidean',
            'num_jobs': 1,
            'cmap': 'Spectral',
            'alpha': 0.7
        },
        verbose=False
    )

    if len(cfg.embeddings_list) != len(cfg.labels_list):
        raise ValueError("Mismatch between number of embeddings and labels in plot_umap_all_epochs.")

    num_epochs = len(cfg.embeddings_list)
    cols = min(num_epochs, cfg.max_cols)
    rows = math.ceil(num_epochs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cfg.figsize[0] * cols, cfg.figsize[1] * rows))
    axes = axes.flatten() if num_epochs > 1 else [axes]

    for i, (embeddings, labels) in enumerate(zip(cfg.embeddings_list, cfg.labels_list)):
        reducer = umap.UMAP(n_neighbors=cfg.num_neighbors, min_dist=cfg.min_dist, metric=cfg.metric, n_jobs=cfg.num_jobs)
        embedding_2d = reducer.fit_transform(embeddings)

        print_verbose(f"[plot_umap_all_epochs_grid] Epoch {i+1}: {tuple(embeddings.shape)}, {tuple(labels.shape)}", verbose)

        ax = axes[i]
        scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap=cfg.cmap, alpha=cfg.alpha)
        ax.set_title(f"Epoch {i+1}")
        ax.axis('off')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.colorbar(scatter, ax=axes, location='right', label='Label')
    plt.show()

def plot_similarity_matrix(embeddings, labels, num_samples, figsize, title, cmap, interpolation, label, verbose=True):
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
    if num_samples is not None and len(embeddings) > num_samples: 
        indices = torch.randperm(len(embeddings))[:num_samples] 
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
    plt.imshow(sim_matrix, cmap=cmap, interpolation=interpolation)
    plt.title(title)
    plt.colorbar(label=label)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_similarity_matrices(similarity_matrix_data, verbose=True, **kwargs):
    for config in similarity_matrix_data:
        cfg = extract_config(
            config,
            required={
                'embeddings', 'labels'
            },
            optional={
                'num_samples': 2000,
                'figsize': (5, 5),
                'title': 'Feature Similarity Matrix', 
                'cmap': 'plasma',
                'interpolation': 'nearest',
                'label': 'Cosine similarity'
            },
            verbose=False
        )
        plot_similarity_matrix(
            cfg.embeddings, cfg.labels, cfg.num_samples, cfg.figsize, cfg.title, 
            cfg.cmap, cfg.interpolation, cfg.label, verbose=verbose
        )

VISUALIZATION_FUNCS = {
    "umap": {
        "func": plot_umaps,
        "required": {"umap_data"}
    },
    "umap_all": {
        "func": plot_umap_all_epochs,
        'required': {'umap_all_data'}
        #"required": {"umap_embeddings_list", "umap_labels_list"}
    },
    "similarity_matrix": {
        "func": plot_similarity_matrices,
        "required": {'similarity_matrix_data'}
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
