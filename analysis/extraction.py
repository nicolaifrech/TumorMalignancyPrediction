import os
import torch.nn as nn

from datasets.loading import get_full_loader
from datasets.extraction import extract_embeddings
from model.loading import load_model
from utils.printing import print_verbose

def extract_best_encoder_embeddings(model, best_model_file, data_path, device, verbose=True):
    # Load best model weights
    model = load_model(model, best_model_file, device=device, verbose=verbose)

    # Remove projection head if present
    if hasattr(model, 'projector'):
        model.projector = nn.Identity()

    # Load full dataset
    full_loader = get_full_loader(data_path)

    # Extract embeddings
    embeddings, labels = extract_embeddings(model, full_loader, device=device, verbose=verbose)

    # Logging
    print_verbose(f"Shape of embeddings: {tuple(embeddings.shape)}", verbose)
    print_verbose(f"Memory usage of embeddings: {embeddings.nbytes / (1024 ** 2):.2f} MB", verbose)

    return embeddings, labels

def extract_encoder_embeddings_over_epochs(
    model,
    checkpoint_dir,
    data_path,
    device,
    verbose=True
):
    checkpoint_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    )

    full_loader = get_full_loader(data_path)

    embeddings_list = []
    labels_list = []

    for ckpt_file in checkpoint_files:
        model_path = os.path.join(checkpoint_dir, ckpt_file)

        # Load model
        model = load_model(model, model_path, device=device, verbose=verbose)
        model.to(device)
        model.eval()

        # Strip projector if present
        if hasattr(model, 'projector'):
            model.projector = nn.Identity()

        # Extract
        embeddings, labels = extract_embeddings(model, full_loader, device=device, verbose=verbose)

        embeddings_list.append(embeddings)
        labels_list.append(labels)

    return embeddings_list, labels_list
