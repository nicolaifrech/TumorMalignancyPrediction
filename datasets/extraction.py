import torch
import numpy as np
from tqdm import tqdm

def extract_outputs(model, loader, device, verbose=True):
    model.to(device)
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        with tqdm(loader, unit='batch', ncols=80,
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]',
                  leave=False) as tepoch:
            for images, labels in tepoch:
                tepoch.set_description("Extracting Embeddings")

                images = images.to(device)
                embeddings = model(images)
 
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels.cpu())

                tepoch.set_postfix(embeddings=len(all_embeddings))

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)
    return all_embeddings, all_labels
