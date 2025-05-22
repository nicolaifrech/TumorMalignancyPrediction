import torch
import numpy as np
from tqdm import tqdm

def extract_embeddings(model, loader, device, verbose=True):
    model.to(device)
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        with tqdm(loader, unit='batch', ncols=80, 
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]', 
            leave=verbose) as tepoch:
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
