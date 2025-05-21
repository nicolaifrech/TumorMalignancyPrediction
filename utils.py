from torchvision import transforms
import math
import torch
from tqdm import tqdm
import numpy as np

def check_config(config, required_keys):
    """
    Check if the given config dictionary contains all required keys.

    Args:
        config (dict): Configuration dictionary to check.
        required_keys (list): List of keys that must be present in the config.

    Raises:
        ValueError: If any required key is missing.
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        missing_keys_str = ', '.join(missing_keys)
        raise ValueError(f"Missing required configuration keys: {missing_keys_str}")
    print(f"Configuration check passed: {len(required_keys)} keys found.")

class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_transforms(split, aug):
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    if split == 'train':
        aug_list = aug.split(',')
        transforms_list = []

        if 'crop' in aug_list:
            transforms_list.append(transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)))
        else:
            transforms_list.append(transforms.Resize(256))
            transforms_list.append(transforms.CenterCrop(224))

        if 'flip' in aug_list:
            transforms_list.append(transforms.RandomHorizontalFlip())

        if 'color' in aug_list:
            transforms_list.append(transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8))

        if 'grayscale' in aug_list:
            transforms_list.append(transforms.RandomGrayscale(p=0.2))

        transforms_list.append(transforms.ToTensor())
        transforms_list.append(normalize)
        transform = transforms.Compose(transforms_list)
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    return transform 

def extract_embeddings(model, loader, device):
    model.to(device)
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        with tqdm(loader, unit='batch', ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]') as tepoch: 
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
