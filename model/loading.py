import torch

from utils.checkpoint import load_checkpoint
from utils.printing import print_verbose

def load_model(model, filepath, device='cpu', verbose=True): 
    checkpoint = load_checkpoint(filepath, map_location=device)

    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError(
            f"Invalid checkpoint format in '{filepath}'. "
            f"Expected dict with key 'model_state_dict'. "
            f"Did you save it using 'save_checkpoint()'?"
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    print_verbose(f"Loaded model from {filepath}", verbose)
    return model
