import random
import torch
import numpy as np

from utils.printing import print_verbose

def set_seed(seed=None, verbose=True):
    """
    Sets the random seed for numpy, random, and torch (CPU & all GPUs).
    If no seed is provided, the randomness will be non-deterministic.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print_verbose(f"🌱 Random seed set to {seed}.", verbose)
    else:
        print_verbose("⚠️ No seed provided — randomness will be non-deterministic.", verbose)
