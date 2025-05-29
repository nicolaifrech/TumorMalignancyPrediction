import os
import json
import random
from datetime import datetime
from typing import List, Dict, Optional

from utils.printing import print_verbose

def create_train_val_test_split(
    all_files: List[str],
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 0,
    split_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, List[str]]:
    """
    Splits a list of filenames into train/val/test and optionally saves the split.

    Args:
        all_files: List of file names (not full paths).
        test_size: Proportion for the test split.
        val_size: Proportion for the validation split.
        seed: Random seed for reproducibility.
        split_name: Optional name for the split (used in filename).
        output_dir: If provided, saves the split as a JSON file here.
        verbose: If True, prints where the split was saved.

    Returns:
        A dict with 'train', 'val', and 'test' lists of filenames.
    """
    assert test_size + val_size < 1.0, "Test and val sizes must leave room for training."

    # Deterministic shuffle
    rnd = random.Random(seed)
    shuffled = all_files.copy()
    rnd.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)

    test_files = shuffled[:n_test]
    val_files = shuffled[n_test:n_test + n_val]
    train_files = shuffled[n_test + n_val:]

    split = {'train': train_files, 'val': val_files, 'test': test_files}

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        if split_name:
            filename = f"split_{split_name}.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"split_{timestamp}.json"

        full_path = os.path.join(output_dir, filename)
        with open(full_path, 'w') as f:
            json.dump(split, f) 
        print_verbose(f"✅ Saved split to: {full_path}")

    return split

def load_or_create_split(
    data_folder: str,
    split_name: str,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 0,
    verbose: bool = True
) -> dict:
    """
    Loads a saved split file if it exists, otherwise generates and saves a new one.

    Args:
        data_folder: Path to dataset image folder (e.g., 'utkface/UTKFace').
        split_name: Name of the split (e.g., 'baseline').
        seed: Seed for deterministic split generation.
        verbose: Whether to print status messages.

    Returns:
        A dictionary with 'train', 'val', 'test' keys and lists of filenames.
    """

    splits_dir = os.path.abspath(os.path.join(data_folder, '..', 'splits'))
    split_file = os.path.join(splits_dir, f"split_{split_name}.json")

    if os.path.exists(split_file): 
        print_verbose(f"✅ Loaded existing split: {split_file}", verbose)
        with open(split_file, 'r') as f:
            return json.load(f)
    else: 
        print_verbose(f"🆕 Split not found — creating: {split_file}", verbose)
        return create_train_val_test_split(
            all_files=[f for f in os.listdir(data_folder) if f.endswith('.jpg')],
            test_size=test_size,
            val_size=val_size,
            seed=seed,
            split_name=split_name,
            output_dir=splits_dir,
            verbose=verbose
        )
