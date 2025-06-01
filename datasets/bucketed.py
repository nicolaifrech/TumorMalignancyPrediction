import torch
from torch.utils.data import Dataset

def compute_bucket_edges(min_value, max_value, num_buckets):
    return torch.linspace(min_value, max_value, steps=num_buckets + 1)[1:-1]  # exclude first and last

def bucketize_targets(targets, num_buckets, min_val=None, max_val=None):
    if min_val is None:
        min_val = float(torch.min(targets))
    if max_val is None:
        max_val = float(torch.max(targets))
    bin_edges = compute_bucket_edges(min_val, max_val, num_buckets)
    return torch.bucketize(targets, bin_edges), bin_edges

class Bucketed(Dataset):
    def __init__(self, base_dataset, num_buckets, min_val=None, max_val=None):
        self.base = base_dataset
        self.num_buckets = num_buckets

        # Estimate range
        targets = torch.tensor([base_dataset[i][1] for i in range(len(base_dataset))], dtype=torch.float)
        self.bin_edges = compute_bucket_edges(
            min_val or float(targets.min()),
            max_val or float(targets.max()),
            num_buckets
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        y = torch.tensor([y], dtype=torch.float)
        y_bucketed = torch.bucketize(y, self.bin_edges)[0]
        return x, y_bucketed
