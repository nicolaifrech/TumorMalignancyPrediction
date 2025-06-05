from torch.utils.data import Dataset

class DiscretizedDataset(Dataset):
    def __init__(self, base_dataset, ordinal_map):
        """
        Wraps a dataset that returns (x, scalar_target) and transforms the target
        into a discrete class index using the given OrdinalMap.

        Args:
            base_dataset (Dataset): A dataset returning (x, scalar_target)
            ordinal_map (OrdinalMap): The mapping to use for discretization
        """
        self.base = base_dataset
        self.map = ordinal_map

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y_scalar = self.base[idx]
        if not torch.is_tensor(y_scalar):
            y_scalar = torch.tensor(y_scalar, dtype=torch.float)
        y_discrete = self.map(y_scalar.unsqueeze(0)).squeeze(0)
        return x, y_discrete
