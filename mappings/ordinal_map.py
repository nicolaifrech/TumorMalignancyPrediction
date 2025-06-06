import torch
from typing import Tuple, Optional
import io

from utils.printing import print_verbose

class OrdinalMap():
    def __init__(self, domain: Tuple[float, float], num_classes: int, map_to_cpu: bool = True):
        self.domain = domain
        self.num_classes = num_classes  
        self.map_to_cpu = map_to_cpu
        
        self.bin_edges = self.compute_bin_edges()
        self.class_centers = self.compute_class_centers() 

    def map_to_index(self, values: torch.Tensor) -> torch.Tensor: 
        if self.map_to_cpu:
            values = values.cpu()
        else:
            self.bin_edges = self.bin_edges.to(values.device)
        values_clipped = torch.clamp(values, min=self.bin_edges[0], max=self.bin_edges[-1])
        index = torch.bucketize(values_clipped, self.bin_edges, right=False) - 1
        return torch.clamp(index, 0, self.num_classes - 1)

    def map_to_center(self, values: torch.Tensor) -> torch.Tensor:
        if self.map_to_cpu:
            values = values.cpu()
        else:
            self.class_centers = self.class_centers.to(values.device)
        return self.class_centers[self.map_to_index(values)]
   
    def __call__(self, values: torch.Tensor) -> torch.Tensor: 
        return self.map_to_center(values)

    def class_distribution(self, values: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Computes class distribution over the given scalar values.

        Args:
            values (torch.Tensor): A tensor of scalar values.
            normalize (bool): If True, returns relative frequencies (sums to 1).
                          If False, returns raw counts.

        Returns:
            torch.Tensor: A 1D tensor of shape (num_classes,)
        """
        indices = self.map_to_index(values)
        counts = torch.bincount(indices, minlength=self.num_classes)
        if normalize:
            return counts.float() / counts.sum()
        return counts

    def describe(self, values: torch.Tensor, file: Optional[str] = None, verbose: bool = True):
        """
        Print (or optionally write) dataset statistics related to the ordinal map.

        Args:
            values (torch.Tensor): scalar labels from the dataset
            file (str or None): optional path to save the output
        """
        out = io.StringIO()

        out.write("OrdinalMap Description\n")
        out.write("----------------------\n")
        out.write(f"Domain: [{self.domain[0]}, {self.domain[1]}]\n")
        out.write(f"Number of classes: {self.num_classes}\n")

        # Class counts
        indices = self.map_to_index(values)
        counts = torch.bincount(indices, minlength=self.num_classes)
        total = counts.sum().item()

        out.write("\nClass-wise bin information:\n")
        out.write(f"{'Class':>5} | {'Bin Range':^16} | {'Center':^10} | {'Count':^6} | {'%':^6}\n")
        out.write("-" * 60 + "\n")
        for i in range(self.num_classes):
            left = self.bin_edges[i]
            right = self.bin_edges[i + 1]
            center = self.class_centers[i]
            count = counts[i].item()
            percent = 100 * count / total if total > 0 else 0.0
            out.write(f"{i:5d} | [{left:6.1f}, {right:6.1f}) | {center:10.2f} | {count:6d} | {percent:5.2f}%\n")

        empty_bins = (counts == 0).sum().item()
        out.write(f"\nEmpty bins: {empty_bins} of {self.num_classes}\n")

        # Out-of-domain values
        below = (values < self.domain[0]).sum().item()
        above = (values > self.domain[1]).sum().item()
        out.write(f"Out-of-domain values: {below} below min, {above} above max\n")

        result = out.getvalue()
        print_verbose(result, verbose)

        if file:
            with open(file, 'w') as f:
                f.write(result)

        return result

    def to(self, device: torch.device):
        self.bin_edges = self.bin_edges.to(device)
        self.class_centers = self.class_centers.to(device)
        return self

    def compute_bin_edges(self) -> torch.Tensor:
        return torch.linspace(self.domain[0], self.domain[1], steps=self.num_classes + 1)

    def compute_class_centers(self) -> torch.Tensor: 
        return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:]) 
