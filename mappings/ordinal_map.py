import torch
from typing import Tuple, Optional
import io

from utils.printing import print_verbose

class OrdinalMap:
    def __init__(self, domain: Tuple[float, float], num_classes: int):
        self.domain = domain
        self.num_classes = num_classes  
        
        self.class_centers = self.compute_class_centers()  
        self.bin_edges = self.compute_bin_edges() 

    def map_to_index(self, value: torch.Tensor) -> torch.Tensor:
        value_clipped = torch.clamp(value, min=self.bin_edges[0], max=self.bin_edges[-1])
        index = torch.bucketize(value_clipped, self.bin_edges, right=False) - 1
        return torch.clamp(index, 0, self.num_classes - 1)

    def map_to_center(self, value: torch.Tensor) -> torch.Tensor:
        return self.class_centers[self.map_to_index(value)]
   
    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return self.map_to_center(value)

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
        out.write(f"Bin edges: {self.bin_edges.tolist()}\n")
        out.write(f"Class centers: {self.class_centers.tolist()}\n")

        # Class counts
        indices = self.map_to_index(values)
        counts = torch.bincount(indices, minlength=self.num_classes)
        total = counts.sum().item()
        out.write("Class distribution:\n")
        for i, count in enumerate(counts.tolist()):
            out.write(f"  Class {i:3}: {count:6} ({100 * count / total:.2f}%)\n")

        empty_bins = (counts == 0).sum().item()
        out.write(f"Empty bins: {empty_bins} of {self.num_classes}\n")

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

    def compute_class_centers(self) -> torch.Tensor:
        return torch.linspace(self.domain[0], self.domain[1], steps=self.num_classes)

    def compute_bin_edges(self) -> torch.Tensor:
        return torch.linspace(self.domain[0], self.domain[1], steps=self.num_classes + 1)
