import torch
import torch.nn as nn

class CompositeLoss(nn.Module):
    def __init__(self, components):
        """
        components: List of (loss_fn, weight, name) tuples.
        - loss_fn: a nn.Module or callable
        - weight: float
        - name: str, used for breakdown logging
        """
        super().__init__()
        self.components = nn.ModuleList([lf for lf, _, _ in components])
        self.weights = [w for _, w, _ in components]
        self.names = [n for _, _, n in components]

    def forward(self, *args, return_components=False, **kwargs):
        total_loss = 0.0
        breakdown = {}
        for loss_fn, weight, name in zip(self.components, self.weights, self.names):
            loss = loss_fn(*args, **kwargs)
            total_loss += weight * loss
            if return_components:
                breakdown[name] = loss.detach().item()
        return (total_loss, breakdown) if return_components else total_loss
