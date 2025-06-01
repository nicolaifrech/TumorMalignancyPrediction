import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, encoder: nn.Module, in_dim: int, out_dim: int, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        self.predictor = nn.Linear(in_dim, out_dim)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Encoder is frozen → skip gradient tracking for it
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.encoder.parameters())):
            x = self.encoder(x)
        return self.predictor(x)
