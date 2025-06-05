# SOURCE: https://github.com/daniel-scholz/address-class-imbalance/blob/main/torch_losses/soft_mcc.py

import torch
import torch.nn as nn

# multi-class versions of the loss
class SoftMCCLossMulti(nn.Module):
    """With logits."""

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        # create soft confusion matrix
        preds = torch.softmax(preds, dim=1)

        # total number of correct predictions, softened by the probability of each class
        c = torch.sum(preds * labels)

        # total number of samples
        s = preds.size(0)

        # number of times each class occured in the labels
        t_k = torch.sum(labels, dim=0)

        # number of times each class was predicted
        p_k = torch.sum(preds, dim=0)

        numerator = c * s - (t_k * p_k).sum()
        denom = (
            torch.sqrt(s**2 - p_k.square().sum())
            * torch.sqrt(s**2 - t_k.square().sum())
            + 1e-8
        )

        soft_mcc = numerator / denom
        return 1 - soft_mcc
