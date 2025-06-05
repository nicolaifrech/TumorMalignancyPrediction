import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_kernel_logits(y_pred: torch.Tensor, class_centers: torch.Tensor) -> torch.Tensor:
    """
    Converts scalar predictions into unnormalized Gaussian logits (not probabilities!).
    These logits can be passed to SoftMCCLossMulti, which will apply softmax.
    """
    spacing = class_centers[1] - class_centers[0]
    sigma = spacing / 2

    dists_sq = (y_pred.unsqueeze(1) - class_centers.unsqueeze(0)) ** 2
    logits = -dists_sq / (2 * sigma ** 2)
    return logits  # [B, C]

class ScalarToClassificationLoss(nn.Module):
    def __init__(self, classification_loss, kernel_fn, ordinal_map, one_hot_encoding=True):
        """
        ordinal_map: provides class_centers
        classification_loss: expects logits, not probabilities
        kernel_fn: must return logits
        """
        super().__init__()
        self.ordinal_map = ordinal_map
        self.classification_loss = classification_loss
        self.kernel_fn = kernel_fn
        self.one_hot_encoding = one_hot_encoding

    def forward(self, y_pred: torch.Tensor, class_targets: torch.Tensor) -> torch.Tensor:
        centers = self.ordinal_map.centers.to(y_pred.device)
        logits = self.kernel_fn(y_pred, centers)

        if self.one_hot_encoding:
            target_probs = F.one_hot(class_targets, num_classes=centers.size(0)).float()
            loss = self.classification_loss(logits, target_probs)
        else:
            loss = self.classification_loss(logits, class_targets)

        return loss
