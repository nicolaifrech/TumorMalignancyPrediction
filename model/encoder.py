import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, backbone_name='resnet18', output_dim=128, pretrained=True):
        super().__init__()

        if backbone_name == 'resnet18':
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
        elif backbone_name == 'resnet50':
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if pretrained else None
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        backbone = getattr(models, backbone_name)(weights=weights)
 
        self.backbone = backbone
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()  # Remove classification layer
 
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projector(features)
        return embeddings
