import torch
import torch.nn as nn

from model.encoder import Encoder

def setup_encoder(backbone_name, data_parallel, pretrained=True, device='cpu'):
    model = Encoder(backbone_name=backbone_name, pretrained=pretrained)
    model = data_parallel(model).to(device) 
    return model
