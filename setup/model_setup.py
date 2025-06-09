import torch
import torch.nn as nn

from model.encoder import Encoder
from model.predictor import Predictor
from model.loading import load_model

def setup_encoder(config):
    device = config['device']
    encoder = Encoder(
        backbone_name=config['backbone_model'],
        output_dim=config.get('output_dim') or config['encoder_output_dim'],
        pretrained=config.get('pretrained', True)
    )
    encoder = config['data_parallel'](encoder).to(device)
    return encoder

def setup_predictor(config):
    device = config['device']

    # Build encoder directly here (on CPU first)
    encoder = Encoder(
        backbone_name=config['backbone_model'],
        output_dim=config.get('output_dim') or config['encoder_output_dim'],
        pretrained=config.get('pretrained', True)
    )
    
    # Load weights if needed (can load on CPU)
    #encoder = load_model(encoder, config['trained_encoder_file'], device='cpu', verbose=config['verbose'])

    # Move encoder to final device
    encoder = encoder.to(device)

    # Build predictor
    predictor = Predictor(
        encoder=encoder,
        input_dim=config['encoder_output_dim'],
        output_dim=config['predictor_output_dim'],
        freeze_encoder=config.get('freeze_encoder', True)
    )

    # Wrap only the full predictor in DataParallel
    predictor = config['data_parallel'](predictor).to(device)

    return predictor

def setup_baseline(config):
    device = config['device']

    core_model = setup_encoder(config) 

    baseline = Predictor(
        encoder=core_model,
        input_dim=config['encoder_output_dim'],
        output_dim=config['predictor_output_dim'],
        freeze_encoder=False,
        device=device
    )
    baseline = config['data_parallel'](baseline).to(device) 
    return baseline
