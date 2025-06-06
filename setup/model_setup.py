import torch
import torch.nn as nn

from model.encoder import Encoder
from model.predictor import Predictor
from model.loading import load_model

#def setup_encoder(backbone_name, data_parallel, output_dim=128, pretrained=True, device='cpu'):
#    encoder = Encoder(backbone_name=backbone_name, output_dim=output_dim, pretrained=pretrained)
#    encoder = data_parallel(encoder).to(device) 
#    return encoder

def setup_encoder(config):
    device = config['device']
    encoder = Encoder(
        backbone_name=config['backbone_model'],
        output_dim=config.get('output_dim', config['encoder_output_dim']),
        pretrained=config.get('pretrained', True)
    )
    encoder = config['data_parallel'](encoder).to(device)
    return encoder

#def setup_predictor(encoder, data_parallel, input_dim=128, output_dim=1, freeze_encoder=True, device='cpu'):
#    predictor = Predictor(encoder, input_dim, output_dim, freeze_encoder, device)
#    predictor = data_parallel(predictor).to(device)
#    return predictor

def setup_predictor(config):
    device = config['device']

    # Build encoder directly here (on CPU first)
    encoder = Encoder(
        backbone_name=config['backbone_model'],
        output_dim=config.get('output_dim', config['encoder_output_dim']),
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

#def setup_predictor(config):
#    device = config['device']
#
#    encoder = setup_encoder({**config, 'device': 'cpu'})
#    encoder = load_model(encoder, config['trained_encoder_file'], 'cpu', config['verbose'])
#
#    predictor = Predictor(
#        encoder=encoder,
#        input_dim=config['encoder_output_dim'],
#        output_dim=config['predictor_output_dim'],
#        freeze_encoder=config.get('freeze_encoder', True) 
#    )
#    predictor = config['data_parallel'](predictor).to(device)
#    return predictor

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
