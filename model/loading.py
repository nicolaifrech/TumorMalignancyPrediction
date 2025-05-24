import torch

from utils.printing import print_verbose

def load_model(model, filepath, output_dim=128, state_dict_key='model_state_dict', device='cpu', verbose=True): 
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint[state_dict_key])
    print_verbose(f"Loaded model from {filepath}, epoch {checkpoint['epoch']}, best loss: {checkpoint['metrics']['train_loss']:.4f}", verbose)
    return model
