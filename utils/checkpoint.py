import torch

def save_checkpoint(model, filepath, epoch=None, metrics=None):
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }, filepath)

def load_checkpoint(filepath, map_location='cpu'):
    checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
    return checkpoint
