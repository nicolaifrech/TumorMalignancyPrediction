import torch

def get_scheduler(scheduler, optimizer, step_size=10, gamma=0.1, epochs=None):
    if scheduler == 'steplr':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        return None

#scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#    optimizer, mode='min', factor=0.5, patience=3, verbose=verbose
#)  
