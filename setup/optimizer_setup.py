import torch

def setup_optimizer(optimizer, model, learning_rate, momentum=0.9):
    if optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
