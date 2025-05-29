import torch

def setup_optimizer(optimizer, model, learning_rate, momentum=0.9, weight_decay=1e-4):
    if optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        return troch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
