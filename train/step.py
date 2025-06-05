import torch

def supervised_step_fn(model, batch, criterion, optimizer, device):
    """
    Generic supervised training step for tasks like classification, regression, or reconstruction.
    Assumes: (input, target) format and model.criterion is defined.
    """
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    return loss.item()

def regression_predictor_step_fn(model, batch, criterion, optimizer, device):
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    return loss.item()

def unsupervised_step_fn(model, batch, criterion, optimizer, device):
    """
    Unsupervised single-input training step.
    Suitable for autoencoders, masked prediction, etc. without targets.
    Assumes: inputs = targets for reconstruction.
    """
    inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
    inputs = inputs.to(device)
    targets = inputs  # For reconstruction-based unsupervised learning

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    return loss.item()

def supervised_two_view_step_fn(model, batch, criterion, optimizer, device):
    """
    Supervised training step for two-view inputs, such as contrastive learning with labels (e.g., SupCon or Rank-N-Contrast).
    Assumes: ((view1, view2), targets) format and model.criterion is defined.
    """
    
    (view1, view2), targets = batch
    view1, view2 = view1.to(device), view2.to(device)
    targets = targets.to(device).float().unsqueeze(1)
    bsz = targets.size(0)

    optimizer.zero_grad()
    inputs = torch.cat((view1, view2), dim=0)       # [2B, C, H, W]
    embeddings = model(inputs)                      # [2B, D]

    embeddings1, embeddings2 = torch.split(embeddings, [bsz, bsz], dim=0)
    features = torch.stack([embeddings1, embeddings2], dim=1) 
    loss = criterion(features, targets)
    loss.backward()
    optimizer.step()

    return loss.item()

def unsupervised_two_view_step_fn(model, batch, criterion, optimizer, device):
    """
    Unsupervised contrastive training step with two views.
    Used in SimCLR, BYOL, etc.
    Assumes: (view1, view2) format and contrastive criterion that does not need labels.
    """
    view1, view2 = batch
    view1, view2 = view1.to(device), view2.to(device)

    bsz = view1.size(0)

    optimizer.zero_grad()
    inputs = torch.cat((view1, view2), dim=0)  # [2 * B, ...]
    embeddings = model(inputs)
    emb1, emb2 = torch.split(embeddings, [bsz, bsz], dim=0)
    features = torch.stack([emb1, emb2], dim=1)  # [B, 2, D]

    loss = criterion(features)  # No labels required
    loss.backward()
    optimizer.step()

    return loss.item()
