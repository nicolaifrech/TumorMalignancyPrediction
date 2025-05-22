import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from utils.extraction import extract_embeddings

def evaluate_scalar_regression(model, loader, device):
    """Evaluates L1 loss for models that output a single scalar."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).float().unsqueeze(1)
            outputs = model(images)
            preds = outputs.mean(dim=1) if outputs.ndim > 1 else outputs
            loss = F.l1_loss(preds.view(-1), targets.view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

# Downstream task metric
def knn_accuracy(embeddings, labels, k=5):
    embeddings_np = embeddings.numpy()
    n = len(embeddings_np)
    split = int(0.8 * n)
    X_train, X_test = embeddings_np[:split], embeddings_np[split:]
    y_train, y_test = labels[:split], labels[split:]

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    return accuracy_score(y_test, preds)

# Embedding quality diagnostics
def embedding_norm(embeddings, labels=None):
    return embeddings.norm(dim=1).mean().item()

def embedding_variance(embeddings, labels=None):
    return embeddings.var(dim=0).mean().item()

# Similarity matrices
# TODO: If wanted, factor out similarity measures and define other measures
def label_similarity(labels):
    label_diff = torch.abs(labels.unsqueeze(0) - labels.unsqueeze(1))
    return -label_diff

def embedding_similarity(embeddings):
    return F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)

# Rank correlation between similarity structures
def spearman_corr(embedding_similarities, label_similarities): 
    return spearmanr(
        label_similarities.flatten().cpu().numpy(),
        embedding_similarities.flatten().cpu().numpy()
    ).correlation

def kendall_corr(embedding_similarities, label_similarities): 
    return kendalltau(
        label_similarities.flatten().cpu().numpy(),
        embedding_similarities.flatten().cpu().numpy()
    ).correlation

METRIC_FUNCS = {
    'knn_accuracy': knn_accuracy,
    'embedding_norm': embedding_norm,
    'embedding_variance': embedding_variance,
    'spearman': spearman_corr,
    'kendall': kendall_corr,
}

def compute_metrics(model, val_loader, device, metric_names, optimizer=None, verbose=True):
    """
    Computes evaluation metrics on the given model and dataloader.

    Args:
        model (torch.nn.Module): The trained model.
        val_loader (DataLoader): Validation dataloader.
        device (torch.device): The device to run computations on.
        metric_names (List[str]): Names of metrics to compute.
        optimizer (torch.optim.Optimizer, optional): Needed for metrics like 'grad_norm' and 'lr'.

    Returns:
        dict: A dictionary mapping metric names to their computed values.
    """
    metrics = {}

    # Always include validation loss
    if 'val_loss' in metric_names:
        metrics['val_loss'] = evaluate_scalar_regression(model, val_loader, device)

    # Optional: include optimizer metrics
    if 'grad_norm' in metric_names and optimizer is not None:
        metrics['grad_norm'] = sum(
            p.grad.norm().item() for p in model.parameters() if p.grad is not None
        )

    if 'lr' in metric_names and optimizer is not None:
        metrics['lr'] = optimizer.param_groups[0]['lr']

    # Extract embeddings and labels
    val_embeddings, val_labels = extract_embeddings(model, val_loader, device, verbose=verbose)
    val_embeddings = torch.tensor(val_embeddings)
    val_labels = torch.tensor(val_labels)

    # Precompute similarities if needed
    needs_sim = any(name in {'spearman', 'kendall'} for name in metric_names)
    if needs_sim:
        emb_sim = embedding_similarity(val_embeddings)
        label_sim = label_similarity(val_labels)

    # Compute requested metrics
    for name in metric_names:
        if name in METRIC_FUNCS:
            if name in {'spearman', 'kendall'}:
                metrics[name] = METRIC_FUNCS[name](emb_sim, label_sim)
            else:
                metrics[name] = METRIC_FUNCS[name](val_embeddings, val_labels)

    return metrics
