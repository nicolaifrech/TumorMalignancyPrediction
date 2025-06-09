import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import entropy as scipy_entropy

from metrics.metric_registry import *

# DIAGNOSTIC METRICS

@metric_registry.register("grad_norm", required=["model"])
def grad_norm(model):
    return sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

@metric_registry.register("lr", required=["optimizer"])
def learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]

@metric_registry.register("parameter_norm", required=["model"])
def parameter_norm(model):
    total = 0.0
    count = 0
    for param in model.parameters():
        total += param.norm().item()
        count += 1
    return total / count if count > 0 else 0.0

@metric_registry.register("embedding_norm", required=["val_outputs_and_labels"])
def embedding_norm(val_outputs_and_labels):
    outputs, _ = val_outputs_and_labels
    norms = torch.norm(outputs, dim=1)
    return norms.mean().item()

@metric_registry.register("embedding_variance", required=["val_outputs_and_labels"])
def embedding_variance(val_outputs_and_labels):
    outputs, _ = val_outputs_and_labels
    variances = torch.var(outputs, dim=0, unbiased=False)
    return variances.mean().item()

@metric_registry.register("dead_relu_ratio", required=["model", "device", "data_loaders"])
def dead_relu_ratio(model, device, data_loaders):
    val_loader = data_loaders["val"]
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())

    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook_fn))

    # Run one pass
    model.eval()
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            model(inputs)
            break  # Use just one batch for efficiency

    for h in hooks:
        h.remove()

    # Compute dead ReLU ratio
    total_neurons = 0
    dead_neurons = 0
    for act in activations:
        total_neurons += act.numel()
        dead_neurons += (act == 0).sum().item()

    return dead_neurons / total_neurons if total_neurons > 0 else 0.0

@metric_registry.register("entropy", required=["val_outputs_and_labels"])
def entropy(val_outputs_and_labels):
    outputs, _ = val_outputs_and_labels
    probs = F.softmax(outputs, dim=1)
    entropy = -(probs * probs.log()).sum(dim=1).mean()
    return entropy.item()

@metric_registry.register("train_val_loss_gap", required=["train_loss", "val_loss"])
def train_val_loss_gap(train_loss, val_loss):
    return abs(val_loss - train_loss)

# VALIDATION LOSS (ADAPTIVE FOR REGRESSION OR CLASSIFICATION)

@metric_registry.register("val_loss", required=["model", "device", "data_loaders", "criterion"])
def val_loss(model, device, data_loaders, criterion):
    model.eval()
    val_loader = data_loaders["val"]

    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(val_loader)

# CLASSIFICATION METRICS 

@metric_registry.register("accuracy", required=["val_outputs_and_labels", 'val_prediction_and_targets'])
def accuracy(val_prediction_and_targets):
    preds, targets = val_prediction_and_targets 
    return accuracy_score(targets.numpy(), preds.numpy())

@metric_registry.register("precision_macro", required=["val_outputs_and_labels", 'val_prediction_and_targets'])
def precision(val_prediction_and_targets):
    preds, targets = val_prediction_and_targets 
    return precision_score(targets.numpy(), preds.numpy(), average="macro", zero_division=0)

@metric_registry.register("recall_macro", required=["val_outputs_and_labels", 'val_prediction_and_targets'])
def recall(val_prediction_and_targets):
    preds, targets = val_prediction_and_targets 
    return recall_score(targets.numpy(), preds.numpy(), average="macro", zero_division=0)

@metric_registry.register("f1_macro", required=["val_outputs_and_labels", 'val_prediction_and_targets'])
def f1(val_prediction_and_targets):
    preds, targets = val_prediction_and_targets 
    return f1_score(targets.numpy(), preds.numpy(), average="macro", zero_division=0)

@metric_registry.register("knn_accuracy", required=[
    "train_outputs_and_labels", "val_outputs_and_labels",
])
def knn_accuracy(train_outputs_and_labels, val_outputs_and_labels):
    train_embs, train_labels = train_outputs_and_labels
    val_embs, val_labels = val_outputs_and_labels

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_embs.numpy(), train_labels.numpy())
    preds = knn.predict(val_embs.numpy())
    return accuracy_score(val_labels.numpy(), preds)

# REGRESSION METRICS

@metric_registry.register("mae", required=["val_outputs_and_labels"])
def mae(val_outputs_and_labels):
    outputs, labels = val_outputs_and_labels
    return mean_absolute_error(labels.numpy(), outputs.numpy())

@metric_registry.register("mse", required=["val_outputs_and_labels"])
def mse(val_outputs_and_labels):
    outputs, labels = val_outputs_and_labels
    return mean_squared_error(labels.numpy(), outputs.numpy())

@metric_registry.register("rmse", required=["val_outputs_and_labels"])
def rmse(val_outputs_and_labels):
    outputs, labels = val_outputs_and_labels
    return mean_squared_error(labels.numpy(), outputs.numpy(), squared=False)

@metric_registry.register("r2", required=["val_outputs_and_labels"])
def r2(val_outputs_and_labels):
    outputs, labels = val_outputs_and_labels
    return r2_score(labels.numpy(), outputs.numpy())

@metric_registry.register("knr_error", required=[
    "train_outputs_and_labels", "val_outputs_and_labels",
])
def knr_error(train_outputs_and_labels, val_outputs_and_labels):
    train_embs, train_targets = train_outputs_and_labels
    val_embs, val_targets = val_outputs_and_labels

    knr = KNeighborsRegressor(n_neighbors=5)
    knr.fit(train_embs.numpy(), train_targets.numpy())
    preds = knr.predict(val_embs.numpy())
    return mean_absolute_error(val_targets.numpy(), preds)

# EMBEDDING METRICS

@metric_registry.register("spearman", required=[
    "val_outputs_and_labels", "val_cosine_similarities", "val_label_l1_distance"
])
def spearman(val_cosine_similarities, val_label_l1_distance, **kwargs):
    return spearmanr(
        val_label_l1_distance.flatten().cpu().numpy(),
        val_cosine_similarities.flatten().cpu().numpy()
    ).correlation

@metric_registry.register("kendall", required=[
    "val_outputs_and_labels", "val_cosine_similarities", "val_label_l1_distance"
])
def kendall(val_cosine_similarities, val_label_l1_distance, **kwargs):
    return kendalltau(
        val_label_l1_distance.flatten().cpu().numpy(),
        val_cosine_similarities.flatten().cpu().numpy()
    ).correlation
