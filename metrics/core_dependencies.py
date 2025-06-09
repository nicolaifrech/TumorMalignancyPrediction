import torch
import torch.nn.functional as F

from metrics import dependency_registry
from datasets.extraction import extract_outputs
from metrics.dependency_registry import *

@dependency_registry.register("val_outputs_and_labels")
def val_outputs_and_labels(ctx):
    model = ctx["model"]
    device = ctx["device"]
    val_loader = ctx["data_loaders"]["val"]
    return extract_outputs(model, val_loader, device, verbose=False)

@dependency_registry.register("val_prediction_and_targets")
def val_prediction_and_targets(ctx):
    outputs, targets = ctx['val_outputs_and_labels']
    preds = outputs.argmax(dim=1)     
    return preds, targets

@dependency_registry.register("train_outputs_and_labels")
def train_outputs_and_labels(ctx):
    model = ctx["model"]
    device = ctx["device"]
    train_loader = ctx["data_loaders"]["train"]
    return extract_outputs(model, train_loader, device, verbose=False)

@dependency_registry.register("val_cosine_similarities")
def val_cosine_similarities(ctx):
    embeddings, _ = ctx["val_outputs_and_labels"]
    return F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)

@dependency_registry.register("val_label_l1_distance")
def val_label_l1_distance(ctx):
    _, labels = ctx["val_outputs_and_labels"]
    return torch.abs(labels.unsqueeze(0) - labels.unsqueeze(1))
