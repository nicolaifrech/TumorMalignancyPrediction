import torch
from tqdm import tqdm

def evaluate_on_loader(model, loader, criterion, device, task_type='regression', return_outputs=False):
    """
    Evaluate the model on the given data loader.

    Args:
        model: The trained model.
        loader: A DataLoader (e.g. test_loader).
        criterion: The loss function.
        device: The device to run evaluation on.
        task_type (str): Either 'regression' or 'classification'.
        return_outputs (bool): Whether to return all predictions and labels.

    Returns:
        dict with 'test_loss', 'test_accuracy' (if classification), 'test_mae' (if regression),
        and optionally predictions/targets
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_correct = 0
    total_absolute_error = 0.0
    total_samples = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating", ncols=80, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            if task_type == 'classification':
                preds = outputs.argmax(dim=1)
                total_correct += (preds == targets).sum().item()

            elif task_type == 'regression':
                abs_error = torch.abs(outputs.view(-1) - targets.view(-1)).sum().item()
                total_absolute_error += abs_error

            if return_outputs:
                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

    results = {
        "test_loss": total_loss / total_samples,
    }

    if task_type == 'classification':
        results["test_accuracy"] = total_correct / total_samples
    elif task_type == 'regression':
        results["test_mae"] = total_absolute_error / total_samples

    if return_outputs:
        results["predictions"] = torch.cat(all_preds)
        results["targets"] = torch.cat(all_targets)

    return results
