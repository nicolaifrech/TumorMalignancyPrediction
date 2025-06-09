from .metric_registry import metric_registry
from .dependency_registry import dependency_registry

class MetricEvaluator:
    def __init__(self, metric_names, data_loaders, device):       
        self.metric_names = metric_names
        self.data_loaders = data_loaders 
        self.device = device 

    def resolve(self, required, context):
        for key in required:
            if key in context:
                continue
            if key in dependency_registry.keys():
                fn = dependency_registry.get(key)
                context[key] = fn(context)
            else:
                raise ValueError(f"Unknown prerequisite: {key}")

    def compute(self, model, optimizer, criterion, train_loss):
        model.eval()

        # Shared context for dependencies and metrics
        context = {
            "model": model,
            "device": self.device,
            "optimizer": optimizer,
            "data_loaders": self.data_loaders,
            "criterion": criterion,
            "train_loss": train_loss
        }

        results = {'train_loss': train_loss}
        for name in self.metric_names:
            fn, required = metric_registry.get(name)
            self.resolve(required, context)
            args = {k: context[k] for k in required}
            results[name] = fn(**args)

        return results
