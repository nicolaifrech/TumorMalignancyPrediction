class MetricRegistry:
    def __init__(self):
        self.metrics = {}

    def register(self, name, required=[]):
        def decorator(fn):
            self.metrics[name] = (fn, required)
            return fn
        return decorator

    def unregister(self, name):
        self.metrics.pop(name, None)

    def get(self, name):
        try:
            return self.metrics[name]
        except KeyError:
            raise KeyError(f"'{name}' not found in metric registry. Available keys: {list(self.metrics)}")

    def list_available(self):
        return list(self.metrics.keys())

metric_registry = MetricRegistry()
