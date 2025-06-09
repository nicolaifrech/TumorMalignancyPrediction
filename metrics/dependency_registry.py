class DependencyRegistry:
    def __init__(self):
        self.deps = {}

    def register(self, name):
        def decorator(fn):
            self.deps[name] = fn
            return fn
        return decorator

    def unregister(self, name):
        self.deps.pop(name, None)

    def get(self, name):
        try:
            return self.deps[name]
        except KeyError:
            raise KeyError(f"'{name}' not found in dependency registry. Available keys: {list(self.deps)}")

    def keys(self):
        return self.deps.keys()

dependency_registry = DependencyRegistry()
