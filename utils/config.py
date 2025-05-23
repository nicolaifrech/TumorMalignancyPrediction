from utils.printing import print_verbose

def check_config(config, required_keys, verbose=True):
    """
    Check if the given config dictionary contains all required keys.

    Args:
        config (dict): Configuration dictionary to check.
        required_keys (Iterable[str]): Keys that must be present in config. 

    Raises:
        ValueError: If any required key is missing.
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        missing_keys_str = ', '.join(missing_keys)
        raise ValueError(f"Missing required configuration keys: {missing_keys_str}")
    print_verbose(f"Configuration check passed: {len(required_keys)} keys found.", verbose)

def extract_config(config, required, optional=None, verbose=True):
    """
    Validates and extracts required and optional config parameters into a namespace-like object.
    """ 

    optional = optional or {}

    # Use existing check logic
    check_config(config, required, verbose) 

    values = {**{k: config[k] for k in required},
              **{k: config.get(k, v) for k, v in optional.items()}}

    return type('ConfigNamespace', (), values)
