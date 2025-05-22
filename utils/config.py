from utils.printing import print_verbose

def check_config(config, required_keys, verbose=True):
    """
    Check if the given config dictionary contains all required keys.

    Args:
        config (dict): Configuration dictionary to check.
        required_keys (list): List of keys that must be present in the config.

    Raises:
        ValueError: If any required key is missing.
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        missing_keys_str = ', '.join(missing_keys)
        raise ValueError(f"Missing required configuration keys: {missing_keys_str}")
    print_verbose(f"Configuration check passed: {len(required_keys)} keys found.", verbose)
