import json
import os

from utils.printing import print_verbose

def save_configuration(config, config_file_name='config.json', verbose=True):
    if 'test_dir' not in config:
        raise KeyError("config must include 'test_dir' before calling save_configuration()")

    # Filter out non-serializable items
    serializable_config = {
        k: v for k, v in config.items()
        if isinstance(v, (str, int, float, bool, list, dict, type(None)))
    }

    save_path = config['test_dir']
    config_file = os.path.join(save_path, config_file_name)
    with open(config_file, "w") as f:
        json.dump(serializable_config, f, indent=4)

    print_verbose(f"📝 Configuration saved to {config_file}", verbose)
