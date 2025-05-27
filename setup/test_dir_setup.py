import os
from datetime import datetime

from utils.printing import print_verbose

def setup_testing_directory(config, create_unique_dir=True):
    if create_unique_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        test_dir_name = f"{config['test_name']}_{timestamp}"
    else:
        test_dir_name = config['test_name']
    test_dir = os.path.join(config['base_dir'], test_dir_name)

    # Create the test directory
    os.makedirs(test_dir, exist_ok=False)

    # Optional: also create subdirs like analysis
    analysis_dir = os.path.join(test_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    # Update the config as a side effect
    config['test_dir_name'] = test_dir_name
    config['test_dir'] = test_dir
    config['analysis_dir'] = analysis_dir
    config['best_model_file'] = os.path.join(test_dir, config.get('best_model_file_name', 'best_model.pth'))
    config['initial_model_file'] = os.path.join(test_dir, config.get('initial_model_file_name', 'initial_model.pth'))

    print_verbose(f"✅ Test directory created: {test_dir}", config['verbose']) 
