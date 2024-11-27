import yaml
import os

def load_config():
    """
    Load the configuration from the YAML file.
    Returns:
        dict: Configuration dictionary.
    """
    # Get the absolute path to the configuration file
    package_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(package_root, "../configs/default.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


