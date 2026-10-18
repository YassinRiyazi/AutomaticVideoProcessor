"""
    Version             : 1.0.0

    Author              : Yassin Riyazi
    Date                : 03.10.2025
    Project             : Automatic Video Processor (AVP)
    File                : BaseUtils/config.py
    License             : GNU GENERAL PUBLIC LICENSE Version 3
    Level access in API : level -1 utility
    Copy Right          : Max Planck Institute for Polymer Research 2025©

    Description: 
        This module provides functions for loading and validating configuration settings from a YAML file.
"""

import os
import yaml

from typing import Union, Dict

def load_config(config_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
                ) -> Dict[str, Union[str, int | float | bool | str]]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

config = load_config()

if __name__ == "__main__":
    # Example usage
    print(config)