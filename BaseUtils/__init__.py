"""
    Should not imported in main code; its the lowest level utility and only should be imported in level one modules.
"""
import os
import yaml
import glob
import colorama
from typing import Union, Dict


def load_config(config_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))) -> Dict[str, Union[str, int]]:
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

def ImageLister(FolderAddress: str,
                frameAddress: str = str(config["frame_folder"]),
                extension: str = str(config["image_extension"])) -> list[str]:
        """
        Listing images in the frames directory
        """
        if not os.path.exists(os.path.join(FolderAddress, frameAddress)):
            raise FileNotFoundError(colorama.Fore.RED + f"No frames directory found in {FolderAddress}. Please extract frames first." + colorama.Style.RESET_ALL)
        
        images = glob.glob(os.path.join(FolderAddress, frameAddress, extension)) 
        if len(images) == 0:
            images = glob.glob(os.path.join(FolderAddress, frameAddress, "*.jpg"))
        elif len(images) == 0:
            raise FileNotFoundError(colorama.Fore.RED + f"No .jpg or .png files found in {os.path.join(FolderAddress, frameAddress)}." + colorama.Style.RESET_ALL)
        images = sorted(images)  # Sorting images before
        return images


if __name__ == "__main__":
    # Example usage
    print(ImageLister(r"D:\Videos\S1_30per_T1_C001H001S0001"))

