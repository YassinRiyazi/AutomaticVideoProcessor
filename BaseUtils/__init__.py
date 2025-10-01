"""
    Should not imported in main code; its the lowest level utility and only should be imported in level one modules.
"""
import re
import os
import cv2
import yaml
import glob
import colorama
import ultralytics
import numpy as np
from typing import Union, Dict, Tuple

if __name__ == "__main__":
    from DropDetection_Sum import detectionV2,Main,DetectEdgeSave,DetectCropSave
else:
    from .DropDetection_Sum import detectionV2,Main,DetectEdgeSave,DetectCropSave

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

def FileIndexChecker( FolderAddress: str,
                     frameAddress: str = str(config["frame_folder"])):
    """
    Checking images are sequentially indexed without missing numbers
    """
    images = ImageLister(FolderAddress=FolderAddress,
                         frameAddress=frameAddress)

    # extracting image index from filename with regex and if there is missing index raise error
    image_indices = [int(re.search(r'(\d+)', os.path.basename(img_file)).group(1)) for img_file in images] # type: ignore
    for i, _ in enumerate(images, start=image_indices[0]):    
        if i != image_indices[i-1]:
            raise ValueError(colorama.Fore.RED + f"Missing image index detected: {i}" + colorama.Style.RESET_ALL)


class DropDetection():
    """
    Placeholder for DropDetection class.
    Actual implementation should be provided here.
    """
    def __init__(self) -> None:
        pass

    def detect_drops(self, frame_path: str):
        """
        Detect drops in the frame.
        """
        pass

    # def __bool__(self):
    #     return True  # Placeholder implementation
    
    @staticmethod
    def horizontal_bound_extractor(result) -> Tuple[int,int]:
        pass

class DropDetection_YOLO(DropDetection):
    """
    Placeholder for DropCounter class.
    Actual implementation should be provided here.
    """
    def __init__(self) -> None:
        addressYOLO = os.path.join(os.path.dirname(__file__), "Weights", f"{config['yolo_name']}.engine")
        self.model = ultralytics.YOLO(addressYOLO, task='detect', verbose=False)

    def detect_drops(self, frame_path: str,
               yolo_conf: float = float(config["yolo_conf"])) -> tuple[list, bool]:
        """
        Detect drops in the frame.
        """
        results = self.model(frame_path, conf=yolo_conf, device="cuda", verbose=False)
        NumberOfObjects = len(results[0].boxes)
        if NumberOfObjects > 1:
            raise ValueError(colorama.Fore.RED + f"More than one object detected in {frame_path}. Expected only one." + colorama.Style.RESET_ALL)
        return results, NumberOfObjects > 0
    
    @staticmethod
    def horizontal_bound_extractor(result) -> Tuple[int,int]:
        x1, _, x2, _ = np.array(result.boxes.xyxy[:, :].cpu().numpy(), dtype=np.float32)[0]
        return (int(x1), int(x2))

    
class DropDetection_SUM(DropDetection):
    """
    Placeholder for DropDetection class.
    Actual implementation should be provided here.
    """
    def __init__(self,
                 scaleDownFactor: int = 5,
                 drop_width: int = 300) -> None:

        self.drop_width= drop_width
        self.scaleDownFactor = scaleDownFactor

    def detect_drops(self, frame_path: str) -> tuple[list[int], bool]:
        """
        Detect drops in the frame.
        """
        detected = True
        image       = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        x2, x1 = detectionV2(image, self.scaleDownFactor, self.drop_width)

        # detecting edge drops 
        if image[-25:-5,0:40].mean() < 20 or image[-25:-5,-40:-1].mean() < 20:
            detected = False
        return [(x1, x2)], detected

    @staticmethod
    def horizontal_bound_extractor(result:Tuple[int,int]) -> Tuple[int,int]:
        return result


if __name__ == "__main__":
    # Example usage
    print(ImageLister(r"D:\Videos\S1_30per_T1_C001H001S0001"))

