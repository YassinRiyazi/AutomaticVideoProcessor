"""
    Version             : 1.0.0

    Author              : Yassin Riyazi
    Date                : 03.10.2025
    Project             : Automatic Video Processor (AVP)
    File                : BaseUtils/drop_detection.py
    License             : GNU GENERAL PUBLIC LICENSE Version 3
    Level access in API : level 0 utility
    Copy Right          : Max Planck Institute for Polymer Research 2025©

    Description: 
        This module provides classes for drop detection using YOLO and SUM methods.    
"""
import  os
import  cv2
import  colorama
import  ultralytics
import  numpy       as      np
from    typing      import  List, Tuple

if __name__ == "__main__":
    from config import config
    from Detection.DropDetection_Sum import detectionV2,Main,DetectEdgeSave,DetectCropSave
    from Detection.DropDetection_Difference import DropPreProcessor as DDdifference
else:
    from .config import config
    from .Detection.DropDetection_Sum import detectionV2,Main,DetectEdgeSave,DetectCropSave
    from .Detection.DropDetection_Difference import DropPreProcessor as DDdifference


class DropDetection():
    """
    Placeholder for DropDetection class.
    Actual implementation should be provided here.
    """
    def __init__(self) -> None:
        pass

    def detect_drops(self, frame_path: str) -> tuple[list, bool]:
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
        addressYOLO = os.path.join(os.path.dirname(__file__),"Detection", "Weights", f"{config['yolo_name']}.engine")
        self.model = ultralytics.YOLO(addressYOLO, task='detect', verbose=False)

    def detect_drops(self, frame_path: str,
               yolo_conf: float = float(config["yolo_conf"])) -> tuple[List, bool]:
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


import re
class DropDetection_Difference(DropDetection):
    """
    Placeholder for DropDetection class.
    Actual implementation should be provided here.
    """
    def __init__(self,
                 ) -> None:
        # check open cv cuda
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            print(colorama.Fore.YELLOW + "OpenCV-CUDA is not available. Falling back to CPU." + colorama.Style.RESET_ALL)
            use_cuda = False
        else:   
            use_cuda = True

        self.holder = DDdifference(use_cuda=use_cuda)

    def detect_drops(self, frame_path: str) -> tuple[list[int], bool]:
        """
        Detect drops in the frame.
        """
        match = re.search(r'(\d+)', frame_path)
        # Extract number
        if match:
            number_str = match.group(1)   # e.g. "0042"
            number_int = int(number_str)  # e.g. 42
            
            # Increment
            new_number_int = number_int + 1
            
            # Preserve zero-padding
            new_number_str = str(new_number_int).zfill(len(number_str))
            
            # Build new filename
            new_frame_path = f"frame_{new_number_str}.png"
            print(new_frame_path)  # frame_0043.png
        

        # finding the number in the name of the frame with regex
        folder = os.path.dirname(frame_path)
        
        

    @staticmethod
    def horizontal_bound_extractor(result:Tuple[int,int]) -> Tuple[int,int]:
        return result




if __name__ == "__main__":
    image = "/media/d2u25/Dont/Teflon-AVP/285/S3-SNr3.06_D/T547_11_68.978021612591/frames_rotated/frame_000006.png"

    detector = DropDetection_Difference()
    bounds, detected = detector.detect_drops(image)
    # img = cv2.imread(image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # detector = DropDetection_SUM()
    # bounds, detected = detector.detect_drops(image)
    # x1,x2 = detector.horizontal_bound_extractor(bounds[0])
    # cv2.rectangle(img, (x1, 0), (x2, img.shape[0]-5), (0, 0, 255), 2)  # Draw rectangle using
    # print(x1,x2)

    # detector = DropDetection_YOLO()
    # bounds, detected = detector.detect_drops(image)
    # x1,x2 = detector.horizontal_bound_extractor(bounds[0])
    # cv2.rectangle(img, (x1, 0), (x2, img.shape[0]-5), (255, 0, 0), 2)  # Draw rectangle using
    # print(x1,x2)

    # import matplotlib.pyplot as plt
    
    # # plotting the bounding box
    

    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()