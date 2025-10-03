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
from    typing      import  List, Tuple, Optional

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

    def detect_drops(self, frame_path: str, yolo_conf: Optional[float] = None) -> tuple[list, bool]:
        """
        Detect drops in the frame.
        """
        pass

    # def __bool__(self):
    #     return True  # Placeholder implementation
    
    @staticmethod
    def horizontal_bound_extractor(result:Tuple[int,int]) -> Tuple[int,int]:
        return result

class DropDetection_YOLO(DropDetection):
    """
    Placeholder for DropCounter class.
    Actual implementation should be provided here.
    """
    def __init__(self) -> None:
        addressYOLO = os.path.join(os.path.dirname(__file__),"Detection", "Weights", f"{config['yolo_name']}.engine")
        self.model = ultralytics.YOLO(addressYOLO, task='detect', verbose=False)

    def detect_drops(self, frame_path: str, yolo_conf: Optional[float] = None) -> tuple[List, bool]:
        """
        Detect drops in the frame.
        """
        if yolo_conf is None:
            yolo_conf = float(config["yolo_conf"])
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
                 scaleDownFactor: int = 1,
                 drop_width: int = 300) -> None:

        self.drop_width= drop_width
        self.scaleDownFactor = scaleDownFactor

    def detect_drops(self, frame_path: str, yolo_conf: Optional[float] = None) -> tuple[list[int], bool]:
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

    
class DropDetection_SUM_YOLO(DropDetection):
    """
    Combines SUM and YOLO drop detection methods.
    """
    def __init__(self,
                 scaleDownFactor: int = 5,
                 drop_width: int = 300) -> None:
        self.Sum = DropDetection_SUM(scaleDownFactor=scaleDownFactor, drop_width=drop_width)
        self.Yolo = DropDetection_YOLO()
    def detect_drops(self, frame_path: str, yolo_conf: Optional[float] = None) -> tuple[list[int], bool]:
        """
        Detect drops in the frame using both SUM and YOLO methods.
        """
        detected = True
        yolo_bounds, yolo_detected = self.Yolo.detect_drops(frame_path, yolo_conf)
        sum_bounds, sum_detected = self.Sum.detect_drops(frame_path)

        image = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        if image[-25:-5,0:40].mean() < 20 or image[-25:-5,-40:-1].mean() < 20:
            detected = False

        if yolo_detected and sum_detected:
            x1_yolo, x2_yolo = self.Yolo.horizontal_bound_extractor(yolo_bounds[0])
            x1_sum, x2_sum = self.Sum.horizontal_bound_extractor(sum_bounds[0])

            # Choose the x1 from SUM and x2 from YOLO to ensure the drop is fully covered
            x1 = x1_yolo
            x2 = max(x2_yolo, x2_sum)
            return [(x1, x2)], True*detected
        
        elif yolo_detected:
            return yolo_bounds, True*detected
        
        elif sum_detected:
            return sum_bounds, True*detected
        
        else:
            return [None,None], False
    

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

    def detect_drops(self, frame_path: str,
                     crop: Optional[Tuple[int, int, int, int]] = None) -> tuple[list[int], bool]:
        """
        Detect drops in the frame.
        """
        detected = True

        current_frame_path = frame_path
        previous_frame_path = None

        folder = os.path.dirname(frame_path)
        namefile = os.path.basename(frame_path)
        match = re.search(r'(\d+)', namefile)
        # Extract number
        if match:
            number_str = match.group(1)   # e.g. "0042"
            number_int = int(number_str)  # e.g. 42
            
            # Increment
            new_number_int = number_int - 1
            
            # Preserve zero-padding
            new_number_str = str(new_number_int).zfill(len(number_str))
            
            # Build new filename
            new_frame_path = f"frame_{new_number_str}.png"
            if os.path.isfile(os.path.join(folder, new_frame_path)):
                previous_frame_path = os.path.join(folder, new_frame_path)
            else:
                raise FileNotFoundError(f"You have to sacrifice the first frame to Drop detection GOD!, {new_frame_path} does not exist in {folder}.")
        else:
            raise FileNotFoundError("No number found in the filename.")

        current_frame = cv2.imread(current_frame_path, cv2.IMREAD_GRAYSCALE)
        previous_frame = cv2.imread(previous_frame_path, cv2.IMREAD_GRAYSCALE)

        if crop is not None:
            x1, y1, x2, y2 = crop
            current_frame = current_frame[y1:y2, x1:x2]
            previous_frame = previous_frame[y1:y2, x1:x2]

        if previous_frame is None or current_frame is None:
            raise ValueError("Could not load previous or current frame as a valid image.")

        x1, x2 = self.holder.process(prev_gray = previous_frame,
                            curr_gray = current_frame)
        
        # detecting edge drops 
        if current_frame[-25:-5,0:40].mean() < 20 or current_frame[-25:-5,-40:-1].mean() < 20:
            detected = False
        return [(x1, x2)], detected

if __name__ == "__main__":
    image = "/media/Dont/Teflon-AVP/285/S3-SDS99_D/T120_01_0.900951687825/frames_rotated/frame_000312.png"

    



    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

    detector = DropDetection_SUM_YOLO()
    bounds, detected = detector.detect_drops(image)
    x1,x2 = detector.horizontal_bound_extractor(bounds[0])
    cv2.rectangle(img, (x1, 0), (x2, img.shape[0]-5), (0, 255, 0), 2)  # Draw rectangle using
    print(x1,x2)


    import matplotlib.pyplot as plt
    
    # plotting the bounding box
    

    plt.imshow(img)
    plt.axis('off')
    plt.show()