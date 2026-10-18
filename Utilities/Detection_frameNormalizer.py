import  os
import  tqdm
# import  cv2
# import  numpy               as      np
# import  matplotlib.pyplot   as      plt
# from    ultralytics         import  YOLO


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import BaseUtils

def singleFolderDropNormalizer(images: list[str],
                               detector = BaseUtils.DropDetection_YOLO):#BaseUtils.DropDetection_SUM
    """
    This function normalizes the drop images in a single folder.
    It removes images that do not meet the criteria defined by the YOLO model.
    """
    in_detector = detector() if callable(detector) else detector

    width = BaseUtils.config["Video"]["cropping_properties"]["WIDTH"]

    Left_boundary = int(BaseUtils.config["single_FolderDropNormalizer"]["Left_boundary_ratio"] * width)
    Right_boundary = int(BaseUtils.config["single_FolderDropNormalizer"]["Right_boundary_ratio"] * width)
    def _forward(images: list[str]):
        """
        Forward pass through the YOLO model.
        Returns the bounding box coordinates.
        """
        for image in tqdm.tqdm(images):
            if not os.path.exists(image):
                # TODO: Log that the image does not exist before continuing
                continue
            results, has_drop = in_detector.detect_drops(image)
            
            #33 Delete empty images
            if not has_drop:
                os.remove(image)
                continue

            for file_idx, res in enumerate(results):
                x1, x2 = in_detector.horizontal_bound_extractor(res)

                if x2 < Left_boundary:
                    return True
                else:
                    os.remove(image)
                    break  # Exit after removing the first invalid image
    def _backward(images: list[str]):
        """
        Backward pass through the YOLO model.
        This function is not used in this context but is kept for consistency.
        """
        for image in reversed(images):
            if os.path.exists(image) is False:
                # TODO: Log that the image does not exist before returning
                return None
            results, has_drop = in_detector.detect_drops(image)
            
            if not has_drop:
                os.remove(image)
                continue

            for file_idx, res in enumerate(results):
                x1, x2 = in_detector.horizontal_bound_extractor(res)
                if x1 > Right_boundary:
                    return True
                else:
                    os.remove(image)
                    break  # Exit after removing the first invalid image
    _forward(images)
    _backward(images)


if __name__ == "__main__":
    image_folder = r"D:\Videos\S1_30per_T1_C001H001S0001"

    images = BaseUtils.ImageLister(FolderAddress=image_folder,
                                       frameAddress=str(BaseUtils.config["rotated_frames_folder"]),)
    singleFolderDropNormalizer(images,BaseUtils.DropDetection_YOLO)

    # singleFolderDropNormalizer(images,BaseUtils.DropDetection_SUM)