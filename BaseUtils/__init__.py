"""
    Should not imported in main code; its the lowest level utility and only should be imported in level one modules.
"""
import re
import os
import glob
import colorama



if __name__ == "__main__":
    from config import config
    from drop_detection import DropDetection_YOLO, DropDetection_SUM, DropDetection_SUM_YOLO # type: ignore
else:
    from .config import config
    from .drop_detection import DropDetection_YOLO, DropDetection_SUM, DropDetection_SUM_YOLO   # type: ignore


def ImageLister(FolderAddress: str,
                frameAddress: str = str(config["frame_folder"]),
                extension: str = str(config["image_extension"])) -> list[str]:
        """
        Listing images in the frames directory
        """
        if not os.path.exists(os.path.join(FolderAddress, frameAddress)):
            raise FileNotFoundError(colorama.Fore.RED + f"No frames directory found in {FolderAddress}. Please extract frames first." + colorama.Style.RESET_ALL)

        images = glob.glob(os.path.join(FolderAddress, frameAddress, f"frame*{extension}"))
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

if __name__ == "__main__":
    # Example usage
    print(ImageLister(r"/media/d2u25/Dont/Teflon-AVP/285/S3-SNr3.06_D/T547_11_68.978021612591"))

