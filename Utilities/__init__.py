"""
    Components:
        + Empty_frames_remover (): Void [Removing empty frames] 
        + Normalizing_drop_Position (): Void []
        + Detect&crop(): Void []
"""
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import BaseUtils

if __name__ == "__main__":
    from Detection_Sparse import Walker
    from Detection_frameNormalizer import singleFolderDropNormalizer
else:
    from .Detection_Sparse import Walker
    from .Detection_frameNormalizer import singleFolderDropNormalizer

image_folder = r"D:\Videos\S1_30per_T1_C001H001S0001"

import BaseUtils.drop_detection as BUDD
def crop_Save(image_folder: str,
                ):
    """
    Crop and save images from the specified folder.
    """
    BUDD.Main(experiment = os.path.join(image_folder, str(BaseUtils.config["rotated_frames_folder"])),
         SaveAddress = os.path.join(image_folder, str(BaseUtils.config["databases_folder"])),
         SaveAddressCSV = os.path.join(image_folder, str(BaseUtils.config["databases_folder"])),
         extension = str(BaseUtils.config["image_extension"]),
         Detect = BUDD.DetectCropSave)
    
def main(image_folder: str,
         ):
    #Cleaning empty frames
    Walker(image_folder,skip = 450,)
    Walker(image_folder,skip = 10)
    BaseUtils.FileIndexChecker(FolderAddress=image_folder,
                                frameAddress=str(BaseUtils.config["rotated_frames_folder"]))

    images = BaseUtils.ImageLister(FolderAddress=image_folder,
                                       frameAddress=str(BaseUtils.config["rotated_frames_folder"]),)
    singleFolderDropNormalizer(images,BaseUtils.DropDetection_YOLO)# type: ignore

    # BaseUtils.FileIndexChecker(FolderAddress=image_folder,
    #                             frameAddress=str(BaseUtils.config["rotated_frames_folder"]))

    crop_Save(image_folder=image_folder)    
if __name__ == "__main__":
    image_folder = r"D:\Videos\S1_30per_T1_C001H001S0001"
    main(image_folder=image_folder)
