"""
    Date    : 2025-09-30
    Author  : Yassin Riyazi
    Project : Automatic Video Processor (AVP)
    File    : AVP.py
    Version : 1.0.0
    License : MIT License

    Description: This is the main file of the Automatic Video Processor (AVP) project. 
    It orchestrates the various components of the AVP system to process videos automatically.

    Components:
        + Frame Extractor.Extractor (Folder Address:str|os.pathlike[str]): Void []

        + Base Line detection.Forward ()

        + Utilities.Empty_frames_remover()
        + Utilities.Normalizing_drop_Position()
        + Utilities.Detect&crop()

        + SROF_4S.DF_generation(): Void []
        + SROF_4S.Result_video_maker(): Void []

        + CleanUP(): Void []
"""
import os
import cv2
import glob
import FrameExtractor
import BaseLine
import Utilities
import CaMeasurer
import shutil
from cleanUp import create_video_from_images


if __name__ == "__main__":
    fe = FrameExtractor.FrameExtractor()
    bld = BaseLine.BaseLine()
    
    # Example usage of the AVP system
    # _folder = [r"D:\Videos\S1_30per_T1_C001H001S0001",
    #            ]
    for _folder in glob.glob(r"D:\Videos\*"):
        if os.path.isfile(os.path.join(_folder,'.done')):
            continue
        else:
            shutil.rmtree(os.path.join(_folder, "frames"), ignore_errors=True)
            shutil.rmtree(os.path.join(_folder, "frames_rotated"), ignore_errors=True)
            shutil.rmtree(os.path.join(_folder, "databases"), ignore_errors=True)
            shutil.rmtree(os.path.join(_folder, "SR_edge"), ignore_errors=True)

        print(f"Processing folder: {_folder}")
        fe.Forward(_folder,
                fps=30)

        bld.Forward(_folder)


        for address in glob.glob(rf"{_folder}\frames_rotated\*.png"):
            image = cv2.imread(address, cv2.IMREAD_UNCHANGED)
            image = image[:,40:-40]
            image = cv2.resize(image, (1245,130))  # Resize to match YOLO input size
            cv2.imwrite(address,image)

        Utilities.main(_folder)

        CaMeasurer.processes_mp(_folder, num_workers=10)

        create_video_from_images(image_folder=os.path.join(_folder, "SR_edge"),
                                  output_video_path=os.path.join(_folder, "result_video.mp4"),
                                  )