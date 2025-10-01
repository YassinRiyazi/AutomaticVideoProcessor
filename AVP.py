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
import glob
import cv2
import FrameExtractor
import BaseLine
import Utilities

if __name__ == "__main__":
    fe = FrameExtractor.FrameExtractor()
    bld = BaseLine.BaseLine()
    
    # Example usage of the AVP system
    _folder = r"D:\Videos\S1_30per_T1_C001H001S0001"

    fe.Forward(_folder,
               fps=30)

    bld.Forward(_folder)


    for address in glob.glob(r"D:\Videos\S1_30per_T1_C001H001S0001\frames_rotated\*.png"):
        image = cv2.imread(address, cv2.IMREAD_UNCHANGED)
        image = image[:,40:-40]
        image = cv2.resize(image, (1245,130))  # Resize to match YOLO input size
        cv2.imwrite(address,image)

    Utilities.main(_folder)