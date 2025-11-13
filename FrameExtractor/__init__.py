"""
    Date    : 2025-09-30
    Author  : Yassin Riyazi
    Project : Automatic Video Processor (AVP)
    File    : FrameExtractor/__init__.py
    Version : 1.0.0
    License : GNU GENERAL PUBLIC LICENSE Version 3

    Components:
        + Frame Generator (Folder Address:str|os.pathlike[str]): Void [Checks frames are existing]
        + Frame B&W maker (Folder Address:str|os.pathlike[str]): Void [Checks and make frames B&W]
        + Bottom_row_checker (Folder Address:str|os.pathlike[str]): Void [Checks bottom row is not white]
        + Frame Health Checker (Folder Address:str|os.pathlike[str]): Void [Checks frames are existing, non zero size, readable]
        + Forward (Folder Address:str|os.pathlike[str]) : Void [Run all above for a selected folder]

    Assumptions:
        there is only one .mp4 file in the folder
        there is no subfolder in the folder
        frames will be saved in a subfolder named 'frames'
        if frames folder already exists, it will not be overwritten unless wipe=True
        frames will be named as 'frame_%06d.png' by default (if output_frame_pattern is None)
        frames will be extracted in grayscale by default (if grayscale=True)
        frames will be checked for health by default (if health_check=True)
        if frames are not in grayscale, they will be converted to grayscale
        if frames are corrupted or zero size, they will be removed
        if frames are not sequentially indexed, an error will be raised
        ffmpeg is installed and added to the system path

    Changelog:
        - 2025-09-30: Initial version
        - 2025-11-06: Adding output folder option
        - 2025-11-12: Automatic fps detection
"""
# import re
import os
import glob
import colorama
import numpy as np
from PIL import Image
if __name__ == "__main__":
    from Video2Jpg import ffmpeg_frame_extractor, init
else:
    from .Video2Jpg import ffmpeg_frame_extractor, init

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import BaseUtils

class FrameExtractor:
    def __init__(self,
                 InPlaceOutput = False) -> None:
        init()
        self.frameAddress = BaseUtils.config["frame_folder"]
        self.InPlaceOutput = InPlaceOutput

    def ffprobe_fps(self, video_path: str) -> float:
        """
        Get the frames per second (fps) of a video using ffprobe.
        """
        import subprocess
        import re
        if not os.path.isfile(video_path):
            try:
                video_path = glob.glob(os.path.join(video_path, "*.mp4"))[0]    
            except:
                raise FileNotFoundError(colorama.Fore.RED + f"Video file not found: {video_path}" + colorama.Style.RESET_ALL)
        command = [
            'ffprobe', '-v', '0', '-of', 'csv=p=0',
            '-select_streams', 'v:0', '-show_entries',
            'stream=r_frame_rate', video_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        fps_info = result.stdout.strip()

        # fps_info is in the form of "num/den"
        num, den = map(int, fps_info.split('/'))
        fps = num / den if den != 0 else 0
        return fps

    def extract_frames(self,
                       FolderAddress: str, 
                       output_frame_pattern: str | None = None, 
                       wipe: bool = False,
                       use_cuda: bool = False,
                       fps: float | int | None = None,
                       grayscale: bool = True,
                       health_check: bool = True
                       ) -> int:
        # checking if adress has no extention then look for .mp4 files
        if os.path.splitext(FolderAddress)[1] == "":
            # raise ValueError(colorama.Fore.RED + f"Please provide a folder address without file extension: {FolderAddress}" + colorama.Style.RESET_ALL)
            video_path = glob.glob(os.path.join(FolderAddress, "*.mp4"))

            if len(video_path) != 1:
                # clean up colorama in case of error
                colorama.init(autoreset=True)
                # cleaning screen
                os.system('cls' if os.name == 'nt' else 'clear')
                raise FileNotFoundError(colorama.Fore.RED + f"No or multiple .mp4 files found in the directory: {FolderAddress}" + colorama.Style.RESET_ALL)

            video_path = video_path[0]
        else:
            video_path = FolderAddress

        if fps is None:
            fps = self.ffprobe_fps(video_path)

        frame_dir = os.path.join(os.path.dirname(video_path), str(self.frameAddress))
        if os.path.exists(frame_dir) and wipe == False:
            print(colorama.Fore.YELLOW + f"Frames directory already exists in {FolderAddress}. Use wipe=True to overwrite." + colorama.Style.RESET_ALL)
            return 0

        ffmpeg_frame_extractor(
            video_path,
            output_frame_pattern=output_frame_pattern,
            fps=fps,
            wipe=wipe,
            use_cuda=use_cuda,
            grayscale=grayscale,
            health_check=health_check
        )
        return 1
    
    def BandGMaker(self, FolderAddress: str):
        """
        Checking images are B&W, if not convert them to B&W
        """
        images = BaseUtils.ImageLister(FolderAddress=FolderAddress)
        
        for img_file in images:    
            img = Image.open(img_file)
            if img.mode == 'L':
                continue  # Already grayscale
            img = img.convert('L')  # Convert to grayscale
            img.save(img_file)

    def HealthChecker(self, FolderAddress: str):
        """
        Checking images are readable and non zero size
        """
        images = BaseUtils.ImageLister(FolderAddress=FolderAddress)
        
        for img_file in (images):    
            try:
                img = Image.open(img_file)
                img.verify()  # Verify that it is, in fact an image
            except (IOError, SyntaxError):
                print(colorama.Fore.RED + f"Corrupted image file detected and removed: {img_file}" + colorama.Style.RESET_ALL)
                os.remove(img_file)
                continue
            
            if os.path.getsize(img_file) == 0:
                print(colorama.Fore.RED + f"Zero-size image file detected and removed: {img_file}" + colorama.Style.RESET_ALL)
                os.remove(img_file)

    def Bottom_row_checker(self, FolderAddress: str):
        images = BaseUtils.ImageLister(FolderAddress=FolderAddress)    

        for img_file in (images[::4]):
            img = Image.open(img_file)
            img_array = np.array(img)
            bottom_row = img_array[-1, :].mean(axis=-1)  # Average across color channels if present
            if bottom_row > 100:  # Threshold for "mostly white"
                raise ValueError(colorama.Fore.RED + f"The sample line is not visible; thus, baseline detection will fail for: {img_file}" + colorama.Style.RESET_ALL)

    def Forward(self,
                FolderAddress: str,
                fps: float | int | None = None,
                out_dir: str | None = None):
        """
        Extract frames and run checks. If out_dir is provided frames will be written
        to that directory (pattern: out_dir/frame_%06d.png) and subsequent checks
        will operate on that directory. Otherwise behavior stays the same.
        """
        # determine the frames directory to use for post-processing checks
        if out_dir:
            frames_folder = out_dir
            output_pattern = os.path.join(frames_folder, 'frame_%06d.png')
        else:
            frames_folder = os.path.join(os.path.dirname(FolderAddress), str(self.frameAddress))
            output_pattern = None

        # run extraction (pass output_pattern when provided)
        _ = self.extract_frames(FolderAddress=FolderAddress, fps=fps, output_frame_pattern=output_pattern)

        # run post extraction checks on the frames folder
        self.BandGMaker(FolderAddress=FolderAddress)
        self.Bottom_row_checker(FolderAddress=FolderAddress)
        self.HealthChecker(FolderAddress=FolderAddress)
        BaseUtils.FileIndexChecker(FolderAddress=FolderAddress)


if __name__ == "__main__":
    fe = FrameExtractor()
    fe.Bottom_row_checker(
        FolderAddress=r"/media/Dont/Teflon-AVP/280/S3-SNr3.02_D/T052_02_12.306287095218",#/media/Dont/Teflon-AVP/280/S3-SNr3.01_D/T111_01_2.416221423374
        
    )