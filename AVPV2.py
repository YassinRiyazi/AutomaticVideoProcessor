"""
    Author: Yassin Riyazi
    Date: 01-07-2025
    Version: 2.0.0

    Description:
        Phase YOLO Frame Normalizer
        This script processes images in a given directory structure, applying YOLO object detection to filter out images
        that do not contain drops. It uses multiprocessing to speed up the processing of multiple experiments.
    
    Change log:
        V2.0.0
            Added Multiprocess

            GPU Utils:96%
            CPU Utils:50%
            5H-> 19M
        V1.0.0
            Initiated
            GPU Utils:40%
            CPU Utils:20%
"""

import  os
import  cv2
import  tqdm
import  numpy               as      np
# import  matplotlib.pyplot   as      plt
from    ultralytics         import  YOLO
import shutil
# import multiprocessing
import BaseLine
import BaseUtils
import CaMeasurer
import FrameExtractor
import glob
import argparse
# import tqdm
# import Utilities

def load_files(ad: str|os.PathLike[str]) -> list[str]:
    valid_extensions = {"tiff", "tif", "png", "jpg", "jpeg", "bmp", "gif", "webp"}  # Common image formats
    FileNames = []
    for file in sorted(os.listdir(ad)):
        try:
            if file.split(".")[-1].lower() in valid_extensions:
                FileNames.append(file)
        except IndexError:
            pass
    return sorted(FileNames)

def _forward(experiment: str|os.PathLike[str], model :YOLO) -> None:
    for i in (load_files(experiment)):
        file_address = os.path.join(experiment,i)
        image       = cv2.imread(file_address)
        
        # Perform batch YOLO prediction
        results = model(image, verbose=False)

        for file_idx, res in enumerate(results):
            if res.boxes.xyxy.shape[0]==0:
                os.remove(file_address)
                continue
            x1, _, x2, _ = np.array(res.boxes.xyxy[:, :].cpu().numpy(), dtype=np.float32)[0]

            if x2 < 1245-40 and 40 < x1:
                return None
            else:
                os.remove(file_address)

def _backward(experiment: str|os.PathLike[str], model:YOLO) -> None:
    for i in (reversed(load_files(experiment))):
        file_address = os.path.join(experiment,i)
        image       = cv2.imread(file_address)
        
        # Perform batch YOLO prediction
        results = model(image, verbose=False)

        for file_idx, res in enumerate(results):
            if res.boxes.xyxy.shape[0]==0:
                # print(f"No drop detected, probably out of scope. {file_address}")
                os.remove(file_address)
                continue
            x1, _, x2, _ = np.array(res.boxes.xyxy[:, :].cpu().numpy(), dtype=np.float32)[0]

            if x2 < 1245-40 and 40 < x1:
                return None
            else:
                os.remove(file_address)

def process_experiment(experiment: str|os.PathLike[str]) -> None:
    yolo_m = YOLO("BaseUtils/Detection/Weights/Gray-320-s.pt", task='detect', verbose=False)
    _forward(experiment,yolo_m.predict)
    _backward(experiment,yolo_m.predict)

def cleanUP(_folder: str|os.PathLike[str]) -> None:
    shutil.rmtree(os.path.join(_folder, "frames"),          ignore_errors=True)
    shutil.rmtree(os.path.join(_folder, "frames_rotated"),  ignore_errors=True)
    shutil.rmtree(os.path.join(_folder, "databases"),       ignore_errors=True)
    shutil.rmtree(os.path.join(_folder, "SR_edge"),         ignore_errors=True)
    shutil.rmtree(os.path.join(_folder, "databases_SR"),         ignore_errors=True)
    os.remove(os.path.join(_folder, 'error_log.txt')) if os.path.isfile(os.path.join(_folder, 'error_log.txt')) else None
    os.remove(os.path.join(_folder, 'result.csv')) if os.path.isfile(os.path.join(_folder, 'result.csv')) else None
    os.remove(os.path.join(_folder, 'result_video.mkv')) if os.path.isfile(os.path.join(_folder, 'result_video.mkv')) else None

def cleanStart(Video_list: list[str]):  
    for _folder in tqdm.tqdm(Video_list):
        cleanUP(_folder)

        os.remove(os.path.join(_folder, '.done')) if os.path.isfile(os.path.join(_folder, '.done')) else None
        
        logs = glob.glob(os.path.join(_folder,'*.log'))
        for log in logs:
            os.remove(log)

def CheckOutputdir(addresses:list[str],)->None:
    for folder in addresses:
        os.makedirs(folder,exist_ok=True)

if __name__ == "__main__":
    import Utilities
    import os
    mainAdress = r"D:\HSC_OPCUA_output"

    Video_list = sorted(glob.glob(os.path.join(mainAdress,'*')))
    Video_list = [folder for folder in Video_list if os.path.isdir(folder)]

    parser = argparse.ArgumentParser(description="Automatic Video Processor (AVP) launcher")
    parser.add_argument("-C", "--clean", action="store_true", help="Run cleanStart and exit")
    parser.add_argument("--video-list", nargs="*", help="Optional list of folders to clean (overrides default glob)")
    args = parser.parse_args()

    if args.clean:
        if args.video_list:
            cleanStart(Video_list=sorted(args.video_list))
        else:
            cleanStart(Video_list)
        print("cleanStart completed, exiting.")

    fe = FrameExtractor.FrameExtractor()
    bld = BaseLine.BaseLine()
    folder_Address = os.path.abspath(os.path.dirname(__file__))
    yolo_Address = os.path.join(folder_Address,f"BaseUtils/Detection/Weights/{BaseUtils.config['yolo_name']}.{BaseUtils.config['extension_yolo']}")
    yolo_m = YOLO(yolo_Address,
                  task='detect',
                  verbose=False)

    for _folder in tqdm.tqdm(Video_list[::]):
        # Phase 1: Frame Extraction
        fe.Forward(_folder)
    
        # Phase 2: Base Line Detection
        bld.Forward(_folder)

        # Phase 3: YOLO-based Frame Normalization
        _forward(os.path.join(_folder, 'frames_rotated'), yolo_m.predict)
        _backward(os.path.join(_folder, 'frames_rotated'),yolo_m.predict)

        # Phase 4: Result Compilation
        images = Utilities.BaseUtils.ImageLister(FolderAddress=_folder,frameAddress=str(Utilities.BaseUtils.config["rotated_frames_folder"]),)
        Utilities.singleFolderDropNormalizer(images,Utilities.BaseUtils.DropDetection_YOLO)# type: ignore
        # TODO: Share resource with YOLO model
        Utilities.crop_Save(image_folder=_folder)    

       
        # Phase 6: 4S-SROF
        os.makedirs(os.path.join(_folder, 'SR_edge'), exist_ok=True)
    

    # Phase 5: Super-Resolution O
    input_folders = sorted(glob.glob(os.path.join(mainAdress,'*',"databases")))
    input_folders = [folder for folder in input_folders if os.path.isdir(folder)]
    output_folders = [i.replace("databases", "databases_SR") for i in input_folders]
    CheckOutputdir(output_folders)
    CaMeasurer.process_folders_parallel(input_folders, output_folders, num_models=12)

    # Phase 6: 4S-SROF
    import multiprocessing
    from CaMeasurer import processes
    # Use multiprocessing to process multiple experiments in parallel with tqdm progress bar
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(processes, Video_list), total=len(Video_list)):
            pass