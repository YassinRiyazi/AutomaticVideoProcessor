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
import glob
import tqdm
import FrameExtractor
import BaseLine
import Utilities
import CaMeasurer
import shutil
from cleanUp import create_video_from_images # type:ignore



def cleanUP(_folder: str|os.PathLike[str]) -> None:

    shutil.rmtree(os.path.join(_folder, "frames"),          ignore_errors=True)
    shutil.rmtree(os.path.join(_folder, "frames_rotated"),  ignore_errors=True)
    shutil.rmtree(os.path.join(_folder, "databases"),       ignore_errors=True)
    shutil.rmtree(os.path.join(_folder, "SR_edge"),         ignore_errors=True)
    os.remove(os.path.join(_folder, 'error_log.txt')) if os.path.isfile(os.path.join(_folder, 'error_log.txt')) else None
    os.remove(os.path.join(_folder, 'result.csv')) if os.path.isfile(os.path.join(_folder, 'result.csv')) else None
    os.remove(os.path.join(_folder, 'result_video.mkv')) if os.path.isfile(os.path.join(_folder, 'result_video.mkv')) else None

def cleanStart(Video_list: list[str] = sorted(glob.glob("/media/Dont/Teflon-AVP/*/*/*"))):  
    for _folder in tqdm.tqdm(Video_list):
        cleanUP(_folder)

        os.remove(os.path.join(_folder, '.done')) if os.path.isfile(os.path.join(_folder, '.done')) else None
        
        logs = glob.glob(os.path.join(_folder,'*.log'))
        for log in logs:
            os.remove(log)

if __name__ == "__main__":
    fe = FrameExtractor.FrameExtractor()
    bld = BaseLine.BaseLine()
    
    Video_list = sorted(glob.glob("/media/Dont/Teflon-AVP/*/*/*"))
    # cleanStart(Video_list=Video_list)

    YOLO = Utilities.YoloWalker(num_workers=8)
    S4 = CaMeasurer.processes_mp_shared( num_workers=8)

    for _folder in Video_list[::5]:  # Process every third folder for testing
        try:
            if os.path.isfile(os.path.join(_folder,'.done')):
                continue

            elif os.path.isfile(os.path.join(_folder,'error_log.txt')):
                print(f"Skipping folder (error log exists): {_folder}")
                continue

            if len(glob.glob(os.path.join(_folder,'*.log'))) > 0:
                print(f"Skipping folder (log files exist): {_folder}")
                continue
 
            else:
                cleanUP(_folder)

            print(f"Processing folder: {_folder}")

            # Phase 1: Frame Extraction
            fe.Forward(_folder)
            # Phase 2: Base Line Detection
            bld.Forward(_folder)
            
            # Phase 3: Utilities
            # TODO: Share resource with YOLO model [Done] Utilities.main(_folder)
            YOLO.run(image_folder =_folder,skip = 40)
            YOLO.run(image_folder =_folder,skip = 5)

            Utilities.BaseUtils.FileIndexChecker(FolderAddress=_folder,frameAddress=str(Utilities.BaseUtils.config["rotated_frames_folder"]))
            images = Utilities.BaseUtils.ImageLister(FolderAddress=_folder,frameAddress=str(Utilities.BaseUtils.config["rotated_frames_folder"]),)
            Utilities.singleFolderDropNormalizer(images,Utilities.BaseUtils.DropDetection_YOLO)# type: ignore
            # TODO: Share resource with YOLO model
            Utilities.crop_Save(image_folder=_folder)    

            os.makedirs(os.path.join(_folder, 'SR_edge'), exist_ok=True)

            # Phase 4: 4S-SROF
            # TODO: Share resources [Done] CaMeasurer.processes_mp(_folder, num_workers=10)
            S4.run(_folder)
            # S4.wait_for_tasks()
            # S4.aggregate_and_save(_folder)
            # break
            # Phase 5: Result Video Maker/Clean Up
            # create_video_from_images(image_folder=os.path.join(_folder, "SR_edge"),
            #                         output_video_path=os.path.join(_folder, "result_video.mkv"),
            #                         )
            if not os.path.isfile(os.path.join(_folder, 'error_log.txt')):
                with open(os.path.join(_folder,'.done'), 'w') as f:
                    f.write('Processing completed successfully.\n')
                # shutil.rmtree(os.path.join(_folder, "SR_edge"),         ignore_errors=True)
            
        except Exception as e:
            import BaseUtils.logException as logException
            
            logger = logException.LogException(base_path=_folder)
            logger.log_exception(e, custom_message=f"Error processing folder: {_folder}", Verbose=True)
            print(f"Error processing folder: {_folder}. Check error_log.txt for details.")
            continue
        
    YOLO.close()
    S4.close()