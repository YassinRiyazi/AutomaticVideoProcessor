"""
    Date    : 2025-09-30
    Author  : Yassin Riyazi
    Project : Automatic Video Processor (AVP)
    File    : AVP_Pass2.py
    Version : 1.0.0
    License : MIT License

    Description: In continuation of AVP.py, this file handles missed frames, 
    2. checks drop position
    3. CA measurement
    4. Add to the results.csv
    5. postprocess the results.csv to fix any position/velocity mismatches.

"""
import re
import os
import glob
from typing import List,Set
import pandas as pd

import tqdm
import FrameExtractor
import BaseLine
import Utilities
import CaMeasurer
import shutil
import argparse
# import sys
from cleanUp import create_video_from_images # type:ignore




pattern = re.compile(r"Exception: ValueError: Image '(frame_\d+\.png)' not found in the CSV\.")
def Reg(path_to_logfile: str, pattern: re.Pattern[str] = pattern) -> Set[str]:
    # Regex pattern to match the ValueError line and extract frame index

    # Read the log file
    with open(path_to_logfile, "r") as f:
        log_data = f.read()

    # Find all frame indices in the log
    frame_numbers = {(m.group(1)) for m in pattern.finditer(log_data)}

    return frame_numbers


if __name__ == "__main__":

    for folder_Address in tqdm.tqdm(sorted(glob.glob("/media/Dont/Teflon-AVP/*/*/*"))):
        error_log_path      = os.path.join(folder_Address, "databases", "error_log.txt")
        if not os.path.isfile(error_log_path):
            continue
        print(f"Processing folder: {folder_Address}")
        
        result_csv_path     = os.path.join(folder_Address, "result.csv")
        detections_csv_path = os.path.join(folder_Address, "databases", "detections.csv")

        df              = pd.read_csv(detections_csv_path)
        df_result       = pd.read_csv(result_csv_path)
        AllImages       = glob.glob(os.path.join(folder_Address, "frames_rotated", "*.png"))    
        AllImages_names = {os.path.basename(img) for img in AllImages}
        ErrorMissing    = Reg(error_log_path)
        AllMissing      = (set(df_result['file number']) | AllImages_names) - (set(df_result['file number']) & AllImages_names)
        
        vv = ErrorMissing | AllMissing
        print("Missing frames:", vv)

        # Ensure every entry in vv exists in df['image']
        missing_not_in_df = vv - set(df['image'].astype(str))
        if missing_not_in_df:
            raise ValueError(f"The following frames from vv are not present in detections.csv: {sorted(missing_not_in_df)}")
        
        ## Step. CA measurement for missing frames

        ## Step. Merging missing frames into result.csv
        # df_result = pd.concat([df_result, missing_frames], ignore_index=True)
        # df_result = df_result.sort_values(by='image').reset_index(drop=True)

        ## Step. DF post processing
        break



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Automatic Video Processor (AVP) launcher")
#     parser.add_argument("-C", "--clean", action="store_true", help="Run cleanStart and exit")
#     parser.add_argument("--video-list", nargs="*", help="Optional list of folders to clean (overrides default glob)")
#     args = parser.parse_args()

#     if args.clean:
#         print("Clean start will not apply in pass 2.")

#     fe = FrameExtractor.FrameExtractor()
#     bld = BaseLine.BaseLine()
    
#     Video_list = sorted(glob.glob("/media/Dont/Teflon-AVP/*/*/*"))


#     YOLO = Utilities.YoloWalker(num_workers=5)
#     S4 = CaMeasurer.processes_mp_shared(num_workers=8)

#     for _folder in Video_list[::]:
#         try:
#             if os.path.isfile(os.path.join(_folder,'.done')):
#                 continue

#             elif os.path.isfile(os.path.join(_folder,'error_log.txt')):
#                 print(f"Skipping folder (error log exists): {_folder}")
#                 continue

#             if len(glob.glob(os.path.join(_folder,'*.log'))) > 0:
#                 print(f"Skipping folder (log files exist): {_folder}")
#                 continue
 
#             else:
#                 cleanUP(_folder)

#             print(f"Processing folder: {_folder}")

#             # Phase 1: Frame Extraction
#             fe.Forward(_folder)
#             # Phase 2: Base Line Detection
#             bld.Forward(_folder)
            
#             # Phase 3: Utilities
#             # TODO: Share resource with YOLO model [Done] Utilities.main(_folder)
#             YOLO.run(image_folder =_folder,skip = 40)
#             YOLO.run(image_folder =_folder,skip = 5)

#             Utilities.BaseUtils.FileIndexChecker(FolderAddress=_folder,frameAddress=str(Utilities.BaseUtils.config["rotated_frames_folder"]))
#             images = Utilities.BaseUtils.ImageLister(FolderAddress=_folder,frameAddress=str(Utilities.BaseUtils.config["rotated_frames_folder"]),)
#             Utilities.singleFolderDropNormalizer(images,Utilities.BaseUtils.DropDetection_YOLO)# type: ignore
#             # TODO: Share resource with YOLO model
#             Utilities.crop_Save(image_folder=_folder)    

#             os.makedirs(os.path.join(_folder, 'SR_edge'), exist_ok=True)

#             # Phase 4: 4S-SROF
#             # TODO: Share resources [Done] CaMeasurer.processes_mp(_folder, num_workers=10)
#             S4.run(_folder)
            
#             _ = Utilities.position_velocity_correction(os.path.join(_folder, 'result.csv'))

#             if not os.path.isfile(os.path.join(_folder, 'error_log.txt')):
#                 with open(os.path.join(_folder,'.done'), 'w') as f:
#                     f.write('Processing completed successfully.\n')
#                 # shutil.rmtree(os.path.join(_folder, "SR_edge"),         ignore_errors=True)
            
#         except Exception as e:
#             import BaseUtils.logException as logException
            
#             logger = logException.LogException(base_path=_folder)
#             logger.log_exception(e, custom_message=f"Error processing folder: {_folder}", Verbose=True)
#             print(f"Error processing folder: {_folder}. Check error_log.txt for details.")
#             continue
        
#     YOLO.close()
#     S4.close()
