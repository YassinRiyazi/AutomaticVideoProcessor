"""
    Author: Sajjad Shumaly
    Maintainer: Yassin Riyazi
    Date: 10.11.2025
    Version: 2.0.0
    Description: 
        CaMeasurer main file
        --------------------
        This file includes the main functions to process the drop videos
        using multiprocessing with persistent YOLO models.  

    Change log:
        V2.0.0
            Seperated the super resolution and 4S-SROF into different modules
"""

import  os
import  cv2
import  tqdm
# import  functools
import  traceback
import  pandas          as  pd
import  numpy           as  np
from   numpy.typing   import  NDArray

from multiprocessing import Pool
from functools import partial


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import BaseUtils

if __name__ == "__main__":
    from    criteria_definition     import *
    from    superResolution         import initiation
    from    BaseUtils.Detection.edgeDetection           import *
    from    processing              import *
    from    visualization           import visualize
    from    Multi_Video                    import *
    from    main                   import *
else:

    from    .criteria_definition    import *
    from    .superResolution        import initiation
    from    BaseUtils.Detection.edgeDetection          import *
    from    .processing             import *
    from    .visualization          import visualize
    from    .main                   import *

from typing import List, Dict, Any
import datetime
import threading
import time

def process_one_file(address:str,
                     file_number: int,
                     name_files: List[str],
                     kernel: NDArray[np.uint8],
                     df: pd.DataFrame) -> Dict[str, Any]:
    """Process a single file (executed inside workers)."""
    
    try:
        arggs = base_function_process(
            df,
            name_files,
            file_number,
            kernel=kernel 
        )

        (i_list, j_list, i_left, j_left, i_right, j_right,
         j_poly_left, i_poly_left, j_poly_right, i_poly_right,
         x_cropped, i_poly_left_rotated, j_poly_left_rotated,
         i_poly_right_rotated, j_poly_right_rotated) = arggs

        distance = (x_cropped) * int(BaseUtils.config['Super_Resolution']['upscale_factor'])
        _address = os.path.join(address, 'SR_edge', os.path.basename(str(name_files[file_number])))


        adv, rec, rec_angle_point, adv_angle_point, contact_line_length, \
        x_center, y_center, middle_angle_degree = visualize(
            _address,
            distance + np.array(i_list), j_list,
            distance + np.array(i_left), j_left,
            distance + np.array(i_right), j_right,
            j_poly_left, distance + np.array(i_poly_left),
            j_poly_right, distance + np.array(i_poly_right),
            x_cropped,
            distance + np.array(i_poly_left_rotated), j_poly_left_rotated,
            distance + np.array(i_poly_right_rotated), j_poly_right_rotated,
            cm_on_pixel=float(BaseUtils.config['Experimetnt_Parameters']['cm_on_pixel_ratio']),
            middle_line_switch=1,
            dpi=100
        )

        return {
            "file": os.path.basename(str(name_files[file_number])),
            "adv": adv,
            "rec": rec,
            "adv_angle_point": adv_angle_point,
            "rec_angle_point": rec_angle_point,
            "contact_line_length": contact_line_length,
            "x_center": x_center,
            "y_center": y_center,
            "middle_angle_degree": middle_angle_degree,
        }

    except Exception as e:
        file_path = str(name_files[file_number])
        dirpath = os.path.dirname(file_path)
        os.makedirs(dirpath, exist_ok=True)
        log_path = os.path.join(dirpath, "error_log.txt")

        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        pid = os.getpid()
        tid = threading.get_ident()
        exc_type = type(e).__name__
        tb_str = traceback.format_exc()
        print(f"Error processing file {file_path}: {exc_type}: {e}")

        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n{'=' * 80}\n")
            log_file.write(f"Timestamp (UTC): {timestamp}\n")
            log_file.write(f"Process ID: {pid}    Thread ID: {tid}\n")
            log_file.write(f"File: {file_path}\n")
            log_file.write(f"Exception: {exc_type}: {e}\n")
            log_file.write("Traceback:\n")
            log_file.write(tb_str)
            log_file.write(f"{'=' * 80}\n")
        return None


def processes(address:str,progress_bar:bool=False) -> None:
    """
    Caution:
        I assumed drop is inside images
        I assumed drop boundaries are inside image
        Images are rotated and leveled
        Images color are inverted (cv2.bitwise_not())
        Images are colored
    """

    if os.path.isfile(os.path.join(address,'SR_result','result.csv')) and os.path.isfile(os.path.join(address,'SR_edge','result.mp4')):
        raise Exception("processes already done")

    df = pd.read_csv(os.path.join(address, BaseUtils.config['databases_folder'],'detections.csv')) # type: ignore

    os.makedirs(os.path.join(address, 'SR_edge'), exist_ok=True)

    fps                         = BaseUtils.config['Experimetnt_Parameters']['fps_experiment']  # fps of the original experiment video
    error_handling_kernel_size  = (5,5)
    kernel                      = np.ones(error_handling_kernel_size,np. uint8)
    name_files                  = BaseUtils.ImageLister(address,'databases_SR')

    results = []
    if progress_bar:
        file_numbers = tqdm.tqdm(range(len(name_files)))
    else:
        file_numbers = range(len(name_files))
    for file_number in file_numbers:
        res = process_one_file(address, file_number, name_files, kernel, df)
        if res is not None:
            results.append(res)

    # Aggregate results
    results = sorted(results, key=lambda r: r["file"])
    processed_number_list = [r["file"] for r in results]
    adv_list = [r["adv"] for r in results]
    rec_list = [r["rec"] for r in results]
    adv_angle_point_list = [r["adv_angle_point"] for r in results]
    rec_angle_point_list = [r["rec_angle_point"] for r in results]
    contact_line_length_list = [r["contact_line_length"] for r in results]
    x_center_list = [r["x_center"] for r in results]
    y_center_list = [r["y_center"] for r in results]
    middle_angle_degree_list = [r["middle_angle_degree"] for r in results]

    vel = []
    for i in range(len(x_center_list) - 1):
        vel.append(x_center_list[i + 1] - x_center_list[i])
    vel = np.array(vel) * fps

    df_out = pd.DataFrame([
        processed_number_list,
        np.arange(0, 1 / fps * len(vel), 1 / fps),
        x_center_list,
        adv_list,
        rec_list,
        contact_line_length_list,
        y_center_list,
        middle_angle_degree_list,
        vel
    ]).T
    df_out = df_out[:-1]

    df_out.columns = [
        'file number', "time (s)", 'x_center (cm)',
        'adv (degree)', 'rec (degree)', 'contact_line_length (cm)',
        'y_center (cm)', 'middle_angle_degree (degree)', 'velocity (cm/s)'
    ]
    df_out.to_csv(os.path.join(address, 'result.csv'), index=False)



if __name__ == "__main__":
    # processes ('/media/d25u2/Dont/Viscosity/280/S5-S2.01_S20/D175220_01', progress_bar=True)

    import glob
    import multiprocessing

    addresses = sorted(glob.glob("/media/d25u2/Dont/Viscosity/*/*/*"))
    addresses = [address for address in addresses if os.path.isfile(os.path.join(address, 'databases', 'detections.csv'))]

    # Use multiprocessing to process multiple experiments in parallel with tqdm progress bar
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(processes, addresses), total=len(addresses)):
            pass


    