import  os
import  cv2
import  tqdm

import  pandas          as  pd
import  numpy           as  np

from    scipy.signal            import savgol_filter
from    ultralytics             import YOLO
from    typing                  import Union, TextIO

if __name__ == "__main__":
    from    criteria_definition     import *
    from    preprocessing           import make_folders
    from    superResolution         import initiation
    from    edgeDetection           import *
    from    processing              import *
    from    visualization           import visualize
    from    main                    import *
else:
    # import os,sys
    # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from    .criteria_definition    import *
    from    .preprocessing          import make_folders
    from    .superResolution        import initiation
    from    .edgeDetection          import *
    from    .processing             import *
    from    .visualization          import visualize
    from    .main                   import *

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import BaseUtils

def logFile(ad: Union[str, os.PathLike[str]])-> TextIO: # 
    log_path = os.path.join(ad, "error_log.txt")
    if os.path.isfile(log_path):
        os.remove(log_path)
    log_file = open(log_path, "a")
    log_file.write("#Initiate\n")
    return log_file

def ensure_directory_exists(directory: str) -> None:
    """
    Ensure that the specified directory exists. 
    If it does not exist, create it along with any necessary parent directories.

    :param directory: Path of the directory to check and create if necessary.
    """
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")

def processes(address:str):
    """
    Caution:
        I assumed drop is inside images
        I assumed drop boundaries are inside image
        Images are rotated and leveled
        Images color are inverted (cv2.bitwise_not())
        Images are colored
    """

    if os.path.isfile(os.path.join(address,'SR_result','result.csv')) and os.path.isfile(os.path.join(address,'SR_edge','result.mp4')):
        return None

    df = pd.read_csv(os.path.join(address, BaseUtils.config['databases_folder'],'detections.csv')) # type: ignore


    make_folders(os.path.join(address, 'SR_edge'))
    # make_folders(os.path.join(address, 'SR_result'))

    fps                         = 4000
    cm_on_pixel_ratio           = 0.0039062
    num_px_ratio                = (0.0039062)/cm_on_pixel_ratio
    error_handling_kernel_size  = (5,5)
    model                       = initiation()
    name_files                  = BaseUtils.ImageLister(address,
                                                        str(BaseUtils.config['databases_folder']))
    kernel                      = np.ones(error_handling_kernel_size,np. uint8)


    adv_list, rec_list, contact_line_length_list, x_center_list, y_center_list, middle_angle_degree_list,processed_number_list=[],[],[],[],[],[],[]
    rec_angle_point_list, adv_angle_point_list=[],[]

    logFile(address)


    for file_number in tqdm.tqdm(range(len(name_files))):
        try:

            arggs = base_function_process(df,address,name_files,file_number,
                                                        model = model, kernel = kernel, num_px_ratio=num_px_ratio)
            i_list, j_list, i_left, j_left, i_right, j_right, j_poly_left, i_poly_left, j_poly_right, i_poly_right, x_cropped, i_poly_left_rotated, j_poly_left_rotated, i_poly_right_rotated, j_poly_right_rotated = arggs
            distance = (x_cropped) * 3
            _address=os.path.join(address,'SR_edge',os.path.basename(name_files[file_number]))
            adv, rec,rec_angle_point, adv_angle_point, contact_line_length, x_center, y_center, middle_angle_degree=visualize(_address, 
                                                                                                                            distance+np.array(i_list),j_list,
                                                                                                                            distance+np.array(i_left),j_left,
                                                                                                                            distance+np.array(i_right),j_right,
                                                                                                                            j_poly_left,distance+np.array(i_poly_left),
                                                                                                                            j_poly_right,distance+np.array(i_poly_right),
                                                                                                                            x_cropped,
                                                                                                                            distance+np.array(i_poly_left_rotated), j_poly_left_rotated,
                                                                                                                            distance+np.array(i_poly_right_rotated),j_poly_right_rotated, 
                                                                                                                            cm_on_pixel=cm_on_pixel_ratio, middle_line_switch=1,
                                                                                                                            dpi = 100)

            processed_number_list.append(os.path.basename(name_files[file_number]))
            adv_list.append(adv)
            rec_list.append(rec)
            adv_angle_point_list.append(adv_angle_point)
            rec_angle_point_list.append(rec_angle_point)
            contact_line_length_list.append(contact_line_length)
            x_center_list.append(x_center)
            y_center_list.append(y_center)
            middle_angle_degree_list.append(middle_angle_degree)

        except Exception as e:
            print(e)
            # Append the error message to a log file
            print(f'File name {os.path.join(address,name_files[file_number])} with shape of :{cv2.imread(os.path.join(address,"drops",name_files[file_number])).shape}')
            with open(os.path.join(address,"error_log.txt"), "a") as log_file:
                log_file.write(f'File name {os.path.join(address,name_files[file_number])} with shape of :{cv2.imread(os.path.join(address,"drops",name_files[file_number])).shape}' + "\n")
            return None
    vel=[]
    # try:
    for i in range(len(x_center_list)-1):
        vel=vel+[x_center_list[i+1]-x_center_list[i]]

    vel=np.array(vel)*fps

    df=pd.DataFrame([processed_number_list, np.arange(0, 1/fps*len(vel), 1/fps), x_center_list,
                     adv_list,rec_list,contact_line_length_list,
                     y_center_list, middle_angle_degree_list, vel]).T
    df=df[:-1]

    df.columns=['file number', "time (s)", 'x_center (cm)',
                'adv (degree)', 'rec (degree)', 'contact_line_length (cm)',
                'y_center (cm)', 'middle_angle_degree (degree)', 'velocity (cm/s)']
    df.to_csv(os.path.join(address, 'result.csv'), index=False)








import multiprocessing as mp
# ---- GLOBAL (per-process)
model = None
kernel = None
num_px_ratio = None
df = None
address = None
cm_on_pixel_ratio = None
fps = None


def _init_worker(shared_df, shared_address, shared_kernel, shared_num_px_ratio, shared_cm_on_pixel_ratio, shared_fps):
    """
    Worker initializer: executed once per process.
    Initializes the model and shared parameters globally.
    """
    global model, kernel, num_px_ratio, df, address, cm_on_pixel_ratio, fps
    model = initiation()  # each worker has its own model
    kernel = shared_kernel
    num_px_ratio = shared_num_px_ratio
    cm_on_pixel_ratio = shared_cm_on_pixel_ratio
    fps = shared_fps
    df = shared_df
    address = shared_address


import functools

def process_one_file(file_number, name_files):
    """Process a single file (executed inside workers)."""
    try:
        arggs = base_function_process(
            df, address, name_files, file_number,
            model=model, kernel=kernel, num_px_ratio=num_px_ratio
        )

        (i_list, j_list, i_left, j_left, i_right, j_right,
         j_poly_left, i_poly_left, j_poly_right, i_poly_right,
         x_cropped, i_poly_left_rotated, j_poly_left_rotated,
         i_poly_right_rotated, j_poly_right_rotated) = arggs

        distance = (x_cropped) * 3
        _address = os.path.join(address, 'SR_edge', os.path.basename(name_files[file_number]))

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
            cm_on_pixel=cm_on_pixel_ratio, middle_line_switch=1, dpi=100
        )

        return {
            "file": os.path.basename(name_files[file_number]),
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
        print(f"Error: {e}")
        file_path = os.path.join(address, "drops", name_files[file_number])
        print(f"File name {file_path} with shape of :{cv2.imread(file_path).shape}")
        with open(os.path.join(address, "error_log.txt"), "a") as log_file:
            log_file.write(f"File name {file_path} with shape of :{cv2.imread(file_path).shape}\n")
        return None



from multiprocessing import Pool
from functools import partial

def processes_mp(shared_address: str, num_workers: int = 15):

    if os.path.isfile(os.path.join(shared_address, 'SR_result', 'result.csv')) and \
       os.path.isfile(os.path.join(shared_address, 'SR_edge', 'result.mp4')):
        return None

    shared_df = pd.read_csv(os.path.join(shared_address, BaseUtils.config['databases_folder'], 'detections.csv'))
    make_folders(os.path.join(shared_address, 'SR_edge'))
    # make_folders(os.path.join(shared_address, 'SR_result'))

    fps = 4000
    cm_on_pixel_ratio = 0.0039062
    num_px_ratio = (0.0039062) / cm_on_pixel_ratio
    error_handling_kernel_size = (5, 5)
    kernel = np.ones(error_handling_kernel_size, np.uint8)
    name_files = BaseUtils.ImageLister(shared_address, str(BaseUtils.config['databases_folder']))

    results = []
    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(shared_df, shared_address, kernel, num_px_ratio, cm_on_pixel_ratio, fps)
    ) as pool:
        # ✅ use partial, not lambda
        worker_func = partial(process_one_file, name_files=name_files)
        for res in tqdm.tqdm(pool.imap_unordered(worker_func, range(len(name_files))),
                             total=len(name_files)):
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
    df_out.to_csv(os.path.join(shared_address, 'result.csv'), index=False)
if __name__ == "__main__":
    
    address  = r"D:\Videos\S1_30per_T1_C001H001S0001"
    processes_mp(address)
