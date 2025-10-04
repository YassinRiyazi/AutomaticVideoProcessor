import  os
import  cv2
import  tqdm
# import  functools
import  traceback
import  pandas          as  pd
import  numpy           as  np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import BaseUtils

if __name__ == "__main__":
    from    criteria_definition     import *
    from    superResolution         import initiation
    from    edgeDetection           import *
    from    processing              import *
    from    visualization           import visualize
    from    main                    import *
else:
    from    .criteria_definition    import *
    from    .superResolution        import initiation
    from    .edgeDetection          import *
    from    .processing             import *
    from    .visualization          import visualize
    from    .main                   import *

import multiprocessing as mp
import platform
system = platform.system().lower()

if system == "linux":   # Ubuntu, Debian, etc.
    mp.set_start_method("spawn", force=True)


# ---- GLOBAL (per-process)
model = None
kernel = None
num_px_ratio = None
df = None
address = None
cm_on_pixel_ratio = None
fps = None

def _init_worker(shared_df: pd.DataFrame, 
                 shared_address: str, 
                 shared_kernel: np.ndarray, 
                 shared_num_px_ratio: float, 
                 shared_cm_on_pixel_ratio: float, 
                 shared_fps: float):
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

def process_one_file(file_number: int, name_files: list) -> dict:
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
        file_path = os.path.join(address, "drops", name_files[file_number])
        error_msg = f"Error processing file {file_path}: {e}"

        # Print full traceback to console
        print(error_msg)
        traceback.print_exc()

        # Check image validity
        img = cv2.imread(file_path)
        if img is not None:
            img_shape = img.shape
        else:
            img_shape = "Image could not be read (None)"
    
        # Write detailed error log
        log_path = os.path.join(address, "error_log.txt")
        with open(log_path, "a+") as log_file:
            log_file.write("\n" + "="*80 + "\n")
            log_file.write(f"File: {file_path}\n")
            log_file.write(f"Image shape: {img_shape}\n")
            log_file.write("Exception traceback:\n")
            log_file.write(traceback.format_exc())  # full traceback
            log_file.write("="*80 + "\n")

        return None

def processes(_address:str):
    """
    Caution:
        I assumed drop is inside images
        I assumed drop boundaries are inside image
        Images are rotated and leveled
        Images color are inverted (cv2.bitwise_not())
        Images are colored
    """
    global model, kernel, num_px_ratio, df, address, cm_on_pixel_ratio, fps
    address = _address

    if os.path.isfile(os.path.join(address,'SR_result','result.csv')) and os.path.isfile(os.path.join(address,'SR_edge','result.mp4')):
        raise Exception("processes already done")

    df = pd.read_csv(os.path.join(address, BaseUtils.config['databases_folder'],'detections.csv')) # type: ignore

    os.makedirs(os.path.join(address, 'SR_edge'), exist_ok=True)

    fps                         = BaseUtils.config['fps_experiment']  # fps of the original experiment video
    cm_on_pixel_ratio           = 0.0039062
    num_px_ratio                = (0.0039062)/cm_on_pixel_ratio
    error_handling_kernel_size  = (5,5)
    model                       = initiation()
    name_files                  = BaseUtils.ImageLister(address,str(BaseUtils.config['databases_folder']))
    kernel                      = np.ones(error_handling_kernel_size,np. uint8)



    results = []
    for file_number in tqdm.tqdm(range(len(name_files))):
        if int(name_files[file_number][-10:-4]) in [321, 322, 323, 324, 325]:
            breakpoint()
        res = process_one_file(file_number, name_files)
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





from multiprocessing import Pool
from functools import partial

def processes_mp(shared_address: str, num_workers: int = 15):

    if os.path.isfile(os.path.join(shared_address,'SR_result','result.csv')) and os.path.isfile(os.path.join(shared_address,'SR_edge','result.mp4')):
        raise Exception("processes already done")

    shared_df = pd.read_csv(os.path.join(shared_address, BaseUtils.config['databases_folder'],'detections.csv')) # type: ignore

    os.makedirs(os.path.join(shared_address, 'SR_edge'), exist_ok=True)

    fps = BaseUtils.config['fps_experiment']  # fps of the original experiment video
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


def single (_address: str, file_number: int, name_files: list[str]):
    global model, kernel, num_px_ratio, df, address, cm_on_pixel_ratio, fps
    address = _address

    df = pd.read_csv(os.path.join(address, BaseUtils.config['databases_folder'],'detections.csv')) # type: ignore

    fps                         = BaseUtils.config['fps_experiment']  # fps of the original experiment video
    cm_on_pixel_ratio           = 0.0039062
    num_px_ratio                = (0.0039062)/cm_on_pixel_ratio
    error_handling_kernel_size  = (5,5)
    model                       = initiation()
    kernel                      = np.ones(error_handling_kernel_size,np. uint8)

    process_one_file(file_number, name_files)


if __name__ == "__main__":
    
    address  = r"/media/Dont/Teflon-AVP/285/S3-SDS99_D/T120_01_0.900951687825"
    # processes(address)
    
    single (address,
            0, ['/media/Dont/Teflon-AVP/285/S3-SDS99_D/T120_01_0.900951687825/databases/frame_000311.png'])