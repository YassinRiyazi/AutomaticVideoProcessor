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
    from    main                    import *
else:

    from    .criteria_definition    import *
    from    .superResolution        import initiation
    from    BaseUtils.Detection.edgeDetection          import *
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

from typing import List, Dict, Any
import datetime
import threading
import time

def process_one_file(file_number: int, name_files: List[str]) -> Dict[str, Any]:
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
            cm_on_pixel=cm_on_pixel_ratio, middle_line_switch=1, dpi=100
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


def _barrier_func():
    """Used to synchronize workers without sending a lambda (picklable)."""
    return None


def _update_worker_globals(args):
    """Executed inside each worker to update global variables between runs."""
    global df, address, kernel, num_px_ratio, cm_on_pixel_ratio, fps
    df, address, kernel, num_px_ratio, cm_on_pixel_ratio, fps = args
    return True


class processes_mp_shared:
    """
    Multiprocessing drop analysis with persistent YOLO models,
    but re-updates all per-run variables (address, df, etc.) each call.
    """

    def __init__(self, num_workers: int = 15):
        self.num_workers = num_workers
        self._pool = None
        self._worker_func = None

    # def _ensure_pool(self):
    #     """Create the pool if not already created."""
    #     if self._pool is None:
    #         print("🚀 Creating persistent worker pool (YOLO initialized once per worker)...")
    #         self._pool = Pool(processes=self.num_workers, initializer=_init_worker, initargs=(None, None, None, None, None, None))
    #     else:
    #         # print("♻️ Reusing existing worker pool (YOLO already loaded).")
    #         pass
    def _ensure_pool(self, init_args=None):
        """Create the pool if not already created. Accept init_args to pass to workers."""
        if self._pool is None:
            print("🚀 Creating persistent worker pool (YOLO initialized once per worker)...")
            # init_args should be a tuple: (shared_df, shared_address, kernel, num_px_ratio, cm_on_pixel_ratio, fps)
            if init_args is None:
                # fallback: no heavy per-worker initialization (not recommended)
                init_args = (None, None, None, None, None, None)
            self._pool = Pool(processes=self.num_workers, initializer=_init_worker, initargs=init_args)
        else:
            # print("♻️ Reusing existing worker pool (YOLO already loaded).")
            pass

    def _update_pool_globals(self, shared_df, shared_address, kernel, num_px_ratio, cm_on_pixel_ratio, fps):
        """Send new global parameters to all workers."""
        # print(f"🔄 Updating worker globals for {shared_address}...")
        args = (shared_df, shared_address, kernel, num_px_ratio, cm_on_pixel_ratio, fps)
        # Send updates to all workers
        list(self._pool.map(_update_worker_globals, [args] * self.num_workers))

    def run(self, shared_address: str):
        """
        Run one processing batch.
        Keeps YOLO loaded across runs, updates per-run globals in all workers.
        """
        # if (os.path.isfile(os.path.join(shared_address, 'SR_result', 'result.csv')) and
        #     os.path.isfile(os.path.join(shared_address, 'SR_edge', 'result.mp4'))):
        #     raise Exception("processes already done")

        shared_df = pd.read_csv(os.path.join(shared_address, str(BaseUtils.config['databases_folder']), 'detections.csv'))
        name_files = BaseUtils.ImageLister(shared_address, str(BaseUtils.config['databases_folder']))
        name_files = [file for file in name_files if os.path.isfile(file)]

        os.makedirs(os.path.join(shared_address, 'SR_edge'), exist_ok=True)

        fps = BaseUtils.config['fps_experiment']
        cm_on_pixel_ratio = 0.0039062
        num_px_ratio = (0.0039062) / cm_on_pixel_ratio
        kernel = np.ones((5, 5), np.uint8)

        # prepare args
        init_args = (shared_df, shared_address, kernel, num_px_ratio, cm_on_pixel_ratio, fps)

            # 1. ensure pool exists
        self._ensure_pool(init_args=init_args)

        # ✅ 2. update globals for all workers
        self._update_pool_globals(shared_df, shared_address, kernel, num_px_ratio, cm_on_pixel_ratio, fps)

        # 3. define worker func
        worker_func = partial(process_one_file, name_files=name_files)

        # 4. run processing
        results = []

        for res in tqdm.tqdm(self._pool.imap_unordered(worker_func, range(len(name_files))),
                             total=len(name_files), desc=f"Processing {shared_address}"):
            if res is not None:
                results.append(res)

        # Step 5: wait for all workers to finish
        self._pool.apply(_barrier_func)

        # Step 6: aggregate results
        if not results:
            raise RuntimeError("No valid results were produced. Check error_log.txt")

        results = sorted(results, key=lambda r: r["file"])

        processed_number_list = [r["file"] for r in results]
        adv_list = [r["adv"] for r in results]
        rec_list = [r["rec"] for r in results]
        contact_line_length_list = [r["contact_line_length"] for r in results]
        x_center_list = [r["x_center"] for r in results]
        y_center_list = [r["y_center"] for r in results]
        middle_angle_degree_list = [r["middle_angle_degree"] for r in results]

        vel = np.diff(x_center_list) * fps

        df_out = pd.DataFrame({
            "file number": processed_number_list[:-1],
            "time (s)": np.arange(0, len(vel)) / fps,
            "x_center (cm)": x_center_list[:-1],
            "adv (degree)": adv_list[:-1],
            "rec (degree)": rec_list[:-1],
            "contact_line_length (cm)": contact_line_length_list[:-1],
            "y_center (cm)": y_center_list[:-1],
            "middle_angle_degree (degree)": middle_angle_degree_list[:-1],
            "velocity (cm/s)": vel,
        })

        df_out.to_csv(os.path.join(shared_address, "result.csv"), index=False)
        # print(f"✅ Finished {len(results)} files for {shared_address}")
        return df_out

    def close(self):
        """Gracefully close and clean up the persistent pool."""
        if self._pool is not None:
            print("🧹 Closing persistent worker pool...")
            try:
                self._pool.close()
                self._pool.join()
            except Exception:
                self._pool.terminate()
            finally:
                self._pool = None
                self._worker_func = None
                print("✅ Pool closed successfully.")

def single(name_files: list[str]) -> Dict[str, Any]:
    """
    Assumption:
        To find the main address of the video it should go up in folder address till it reach the folder including the main video

    """
    global model, kernel, num_px_ratio, df, address, cm_on_pixel_ratio, fps


    fps                         = BaseUtils.config['fps_experiment']  # fps of the original experiment video
    cm_on_pixel_ratio           = 0.0039062
    num_px_ratio                = (0.0039062)/cm_on_pixel_ratio
    error_handling_kernel_size  = (5,5)
    model                       = model
    kernel                      = np.ones(error_handling_kernel_size,np. uint8)
    model                       = initiation()

    results:Dict[str, Any] = {}
    for index, image in enumerate(name_files):
        # climb parent folders until we find the folder that contains the databases folder
        curr = os.path.dirname(image)
        while True:
            if os.path.isfile(os.path.join(curr, 'video.mp4')):
                address = curr
                break
            curr = os.path.dirname(curr)
            if len(curr.split(os.sep)) < 2:
                raise FileNotFoundError("Could not find the main video folder containing 'video.mp4'.")

        df = pd.read_csv(os.path.join(address, BaseUtils.config['databases_folder'],'detections.csv')) # type: ignore

        result = process_one_file(index, name_files)
        results[image] = result
    return results


if __name__ == "__main__":
    
    # address= r"/media/Dont/Teflon-AVP/280/S3-SNr3.07_D/T105_06_79.813535314440"
    # processes(address)
    
    vv = single (['/media/Dont/Teflon-AVP/280/S2-SNr2.1_D/T528_03_4.460000000000/databases/frame_000001.png'])
    
    print(vv)