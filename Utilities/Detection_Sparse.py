# """
#     Author: Yassin Riyazi
#     Date: 01-07-2025
#     Description: Detects drops in video frames using YOLO.

#     Version: 2.0.0
#     License: GNU General Public License v3.0

#     Changes:
#     - 06-10-2025: Refactored to use multiprocessing with a persistent pool.
#     - 01-07-2025: Initial version.
# """
# import  os
# import  colorama
# import  BaseUtils
# import  multiprocessing     as      mp
# from    multiprocessing     import  Pool, cpu_count
# from    tqdm                import  tqdm
# from    typing              import  List, Tuple, TypeAlias, Optional
# from    concurrent.futures  import  ThreadPoolExecutor

# listImages: TypeAlias = List[str]
# WalkerData: TypeAlias = Tuple[int, int, listImages, int]


# def safe_delete(file: str) -> None:
#     """
#     Safely deletes a file if it exists.

#     Args:
#         file (str): Path to the file to delete.

#     Returns:
#         None: None

#     Raises:
#         Exception: If any unexpected error occurs while deleting the file.
#     """
#     try:
#         os.remove(file)
#     except FileNotFoundError:
#         pass
#     except Exception as e:
#         print(f"Error deleting {file}: {e}")

# def delInRange(_start: int, _end: int, _list_addresses: listImages, max_threads: int = 8) -> None:
#     """
#     Deletes a range of files from a list using multithreading.

#     Args:
#         _start (int): Start index of the file range to delete.
#         _end (int): End index (exclusive) of the file range to delete.
#         _list_addresses (list): List of file paths.
#         max_threads (int, optional): Maximum number of threads to use. Defaults to 8.

#     Returns:
#         None: None

#     Raises:
#         Exception: If any unexpected error occurs during file deletion.
#     """
#     files_to_delete = _list_addresses[_start:_end]
#     with ThreadPoolExecutor(max_workers=max_threads) as executor:
#         executor.map(safe_delete, files_to_delete)
    
#     return None

# def detect_and_filter_NGlobal(index_range: WalkerData,
#                             detector = BaseUtils.DropDetection_SUM_YOLO) -> None:
#     """
#     Worker function for a process that detects drops in a batch of frames using YOLO.
#     Deletes all frames in the range if no drops are detected in the first and last frames.

#     Args:
#         index_range (tuple): Contains (start_idx, end_idx, frame_list, skip, yolo_conf)
#             - start_idx (int): Start index for this worker
#             - end_idx (int): End index (exclusive)
#             - frame_list (list): List of all frame paths
#             - skip (int): Step size (interval between frames)
#             - yolo_conf (float): YOLO confidence threshold
#     """
#     assert not isinstance(detector, BaseUtils.DropDetection_SUM_YOLO), colorama.Fore.RED + "detector must not be an instance of DropDetection" + colorama.Style.RESET_ALL

#     start_idx, end_idx, frame_list, skip = index_range
#     in_detector = detector() if callable(detector) else detector

#     for i in range(start_idx, end_idx, skip):
#         # frame1 = cv2.imread(frame_list[i])
#         # frame2 = cv2.imread(frame_list[i + skip - 1])
#         frame1 = frame_list[i]
#         frame2 = frame_list[i + skip - 1]

#         # Run YOLO detection on both frames
#         result1, has_drop1 = in_detector.detect_drops(frame1)
#         result2, has_drop2 = in_detector.detect_drops(frame2)
#         del result1, result2

#         # If neither frame has drops, delete the entire range
#         if not has_drop1 and not has_drop2:
#             delInRange(i, i + skip - 1, frame_list)

# def detect_and_filter_batch(index_range: WalkerData,
#                             detector = BaseUtils.DropDetection_SUM_YOLO) -> None:
#     """
#     Worker function for a process that detects drops in a batch of frames using YOLO.
#     Deletes all frames in the range if no drops are detected in the first and last frames.

#     Args:
#         index_range (tuple): Contains (start_idx, end_idx, frame_list, skip, yolo_conf)
#             - start_idx (int): Start index for this worker
#             - end_idx (int): End index (exclusive)
#             - frame_list (list): List of all frame paths
#             - skip (int): Step size (interval between frames)
#             - yolo_conf (float): YOLO confidence threshold
#     """
#     global in_detector
#     assert not isinstance(detector, BaseUtils.DropDetection_SUM_YOLO), colorama.Fore.RED + "detector must not be an instance of DropDetection" + colorama.Style.RESET_ALL

#     start_idx, end_idx, frame_list, skip = index_range
#     in_detector = detector() if callable(detector) else detector

#     for i in range(start_idx, end_idx, skip):
#         # frame1 = cv2.imread(frame_list[i])
#         # frame2 = cv2.imread(frame_list[i + skip - 1])
#         frame1 = frame_list[i]
#         frame2 = frame_list[i + skip - 1]

#         # Run YOLO detection on both frames
#         result1, has_drop1 = in_detector.detect_drops(frame1)
#         result2, has_drop2 = in_detector.detect_drops(frame2)
#         del result1, result2

#         # If neither frame has drops, delete the entire range
#         if not has_drop1 and not has_drop2:
#             delInRange(i, i + skip - 1, frame_list)

# def Walker(image_folder: str,
#            skip: int = 90,
#            num_workers: int = cpu_count() // 2,
#            ) -> None:
#     """
#     Walk through all images in a folder in steps of `skip` frames.
#     Uses multiprocessing to detect drops with YOLO and deletes frame ranges without drops.

#     Args:
#         image_folder (str): Path to the folder containing image frames.
#         skip (int, optional): Frame step size. Defaults to 90.
#         yolo_conf (float, optional): YOLO confidence threshold. Defaults to 0.6.
#         num_workers (int, optional): Number of parallel processes. Defaults to half of CPU cores.
        
#     Returns:
#         None: None

#     Example:
#         >>> Walker("extracted_frames", skip=30, yolo_conf=0.5)
#     """
#     frame_list = BaseUtils.ImageLister(FolderAddress=image_folder,
#                                        frameAddress=str(BaseUtils.config["rotated_frames_folder"]),)

#     # Create a list of indices at intervals of `skip`
#     total_indices   = list(range(0, len(frame_list) - skip, skip))
#     chunk_size      = len(total_indices) // num_workers + 1

#     # Prepare workload for each worker
#     tasks:List[WalkerData] = []
#     for w in range(num_workers):
#         start = w * chunk_size
#         end = min((w + 1) * chunk_size, len(total_indices))
#         if start >= end:
#             continue
#         # Each task includes its start and end index and other parameters
#         tasks.append((total_indices[start], total_indices[end - 1] + 1, frame_list, skip))

#     print(f"Distributing {len(total_indices)} frame pairs among {len(tasks)} processes...")

#     # Run detection tasks in parallel using a process pool
#     with Pool(processes=num_workers) as pool:
#         list(tqdm(pool.imap_unordered(detect_and_filter_batch, tasks), total=len(tasks)))



# # global in each worker process (set by initializer)
# in_detector = None        # per-worker global (set by initializer)
# _worker_idx = None        # optional index in each worker (not used by default)

# def _init_detector(detector_factory_or_instance, worker_index: int = 0, use_gpu_cleanup: bool = False):
#     """
#     Pool initializer. Called once per worker process.

#     Parameters:
#       detector_factory_or_instance: callable (class/factory) OR picklable instance.
#         - If callable: this will be called once per worker to create the detector instance.
#         - If instance: it will be unpickled into the worker process (less ideal for big models).
#       worker_index: optional integer, useful for per-worker GPU assignment.
#       use_gpu_cleanup: if True and detector uses torch, we will call torch.cuda.empty_cache() periodically.
#     """
#     global in_detector, _worker_idx, _use_gpu_cleanup
#     _worker_idx = worker_index
#     _use_gpu_cleanup = use_gpu_cleanup

#     # instantiate once per worker
#     try:
#         in_detector = detector_factory_or_instance() if callable(detector_factory_or_instance) else detector_factory_or_instance
#         # If detector uses PyTorch, ensure eval and no_grad if not already done by the detector
#         # e.g., if in_detector has .model (torch.nn.Module), you can call:
#         #    in_detector.model.eval()
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         raise RuntimeError(f"_init_detector failed in worker {worker_index}: {e}") from e


# class Walker_mp:
#     def __init__(self, num_workers: Optional[int] = None, use_gpu_cleanup: bool = False):
#         self.num_workers = num_workers or max(1, cpu_count() // 2)
#         self._pool: Optional[Pool] = None
#         self._detector_arg = None
#         self._use_gpu_cleanup = use_gpu_cleanup

#     def get_pool(self, detector, recreate_if_dead: bool = True):
#         """
#         Create or reuse a persistent Pool initialized with the given detector (callable or instance).
#         detector should preferably be a class or factory (callable). If you pass an instance,
#         it will be pickled/unpickled into each worker (memory duplication).
#         """
#         self._detector_arg = detector

#         # If pool exists but seems dead, close and recreate
#         if self._pool is not None:
#             try:
#                 if self._pool._state != mp.pool.RUN:
#                     self.close()
#             except Exception:
#                 self.close()

#         if self._pool is None:
#             print("Walker_mp: creating new worker pool...")
#             # build initargs: pass detector and a worker index (we pass only detector here; worker index can't be passed per process easily,
#             # but you can set environment variables per worker if needed by using os.environ before pool creation or by constructing detector to inspect PID)
#             # We'll pass detector and use_gpu_cleanup flag
#             # NOTE: initargs is the same for all workers; the detector factory should internally handle per-worker GPU assignment if needed.
#             self._pool = Pool(
#                 processes=self.num_workers,
#                 initializer=_init_detector,
#                 initargs=(detector, 0, self._use_gpu_cleanup)
#             )
#         else:
#             print("Walker_mp: reusing existing worker pool...")

#         return self._pool

#     def Walker(self, image_folder: str, skip: int = 90, detector = BaseUtils.DropDetection_SUM_YOLO, num_workers: Optional[int] = None):
#         """
#         Partition frames into tasks and run detect_and_filter_batch using a persistent pool.
#         detector should be a callable (class/factory) so each worker makes its own instance once.
#         """

#         if num_workers is not None:
#             # allow temporarily overriding workers for this run
#             self.num_workers = num_workers

#         frame_list = BaseUtils.ImageLister(FolderAddress=image_folder,
#                                            frameAddress=str(BaseUtils.config["rotated_frames_folder"]))

#         total_indices = list(range(0, len(frame_list) - skip, skip))
#         if len(total_indices) == 0:
#             print("Walker_mp: no frames to process (len(frame_list) < skip).")
#             return

#         chunk_size = len(total_indices) // self.num_workers + 1

#         tasks: List[WalkerData] = []
#         for w in range(self.num_workers):
#             start = w * chunk_size
#             end = min((w + 1) * chunk_size, len(total_indices))
#             if start >= end:
#                 continue
#             tasks.append((total_indices[start], total_indices[end - 1] + 1, frame_list, skip))

#         print(f"Distributing {len(total_indices)} frame pairs among {len(tasks)} processes...")

#         # ensure pool exists and get it
#         pool = self.get_pool(detector)

#         # run tasks and wait for completion
#         try:
#             list(tqdm(pool.imap_unordered(detect_and_filter_batch, tasks), total=len(tasks)))
#         except Exception:
#             import traceback
#             traceback.print_exc()
#             # on error, close to avoid future hung pool
#             self.close()
#             raise

#     def close(self):
#         """Gracefully terminate the pool (when you are done)."""
#         if self._pool is not None:
#             print("Walker_mp: closing persistent pool...")
#             try:
#                 self._pool.close()
#                 self._pool.join()
#             except Exception:
#                 try:
#                     self._pool.terminate()
#                 except Exception:
#                     pass
#             finally:
#                 self._pool = None
#                 self._detector_arg = None

        


# if __name__ == "__main__":
#     image_folder = r"D:\Videos\S1_30per_T1_C001H001S0001"

#     Walker(image_folder,skip = 450,
#            )
#     Walker(image_folder,skip = 10)


"""
    Author: Yassin Riyazi
    Date: 01-07-2025
    Description: Detects drops in video frames using YOLO with a persistent multiprocessing pool.

    Version: 3.0.0
    License: GNU General Public License v3.0

    Changes:
    - 06-10-2025: Refactored into a YoloWalker class to manage a persistent worker pool,
                  ensuring the YOLO model is initialized only once per worker process.
    - 06-10-2025: Refactored to use multiprocessing with a persistent pool.
    - 01-07-2025: Initial version.
"""
import  os
import  colorama
import  BaseUtils
import  multiprocessing     as      mp
from    multiprocessing     import  Pool, cpu_count
from    tqdm                import  tqdm
from    typing              import  List, Tuple, TypeAlias, Optional, Callable
from    concurrent.futures  import  ThreadPoolExecutor

# --- Type Aliases for Clarity ---
ListImages: TypeAlias = List[str]
# (start_index, end_index, list_of_all_frames, skip_interval)
TaskData: TypeAlias = Tuple[int, int, ListImages, int]

# --- Global variable for each worker process ---
# This holds the detector instance, initialized once per process.
worker_detector_instance = None

# --- Helper Functions ---

def safe_delete(file: str) -> None:
    """
    Safely deletes a file if it exists.
    """
    try:
        os.remove(file)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error deleting {file}: {e}")

def del_in_range(start_idx: int, end_idx: int, list_addresses: ListImages, max_threads: int = 8) -> None:
    """
    Deletes a range of files from a list using multithreading for I/O efficiency.
    """
    files_to_delete = list_addresses[start_idx:end_idx]
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        executor.map(safe_delete, files_to_delete)

# --- Multiprocessing Worker Functions ---

def _init_worker(detector_factory: Callable) -> None:
    """
    Initializer for each worker in the pool.
    This function is called once per worker process when the pool is created.
    """
    global worker_detector_instance
    print(f"{colorama.Fore.GREEN}Initializing detector in worker process: {os.getpid()}...{colorama.Style.RESET_ALL}")
    if callable(detector_factory):
        # Create an instance of the detector for this worker
        worker_detector_instance = detector_factory()
    else:
        # If a pre-initialized instance was passed (not recommended for large models)
        worker_detector_instance = detector_factory

def _process_batch(task_data: TaskData) -> None:
    """
    The target function for each worker process. It processes a chunk of frames.
    It relies on the 'worker_detector_instance' global variable initialized by _init_worker.
    """
    global worker_detector_instance
    if worker_detector_instance is None:
        raise RuntimeError("Detector was not initialized in the worker process. This should not happen.")

    start_idx, end_idx, frame_list, skip = task_data

    # The loop processes pairs of frames separated by 'skip'
    for i in range(start_idx, end_idx, skip):
        # Ensure we don't go out of bounds
        if i + skip > len(frame_list):
            continue
            
        frame_path1 = frame_list[i]
        frame_path2 = frame_list[i + skip - 1]

        # Run YOLO detection on the first and last frame of the chunk
        _, has_drop1 = worker_detector_instance.detect_drops(frame_path1)
        _, has_drop2 = worker_detector_instance.detect_drops(frame_path2)

        # If neither frame has drops, delete the entire range of frames
        if not has_drop1 and not has_drop2:
            del_in_range(i, i + skip, frame_list) # Corrected to delete the whole chunk


# --- Main Class for Managing the Process ---

class YoloWalker:
    """
    Manages a persistent multiprocessing pool to efficiently process video frames with YOLO.

    By creating an instance of this class, you can call the `run` method multiple times
    without the overhead of creating new processes and re-initializing the YOLO model each time.
    """
    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers or max(1, cpu_count() // 2)
        self._pool: Optional[Pool] = None
        self._detector_factory: Optional[Callable] = None
        print(f"YoloWalker initialized to use up to {self.num_workers} worker processes.")

    def _get_or_create_pool(self, detector_factory: Callable) -> Pool:
        """
        Creates a new pool if one doesn't exist or if the detector has changed.
        Otherwise, it returns the existing, persistent pool.
        """
        # If the pool exists but the detector model is different, we must recreate it.
        if self._pool and self._detector_factory != detector_factory:
            print(f"{colorama.Fore.YELLOW}Detector has changed. Recreating worker pool...{colorama.Style.RESET_ALL}")
            self.close()

        if self._pool is None:
            print(f"Creating new worker pool with {self.num_workers} processes...")
            self._detector_factory = detector_factory
            # The initializer runs '_init_worker' in each new process, passing 'detector_factory' to it.
            self._pool = Pool(
                processes=self.num_workers,
                initializer=_init_worker,
                initargs=(self._detector_factory,)
            )
        return self._pool

    def run(self, image_folder: str, skip: int = 90, detector_factory: Callable = BaseUtils.DropDetection_SUM_YOLO) -> None:
        """
        Walks through image frames, distributing the detection work across the managed pool.

        Args:
            image_folder (str): Path to the folder containing image frames.
            skip (int): The number of frames in each chunk to check.
            detector_factory (Callable): A class or function that returns a detector instance (e.g., YOLO model).
        """
        pool = self._get_or_create_pool(detector_factory)

        frame_list = BaseUtils.ImageLister(
            FolderAddress=image_folder,
            frameAddress=str(BaseUtils.config["rotated_frames_folder"]),
        )
        
        if len(frame_list) < skip:
            print(f"{colorama.Fore.YELLOW}Warning: Not enough frames ({len(frame_list)}) to process with a skip of {skip}.{colorama.Style.RESET_ALL}")
            return

        # Create a list of start indices for each chunk
        total_indices = list(range(0, len(frame_list), skip))
        chunk_size = (len(total_indices) + self.num_workers - 1) // self.num_workers # More robust chunking

        # Prepare workload for each worker
        tasks: List[TaskData] = []
        for i in range(0, len(total_indices), chunk_size):
            batch_indices = total_indices[i:i + chunk_size]
            if not batch_indices:
                continue
            
            start_idx = batch_indices[0]
            # The end index is the start of the last chunk in this batch + skip
            end_idx = batch_indices[-1] + skip
            tasks.append((start_idx, end_idx, frame_list, skip))

        print(f"Distributing {len(frame_list)} frames into {len(total_indices)} chunks across {len(tasks)} tasks...")

        # Execute the tasks in parallel and show progress
        # The list() call ensures we wait for all tasks to complete.
        list(tqdm(pool.imap_unordered(_process_batch, tasks), total=len(tasks), desc="Processing Frames"))
        print(f"{colorama.Fore.CYAN}Processing for '{image_folder}' complete.{colorama.Style.RESET_ALL}")


    def close(self) -> None:
        """
        Gracefully shuts down the worker pool. Call this when you're finished.
        """
        if self._pool:
            print("Closing persistent worker pool...")
            self._pool.close()
            self._pool.join()
            self._pool = None
            self._detector_factory = None


# --- Example Usage ---

if __name__ == "__main__":
    # Best practice: Put your execution logic inside the main guard.
    image_folder = r"D:\Videos\S1_30per_T1_C001H001S0001"
    
    # 1. Initialize the manager once
    walker_manager = YoloWalker(num_workers=4)

    try:
        # 2. Run the processing loop as many times as you need
        # The YOLO model will be loaded only on the first call.
        print("\n--- First Run (High Skip) ---")
        walker_manager.run(image_folder, skip=450)

        print("\n--- Second Run (Low Skip) ---")
        # The pool and initialized models are reused, making this call much faster to start.
        walker_manager.run(image_folder, skip=10)

    except Exception as e:
        print(f"{colorama.Fore.RED}An error occurred: {e}{colorama.Style.RESET_ALL}")
    finally:
        # 3. Clean up the worker processes when you are completely done.
        walker_manager.close()