"""
    Author: Yassin Riyazi
    Date: 01-07-2025
    Description: Detects drops in video frames using YOLO.
"""
import os
import cv2
import colorama
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import List, Tuple, TypeAlias
listImages: TypeAlias = List[str]

from concurrent.futures import ThreadPoolExecutor

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import BaseUtils

def safe_delete(file: str) -> None:
    """
    Safely deletes a file if it exists.

    Args:
        file (str): Path to the file to delete.

    Returns:
        None: None

    Raises:
        Exception: If any unexpected error occurs while deleting the file.
    """
    try:
        os.remove(file)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error deleting {file}: {e}")

def delInRange(_start: int, _end: int, _list_addresses: listImages, max_threads: int = 8) -> None:
    """
    Deletes a range of files from a list using multithreading.

    Args:
        _start (int): Start index of the file range to delete.
        _end (int): End index (exclusive) of the file range to delete.
        _list_addresses (list): List of file paths.
        max_threads (int, optional): Maximum number of threads to use. Defaults to 8.

    Returns:
        None: None

    Raises:
        Exception: If any unexpected error occurs during file deletion.
    """
    files_to_delete = _list_addresses[_start:_end]
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        executor.map(safe_delete, files_to_delete)
    
    return None

def detect_and_filter_batch(index_range: Tuple[int, int, listImages, int],
                            detector = BaseUtils.DropDetection_SUM_YOLO) -> None:
    """
    Worker function for a process that detects drops in a batch of frames using YOLO.
    Deletes all frames in the range if no drops are detected in the first and last frames.

    Args:
        index_range (tuple): Contains (start_idx, end_idx, frame_list, skip, yolo_conf)
            - start_idx (int): Start index for this worker
            - end_idx (int): End index (exclusive)
            - frame_list (list): List of all frame paths
            - skip (int): Step size (interval between frames)
            - yolo_conf (float): YOLO confidence threshold
    """
    assert not isinstance(detector, BaseUtils.DropDetection_SUM_YOLO), colorama.Fore.RED + "detector must not be an instance of DropDetection" + colorama.Style.RESET_ALL

    start_idx, end_idx, frame_list, skip = index_range
    in_detector = detector() if callable(detector) else detector

    for i in range(start_idx, end_idx, skip):
        # frame1 = cv2.imread(frame_list[i])
        # frame2 = cv2.imread(frame_list[i + skip - 1])
        frame1 = frame_list[i]
        frame2 = frame_list[i + skip - 1]

        # Run YOLO detection on both frames
        result1, has_drop1 = in_detector.detect_drops(frame1)
        result2, has_drop2 = in_detector.detect_drops(frame2)

        # If neither frame has drops, delete the entire range
        if not has_drop1 and not has_drop2:
            delInRange(i, i + skip - 1, frame_list)


def Walker(image_folder: str,
           skip: int = 90,
           num_workers: int = cpu_count() // 2,
           ) -> None:
    """
    Walk through all images in a folder in steps of `skip` frames.
    Uses multiprocessing to detect drops with YOLO and deletes frame ranges without drops.

    Args:
        image_folder (str): Path to the folder containing image frames.
        skip (int, optional): Frame step size. Defaults to 90.
        yolo_conf (float, optional): YOLO confidence threshold. Defaults to 0.6.
        num_workers (int, optional): Number of parallel processes. Defaults to half of CPU cores.
        
    Returns:
        None: None

    Example:
        >>> Walker("extracted_frames", skip=30, yolo_conf=0.5)
    """
    frame_list = BaseUtils.ImageLister(FolderAddress=image_folder,
                                       frameAddress=str(BaseUtils.config["rotated_frames_folder"]),)

    # Create a list of indices at intervals of `skip`
    total_indices   = list(range(0, len(frame_list) - skip, skip))
    chunk_size      = len(total_indices) // num_workers + 1

    # Prepare workload for each worker
    tasks:List[Tuple[int, int, listImages, int, float]] = []
    for w in range(num_workers):
        start = w * chunk_size
        end = min((w + 1) * chunk_size, len(total_indices))
        if start >= end:
            continue
        # Each task includes its start and end index and other parameters
        tasks.append((total_indices[start], total_indices[end - 1] + 1, frame_list, skip))

    print(f"Distributing {len(total_indices)} frame pairs among {len(tasks)} processes...")

    # Run detection tasks in parallel using a process pool
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(detect_and_filter_batch, tasks), total=len(tasks)))

if __name__ == "__main__":
    image_folder = r"D:\Videos\S1_30per_T1_C001H001S0001"

    Walker(image_folder,skip = 450,
           )
    Walker(image_folder,skip = 10)


