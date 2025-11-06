import FrameExtractor
import os
import glob
import tqdm

"""
    Author: Yassin Riyazi
    Date: 01-07-2025
    Description: Detects drops in video frames using YOLO.
"""
import cv2
from ultralytics import YOLO
from send2trash import send2trash
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import TypeAlias, Tuple, List

YOLO_struct: TypeAlias = Tuple[int,int, list[str], int, float]


import re
import shutil


import Utilities
import subprocess

def safe_delete(file: str
                ) -> None:
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

def delInRange(_start: int, _end: int, _list_address: list[str], max_threads: int = 8
               ) -> None:
    """
    Deletes a range of files from a list using multithreading.

    Args:
        _start (int): Start index of the file range to delete.
        _end (int): End index (exclusive) of the file range to delete.
        _list_address (list): List of file paths.
        max_threads (int, optional): Maximum number of threads to use. Defaults to 8.

    Returns:
        None: None

    Raises:
        Exception: If any unexpected error occurs during file deletion.
    """
    files_to_delete = _list_address[_start:_end]
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        executor.map(safe_delete, files_to_delete)

def detect_and_filter_batch(index_range: YOLO_struct) -> None:
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
    start_idx, end_idx, frame_list, skip, yolo_conf = index_range

    # Load YOLO model once per process
    YOLO_path = os.path.join(os.path.dirname(__file__), "BaseUtils", "Detection", "Weights", "Gray-320-s.engine")
    model = YOLO(YOLO_path, task='detect', verbose=False)

    for i in range(start_idx, end_idx, skip):
        frame1 = cv2.imread(frame_list[i])
        frame2 = cv2.imread(frame_list[i + skip - 1])

        # Run YOLO detection on both frames
        result1 = model(frame1, conf=yolo_conf, device="cuda", verbose=False) # type: ignore
        result2 = model(frame2, conf=yolo_conf, device="cuda", verbose=False) # type: ignore

        has_drop1 = len(result1[0].boxes) > 0
        has_drop2 = len(result2[0].boxes) > 0

        # If neither frame has drops, delete the entire range
        if not has_drop1 and not has_drop2:
            delInRange(i, i + skip - 1, frame_list)

def Walker(image_folder: str,
           skip: int = 90,
           yolo_conf: float = 0.6,
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
    frame_list = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    if len(frame_list) == 0:
        frame_list = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    elif len(frame_list) == 0:
        raise ValueError(f"No image files found in {image_folder}")

    # Create a list of indices at intervals of `skip`
    total_indices = list(range(0, len(frame_list) - skip, skip))
    chunk_size = len(total_indices) // num_workers + 1

    # Prepare workload for each worker
    tasks: List[YOLO_struct] = []
    for w in range(num_workers):
        start = w * chunk_size
        end = min((w + 1) * chunk_size, len(total_indices))
        if start >= end:
            continue
        # Each task includes its start and end index and other parameters
        tasks.append((total_indices[start], total_indices[end - 1] + 1, frame_list, skip, yolo_conf))

    print(f"Distributing {len(total_indices)} frame pairs among {len(tasks)} processes...")

    # Run detection tasks in parallel using a process pool
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(detect_and_filter_batch, tasks), total=len(tasks)))

def MoveLeftFrames(FolderAddress: str, least_length=50, Name_="") -> None:
    """
    Moves frames from FolderAddress into subfolders inside destination_folder.
    Sequentially numbered frames are placed in the same subfolder,
    while non-sequential frames start a new subfolder.

    Args:
        FolderAddress (str): Path to the folder containing image frames.
        destination_folder (str): Path to the destination folder.

    Returns:
        None
    """
    if Name_ == "":
        Name_ = "Drop"
    # Find all frames (jpg or png)
    frame_list = sorted(glob.glob(os.path.join(FolderAddress, "*.jpg")))
    if len(frame_list) == 0:
        frame_list = sorted(glob.glob(os.path.join(FolderAddress, "*.png")))
    if len(frame_list) == 0:
        raise ValueError(f"No image files found in {FolderAddress}")

    # Extract frame numbers using regex
    def extract_index(filename):
        match = re.search(r'(\d+)', os.path.basename(filename))
        return int(match.group(1)) if match else -1

    # Sort by numeric index
    frame_list.sort(key=lambda x: extract_index(x))

    # Group frames by sequential indices
    groups = []
    current_group = [frame_list[0]]
    prev_idx = extract_index(frame_list[0])

    for frame_path in frame_list[1:]:
        idx = extract_index(frame_path)
        if idx == prev_idx + 1:
            current_group.append(frame_path)
        else:
            groups.append(current_group)
            current_group = [frame_path]
        prev_idx = idx
    
    groups.append(current_group)  # Add last group

    # Move each group into a separate folder
    index = 1
    for group in groups:
        if len(group) < least_length:
            continue  # Skip groups with less than 50 frames

        group_folder = os.path.join(FolderAddress, f"{Name_}_{index:02d}")
        os.makedirs(group_folder, exist_ok=True)
        for frame_path in group:
            shutil.move(frame_path, os.path.join(group_folder, os.path.basename(frame_path)))
        index += 1

if __name__ == "__main__":
    fe = FrameExtractor.FrameExtractor()

    video_files = sorted(glob.glob(r"/media/d25u2/Dont/Viscosity/*/*/*.mp4"))

    for video_path in tqdm(video_files):
        if not os.path.exists(video_path):
            continue

        print(f"Processing video: {video_path}")
        folder_path = os.path.splitext(video_path)[0]
        video_name = os.path.basename(folder_path)
        # folder_path = os.path.join(folder_path[0], video_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
       
        # Phase 1: Extracting frames from video
        fe.extract_frames(video_path, 30.0, os.path.join(folder_path, 'frame_%06d.png'))

        # Phase 2: Walking through frames and deleting non-drop frames
        Walker(folder_path,skip = 450)
        Walker(folder_path,skip = 10)
        Walker(folder_path,skip = 5)

        # Phase 3: Moving left frames into segmented folders
        MoveLeftFrames(folder_path, Name_=video_name)

        # Phase 4: Making video of each segmented folder
        segmented_folders = sorted(glob.glob(os.path.join(folder_path, f"{video_name}_*")))
        for segmented_folder in segmented_folders:
            saveVideoDirectory = os.path.normpath(os.path.join(os.path.dirname(folder_path),
                                                            #    '..',
                                                               f"{os.path.basename(segmented_folder)}.mp4"))
            
            Utilities.create_video_from_images(image_folder=segmented_folder,
                                              output_video_path=saveVideoDirectory,
                                              extension="png",
                                              fps=30)
            
            subprocess.run(["rm", "-rf", segmented_folder])
