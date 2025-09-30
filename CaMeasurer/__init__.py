from .preprocessing import *
from .imageloader import *
from .superResolution import *
from .edgeDetection import *
from .processing import *
from .visualization import *
from .criteria_definition import *
from .main import *
import os
from typing import TextIO, Union

def logFile(ad: Union[str, os.PathLike]): # -> TextIO
    log_path = os.path.join(ad, "error_log.txt")
    if os.path.isfile(log_path):
        os.remove(log_path)
    log_file = open(log_path, "a")
    log_file.write("#Initiate\n")
    # return log_file

def get_subdirectories(root_dir, max_depth=2):
    directories = []
    for root, dirs, _ in sorted(os.walk(root_dir)):
        if root == root_dir:
            continue  # Skip the root directory itself
        depth = root[len(root_dir):].count(os.sep)
        if depth < max_depth:
            directories.append(root)
        else:
            del dirs[:]  # Stop descending further
    return directories



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

def writter(_adress):
    with open(_adress, 'w') as output:
        pass