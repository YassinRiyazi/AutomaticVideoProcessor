"""
    Edited by: Yassin Riyazi
    Main Author: Sajjad Shumaly
    Date: 01-07-2025
    Description: This script implements a PyTorch model for single-channel image super-resolution.

    Changelog:
        - Converted the Tensorflow model to PyTorch format.
"""
import os
import cv2
import torch
import numpy    as np
import torch.nn as nn
from numpy.typing import NDArray
# from typing import Tuple 

# Define the equivalent PyTorch model
class PyTorchModel(nn.Module):

    def __init__(self) -> None:
        """
        A PyTorch convolutional neural network for single-channel image super-resolution.

        Architecture:
            - Conv2d(1 → 64) with kernel size 5
            - Conv2d(64 → 64) with kernel size 3
            - Conv2d(64 → 32) with kernel size 3
            - Conv2d(32 → 9) with kernel size 3
            - PixelShuffle with upscale factor 3

        Activation:
            - ReLU is used after each convolution.

        Output:
            - A super-resolved image with spatial resolution increased by a factor of 3.

        Notes:
            - The final convolution outputs 9 channels, which are reshaped via PixelShuffle (3× upscaling).
        """
        super(PyTorchModel, self).__init__() # type: ignore
        self.conv1 = nn.Conv2d(1,   64,     kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(64,  64,     kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(64,  32,     kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(32,  9,      kernel_size=3, padding="same")
        self.pixel_shuffle = nn.PixelShuffle(3)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pixel_shuffle(x)
        return x

class initiation():
    def __init__(self,_cuda: bool = True):
        """
        Wrapper class for initializing and using a pre-trained PyTorch super-resolution model.

        This class loads a pre-trained model from disk and provides a `forward` method to apply
        super-resolution to a single-channel input image represented as a NumPy array.

        Parameters:
            _cuda (bool): Whether to run inference on GPU (default: True).

        Attributes:
            sup_res_model (torch.nn.Module): The loaded and ready-to-use super-resolution model.

        Methods:
            forward(input_tensor): Apply the model to a given input image.
        
        Example:
            >>> model = initiation()
            >>> output = model.forward(input_array)

        Notes:
            - The input tensor must be a 2D NumPy array (grayscale image).
            - Output is a 2D uint8 NumPy array representing the upscaled image.
        """

        self._cuda = _cuda
        self.initiate_torch()

    def forward(self, input_tensor: NDArray[np.float32]) -> NDArray[np.uint8]:
        """
        Apply the super-resolution model to the input image(s).

        Parameters:
            input_tensor (np.ndarray): 2D or 3D NumPy array. If 2D (H, W), single image. If 3D (N, H, W), batch of N images.

        Returns:
            np.ndarray: 2D or 3D uint8 array of the super-resolved image(s).

        Raises:
            ValueError: If input_tensor is not 2D or 3D.

        Notes:
            - The model expects input with values normalized to [0, 1].
            - Output is rescaled to [0, 255] and clipped.
        """

        
        data = torch.from_numpy(input_tensor).to(dtype=torch.float32)
        if self._cuda:
            data = data.cuda()
        with torch.inference_mode():
            out_img_y = self.sup_res_model(data)

        out_img_y = (out_img_y.detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        
        return out_img_y

    def initiate_torch(self,):
        """
        Initialize the PyTorch model and load pre-trained weights from disk.

        Notes:
            - The model weights are loaded from '../models/converted_model.pt'
            - The model is set to evaluation mode to disable dropout/batchnorm updates.
        """
        self.sup_res_model = PyTorchModel()
        if self._cuda:
            self.sup_res_model = self.sup_res_model.cuda()
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.sup_res_model.load_state_dict(torch.load(os.path.join(script_dir,'models','converted_model.pt'), weights_only=True))
        self.sup_res_model.eval()

# Upscale the image using the optimized model and OpenCV
def upscale_image(model: torch.nn.Module, 
                  img: NDArray[np.uint8] | list[NDArray[np.uint8]], 
                  kernel: NDArray[np.uint8],
                  output_paths: list[str] | None = None
                  ) -> NDArray[np.uint8] | list[NDArray[np.uint8]] | None:
    """
    Apply super-resolution and postprocessing to input RGB image(s), and optionally save.

    Args:
        model: A model object with a `.forward()` method for Y-channel upscaling.
        img (np.ndarray or list of np.ndarray): RGB input image(s). If list, batch process.
        kernel (np.ndarray): Morphological kernel (e.g., cv2.getStructuringElement).
        output_paths (list of str, optional): Paths to save the upscaled images. If provided, saves and returns None.

    Returns:
        np.ndarray or list of np.ndarray or None: Grayscale post-processed image(s), or None if saved.

    Authors:
        - Yassin Riyazi (edited for clarity and structure)
        - Sajjad Shumaly
    """
    if not isinstance(img, list):
        imgs = [img]
        single = True
    else:
        imgs = img
        single = False
    
    if output_paths is not None and len(output_paths) != len(imgs):
        raise ValueError("output_paths must have the same length as imgs")
    
    # Prepare Y channels
    y_norms = []
    crs:list[NDArray[np.uint8]] = []
    cbs:list[NDArray[np.uint8]] = []
    for im in imgs:
        # Convert to YCrCb and split channels
        # im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        img_y_cr_cb = cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(img_y_cr_cb)

        # Normalize Y channel
        y_norm = np.empty_like(y, dtype=np.float32)
        cv2.normalize(src=y, dst=y_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        y_norms.append(y_norm)
        crs.append(cr)
        cbs.append(cb)
    
    # Batch process Y channels
    batch_y = np.stack(y_norms, axis=0)  # (N, H, W)
    out_ys = model.forward(batch_y)  # (N, H', W')

    
    results: list[NDArray[np.uint8]] = []
    for i, out_y in enumerate(out_ys):
        # Resize Cr/Cb to match upscaled Y
        h, w = out_y.shape
        cr_up = cv2.resize(crs[i], (w, h), interpolation=cv2.INTER_CUBIC)
        cb_up = cv2.resize(cbs[i], (w, h), interpolation=cv2.INTER_CUBIC)

        # Merge YCrCb and convert to RGB
        merged_ycrcb = cv2.merge([out_y, cr_up, cb_up])
        rgb_upscaled = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2RGB)
        gray = cv2.cvtColor(rgb_upscaled, cv2.COLOR_RGB2GRAY)

        # Gaussian blur
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Morphological close operation
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        gray = gray.astype(np.uint8)

        if output_paths is not None:
            cv2.imwrite(output_paths[i], gray)
        else:
            results.append(gray)
    
    if output_paths is not None:
        return None
    else:
        return results

import glob
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Lock
import time, tqdm

def process_folder_parallel(input_folder: str, output_folder: str, num_models: int = 12) -> None:
    """
    Process images in parallel using multiple GPU models.
    
    Args:
        input_folder: Path to folder containing input images
        output_folder: Path to folder for output images
        num_models: Number of parallel models to run (default 12)
    """
    

    # Initialize models
    models = [initiation(_cuda=True) for _ in range(num_models)]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of all images
    image_files = glob.glob(os.path.join(input_folder, "*.png")) + \
                 glob.glob(os.path.join(input_folder, "*.jpg"))
    
    # Create thread-safe queue and lock
    image_queue = Queue()
    print_lock = Lock()
    
    def worker(model_id: int) -> None:
        model = models[model_id]
        while True:
            try:
                img_path = image_queue.get_nowait()
            except:
                break
                
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    with print_lock:
                        print(f"Failed to read {img_path}")
                    continue
                
                # Create output path
                rel_path = os.path.relpath(img_path, input_folder)
                out_path = os.path.join(output_folder, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                
                # Process image
                upscale_image(model, img, kernel, output_paths=[out_path])
                
                with print_lock:
                    # print(f"Model {model_id} processed {rel_path}")
                    pass
            
            except Exception as e:
                with print_lock:
                    print(f"Error processing {img_path}: {str(e)}")
            
            finally:
                image_queue.task_done()
    
    # Fill queue with image paths
    for img_path in image_files:
        image_queue.put(img_path)
    
    print(f"Processing {len(image_files)} images with {num_models} parallel models...")
    start_time = time.time()
    
    # Start worker threads
    with ThreadPoolExecutor(max_workers=num_models) as executor:
        for i in range(num_models):
            executor.submit(worker, i)
    
    # Wait for completion
    image_queue.join()
    
    print(f"Finished processing in {time.time() - start_time:.2f} seconds")

def process_folders_parallel(input_folders: list[str],
                             output_folders: list[str],
                             num_models: int = 12,
                             verbose: bool = False) -> None:
    """
    Process multiple folders of images in parallel using multiple GPU models.
    Models are kept alive between folders to maximize efficiency.
    
    Args:
        input_folders: List of paths to folders containing input images
        output_folders: List of paths to folders for output images
        num_models: Number of parallel models to run (default 12)
    """
    if len(input_folders) != len(output_folders):
        raise ValueError("input_folders and output_folders must have same length")

    # Initialize models once and keep them alive
    print("Initializing GPU models...")
    models = [initiation(_cuda=True) for _ in range(num_models)]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Process each folder pair
    total_start_time = time.time()
    total_images = 0
    
    for folder_idx, (in_folder, out_folder) in tqdm.tqdm(enumerate(zip(input_folders, output_folders)), total=len(input_folders), ):
        if verbose:
            print(f"Found {len(input_folders)} folders to process:")
            for i, folder in enumerate(input_folders):
                print(f"{i+1}. {folder} -> {output_folders[i]}")
            print()

        # Create output directory
        os.makedirs(out_folder, exist_ok=True)
        
        # Get list of all images in this folder
        image_files = glob.glob(os.path.join(in_folder, "*.png")) + \
                     glob.glob(os.path.join(in_folder, "*.jpg"))
        
        if not image_files:
            print(f"No images found in {in_folder}")
            continue
            
        # Create thread-safe queue and lock for this folder
        image_queue: Queue[str] = Queue()
        print_lock = Lock()
        folder_start_time = time.time()
        
        def worker(model_id: int) -> None:
            model = models[model_id]
            while True:
                try:
                    img_path = image_queue.get_nowait()
                except:
                    break
                    
                try:
                    # Read image
                    img = cv2.imread(img_path)
                    if img is None:
                        with print_lock:
                            print(f"Failed to read {img_path}")
                        continue
                    
                    # Create output path
                    rel_path = os.path.relpath(img_path, in_folder)
                    out_path = os.path.join(out_folder, rel_path)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    
                    # Process image
                    upscale_image(model, img, kernel, output_paths=[out_path])
                    
                    with print_lock:
                        # print(f"Model {model_id} processed {rel_path}")
                        pass
                
                except Exception as e:
                    with print_lock:
                        print(f"Error processing {img_path}: {str(e)}")
                
                finally:
                    image_queue.task_done()
        
        # Fill queue with image paths for this folder
        for img_path in image_files:
            image_queue.put(img_path)
        
        folder_image_count = len(image_files)
        total_images += folder_image_count
        
        # Process this folder with worker threads
        with ThreadPoolExecutor(max_workers=num_models) as executor:
            for i in range(num_models):
                executor.submit(worker, i)
        
        # Wait for folder completion
        image_queue.join()
        
        folder_time = time.time() - folder_start_time
        if verbose:
            print(f"\nProcessing folder {folder_idx + 1}/{len(input_folders)}: {in_folder}")
            print(f"Found {folder_image_count} images to process...")
            print(f"Folder completed in {folder_time:.2f} seconds ({folder_image_count / folder_time:.1f} images/sec)")
    
    total_time = time.time() - total_start_time
    print(f"\nAll folders completed!")
    print(f"Total images processed: {total_images}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Overall speed: {total_images / total_time:.1f} images/sec")

if __name__ == "__main__":
    # Get all database folders recursively
    input_folders = sorted(glob.glob("/media/d25u2/Dont/Viscosity/*/*/*/databases"))
    input_folders = [folder for folder in input_folders if os.path.isdir(folder)]
    output_folders = [i.replace("databases", "databases_SR") for i in input_folders]

    

    process_folders_parallel(input_folders, output_folders, num_models=12)
    