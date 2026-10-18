"""
    Author:         Yassin Riyazi
    Date:           01.07.2025

    Description: 
        Detects drops in video frames using frame differencing and morphological operations.
"""

import cv2 
from typing import Tuple

class DropPreProcessor:
    """
    Preprocesses grayscale video frames to detect moving objects (e.g., drops) via frame differencing.
    
    This class supports both CPU and CUDA-accelerated pipelines using OpenCV.
    """

    def __init__(self, kernel_size: tuple[int, int] = (5, 5), 
                 threshold_val: int = 30, 
                 use_cuda: bool = False) -> None:
        """
        Initializes the preprocessor with morphological filters and CUDA setup.

        Args:
            kernel_size (tuple[int, int]): Size of the structuring element for morphological operations.
            threshold_val (int): Threshold value used to binarized the frame difference.
            use_cuda (bool): Whether to use CUDA acceleration if available.
        """
        self.use_cuda       = use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.kernel         = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        self.threshold_val  = threshold_val

        if self.use_cuda:
            # Initialize CUDA-based morphological filters and GPU memory
            self.morph_open     = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, cv2.CV_8UC1, self.kernel)        # type: ignore
            self.morph_dilate   = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC1, self.kernel)      # type: ignore
            self.gpu_prev       = cv2.cuda_GpuMat() # type: ignore  
            self.gpu_curr       = cv2.cuda_GpuMat() # type: ignore
        else:
            # CUDA not used; filters set to None
            self.morph_open     = None
            self.morph_dilate   = None

    def process(self,
                prev_gray: cv2.Mat, 
                curr_gray: cv2.Mat) -> Tuple[int, int]:
        """
        Processes a pair of consecutive grayscale frames to extract contours of moving objects.

        Args:
            prev_gray (np.ndarray): Previous grayscale frame.
            curr_gray (np.ndarray): Current grayscale frame.

        Returns:
            list: List of detected contours representing motion between frames.
        """
        if self.use_cuda:
            # Upload frames to GPU memory
            self.gpu_prev.upload(prev_gray) # type: ignore
            self.gpu_curr.upload(curr_gray) # type: ignore

            # Compute absolute difference and threshold on GPU
            diff_gpu = cv2.cuda.absdiff(self.gpu_prev, self.gpu_curr) # type: ignore
            _, thresh_gpu = cv2.cuda.threshold(diff_gpu, self.threshold_val, 255, cv2.THRESH_BINARY) # type: ignore

            # Apply morphological opening and dilation to reduce noise and connect components
            opened_gpu = self.morph_open.apply(thresh_gpu) # type: ignore
            dilated_gpu = self.morph_dilate.apply(opened_gpu) # type: ignore

            # Download the result back to CPU for contour detection
            dilated = dilated_gpu.download()# type: ignore
        else:
            # CPU fallback: frame differencing and morphology
            diff        = cv2.absdiff(prev_gray, curr_gray)
            _, thresh   = cv2.threshold(diff, self.threshold_val, 255, cv2.THRESH_BINARY)
            opened      = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
            dilated     = cv2.dilate(opened, self.kernel, iterations=2)

        # Contour detection using CPU
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)

        return (x, x + w)
