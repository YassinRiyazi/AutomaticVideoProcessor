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

# Define the equivalent PyTorch model
class PyTorchModel(nn.Module):
    
    def __init__(self):
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
        super(PyTorchModel, self).__init__()
        self.conv1 = nn.Conv2d(1,   64,     kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(64,  64,     kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(64,  32,     kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(32,  9,      kernel_size=3, padding="same")
        self.pixel_shuffle = nn.PixelShuffle(3)
        self.relu = nn.ReLU()
    
    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pixel_shuffle(x)
        return x

class initiation():
    def __init__(self,_cuda = True):
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

    def forward(self,input_tensor):
        """
        Apply the super-resolution model to the input image.

        Parameters:
            input_tensor (np.ndarray): 2D NumPy array representing a grayscale image.

        Returns:
            np.ndarray: 2D uint8 NumPy array of the super-resolved image.

        Raises:
            ValueError: If input_tensor is not a 2D array.

        Notes:
            - The model expects a (1, H, W) input with values normalized to [0, 1].
            - Output is rescaled to [0, 255] and clipped.
        """
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add channel dimension
        data = torch.from_numpy(input_tensor).to(dtype=torch.float32)
        if self._cuda:
            data = data.cuda()
        with torch.inference_mode():
            out_img_y = self.sup_res_model(data)

        out_img_y = (out_img_y.detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")

        
        return out_img_y[0,0,:,:]

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

        self.sup_res_model.load_state_dict(torch.load(os.path.join(script_dir,'..','models','converted_model.pt'), weights_only=True))
        self.sup_res_model.eval()

# Upscale the image using the optimized model and OpenCV
def upscale_image(model, img, kernel):
    """
    Apply super-resolution and postprocessing to an input RGB image.

    Args:
        model: A model object with a `.forward()` method for Y-channel upscaling.
        img (np.ndarray): RGB input image.
        kernel (np.ndarray): Morphological kernel (e.g., cv2.getStructuringElement).

    Returns:
        np.ndarray: Grayscale post-processed image.

    Authors:
        - Yassin Riyazi (edited for clarity and structure)
        - Sajjad Shumaly
    """
    # Convert to YCrCb and split channels
    img_y_cr_cb     = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb       = cv2.split(img_y_cr_cb)

    # Normalize and expand Y channel
    y_norm          = cv2.normalize(y, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    input_tensor    = np.expand_dims(y_norm, axis=0)

    # Run model
    out_y           = model.forward(input_tensor)

    # Resize Cr/Cb to match upscaled Y
    h, w            = out_y.shape
    cr_up           = cv2.resize(cr, (w, h), interpolation=cv2.INTER_CUBIC)
    cb_up           = cv2.resize(cb, (w, h), interpolation=cv2.INTER_CUBIC)

    # Merge YCrCb and convert to RGB
    merged_ycrcb    = cv2.merge([out_y, cr_up, cb_up])
    rgb_upscaled    = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2RGB)
    gray            = cv2.cvtColor(rgb_upscaled, cv2.COLOR_RGB2GRAY)

    # Gaussian blur
    gray            = cv2.GaussianBlur(gray, (3, 3), 0)

    # Morphological close operation
    gray            = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    return gray
