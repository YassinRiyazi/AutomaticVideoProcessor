"""
    Date    : 2025-09-30
    Author  : Yassin Riyazi
    Project : Automatic Video Processor (AVP)
    File    : BaseLine/__init__.py
    Version : 1.0.0
    License : GNU GENERAL PUBLIC LICENSE Version 3

"""

import  os
import  cv2
import  glob
import  matplotlib
import  multiprocessing
 
import  numpy               as      np
from multiprocessing        import  pool
matplotlib.use('Agg')  # For file output only, no GUI
import  matplotlib.pyplot as plt
from    skimage.measure     import  ransac, LineModelND

from numpy.typing import ArrayLike,NDArray
from typing import Tuple, TypeAlias
ImageSize: TypeAlias = Tuple[int, int]

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import BaseUtils

def fit_and_rotate_image(image_path: os.PathLike[str],
                         results: bool = True,
                         focus_ratio: float = 0.3
                         ) -> tuple[float, ImageSize, NDArray[np.uint8]]:
    """
    Fits a robust line to the bottom edges of an image and rotates the image to level the surface.
    
    Args:
        image_path (os.PathLike): Path to the input image.
        results (bool, optional): If True, saves a diagnostic plot.
        focus_ratio (float, optional): Portion of the image height to analyze from the bottom. Default 0.3.
    
    Returns:
        tuple:
            - angle (float): Rotation angle in degrees.
            - image_shape (tuple): Original image shape.
            - rotated_image (NDArray[np.uint8]): Rotated image.

    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/ContactAngle/BaseLine/doc/result.png" alt="Italian Trulli">

    """

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found or unable to load: {image_path}")
    h, w = image.shape#[:2]

    # Focus on the bottom region
    focus_height = int(h * focus_ratio)
    bottom_region = image[h - focus_height:h, :]

    # Preprocess to stabilize edges
    blurred = cv2.GaussianBlur(bottom_region, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Get edge coordinates
    y_indices, x_indices = np.where(edges > 0)

    points = np.column_stack((x_indices, y_indices))

    # Fit robust line using RANSAC
    model, inliers = ransac(
        points, LineModelND,
        min_samples=2,
        residual_threshold=1.0,  # tighter fit
        max_trials=5000          # more attempts
    )

    # Compute line endpoints
    line_x = np.array([min(x_indices), max(x_indices)])
    line_y = model.predict_y(line_x)

    # Adjust for cropped region
    line_y += (h - focus_height)

    # Compute angle
    dx = line_x[1] - line_x[0]
    dy = line_y[1] - line_y[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotate around bottom-center to preserve surface alignment
    center = (w // 2, h - 1)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(255,))
    # Optional visualization
    if results:
        
        plt.figure(figsize=(10, 5))                         # type: ignore
        plt.subplot(1, 2, 1)                                # type: ignore
        plt.imshow(image, cmap='gray')                      # type: ignore
        plt.plot(line_x, line_y, color='red', linewidth=2)  # type: ignore
        plt.title("Detected Line")                          # type: ignore
        plt.subplot(1, 2, 2)                                # type: ignore
        plt.imshow(rotated_image, cmap='gray')              # type: ignore
        plt.title("Rotated Image")                          # type: ignore
        save_dir = os.path.join(os.path.dirname(image_path))
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "Rotation.png"), dpi=300) # type: ignore
        plt.close()

    return angle, image.shape, rotated_image

def fit_image(image: cv2.Mat, black_base_line=10):
    """
    Fits a line to the detected edges in a grayscale image using RANSAC, and computes 
    the vertical offset from the fitted line to a given black baseline.

    Args:
        image (cv2.Mat): Grayscale input image.
        black_base_line (int, optional): Reference baseline offset in pixels. Defaults to 10.

    Returns:
        int: Vertical offset (in pixels) between the fitted line's center height and the black baseline.
    """
    # Detect edges using Canny edge detector
    edges = cv2.Canny(image, 50, 150)
    
    # Find coordinates of non-zero (edge) pixels
    y_indices, x_indices = np.where(edges > 0)
    points = np.column_stack((x_indices, y_indices))  # Shape: (N, 2)

    # Fit a robust line to the edge points using RANSAC (to handle outliers)
    model, inliers = ransac(points, LineModelND, min_samples=2,
                            residual_threshold=2, max_trials=1000)
    
    # Define X-range of the line (min to max X in the edge points)
    line_x = np.array([min(x_indices), max(x_indices)])
    
    # Predict corresponding Y values from the fitted line model
    line_y = model.predict_y(line_x)
    
    # Compute angle of the line (not used in return, but may be useful for debugging)
    dx = line_x[1] - line_x[0]
    dy = line_y[1] - line_y[0]
    angle = np.degrees(np.arctan2(dy, dx))  # Angle of the fitted line in degrees

    # Compute average height of the line and subtract the baseline
    return int((line_y[1] + line_y[0]) // 2) - black_base_line

def line_finder(image:cv2.Mat, rotation_matrix:cv2.Mat, black_base_line:int = 10) -> int:
    """
    Finds the height of the line in the image after applying a rotation matrix.
    Args:
        image (cv2.Mat): Input image in **grayscale**.
        rotation_matrix (cv2.Mat): Rotation matrix to apply to the image.
        black_base_line (int): The baseline height to subtract from the line height.
    Returns:
        int: Height of the line in the rotated image.
    """
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    (h, w) = image.shape
    # rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=255)
    cropped_height = fit_image(rotated_image, black_base_line=black_base_line)
    return cropped_height

def process_image(filepath: str, 
                  rotation_matrix: np.ndarray,
                  cropped_height,
                  output_path: str = None,
                  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) -> None:
    """
    Processes an image by applying a rotation matrix and saving the result.
    Args:
        file (str): Path to the input image file.
        rotation_matrix (cv2.Mat): Rotation matrix to apply to the image.
    Returns:
        None: The function saves the processed image to the same path.

    Calling image[cropped_height+10:, :] = 0  before image rotation make weird artifacts
    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/ContactAngle/BaseLine/doc/rotationweirdartifacts.png" alt="Italian Trulli">
    """
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    (w, h) = image.shape[:2]
    if output_path is None:
        output_path = os.path.dirname(filepath).replace("frames", "frames_rotated")
        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)

    # rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image   = cv2.warpAffine(image, rotation_matrix, (h, w ),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    rotated_image[cropped_height+10:, :] = 0  # Set the top part of the image to black

    # TODO: normalize bottom row
    _rotated_image = bottom_row_unifierGRAY(rotated_image, target_height=130)
    
     ## Close operation fills small dark holes # Kernel size depends on spot size
    # _rotated_image = cv2.morphologyEx(_rotated_image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(output_path, os.path.basename(filepath)), _rotated_image)

def folderBaseLineNormalizer(experiment: os.PathLike[str]|str, 
                             output_path: os.PathLike[str] | None = None):
        files = BaseUtils.ImageLister(experiment)

        output_path = os.path.join(experiment, BaseUtils.config["rotated_frames_folder"])


        if len(glob.glob(os.path.join(output_path, BaseUtils.config["image_extension"]))) == len(files):
            pass
        else:
            if not os.path.isdir(output_path):
                os.makedirs(output_path, exist_ok=True)
        
        image = cv2.imread(os.path.join(experiment, files[2]), cv2.IMREAD_GRAYSCALE)
        (h, w) = image.shape
        center = (w // 2, h // 2)
        angle,_shape, rotated_image = fit_and_rotate_image(os.path.join(experiment, files[2]),results=True)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        with multiprocessing.Pool(processes=int(multiprocessing.cpu_count()*0.75)) as pool: #
            cropped_height_list = pool.starmap(line_finder, [(file, rotation_matrix) for file in files])
        cropped_height = np.array(cropped_height_list).mean().astype(np.int16)
        rotation_matrix = cv2.getRotationMatrix2D((w // 2, cropped_height+10), angle, 1.0)
        print(f"Rotation angle: {angle:.2f} degrees, cropped_height: {cropped_height}")
        with multiprocessing.Pool(processes=int(multiprocessing.cpu_count() * 0.75)) as pool:
            pool.starmap(process_image, [(file, rotation_matrix,cropped_height) for file in files])

def bottom_row_unifierGRAY(image:cv2.Mat,target_height=130) -> cv2.Mat:
    """
    Unifies the bottom rows of an image to a specified target height.
    args:
        image (cv2.Mat): Input image to process.
        target_height (int): Desired height of the output image. Default is 130 pixels.
    Returns:
        cv2.Mat: Processed image with unified bottom rows.
    caution:
        Resizing is mistake. 
        Do the summation in loop and stop when sum is more than one
    """
    ## Step 1: Resize the image if necessary
    resized_image   = image
    # resized_image   = cv2.morphologyEx(resized_image, cv2.MORPH_CLOSE, kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    resized_image   = image[:,::50]
    vv              = resized_image.sum(axis=1)
    height = len(vv)

    for i in range(height-1, 0, -1):
        if vv[i] > 2:
            i -= 1
            break
    padding_top = target_height - i
    image = cv2.copyMakeBorder(image[:i-1,:], padding_top, 0, 0, 0, cv2.BORDER_CONSTANT, None, value = 255)
    image = cv2.copyMakeBorder(image[:,:], 0, 5, 0, 0, cv2.BORDER_CONSTANT, None, value = 0)

    return image

if __name__ == "__main__":
    
    "# first remove the drop from the histogram and then find the angle"
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import BaseUtils

    images = BaseUtils.ImageLister(r"D:\Videos\S1_30per_T1_C001H001S0001")

    # fit_and_rotate_image(images[0],
    #                      results=True,
    #                      focus_ratio=0.3)
    folderBaseLineNormalizer(experiment = r"D:\Videos\S1_30per_T1_C001H001S0001")
    
