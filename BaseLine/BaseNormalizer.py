"""
    Date    : 2025-09-30
    Author  : Yassin Riyazi
    Project : Automatic Video Processor (AVP)
    File    : BaseLine/__init__.py
    Version : 1.0.0
    License : GNU GENERAL PUBLIC LICENSE Version 3

    ChangeLog:
        V1.0.1 29.10.2025 : Minor bug fix, problem of making name file in windows.
        V1.0.0 30.09.2025 : Initial version.

"""

import  os
import  cv2
import  glob
import  matplotlib
import  multiprocessing
 
import  numpy               as      np
matplotlib.use('Agg')  # For file output only, no GUI
import  matplotlib.pyplot as plt
from    skimage.measure     import  ransac, LineModelND # type: ignore

from numpy.typing import NDArray
from typing import Any, Tuple, TypeAlias
ImageSize: TypeAlias = Tuple[int, int]

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import BaseUtils

def load_cropped_image(image_path: str,
                       yolo_hint: int | None = None) -> NDArray[np.uint8]:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)       
    if yolo_hint is not None:
        image = image[:yolo_hint, :].astype(np.uint8)
    return image

def fit_and_rotate_image(image_path: str,
                         results: bool = True,
                         focus_ratio: float = 0.3,
                         yolo_hint: int | None = None
                         ) -> tuple[float, ImageSize, NDArray[np.uint8]]:
    """
    Fits a robust line to the bottom edges of an image and rotates the image to level the surface.
    
    Args:
        image_path (os.PathLike): Path to the input image.
        results (bool, optional): If True, saves a diagnostic plot.
        focus_ratio (float, optional): Portion of the image height to analyze from the bottom. Default 0.3.
        yolo_hint (int | None, optional): Vertical position hint from YOLO to guide line fitting. Default None.
    
    Returns:
        tuple:
            - angle (float): Rotation angle in degrees.
            - image_shape (tuple): Original image shape.
            - rotated_image (NDArray[np.uint8]): Rotated image.

    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/ContactAngle/BaseLine/doc/result.png" alt="Italian Trulli">

    """
    image = load_cropped_image(image_path, yolo_hint)  # type: ignore
    if image is None:
        raise FileNotFoundError(f"Image not found or unable to load: {image_path}")
    h, w = image.shape[:2]

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
    model, inliers = ransac( # type: ignore
        points, LineModelND,
        min_samples=2,
        residual_threshold=1.0,  # tighter fit
        max_trials=5000          # more attempts
    )

    # Compute line endpoints
    line_x = np.array([min(x_indices), max(x_indices)])
    line_y = model.predict_y(line_x)# type: ignore

    # Adjust for cropped region
    line_y += (h - focus_height)# type: ignore

    # Compute angle
    dx = line_x[1] - line_x[0]
    dy = line_y[1] - line_y[0]# type: ignore
    angle = np.degrees(np.arctan2(dy, dx))# type: ignore

    # Rotate around bottom-center to preserve surface alignment
    center = (w // 2, h - 1)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(255,)).astype(np.uint8)
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

def fit_image(image: NDArray[np.uint8],
              black_base_line: int = 10,
              yolo_hint: int | None = None,
              search_margin: int = 80) -> int:
    """
    Finds the topmost horizontal surface line by scanning edge rows top-to-bottom.

    When `yolo_hint` is provided (the rotated-space y2 of the drop bounding box), the
    search is restricted to [yolo_hint - search_margin, yolo_hint + search_margin] so
    the function locks onto the substrate surface rather than deeper lines.
    Falls back to RANSAC if no usable edges are found in the search window.

    Args:
        image (cv2.Mat): Grayscale input image (already rotated).
        black_base_line (int): Offset subtracted from the detected row. Defaults to 10.
        yolo_hint (int | None): Estimated y-coordinate of the drop-surface contact in
            rotated-image space. If None the full image height is searched.
        search_margin (int): Pixels above/below `yolo_hint` to search. Defaults to 80.

    Returns:
        int: Row index of the topmost prominent edge minus black_base_line.
    """
    # Detect edges
    edges = cv2.Canny(image, 50, 150)

    h = image.shape[0]

    # Determine vertical search window
    if yolo_hint is not None:
        row_start = max(0, yolo_hint - search_margin)
        row_end   = min(h, yolo_hint + search_margin)
    else:
        row_start, row_end = 0, h

    # Per-row edge counts within the window
    row_sums = edges[row_start:row_end, :].sum(axis=1)

    if row_sums.max() > 0:
        threshold = row_sums.max() * 0.30
        # Scan top-to-bottom and take the first row that crosses the threshold
        for local_i, val in enumerate(row_sums):
            if val >= threshold:
                return (row_start + local_i) - black_base_line

    # --- Fallback: RANSAC on all edges in the window ---
    y_indices, x_indices = np.where(edges[row_start:row_end, :] > 0)
    if len(y_indices) < 2:
        # Last resort: return middle of the window
        return (row_start + (row_end - row_start) // 2) - black_base_line

    y_indices = y_indices + row_start  # adjust back to full-image coordinates
    points = np.column_stack((x_indices, y_indices))  # Shape: (N, 2)

    # Fit a robust line to the edge points using RANSAC (to handle outliers)
    model, inliers = ransac(points, LineModelND, min_samples=2,  # type: ignore# type: ignore
                            residual_threshold=2, max_trials=1000)
    
    # Define X-range of the line (min to max X in the edge points)
    line_x = np.array([min(x_indices), max(x_indices)])
    line_y = model.predict_y(line_x)  # type: ignore

    return int((line_y[1] + line_y[0]) // 2) - black_base_line  # type: ignore

def line_finder(image_address: str,
                rotation_matrix: cv2.Mat,
                black_base_line: int = 10,
                yolo_hint: int | None = None) -> int:
    """
    Finds the height of the topmost surface line in the image after applying a rotation matrix.
    Args:
        image_address (str): Path to the input grayscale image.
        rotation_matrix (cv2.Mat): Rotation matrix to apply to the image.
        black_base_line (int): Offset subtracted from the detected row. Defaults to 10.
        yolo_hint (int | None): Estimated y-coordinate of the drop-surface contact in
            rotated-image space, used to restrict the edge search window.
    Returns:
        int: Row index (in the rotated image) of the detected topmost surface line.
    """
    # image = cv2.imread(image_address, cv2.IMREAD_GRAYSCALE)
    image = load_cropped_image(image_address, yolo_hint)  # type: ignore

    if image is None:
        raise FileNotFoundError("Image not found or unable to load.")
    (h, w) = image.shape[:2]
    # rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=255)
    cropped_height = fit_image(rotated_image.astype(np.uint8),
                               black_base_line=black_base_line,
                               yolo_hint=yolo_hint)
    return cropped_height

def process_image(filepath: str, 
                  rotation_matrix: NDArray[np.float64],
                  cropped_height:int,
                  output_path: str|None = None,
                  yolo_hint: int | None = None
                  ) -> None:
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
    # image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image = load_cropped_image(filepath, yolo_hint)  # type: ignore

    if image is None:
        raise FileNotFoundError(f"Image not found or unable to load: {filepath}")
    (w, h) = image.shape[:2]
    if output_path is None:
        output_path = os.path.dirname(filepath)
        parts = output_path.split("frames")

        if len(parts) > 2:
            # Replace only the second occurrence
            output_path = "frames".join(parts[:2]) + "frames_rotated" + "frames".join(parts[2:])
        else:
            # Replace the only or last occurrence
            output_path = output_path.replace("frames", "frames_rotated", 1)

        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)

    # rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image   = cv2.warpAffine(image, rotation_matrix, (h, w ),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255) # type: ignore
    rotated_image[cropped_height+10:, :] = 0  # Set the top part of the image to black

    # TODO: normalize bottom row [Done]
    _rotated_image = bottom_row_unifierGRAY(rotated_image.astype(np.uint8), target_height=w)
    
    # Close operation fills small dark holes # Kernel size depends on spot size
    # kernel:NDArray[np.uint8] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)).astype(np.uint8)
    # _rotated_image = cv2.morphologyEx(_rotated_image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(output_path, os.path.basename(filepath)), _rotated_image)

def _estimate_baseline_from_yolo(files: list[str],
                                  n_samples: int = 50) -> int | None:
    """
    Estimates the substrate surface row in rotated-image space by running the YOLO
    drop detector on a few mid-section frames and projecting the bounding-box bottom
    edge (y2) through the rotation matrix.

    This gives `folderBaseLineNormalizer` a spatial hint so that `fit_image` searches
    only near the actual drop-substrate contact line, rather than latching onto lower
    or deeper features in the image.

    Args:
        files (list[str]): Sorted list of full frame paths for the experiment.
        rotation_matrix (NDArray[Any]): 2x3 affine rotation matrix already
            computed from the first rotation step.
        n_samples (int): Number of frames to sample from the middle third. Defaults to 50.

    Returns:
        int | None: Median rotated y2 coordinate, or None if YOLO is unavailable or
            yields no detections.
    """
    try:
        detector = BaseUtils.DropDetection_YOLO()
    except Exception:
        return None

    n = len(files)
    third = max(1, n // 3)
    sample_indices = np.linspace(third, 2 * third, n_samples, dtype=int)

    y2_rotated_vals: list[int] = []
    for idx in sample_indices:
        frame_path = files[int(idx)]
        try:
            box, detected = detector.detect_drops(frame_path)
            if not detected or box is None:
                continue
            x1, y1, x2, y2 = BaseUtils.DropDetection_YOLO.bound_extractor(box)
            y2_rotated_vals.append(y2)
        except Exception:
            continue

    if not y2_rotated_vals:
        return None

    # iterate over all images and crop them to the y2_rotated_vals and then find the line and take the median of the line heights as the final y2_rotated_vals
    # y22 = int(np.array(y2_rotated_vals).mean())  # +30 px safety margin so the substrate surface line is not clipped
    y22 = sorted(y2_rotated_vals)[0] + 15  # type: ignore
    for file in files:
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)       
        image = image[:y22, :].astype(np.uint8)  # crop to the max y2 + margin
        
        # these lines are to avoid the weird yolo detection failailire which lead to flase negatives. 
        # by making the top and bottom 3 rows white we ensure that the line fitting will not latch onto these artifacts
        image[:, :3] = 255  # Set the leftmost 3 columns to white
        image[:, -3:] = 255  # Set the rightmost 3 columns to white

        cv2.imwrite(file, image)  # overwrite the original image with the cropped one

    return None
    # return int(np.median(y2_rotated_vals))

def folderBaseLineNormalizer(experiment: str, 
                             output_path: str | None = None,
                             verbose: bool = False
                             ) -> None:
        files = BaseUtils.ImageLister(experiment)

        output_path = os.path.join(experiment, str(BaseUtils.config["rotated_frames_folder"]))

        if len(glob.glob(os.path.join(output_path, str(BaseUtils.config["image_extension"])))) == len(files):
            pass
        else:
            if not os.path.isdir(output_path):
                os.makedirs(output_path, exist_ok=True)
        
        image = cv2.imread(os.path.join(experiment, files[2]), cv2.IMREAD_GRAYSCALE)
        
        # Get a hint of the drop's vertical position from YOLO to guide the line fitting
        _ = _estimate_baseline_from_yolo(files)
        yolo_hint = None
        # add a margin to be safe and avoid the drop itself
        # yolo_hint = yolo_hint + 7 if yolo_hint is not None else None
        # crop image to focus on the top part where the surface line is expected, this also makes the line fitting more robust and faster
        if yolo_hint is not None:
            row_start = max(0, yolo_hint - 100)  # type: ignore
        else:
            row_start = 0
        image = image[row_start:, :]

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        angle,_shape, rotated_image = fit_and_rotate_image(os.path.join(experiment, files[2]),results=True, yolo_hint=yolo_hint) # type: ignore
        del _shape, rotated_image
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        if verbose:
            print(f"YOLO baseline hint (rotated y2): {yolo_hint}")

        with multiprocessing.Pool(processes=int(multiprocessing.cpu_count()*0.75)) as pool: #
            cropped_height_list = pool.starmap(line_finder, [(file, rotation_matrix, 10, yolo_hint) for file in files])
        cropped_height = np.array(cropped_height_list).mean().astype(np.int16)
        rotation_matrix = cv2.getRotationMatrix2D((w // 2, cropped_height+10), angle, 1.0)
        if verbose:
            print(f"Rotation angle: {angle:.2f} degrees, cropped_height: {cropped_height}")
        with multiprocessing.Pool(processes=int(multiprocessing.cpu_count() * 0.75)) as pool:
            pool.starmap(process_image, [(file, rotation_matrix,cropped_height) for file in files])

def bottom_row_unifierGRAY(image:NDArray[np.uint8],target_height:int,
                           pad_bottom:int = 5) -> NDArray[np.uint8]:
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
    
    i = None
    for i in range(height-1, 0, -1):
        if vv[i] > 2:
            i -= 1
            break
    if i is None:
        raise ValueError("No non-zero rows found in the image.")
    
    padding_top = target_height - i
    image = cv2.copyMakeBorder(image[:i-1,:], padding_top, 0, 0, 0, cv2.BORDER_CONSTANT, None, value = 255) # type: ignore
    image = cv2.copyMakeBorder(image[:,:], 0, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, None, value = 0)# type: ignore

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
    
