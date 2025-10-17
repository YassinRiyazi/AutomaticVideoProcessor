"""
    Authors:
        - Yassin Riyazi (Removing two layer polynomial and adaptive pixel selection, because Euclidean distance is not working well for all images)
        - Sajjad Shumaly
"""

import os
import cv2
import torch
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

if __name__ == "__main__":
    from criteria_definition   import *
    from superResolution         import upscale_image
    from BaseUtils.Detection.edgeDetection           import edge_extraction, Advancing_pixel_selection_Euclidean, Receding_pixel_selection_Euclidean
    from BaseUtils.Detection.LightSourceReflectionRemoving import LightSourceReflectionRemover
    from processing              import poly_fitting
    from criteria_definition    import right_angle, left_angle
else:
    import os,sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from criteria_definition   import *
    from superResolution         import upscale_image
    from BaseUtils.Detection.edgeDetection           import edge_extraction, Advancing_pixel_selection_Euclidean, Receding_pixel_selection_Euclidean
    from BaseUtils.Detection.LightSourceReflectionRemoving import LightSourceReflectionRemover
    from processing              import poly_fitting
    from criteria_definition    import right_angle, left_angle

def read_csv_for_endpoint_beginning(df: pd.DataFrame|None, image_name: str) -> list[int]:
    """
    Reads a CSV file and returns the endpoint and beginning values for a given image name.

    Args:
        csv_path (str): Path to the CSV file containing the image data.
        image_name (str): Name of the image to look up (e.g., 'image001.jpg').

    Returns:
        list: A list containing [endpoint, beginning].

    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
        ValueError: If the image is not found or the row does not contain exactly two integers.
    
    Author:
        - Yassin Riyazi
    """
    # accessing global dataframe 'df'
    if df is None:
        df = pd.read_csv(os.path.join(os.path.dirname(image_name), 'detections.csv')) # type: ignore
    image_name = os.path.basename(image_name)

    match = df[df['image'] == image_name]
        
    if match.empty:
        raise ValueError(f"Image '{image_name}' not found in the CSV.")

    row = match.iloc[0]
    endpoint = int(row['endpoint'])
    beginning = int(row['beginning'])

    return [endpoint, beginning]


def polyOrderDecider(degree:float|NDArray[np.float64],
                     num_px_ratio:float) -> Tuple[int,int]:
    if degree<=60 :
        pixelNum=int(60*num_px_ratio)
        polyOrder=2
    elif 60<degree<=105:
        pixelNum=int(85*num_px_ratio)
        polyOrder=2
    elif 105<degree<=135:
        pixelNum=int(125*num_px_ratio)  #175
        polyOrder=3
    elif 135<degree:
        pixelNum=int(145*num_px_ratio) #215
        polyOrder=4
    else:
        raise ValueError("Angle is not in the expected range.")
    return pixelNum,polyOrder

def base_function_process(df: pd.DataFrame, 
                          ad: str,
                          name_files: list[str],
                          file_number: int, 
                          model:torch.nn.Module, 
                          kernel: NDArray[np.uint8],
                          num_px_ratio: float,
                          left_polynomial_degree: int = 3,
                          right_polynomial_degree: int = 2
                          ):
    """
    1.  Loading data
    1.1.Loading the image
    1.2.cropping the base line
    1.3.loading the x1, x2 positions
    2.  Super-resolution
    3.  Extracting whole edge points
    4.  Extracting advancing and receding points

    Test:
        Removing two layer polynomial 
        Removing super-resolution from this section
    """
    # 1. Loading data

    File_address = name_files[file_number]

    just_drop       = cv2.imread(name_files[file_number])
    if just_drop is None:
        raise FileNotFoundError(f"Image not found or unable to read: {File_address}")
    just_drop       = just_drop[:-5,:,:]

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # just_drop = cv2.morphologyEx(just_drop, cv2.MORPH_CLOSE, kernel)

    x1, x2 = read_csv_for_endpoint_beginning(df,name_files[file_number])
    del x2 

    # 2.  Supper resolution
    just_drop = just_drop.astype(np.uint8)
    upscaled_image  = upscale_image(model, just_drop, kernel)


    # 3.  Extracting whole edge points
    # upscaled_image  = LightSourceReflectionRemover(upscaled_image.astype(np.uint8))
    i_list, j_list  = edge_extraction(upscaled_image.astype(np.int8), thr=30)
    # 4.  Extracting advancing and receding points   
    left_number_of_pixels   = int(64*num_px_ratio)
    right_number_of_pixels  = int(65*num_px_ratio)
    i_left, j_left          = Advancing_pixel_selection_Euclidean(i_list,j_list, left_number_of_pixels=left_number_of_pixels)
    i_right, j_right        = Receding_pixel_selection_Euclidean(i_list,j_list, right_number_of_pixels=right_number_of_pixels)

    #rotation for fitting, it can increase the accuracy to rotate 90 degrees then fit the polynomial
    i_left_rotated,j_left_rotated=j_left,i_left       
    i_right_rotated,j_right_rotated=j_right,i_right   

    
    i_poly_left_rotated, j_poly_left_rotated    = poly_fitting(i_left_rotated,j_left_rotated,left_polynomial_degree,left_number_of_pixels)
    i_poly_right_rotated, j_poly_right_rotated  = poly_fitting(i_right_rotated,j_right_rotated,right_polynomial_degree,right_number_of_pixels)

    right_angle_degree,right_angle_point        = right_angle(i_poly_right_rotated, j_poly_right_rotated,1)
    left_angle_degree,left_angle_point          = left_angle(i_poly_left_rotated, j_poly_left_rotated,1)
    del right_angle_point, left_angle_point
    
    left_number_of_pixels,  left_polynomial_degree = polyOrderDecider(left_angle_degree,    num_px_ratio)
    right_number_of_pixels, right_polynomial_degree= polyOrderDecider(right_angle_degree,   num_px_ratio)


    # 9. extracting the desired number of pixels as input of the polynomial fitting 
    i_left, j_left      = Advancing_pixel_selection_Euclidean(i_list,j_list, left_number_of_pixels=left_number_of_pixels)
    i_right, j_right    = Receding_pixel_selection_Euclidean(i_list,j_list, right_number_of_pixels=right_number_of_pixels)

    # 10. rotation for fitting, it can increase the accuracy to rotate 90 degrees and then fit the polynomial
    i_left_rotated,j_left_rotated=j_left,i_left       
    i_right_rotated,j_right_rotated=j_right,i_right 

    i_poly_left_rotated, j_poly_left_rotated    = poly_fitting(i_left_rotated,j_left_rotated,left_polynomial_degree,left_number_of_pixels)
    i_poly_right_rotated, j_poly_right_rotated  = poly_fitting(i_right_rotated,j_right_rotated,right_polynomial_degree,right_number_of_pixels)

    j_poly_left=i_poly_left_rotated
    i_poly_left=j_poly_left_rotated
    j_poly_right=i_poly_right_rotated
    i_poly_right=j_poly_right_rotated
    # x_cropped=dim[0]
    x_cropped = x1
    return i_list, j_list, i_left, j_left, i_right, j_right, j_poly_left, i_poly_left, j_poly_right, i_poly_right, x_cropped, i_poly_left_rotated, j_poly_left_rotated, i_poly_right_rotated, j_poly_right_rotated