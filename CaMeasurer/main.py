import os
import cv2
import CaMeasurer
import pandas as pd
import numpy as np
from .criteria_definition   import *

def read_four_integers(file_path:str) -> list[int]:
    """
        Legacy code to read YOLO x1 x2 position from a text file.
    """
    with open(file_path, 'r') as file:
        line = file.readline()
        numbers = list(map(int, line.strip().split()))
        if len(numbers) != 2:
            raise ValueError("The file does not contain exactly four integers.")
        return numbers



def read_csv_for_endpoint_beginning(df, image_name):
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
    # try:
    #     # df = pd.read_csv(csv_path)
    # except FileNotFoundError:
    #     raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    match = df[df['image'] == image_name]
    if match.empty:
        raise ValueError(f"Image '{image_name}' not found in the CSV.")

    row = match.iloc[0]
    endpoint = int(row['endpoint'])
    beginning = int(row['beginning'])

    return [endpoint, beginning]


def polyOrderDecider(degree,num_px_ratio):
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
    return pixelNum,polyOrder


def base_function_process(df,ad,name_files,file_number, model, kernel, num_px_ratio, left_polynomial_degree=3, right_polynomial_degree=2):
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

    Authors:
        - Yassin Riyazi (Removing two layer polynomial and adaptive pixel selection, because Euclidean distance is not working well for all images)
        - Sajjad Shumaly
    """
    # 1. Loading data
    just_drop       = cv2.imread(os.path.join(ad,name_files[file_number]))
    just_drop       = just_drop[:-2,:,:]

    # x1              = read_four_integers(os.path.join(ad,name_files[file_number]).replace("jpg","txt"))[0]
    x1, x2 = read_csv_for_endpoint_beginning(df, name_files[file_number])

    # 2.  Supper resulotion
    upscaled_image  = CaMeasurer.upscale_image(model, just_drop, kernel)

    # 3.  Extracting whole edge points
    i_list, j_list  = CaMeasurer.edge_extraction( upscaled_image, thr=30)

    # 4.  Extracting advancing and receding points   
    left_number_of_pixels   = int(64*num_px_ratio)
    right_number_of_pixels  = int(65*num_px_ratio)
    i_left, j_left          = CaMeasurer.Advancing_pixel_selection_Euclidean(i_list,j_list, left_number_of_pixels=left_number_of_pixels)
    i_right, j_right        = CaMeasurer.Receding_pixel_selection_Euclidean(i_list,j_list, right_number_of_pixels=right_number_of_pixels)

    #rotation for fitting, it can increase the accuracy to rotate 90 degrees then fit the polynomial
    i_left_rotated,j_left_rotated=j_left,i_left       
    i_right_rotated,j_right_rotated=j_right,i_right   

    
    i_poly_left_rotated, j_poly_left_rotated    = CaMeasurer.poly_fitting(i_left_rotated,j_left_rotated,left_polynomial_degree,left_number_of_pixels)
    i_poly_right_rotated, j_poly_right_rotated  = CaMeasurer.poly_fitting(i_right_rotated,j_right_rotated,right_polynomial_degree,right_number_of_pixels)

    # right_angle_degree,right_angle_point        = right_angle(i_poly_right_rotated, j_poly_right_rotated,1)
    # left_angle_degree,left_angle_point          = left_angle(i_poly_left_rotated, j_poly_left_rotated,1)
    
    # left_number_of_pixels,  left_polynomial_degree = polyOrderDecider(left_angle_degree,    num_px_ratio)
    # right_number_of_pixels, right_polynomial_degree= polyOrderDecider(right_angle_degree,   num_px_ratio)


    #9. extracting the desired number of pixels as input of the polynomial fitting 
    # i_left, j_left      = CaMeasurer.Advancing_pixel_selection_Euclidean(i_list,j_list, left_number_of_pixels=left_number_of_pixels)
    # i_right, j_right    = CaMeasurer.Receding_pixel_selection_Euclidean(i_list,j_list, right_number_of_pixels=right_number_of_pixels)

    #10. rotation for fitting, it can increase the accuracy to rotate 90 degrees and then fit the polynomial
    # i_left_rotated,j_left_rotated=j_left,i_left       
    # i_right_rotated,j_right_rotated=j_right,i_right 

    # i_poly_left_rotated, j_poly_left_rotated    = CaMeasurer.poly_fitting(i_left_rotated,j_left_rotated,left_polynomial_degree,left_number_of_pixels)
    # i_poly_right_rotated, j_poly_right_rotated  = CaMeasurer.poly_fitting(i_right_rotated,j_right_rotated,right_polynomial_degree,right_number_of_pixels)

    j_poly_left=i_poly_left_rotated
    i_poly_left=j_poly_left_rotated
    j_poly_right=i_poly_right_rotated
    i_poly_right=j_poly_right_rotated
    # x_cropped=dim[0]
    x_cropped = x1
    return i_list, j_list, i_left, j_left, i_right, j_right, j_poly_left, i_poly_left, j_poly_right, i_poly_right, x_cropped, i_poly_left_rotated, j_poly_left_rotated, i_poly_right_rotated, j_poly_right_rotated