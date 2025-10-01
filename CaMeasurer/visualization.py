"""
    Editor: Yassin Riyazi
    Main author: Sajjad Shumaly
    Date: 01-07-2025
    Description: Visualization functions for the CaMeasurer module.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
if __name__ == "__main__":
    import  cv2
    from    criteria_definition import left_angle, right_angle, middle_angle
    from    processing import poly_fitting
    from    edgeDetection import *

else:
    # import os,sys
    # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from .criteria_definition import left_angle, right_angle, middle_angle
    from .processing import poly_fitting


plt.rcParams["figure.figsize"] = (20,15)

# For calling Matplotlib headlessly
import matplotlib
matplotlib.use('Agg')


def horizontal_center(i_list: list, j_list: list, intersection_margin: int = 4) -> tuple:
    """
    Calculate the horizontal center of a shape defined by i_list and j_list coordinates.
    The intersection margin is a margin from the top edge to prevent errors in special cases.

    Args:
        i_list (list): List of i-coordinates (horizontal).
        j_list (list): List of j-coordinates (vertical).
        intersection_margin (int): Margin to avoid edge cases.

    Returns:
        tuple: (horizontal_center, mean_list, j_location_list)

    Authors:
        - Yassin Riyazi (Improved horizontal center calculation)
        - Sajjad Shumaly
    """
    # Convert inputs to numpy arrays
    i_list, j_list = np.array(i_list), np.array(j_list)

    # Split into left and right based on the vertical middle
    i_middle_vertical = int(np.mean(i_list[j_list == j_list.max()]))
    left_mask = i_list <= i_middle_vertical
    i_left, j_left = i_list[left_mask], j_list[left_mask]
    i_right, j_right = i_list[~left_mask], j_list[~left_mask]

    def calculate_extremes(j_vals, i_vals, is_left, j_ref):
        """
        Helper function to find the extreme (leftmost/rightmost) pixel for a given j-coordinate.
        """
        i_out = []
        for j in j_vals:
            i_pixels = i_vals[j_ref == j]
            if i_pixels.size > 0:
                i_value = i_pixels.max() if is_left else i_pixels.min()
                # Adjust for continuous pixels
                for i in range(len(i_pixels)):
                    target = i_pixels.max() - i if is_left else i_pixels.min() + i
                    if target not in i_pixels:
                        i_out.append(target - 1 if is_left else target + 1)
                        break
                else:
                    i_out.append(i_value)
            else:
                i_out.append(np.nan)
        return np.array(i_out)

    def compute_weighted_mean(j_vals, i_left_vals, i_right_vals):
        """
        Helper function to compute weighted mean for horizontal center calculations.
        """
        mean_list, j_loc_list, sum_weighted, total_weight = [], [], 0, 0
        for idx, j in enumerate(j_vals):
            left_pix, right_pix = i_left_vals[idx], i_right_vals[idx]
            if not (np.isnan(left_pix) or np.isnan(right_pix)):
                weight = abs(right_pix - left_pix)
                mean = np.mean([right_pix, left_pix])
                mean_list.append(mean)
                j_loc_list.append(j)
                sum_weighted += weight * mean
                total_weight += weight
        return mean_list, j_loc_list, sum_weighted, total_weight

    # Calculate extremes for left and right sides
    j_range = range(max(j_left) - intersection_margin + 1)
    i_left_ext = calculate_extremes(j_range, i_left, True, j_left)
    i_right_ext = calculate_extremes(j_range, i_right, False, j_right)

    # Compute weighted mean
    mean_list, j_location_list, sum_all, total_weight = compute_weighted_mean(j_range, i_left_ext, i_right_ext)

    # Calculate horizontal center
    horizontal_center = sum_all / total_weight if total_weight != 0 else 0

    return horizontal_center, mean_list, j_location_list

def vertical_center(i_list, j_list, intersection_margin=4):
    """
    Calculate the vertical center of a shape defined by i_list and j_list coordinates.
    The intersection margin is a margin from the left side to prevent errors in special cases.

    Args:
        i_list (list): List of i-coordinates (horizontal).
        j_list (list): List of j-coordinates (vertical).
        intersection_margin (int): Margin to avoid edge cases.

    Returns:
        tuple: (vertical_center, i_location_list, mean_list)

    Authors:
        - Yassin Riyazi (Improved vertical center calculation)
        - Sajjad Shumaly
    """
    # Convert inputs to numpy arrays
    i_list, j_list = np.array(i_list), np.array(j_list)

    # Split into left and right based on the vertical middle
    i_middle_vertical = int(np.mean(i_list[j_list == j_list.max()]))
    left_mask = i_list <= i_middle_vertical
    i_left, j_left = i_list[left_mask], j_list[left_mask]
    i_right, j_right = i_list[~left_mask], j_list[~left_mask]

    # Split left into up and down
    j_middle_left = int(np.mean(j_left[i_left == i_left.min()]))
    left_down_mask = j_left <= j_middle_left
    i_left_down, j_left_down = i_left[left_down_mask], j_left[left_down_mask]
    i_left_up, j_left_up = i_left[~left_down_mask], j_left[~left_down_mask]

    # Split right into up and down
    j_middle_right = int(np.mean(j_right[i_right == i_right.max()]))
    right_down_mask = j_right <= j_middle_right
    i_right_down, j_right_down = i_right[right_down_mask], j_right[right_down_mask]
    i_right_up, j_right_up = i_right[~right_down_mask], j_right[~right_down_mask]

    def calculate_extremes(i_vals, j_vals, is_upper, i_ref):
        """
        Helper function to find the extreme (uppermost/lowermost) pixel for a given i-coordinate.
        """
        j_out = []
        for i in i_vals:
            j_pixels = j_vals[i_ref == i]
            if j_pixels.size > 0:
                j_value = j_pixels.min() if is_upper else j_pixels.max()
                # Adjust for continuous pixels
                for j in range(len(j_pixels)):
                    target = j_pixels.min() + j if is_upper else j_pixels.max() - j
                    if target not in j_pixels:
                        j_out.append(target - 1 if is_upper else target + 1)
                        break
                else:
                    j_out.append(j_value)
            else:
                j_out.append(np.nan)
        return np.array(j_out)

    def compute_weighted_mean(i_vals, j_up, j_down, i_range, is_simple=False):
        """
        Helper function to compute weighted mean for intersection or simple calculations.
        """
        mean_list, i_loc_list, sum_weighted, total_weight = [], [], 0, 0
        for i in i_range:
            up_pix, down_pix = j_up[list(i_vals).index(i)], j_down[list(i_vals).index(i)]
            if not (np.isnan(up_pix) or np.isnan(down_pix)):
                weight = abs(up_pix - (0 if is_simple else down_pix))
                mean = np.mean([up_pix, 0 if is_simple else down_pix])
                mean_list.append(mean)
                i_loc_list.append(i)
                sum_weighted += weight * mean
                total_weight += weight
        return mean_list, i_loc_list, sum_weighted, total_weight

    # Left side calculations
    i_left_range_inter = range(min(i_left_down) + intersection_margin, max(i_left_down))
    i_left_range_simple = range(max(i_left_down), max(i_left_up))
    
    j_left_down_ext = calculate_extremes(i_left_range_inter, j_left_down, False, i_left_down)
    j_left_up_ext = calculate_extremes(i_left_range_inter, j_left_up, True, i_left_up)
    mean_left_inter, i_loc_left_inter, sum_left_inter, weight_left_inter = compute_weighted_mean(
        i_left_range_inter, j_left_up_ext, j_left_down_ext, i_left_range_inter
    )
    
    j_left_up_simple = calculate_extremes(i_left_range_simple, j_left_up, True, i_left_up)
    mean_left_simple, i_loc_left_simple, sum_left_simple, weight_left_simple = compute_weighted_mean(
        i_left_range_simple, j_left_up_simple, np.zeros_like(j_left_up_simple), i_left_range_simple, True
    )

    # Right side calculations
    i_right_range_inter = range(min(i_right_down), max(i_right_down) - intersection_margin)
    i_right_range_simple = range(min(i_right_up), min(i_right_down))
    
    j_right_down_ext = calculate_extremes(i_right_range_inter, j_right_down, False, i_right_down)
    j_right_up_ext = calculate_extremes(i_right_range_inter, j_right_up, True, i_right_up)
    mean_right_inter, i_loc_right_inter, sum_right_inter, weight_right_inter = compute_weighted_mean(
        i_right_range_inter, j_right_up_ext, j_right_down_ext, i_right_range_inter
    )
    
    j_right_up_simple = calculate_extremes(i_right_range_simple, j_right_up, True, i_right_up)
    mean_right_simple, i_loc_right_simple, sum_right_simple, weight_right_simple = compute_weighted_mean(
        i_right_range_simple, j_right_up_simple, np.zeros_like(j_right_up_simple), i_right_range_simple, True
    )

    # Combine results
    sum_all = sum_left_inter + sum_left_simple + sum_right_inter + sum_right_simple
    total_weight = weight_left_inter + weight_left_simple + weight_right_inter + weight_right_simple
    vertical_center = sum_all / total_weight if total_weight != 0 else 0
    
    i_location_list = i_loc_right_simple + i_loc_right_inter + i_loc_left_simple + i_loc_left_inter
    mean_list = mean_right_simple + mean_right_inter + mean_left_simple + mean_left_inter

    return vertical_center, i_location_list, mean_list

def visualize(save_address:str , 
              i_list:NDArray[np.int64],                 j_list:NDArray[np.int64],
              i_left:NDArray[np.int64],                 j_left:NDArray[np.int64],
              i_right:NDArray[np.int64],                j_right:NDArray[np.int64],
              j_poly_left:NDArray[np.float64],          i_poly_left:NDArray[np.float64],
              j_poly_right:NDArray[np.float64],         i_poly_right:NDArray[np.float64],
              x_cropped:int,
              i_poly_left_rotated:NDArray[np.float64],  j_poly_left_rotated:NDArray[np.float64], 
              i_poly_right_rotated:NDArray[np.float64], j_poly_right_rotated:NDArray[np.float64],
              cm_on_pixel:float=5/1280, middle_line_switch:int=0, dpi:int=100):
    """
    Visualize the contact angle measurement results and save the figure.

    Author:
        - Sajjad Shumaly
    """

    font_size=14
    upscale_factor=3
    conversion_factor=cm_on_pixel/upscale_factor

    fig, ax = plt.subplots(figsize=(15, 10),dpi=dpi)  # Use subplot
    ax.clear()

    # Drop shape
    ax.plot(i_list, j_list, '.', color='black')

    # Contact angle edge
    ax.plot(i_left,     j_left,         '.', color='red', markersize=12)
    ax.plot(i_right,    j_right,        '.', color='red', markersize=12)

    # Poly fit
    ax.plot(i_poly_left,        j_poly_left, '--', color='yellow', linewidth=4)
    ax.plot(i_poly_right,       j_poly_right, '--', color='yellow', linewidth=4)

    # Left angle
    left_angle_degree, left_angle_point = left_angle(i_poly_left_rotated, j_poly_left_rotated, 1)

    ax.plot([i_poly_left[0]+20, i_poly_left[0]], [j_poly_left[0], j_poly_left[0]], linewidth=3, color='blue')

    m = np.tan(left_angle_degree * (np.pi / 180))

    ax.plot([i_poly_left[0], i_poly_left[0] + (1/m) * j_poly_left[20]], [j_poly_left[0], j_poly_left[20]], linewidth=3, color='blue')
    ax.text(i_poly_left[0], j_poly_left[0] - 12, 'Advancing=' + str(round(left_angle_degree, 2)), color="blue", fontsize=font_size)

    # Right angle
    right_angle_degree, right_angle_point = right_angle(i_poly_right_rotated, j_poly_right_rotated, 1)
    ax.plot([i_poly_right[0]-20, i_poly_right[0]], [j_poly_right[0], j_poly_right[0]], linewidth=3, color='blue')
    m = np.tan(right_angle_degree * (np.pi / 180))
    ax.plot([i_poly_right[0], i_poly_right[0] - (1/m) * j_poly_right[20]], [j_poly_right[0], j_poly_right[20]], linewidth=3, color='blue')
    ax.text(i_poly_right[0] - 65, j_poly_right[0] - 12, 'Receding=' + str(round(right_angle_degree, 2)), color="blue", fontsize=font_size)

    # Contact line
    contact_line_length = (right_angle_point - left_angle_point) * conversion_factor
    ax.plot([(x_cropped * 3) + np.array(left_angle_point), (x_cropped * 3) + np.array(right_angle_point)], [0, 0], '--', linewidth=1, color='red')
    ax.text(((x_cropped * 3) + np.array(right_angle_point) + (x_cropped * 3) + np.array(left_angle_point)) / 2 - 60, j_poly_right[0] - 12,
            'Contact line length=' + str(round(contact_line_length, 3)) + ' cm', color="red", fontsize=font_size)
    right_angle_point = ((1280) * 3 - right_angle_point - (x_cropped) * 3) * conversion_factor
    left_angle_point = ((1280) * 3 - left_angle_point + -(x_cropped) * 3) * conversion_factor

    # Centers
    v_center, *_ = vertical_center(i_list, j_list)
    h_center, i_mean, j_mean = horizontal_center(i_list, j_list)
    ax.plot([h_center, h_center], [min(j_list), j_list[i_list == int(h_center)][0]], '--', color='green')
    drop_height = abs(min(j_list) - j_list[i_list == int(h_center)][0]) * conversion_factor
    i_text_horizontal = (j_list[i_list == int(h_center)][0] + v_center) / 2
    ax.text(h_center + 5, i_text_horizontal, str(round(drop_height, 3)) + ' cm', color="green", fontsize=font_size)

    # Middle line
    i_middle_line, j_middle_line    = poly_fitting(i_mean, j_mean, polynomial_degree=1, line_space=100)
    middle_angle_degree             = middle_angle(i_middle_line, j_middle_line)
    if middle_line_switch != 0:
        i_middle_line, j_middle_line = poly_fitting(i_mean, j_mean, polynomial_degree=1, line_space=100)
        middle_angle_degree = middle_angle(i_middle_line, j_middle_line)
        i2_middle_line = min(i_middle_line[j_middle_line <= j_list[i_list == int(h_center)][0]])
        ax.plot([i_middle_line[-1], i2_middle_line], [0, j_middle_line[i_middle_line == i2_middle_line][0]], '-', color='black')
        ax.text(i2_middle_line - 35, j_middle_line[i_middle_line == i2_middle_line][0] - 20,
                'Angle=' + str(round(middle_angle_degree[0], 2)), color="black", fontsize=font_size)

    # Vertical center
    v_center, i_mean, j_mean = vertical_center(i_list, j_list)
    ax.plot([min(i_list[j_list == int(v_center)]), max(i_list[j_list == int(v_center)])], [v_center, v_center], '--', color='green')
    i_text_vertical = (min(i_list) + h_center) / 2
    drop_length = abs(min(i_list[j_list == int(v_center)]) - max(i_list[j_list == int(v_center)])) * conversion_factor
    ax.text(i_text_vertical, v_center + 5, str(round(drop_length, 3)) + ' cm', color="green", fontsize=font_size)

    # Center point
    x_center = ((1280) * 3 - h_center) * conversion_factor
    y_center = v_center * conversion_factor
    ax.plot(h_center, v_center, '.', color='blue', markersize=14)
    ax.text(h_center + 5, v_center + 5,
            'Center= [x=' + str(round(x_center, 3)) + ' cm, y=' + str(round(y_center, 3)) + ' cm]', color="blue", fontsize=font_size)

    # ax.axis('equal')
    ax.set_ylim(-30, 300)  # Set y limit as requested
    ax.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    fig.savefig(save_address,dpi=dpi)
    plt.close(fig)

    return left_angle_degree, right_angle_degree, right_angle_point, left_angle_point, contact_line_length, x_center, y_center, middle_angle_degree[0]

if __name__ == "__main__":
    

    left_polynomial_degree  = 3
    right_polynomial_degree = 2

    cm_on_pixel_ratio           = 0.0039062
    num_px_ratio                = (0.0039062)/cm_on_pixel_ratio

    _image = cv2.imread('src/PyThon/ContactAngle/CaMeasurer/doc/FullFrame.png')
    if _image is None:
        raise ValueError("Image not found or unable to load.")
    elif len(_image.shape) > 2:
        # raise ValueError("Invalid image format.")
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
    
    i_list, j_list  = edge_extraction(_image, thr=30)


    left_number_of_pixels   = int(64*num_px_ratio)
    right_number_of_pixels  = int(65*num_px_ratio)
    i_left, j_left          = Advancing_pixel_selection_Euclidean(i_list,j_list, left_number_of_pixels=left_number_of_pixels)
    i_right, j_right        = Receding_pixel_selection_Euclidean(i_list,j_list, right_number_of_pixels=right_number_of_pixels)

    
    i_poly_left_rotated, j_poly_left_rotated    = poly_fitting(j_left,i_left,left_polynomial_degree,left_number_of_pixels)
    i_poly_right_rotated, j_poly_right_rotated  = poly_fitting(i_right,i_right,right_polynomial_degree,right_number_of_pixels)


    horizontal_center, mean_list, j_location_list = horizontal_center(i_list, j_list, intersection_margin=4)
    import matplotlib.pyplot as plt 

    plt.figure(figsize=(10, 6))
    plt.plot(horizontal_center, mean_list, label='Mean Profile', color='blue')
    plt.scatter(horizontal_center, j_location_list, label='J Locations', color='red')
    plt.title('Horizontal Center and Mean Profile')
    plt.xlabel('Horizontal Center')
    plt.ylabel('Mean Profile / J Locations')
    plt.legend()
    plt.grid()
    plt.show()
