"""
    Author: Sajjad Shumaly
    Date: 01-07-2022
"""
import  numpy           as      np
from    typing          import  List, Tuple # type:ignore
from    numpy.typing    import  NDArray

def left_angle(i_poly_left:NDArray[np.float64], 
               j_poly_left:NDArray[np.float64], 
               tan_pixel_number:int=1
               ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate the advancing angle and pixel position from polynomial fitted data.
    args:
        i_poly_left (np.ndarray): x-coordinates of the left polynomial fitted data.
        j_poly_left (np.ndarray): y-coordinates of the left polynomial fitted data.
        tan_pixel_number (int): Index of the pixel used for tangent calculation (default: 1).
    returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the advancing angle and the pixel position.
    
    Author:
        - Sajjad Shumaly
    """
    if len(i_poly_left) < 2 or len(j_poly_left) < 2:
        raise ValueError("Input arrays must have at least two elements.")
    if tan_pixel_number < 1 or tan_pixel_number >= len(i_poly_left):
        raise ValueError("tan_pixel_number must be at least 1 and less than the length of the input arrays.")

    dx = i_poly_left[tan_pixel_number] - i_poly_left[0]
    dy = j_poly_left[tan_pixel_number] - j_poly_left[0]

    if np.isclose(dx, 0) and np.isclose(dy, 0):
        raise ValueError("dx and dy cannot both be zero for angle calculation.")

    radian_angle = np.arctan(dy / dx)
    horizontal_angle = np.degrees(radian_angle)
    left_angle = 90 - horizontal_angle
    left_pixel_position = j_poly_left[0]
    return left_angle, left_pixel_position

def right_angle(i_poly_right:NDArray[np.float64], 
                j_poly_right:NDArray[np.float64],
                tan_pixel_number:int=1
                ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate the receding angle and pixel position from polynomial fitted data.    
    args:
        i_poly_right (NDArray[np.float64]): x-coordinates of the right polynomial fitted data.
        j_poly_right (NDArray[np.float64]): y-coordinates of the right polynomial fitted data.
        tan_pixel_number (int): Index of the pixel used for tangent calculation (default: 1).
    returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the receding angle and the pixel position
    
    Changelog:
        - 2022-07-01: Initial implementation. [Sajjad Shumaly]
    """
    if len(i_poly_right) < 2 or len(j_poly_right) < 2:
        raise ValueError("Input arrays must have at least two elements.")
    if tan_pixel_number < 1 or tan_pixel_number >= len(i_poly_right):
        raise ValueError("tan_pixel_number must be at least 1 and less than the length of the input arrays.")

    dx = i_poly_right[tan_pixel_number] - i_poly_right[0]
    dy = j_poly_right[tan_pixel_number] - j_poly_right[0]

    if np.isclose(dx, 0) and np.isclose(dy, 0):
        raise ValueError("dx and dy cannot both be zero for angle calculation.")


    radian_angle = np.arctan(dy / dx)
    horizontal_angle = np.degrees(radian_angle)
    right_angle = 90 + horizontal_angle
    right_pixel_position = j_poly_right[0]
    return right_angle, right_pixel_position

def middle_angle(i_poly_right: NDArray[np.float64], 
                 j_poly_right: NDArray[np.float64]
                 ) -> Tuple[float|NDArray[np.float64], float|NDArray[np.float64]]:
    """
    Version 1.1.0

    Calculate the middle angle and pixel position from polynomial fitted data.

    Args:
        i_poly_right (NDArray[np.float64]): x-coordinates of the right polynomial fitted data.
        j_poly_right (NDArray[np.float64]): y-coordinates of the right polynomial fitted data.

    Returns:
        Tuple[float, float]: A tuple containing the middle angle and the pixel position.

    Changelog:
        - 2022-07-01: Initial implementation. [Sajjad Shumaly]
        - 2025-10-16: [Yassin Riyazi]
            - [BUGFIX] Uninitialized middle_angle
                if dx<0:
                    middle_angle=-horizontal_angle
                if dx>0:
                    middle_angle=180+90-horizontal_angle
            - Using native rad2deg conversion
            - Added assertions for input validation
            - checking dx and dy for zero to avoid division by zero
    """
    assert len(i_poly_right) == len(j_poly_right), "Input arrays must have the same length."
    assert len(i_poly_right) >= 2, "Input arrays must have at least two elements."

    dx = i_poly_right[-2] - i_poly_right[-1]
    dy = j_poly_right[-2] - j_poly_right[-1]

    if np.isclose(dx, 0) and np.isclose(dy, 0):
        raise ValueError("dx and dy cannot both be zero for angle calculation.")

    radian_angle = np.arctan(dy / dx)
    horizontal_angle = np.degrees(radian_angle)

    if dx < 0:
        middle_angle = -horizontal_angle
    else:
        middle_angle = 270 - horizontal_angle

    middle_pixel_position = i_poly_right[-1]
    return middle_angle, middle_pixel_position
