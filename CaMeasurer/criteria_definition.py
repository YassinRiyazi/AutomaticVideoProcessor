"""
    Author: Sajjad Shumaly
    Date: 01-07-2022
"""
import  warnings
import  numpy           as      np
from    typing          import  List, Tuple
from    numpy.typing    import  NDArray

def left_angle( i_poly_left:NDArray[np.float64], 
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
    dx=i_poly_left[tan_pixel_number]-i_poly_left[0]
    dy=j_poly_left[tan_pixel_number]-j_poly_left[0]
    gradian=np.arctan((dy)/(dx))
    horizontal_angle=gradian*180/np.pi
    left_angle=90-horizontal_angle
    left_pixel_position=j_poly_left[0]    
    return(left_angle,left_pixel_position)

def right_angle( i_poly_right:NDArray[np.float64], j_poly_right:NDArray[np.float64], tan_pixel_number:int=1
                ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate the receding angle and pixel position from polynomial fitted data.    
    args:
        i_poly_right (NDArray[np.float64]): x-coordinates of the right polynomial fitted data.
        j_poly_right (NDArray[np.float64]): y-coordinates of the right polynomial fitted data.
        tan_pixel_number (int): Index of the pixel used for tangent calculation (default: 1).
    returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the receding angle and the pixel position
    
    Author:
        - Sajjad Shumaly
    """
    dx=i_poly_right[tan_pixel_number]-i_poly_right[0]
    dy=j_poly_right[tan_pixel_number]-j_poly_right[0]
    gradian=np.arctan((dy)/(dx))
    horizontal_angle=gradian*180/np.pi
    right_angle=90+horizontal_angle
    right_pixel_position=j_poly_right[0]
    return(right_angle,right_pixel_position)

def middle_angle( i_poly_right:NDArray[np.float64], 
                 j_poly_right:NDArray[np.float64]
                 ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate the middle angle and pixel position from polynomial fitted data.
    args:
        i_poly_right (NDArray[np.float64]): x-coordinates of the right polynomial fitted data.
        j_poly_right (NDArray[np.float64]): y-coordinates of the right polynomial fitted data.
    returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the middle angle and the pixel position.

    Author:
        - Sajjad Shumaly
    """
    dx=i_poly_right[-2]-i_poly_right[-1]
    dy=j_poly_right[-2]-j_poly_right[-1]
    gradian=np.arctan((dy)/(dx))
    horizontal_angle=gradian*180/np.pi

    if dx<0:
        middle_angle=-horizontal_angle

    if dx>0:
        middle_angle=180+90-horizontal_angle

    middle_pixel_position=i_poly_right[-1]
    return(middle_angle,middle_pixel_position)


