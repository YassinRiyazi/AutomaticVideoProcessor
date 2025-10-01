import warnings
import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray

def poly_fitting(i: List[int], j: List[int], polynomial_degree:int=3, line_space:int=100
                 ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Fit a polynomial to data and generate interpolated points.

    Parameters:
        i (array-like): Input x-coordinates.
        j (array-like): Input y-coordinates.
        polynomial_degree (int): Degree of the polynomial to fit (default: 3).
        line_space (int): Number of points for interpolation (default: 100).

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Interpolated x and y values.

    Authors:
        - Yassin Riyazi (Improved polynomial fitting and interpolation)
        - Sajjad Shumaly
    """
    if len(i) == 0 or len(j) == 0:
        raise ValueError("Input arrays for polynomial fitting are empty.")
    if len(i) != len(j):
        raise ValueError("Input arrays must have the same length.")

    # Convert inputs to NumPy arrays
    i = np.asarray(i)
    j = np.asarray(j)

    # Input validation
    if len(i) < polynomial_degree + 1:
        raise ValueError(f"Number of data points ({len(i)}) must be at least polynomial_degree + 1 ({polynomial_degree + 1})")

    # Suppress warnings from polyfit (general suppression to avoid RankWarning issue)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        y_poly_equation = np.poly1d(np.polyfit(i, j, polynomial_degree))

    # Generate interpolated points
    x_poly = np.linspace(i.min(), i.max(), line_space, dtype=np.float64)
    y_poly = y_poly_equation(x_poly)

    return x_poly, y_poly