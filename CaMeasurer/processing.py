import warnings
import numpy as np

def poly_fitting(i, j, polynomial_degree=3, line_space=100) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a polynomial to data and generate interpolated points.

    Parameters:
        i (array-like): Input x-coordinates.
        j (array-like): Input y-coordinates.
        polynomial_degree (int): Degree of the polynomial to fit (default: 3).
        line_space (int): Number of points for interpolation (default: 100).

    Returns:
        tuple[np.ndarray, np.ndarray]: Interpolated x and y values.

    Authors:
        - Yassin Riyazi (Improved polynomial fitting and interpolation)
        - Sajjad Shumaly
    """
    # Convert inputs to NumPy arrays
    i = np.asarray(i)
    j = np.asarray(j)

    # Input validation
    if len(i) < polynomial_degree + 1:
        raise ValueError(f"Number of data points ({len(i)}) must be at least polynomial_degree + 1 ({polynomial_degree + 1})")

    # Suppress warnings from polyfit (general suppression to avoid RankWarning issue)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        y_polyequation = np.poly1d(np.polyfit(i, j, polynomial_degree))

    # Generate interpolated points
    x_poly = np.linspace(i.min(), i.max(), line_space, dtype=np.float64)
    y_poly = y_polyequation(x_poly)

    return x_poly, y_poly