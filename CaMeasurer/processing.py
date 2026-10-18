import  warnings
import  numpy           as      np
from    typing          import  Tuple
from    numpy.linalg    import  LinAlgError
from    numpy.typing    import  NDArray

def poly_fitting(i: NDArray[np.int64], 
                 j: NDArray[np.int64], 
                 polynomial_degree:int=3, 
                 line_space:int=100
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
    
    assert len(i) > 0 and len(j) > 0, "Input arrays for polynomial fitting are empty."
    assert len(i) == len(j), "Input arrays must have the same length."
    assert len(i) >= polynomial_degree + 1, f"Number of data points ({len(i)}) must be at least polynomial_degree + 1 ({polynomial_degree + 1})"

    if isinstance(i, list):
        i = np.array(i, dtype=np.int64)
        j = np.array(j, dtype=np.int64)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y_poly_equation = np.poly1d(np.polyfit(i, j, polynomial_degree))

    except LinAlgError as e:
        raise ValueError(f"Linear algebra error during polynomial fitting: {e}")

    # Generate interpolated points
    x_poly = np.linspace(i.min(), i.max(), line_space, dtype=np.float64)
    y_poly = y_poly_equation(x_poly)

    return x_poly, y_poly
    