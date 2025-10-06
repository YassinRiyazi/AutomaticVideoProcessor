"""
    Author:     Yassin Riyazi
    Date:       15-08-2025
    Purpose:    Process drop images to extract and sort edge points, then visualize them.

    TODO:
        - [05-09-2025] Order edge points based on distance from the leftmost point, then from the rightmost point.
        - [05-09-2025] Implement nearest-neighbor sorting algorithm for edge points. Double sort implementation.

    Learned:
        - Use npt.NDArray[np.float64] for type hinting numpy arrays, npt.NDArray[np.int8] does weird shit.
        for instance it returned NaN for distance calculation though the destination matrix was float64.

"""
import  os
import  sys
import  glob


import  cv2
import  numpy.typing     as  npt
import  numpy               as  np
import  matplotlib.pyplot   as  plt

if __name__ == "__main__":
    from edgeDetection import edge_extraction
else:
    from .edgeDetection import edge_extraction

import matplotlib
matplotlib.use('TkAgg')

def LoadImage(address:str):
    _image = cv2.imread(address)
    if _image is None:
        raise ValueError("Image not found or unable to load.")
    elif len(_image.shape) > 2:
        # raise ValueError("Invalid image format.")
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
    return _image

def _edgeSort(i_array: npt.NDArray[np.int_], j_array: npt.NDArray[np.int_],
              origin_x: int, origin_y: int = 0, Reverse: bool = False
              ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """
    Arrange points in a specific order.
    Args:   
        i_array (np.array): The x-coordinates of the points.
        j_array (np.array): The y-coordinates of the points.
        origin_x (int): The x-coordinate of the origin point.
        origin_y (int): The y-coordinate of the origin point.
    
    """
    # Vectorized Euclidean distance calculation
    distances = np.sqrt((i_array - origin_x)**2 + (j_array - origin_y)**2)

    sorted_indices = np.argsort(distances)
    
    if Reverse:
        sorted_indices = sorted_indices[::-1]

    return i_array[sorted_indices], j_array[sorted_indices]

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
def EdgePointSorter(_image: npt.NDArray[np.uint8]
                    ) -> npt.NDArray[np.int_]:
    """
    Arrange points in a specific order.
    Args:
        _image (np.array): The input image from which to extract and sort points.

    Returns:
        tuple[np.ndarray, np.ndarray]: The sorted x and y coordinates of the points.

    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/ContactAngle/DropDetection/doc/DropCoordinateSystem-output_Unordered.png" alt="Italian Trulli" style="width: 800px; height: auto;">
    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/ContactAngle/DropDetection/doc/DropCoordinateSystem-output_Ordered.png" alt="Italian Trulli" style="width: 800px; height: auto;">
    """
    _image = cv2.morphologyEx(_image, cv2.MORPH_CLOSE, kernel)
    _image = cv2.bitwise_not(_image)
    res = edge_extraction(_image, thr=10)
    i_array, j_array = res
    i_array, j_array = _edgeSort(i_array, j_array, origin_x=i_array.min())

    dataLength = len(i_array)
    X = np.zeros(shape=(dataLength, 2), dtype=np.float64) # When I changed the type from int to float64, the error was resolved. I had NaN inside Distance matrix before
    X[:, 0], X[:, 1] = i_array, j_array

    distances = np.zeros(shape=(dataLength, dataLength), dtype=np.float64)
    # Compute distances for upper triangle (i < j)
    for i in range(dataLength):
        for j in range(i + 1, dataLength):
            distances[i, j] = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            distances[j, i] = distances[i, j]  # Mirror to lower triangle
    

    # Find the starting point closest to (0,0)
    
    current_idx = np.argmin(j_array)  # Index of point closest to (0,0)

    # Initialize variables for nearest-neighbor path
    sorted_indices = []
    visited = np.zeros(dataLength, dtype=bool)
    sorted_indices.append(current_idx)
    visited[current_idx] = True

    # Greedy nearest-neighbor sorting
    for _ in range(dataLength - 1):
        # Get distances from the current point to all unvisited points
        dist_to_others = distances[current_idx].copy()
        dist_to_others[visited] = np.inf  # Ignore visited points
        next_idx = np.argmin(dist_to_others)  # Find closest unvisited point
        if np.isinf(dist_to_others[next_idx]):  # No unvisited points left
            break
        sorted_indices.append(next_idx)
        visited[next_idx] = True
        current_idx = next_idx

    sorted_indices = np.array(sorted_indices, dtype=np.int_)
    X[:, 0], X[:, 1] = i_array[sorted_indices], j_array[sorted_indices]
    return X

def EdgePointSorter2(_image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    _image = cv2.morphologyEx(_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cv2.bitwise_not(_image, _image)
    i_array, j_array = edge_extraction(_image, thr=10)
    i_array, j_array = _edgeSort(i_array, j_array, origin_x=i_array.min())

    X = np.column_stack((i_array, j_array)).astype(np.float64)

    # Start with the lowest y point
    current_idx = np.argmin(j_array)

    sorted_indices = [current_idx]
    visited = np.zeros(len(X), dtype=bool)
    visited[current_idx] = True

    for _ in range(len(X) - 1):
        # Compute distances only to unvisited points
        diff = X[~visited] - X[current_idx]
        dist_sq = np.einsum('ij,ij->i', diff, diff)  # fast squared distance
        next_rel = np.argmin(dist_sq)
        next_idx = np.flatnonzero(~visited)[next_rel]

        sorted_indices.append(next_idx)
        visited[next_idx] = True
        current_idx = next_idx

    return X[sorted_indices]
def Plotter(_image: np.ndarray,
            i_array: np.ndarray, j_array: np.ndarray,
            SaveAddress: str = "src/PyThon/ContactAngle/output.png") -> None:
    """
    Plot the image and the detected points.
    Args:
        _image: The input image.
        i_array: The x-coordinates of the detected points.
        j_array: The y-coordinates of the detected points.
        SaveAddress: The file path to save the output image.
    Returns:
        None: None
    """
    
    colors = np.linspace(0, 1, len(i_array))

    plt.figure(figsize=(10, 10))
    plt.imshow(_image[::-1, :], cmap='gray')

    plt.scatter(i_array, j_array, c=colors, cmap='RdBu', s=6)

    plt.gca().invert_yaxis()
    plt.axis('off')
    # plt.savefig(SaveAddress, bbox_inches='tight', pad_inches=0)
    # plt.close()
    plt.show()

import numba as nb
@nb.njit(fastmath=True, cache=True)
def greedy_sort(X):
    n = X.shape[0]
    visited = np.zeros(n, dtype=np.uint8)
    path = np.empty(n, dtype=np.int64)

    # start with lowest y
    current = np.argmin(X[:, 1])
    path[0] = current
    visited[current] = 1

    for k in range(1, n):
        best = -1
        best_dist = 1e20
        for j in range(n):
            if visited[j]:
                continue
            dx = X[current, 0] - X[j, 0]
            dy = X[current, 1] - X[j, 1]
            d2 = dx * dx + dy * dy
            if d2 < best_dist:
                best_dist = d2
                best = j
        path[k] = best
        visited[best] = 1
        current = best
    return path


def EdgePointSorter_Numba(_image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    """
        For <2k points, Numba greedy_sort is excellent (no dependency on scipy).

        For >5k points, KDTree usually wins.
    """
    _image = cv2.morphologyEx(_image, cv2.MORPH_CLOSE, kernel)
    _image = cv2.bitwise_not(_image)
    i_array, j_array = edge_extraction(_image, thr=10)
    i_array, j_array = _edgeSort(i_array, j_array, origin_x=i_array.min())

    X = np.column_stack((i_array, j_array)).astype(np.float64)
    order = greedy_sort(X)
    return X[order]

if __name__ == "__main__":
    from  DropDetection_Sum import detectionV2
    _image = LoadImage("/media/Dont/Teflon-AVP/280/S2-SNr2.1_D/T528_01_4.460000000000/SR_edge/frame_000002.png")
    x2, x1 = detectionV2(_image, scaleDownFactor=1, drop_width=300)
    # X = EdgePointSorter(_image[:-1,x1-5:x2+5])
    X = EdgePointSorter_Numba(_image[:-1,x1-9:x2+9])
    i_array, j_array = X[:, 0]-9, X[:, 1]




    plt.figure(figsize=(10, 10))
    plt.imshow(_image[::-1, :], cmap='gray')

    mid = len(i_array) #// 2
    colors = np.linspace(0, 1, mid)
    plt.scatter(i_array[:mid]+x1, j_array[:mid]+1, c=colors, cmap='RdBu', s=6)

    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()

    # Plotter(_image,
    #         i_array, j_array,
    #         SaveAddress=f"src/PyThon/ContactAngle/DropSamples/output_Ordered_frame_000001_fin.png")
