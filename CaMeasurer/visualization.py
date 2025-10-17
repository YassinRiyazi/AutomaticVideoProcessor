"""
    Editor: Yassin Riyazi
    Main author: Sajjad Shumaly
    Date: 01-07-2025
    Description: Visualization functions for the CaMeasurer module.
"""
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Tuple, List
if __name__ == "__main__":

    import  cv2
    from    criteria_definition import left_angle, right_angle, middle_angle
    from    processing import poly_fitting
    import matplotlib.pyplot as plt 

    import os,sys
    # print()
    sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + '/../BaseUtils/Detection'))
    from edgeDetection import *
else:
    import os,sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from criteria_definition import left_angle, right_angle, middle_angle
    from processing import poly_fitting
    # For calling Matplotlib headlessly
    import matplotlib
    matplotlib.use('Agg')


plt.rcParams["figure.figsize"] = (20,15)


# def horizontal_center(i_list: NDArray[np.int64], j_list: NDArray[np.int64],
#                       intersection_margin: int = 4) -> Tuple:
#     """
#     Calculate the horizontal center of a shape defined by i_list and j_list coordinates.
#     The intersection margin is a margin from the top edge to prevent errors in special cases.

#     Args:
#         i_list (NDArray[np.int64]): List of i-coordinates (horizontal).
#         j_list (NDArray[np.int64]): List of j-coordinates (vertical).
#         intersection_margin (int): Margin to avoid edge cases.

#     Returns:
#         tuple: (horizontal_center, mean_list, j_location_list)

#     Authors:
#         - Yassin Riyazi (Improved horizontal center calculation)
#         - Sajjad Shumaly
#     """

#     # Split into left and right based on the vertical middle
#     i_middle_vertical = int(np.mean(i_list[j_list == j_list.max()]))
#     left_mask = i_list <= i_middle_vertical
#     i_left, j_left = i_list[left_mask], j_list[left_mask]
#     i_right, j_right = i_list[~left_mask], j_list[~left_mask]

#     def calculate_extremes(j_vals, i_vals, is_left, j_ref):
#         """
#         Helper function to find the extreme (leftmost/rightmost) pixel for a given j-coordinate.
#         """
#         i_out = []
#         for j in j_vals:
#             i_pixels = i_vals[j_ref == j]
#             if i_pixels.size > 0:
#                 i_value = i_pixels.max() if is_left else i_pixels.min()
#                 # Adjust for continuous pixels
#                 for i in range(len(i_pixels)):
#                     target = i_pixels.max() - i if is_left else i_pixels.min() + i
#                     if target not in i_pixels:
#                         i_out.append(target - 1 if is_left else target + 1)
#                         break
#                 else:
#                     i_out.append(i_value)
#             else:
#                 i_out.append(np.nan)
#         return np.array(i_out)

#     def compute_weighted_mean(j_vals, i_left_vals, i_right_vals):
#         """
#         Helper function to compute weighted mean for horizontal center calculations.
#         """
#         mean_list, j_loc_list, sum_weighted, total_weight = [], [], 0, 0
#         for idx, j in enumerate(j_vals):
#             left_pix, right_pix = i_left_vals[idx], i_right_vals[idx]
#             if not (np.isnan(left_pix) or np.isnan(right_pix)):
#                 weight = abs(right_pix - left_pix)
#                 mean = np.mean([right_pix, left_pix])
#                 mean_list.append(mean)
#                 j_loc_list.append(j)
#                 sum_weighted += weight * mean
#                 total_weight += weight
#         return mean_list, j_loc_list, sum_weighted, total_weight

#     # Calculate extremes for left and right sides
#     j_range = range(max(j_left) - intersection_margin + 1)
#     i_left_ext = calculate_extremes(j_range, i_left, True, j_left)
#     i_right_ext = calculate_extremes(j_range, i_right, False, j_right)

#     # Compute weighted mean
#     mean_list, j_location_list, sum_all, total_weight = compute_weighted_mean(j_range, i_left_ext, i_right_ext)

#     # Calculate horizontal center
#     horizontal_center = sum_all / total_weight if total_weight != 0 else 0

#     return horizontal_center, mean_list, j_location_list

def horizontal_center(i_list: NDArray[np.int64], j_list: NDArray[np.int64],
                      intersection_margin: int = 4) -> Tuple[float, NDArray, NDArray]:
    """
    Calculate the horizontal center for a shape given lists of (i, j) pixel coordinates.
    This preserves the original "continuous pixel" adjustment logic.

    Args:
        i_list: array of i-coordinates (columns).
        j_list: array of j-coordinates (rows).
        intersection_margin: margin from the top to avoid edge rows.

    Returns:
        horizontal_center (float), mean_list (np.ndarray of means per valid row),
        j_location_list (np.ndarray of corresponding j values).
    """
    # ensure arrays
    i = np.asarray(i_list)
    j = np.asarray(j_list)

    if i.size == 0 or j.size == 0:
        return 0.0, np.array([], dtype=float), np.array([], dtype=int)

    # compute vertical middle using the topmost row (original logic)
    top_j = j.max()
    top_mask = (j == top_j)
    if not top_mask.any():
        return 0.0, np.array([], dtype=float), np.array([], dtype=int)
    i_middle_vertical = int(np.mean(i[top_mask]))

    # split left / right once
    left_mask = i <= i_middle_vertical
    i_left, j_left = i[left_mask], j[left_mask]
    i_right, j_right = i[~left_mask], j[~left_mask]

    # helper: group i-values by j for a side (returns dict: j_value -> sorted unique i-values)
    def group_by_j(i_vals, j_vals):
        if i_vals.size == 0:
            return {}
        order = np.argsort(j_vals)
        j_sorted = j_vals[order]
        i_sorted = i_vals[order]
        unique_js, idx_start, counts = np.unique(j_sorted, return_index=True, return_counts=True)
        idx_end = idx_start + counts
        groups = {int(jv): np.unique(i_sorted[s:e]) for jv, s, e in zip(unique_js, idx_start, idx_end)}
        return groups

    groups_left = group_by_j(i_left, j_left)
    groups_right = group_by_j(i_right, j_right)

    # j-range to iterate (preserve original range logic)
    if j_left.size == 0:
        return 0.0, np.array([], dtype=float), np.array([], dtype=int)
    max_j_left = int(j_left.max())
    end = max_j_left - intersection_margin + 1
    if end <= 0:
        return 0.0, np.array([], dtype=float), np.array([], dtype=int)
    j_range = np.arange(end, dtype=int)

    # compute extremes for each j in j_range
    i_left_ext = np.full(j_range.shape, np.nan, dtype=float)
    i_right_ext = np.full(j_range.shape, np.nan, dtype=float)

    for idx, jj in enumerate(j_range):
        # left side (find "extreme" as original function does)
        arr = groups_left.get(int(jj))
        if arr is not None and arr.size > 0:
            # original logic checks targets = max - i for i in range(len(i_pixels))
            m = int(arr.max())
            expected = m - np.arange(arr.size, dtype=int)
            present = np.isin(expected, arr)
            # find first missing in expected sequence
            if not present.all():
                first_missing_idx = int(np.where(~present)[0][0])
                i_left_ext[idx] = expected[first_missing_idx] - 1
            else:
                i_left_ext[idx] = m

        # right side
        arr = groups_right.get(int(jj))
        if arr is not None and arr.size > 0:
            mmin = int(arr.min())
            expected = mmin + np.arange(arr.size, dtype=int)
            present = np.isin(expected, arr)
            if not present.all():
                first_missing_idx = int(np.where(~present)[0][0])
                i_right_ext[idx] = expected[first_missing_idx] + 1
            else:
                i_right_ext[idx] = arr.min()

    # compute weighted mean vectorized
    valid_mask = ~np.isnan(i_left_ext) & ~np.isnan(i_right_ext)
    if not valid_mask.any():
        return 0.0, np.array([], dtype=float), np.array([], dtype=int)

    left_vals = i_left_ext[valid_mask]
    right_vals = i_right_ext[valid_mask]
    widths = np.abs(right_vals - left_vals)
    means = (right_vals + left_vals) / 2.0

    total_weight = widths.sum()
    if total_weight == 0:
        horizontal_center = 0.0
    else:
        horizontal_center = (widths * means).sum() / total_weight

    j_locations = j_range[valid_mask]

    return float(horizontal_center), means, j_locations



def vertical_center(i_list: NDArray[np.float64], j_list: NDArray[np.float64],
                    intersection_margin: int = 4) -> Tuple[float, NDArray, NDArray]:
    """
    Calculate the vertical center for a shape given lists of (i, j) pixel coordinates.

    This preserves the original "continuous pixel" adjustment logic.

    Args:
        i_list: array of i-coordinates (horizontal).
        j_list: array of j-coordinates (vertical).
        intersection_margin: margin from the left/right edges to avoid edge rows.

    Returns:
        vertical_center (float),
        i_location_list (np.ndarray[int]): i positions that contributed,
        mean_list (np.ndarray[float]): corresponding vertical means per i.
    """
    # Ensure numpy arrays
    i = np.asarray(i_list)
    j = np.asarray(j_list)

    if i.size == 0 or j.size == 0:
        return 0.0, np.array([], dtype=int), np.array([], dtype=float)

    # compute vertical middle using the topmost row (original logic)
    top_j = j.max()
    top_mask = (j == top_j)
    if not top_mask.any():
        return 0.0, np.array([], dtype=int), np.array([], dtype=float)
    i_middle_vertical = int(np.mean(i[top_mask]))

    # split left / right once
    left_mask = i <= i_middle_vertical
    i_left, j_left = i[left_mask], j[left_mask]
    i_right, j_right = i[~left_mask], j[~left_mask]

    # if either side empty, still proceed but many ranges will be empty
    def group_by_i(i_vals, j_vals):
        """Return dict: int(i) -> sorted unique j-values (ints)."""
        if i_vals.size == 0:
            return {}
        order = np.argsort(i_vals)
        i_s = i_vals[order].astype(int)
        j_s = j_vals[order].astype(int)
        uniq_i, idx_start, counts = np.unique(i_s, return_index=True, return_counts=True)
        idx_end = idx_start + counts
        return {int(iv): np.unique(j_s[s:e]) for iv, s, e in zip(uniq_i, idx_start, idx_end)}

    groups_left = group_by_i(i_left, j_left)
    groups_right = group_by_i(i_right, j_right)

    # For splitting up/down we need j_middles computed from extreme i positions
    # Left split
    if i_left.size == 0:
        i_left_min = i_left_max = None
    else:
        # i_left.min() used when selecting j positions
        i_left_min = int(i_left.min())
        i_left_max = int(i_left.max())
    if i_right.size == 0:
        i_right_min = i_right_max = None
    else:
        i_right_min = int(i_right.min())
        i_right_max = int(i_right.max())

    # compute j_middle_left: mean of j where i == min(i_left)
    if i_left.size == 0 or i_left_min is None:
        j_middle_left = None
    else:
        jl = j_left[i_left.astype(int) == i_left_min]
        j_middle_left = int(np.mean(jl)) if jl.size > 0 else None

    # compute j_middle_right: mean of j where i == max(i_right)
    if i_right.size == 0 or i_right_max is None:
        j_middle_right = None
    else:
        jr = j_right[i_right.astype(int) == i_right_max]
        j_middle_right = int(np.mean(jr)) if jr.size > 0 else None

    # Now split left groups into up/down by comparing j to j_middle_left
    # We'll build groups for left_up, left_down, right_up, right_down as dicts keyed by i
    groups_left_up = {}
    groups_left_down = {}
    if j_middle_left is not None:
        for ii, arr in groups_left.items():
            if arr.size == 0:
                continue
            # arr contains j-values; split by <= j_middle_left => down, else up
            down = arr[arr <= j_middle_left]
            up = arr[arr > j_middle_left]
            if down.size > 0:
                groups_left_down[ii] = down
            if up.size > 0:
                groups_left_up[ii] = up

    groups_right_up = {}
    groups_right_down = {}
    if j_middle_right is not None:
        for ii, arr in groups_right.items():
            if arr.size == 0:
                continue
            down = arr[arr <= j_middle_right]
            up = arr[arr > j_middle_right]
            if down.size > 0:
                groups_right_down[ii] = down
            if up.size > 0:
                groups_right_up[ii] = up

    # helper to compute extreme j for each integer i in a requested range
    def calculate_extremes_i(i_range: np.ndarray, groups: dict, is_upper: bool) -> np.ndarray:
        """
        For each i in i_range, return the extreme j (uppermost if is_upper True -> min,
        else lowermost -> max). Apply same continuity adjustment as original code.
        """
        out = np.full(i_range.shape, np.nan, dtype=float)
        for idx, ii in enumerate(i_range):
            arr = groups.get(int(ii))
            if arr is None or arr.size == 0:
                continue
            arr = np.sort(np.unique(arr)).astype(int)
            if is_upper:
                m = int(arr.min())
                count = arr.size
                expected = m + np.arange(count, dtype=int)
                present = np.isin(expected, arr)
                if not present.all():
                    first_missing = int(np.where(~present)[0][0])
                    out[idx] = expected[first_missing] - 1
                else:
                    out[idx] = m
            else:
                m = int(arr.max())
                count = arr.size
                expected = m - np.arange(count, dtype=int)
                present = np.isin(expected, arr)
                if not present.all():
                    first_missing = int(np.where(~present)[0][0])
                    out[idx] = expected[first_missing] + 1
                else:
                    out[idx] = m
        return out

    # helper to compute weighted mean over ranges
    def compute_weighted(i_range: np.ndarray, up_arr: np.ndarray, down_arr: np.ndarray, is_simple: bool = False):
        """
        up_arr, down_arr: arrays of same shape as i_range containing j values or nan.
        if is_simple True, down_arr is treated as zeros (used in 'simple' parts).
        Returns: mean_list (np.ndarray), i_list (np.ndarray), sum_weighted (float), total_weight (float)
        """
        if i_range.size == 0:
            return np.array([], dtype=float), np.array([], dtype=int), 0.0, 0.0

        if is_simple:
            valid = ~np.isnan(up_arr)
            up_vals = up_arr[valid]
            down_vals = np.zeros_like(up_vals)
            i_vals = i_range[valid]
        else:
            valid = ~np.isnan(up_arr) & ~np.isnan(down_arr)
            up_vals = up_arr[valid]
            down_vals = down_arr[valid]
            i_vals = i_range[valid]

        if i_vals.size == 0:
            return np.array([], dtype=float), np.array([], dtype=int), 0.0, 0.0

        weights = np.abs(up_vals - down_vals)
        means = (up_vals + down_vals) / 2.0
        total_weight = float(weights.sum())
        sum_weighted = float((weights * means).sum())
        return means, i_vals.astype(int), sum_weighted, total_weight

    # Build ranges (matching original range(...) behavior: start inclusive, stop exclusive)
    # Left intersection and simple ranges
    left_down_keys = np.array(sorted(groups_left_down.keys()), dtype=int) if groups_left_down else np.array([], dtype=int)
    left_up_keys = np.array(sorted(groups_left_up.keys()), dtype=int) if groups_left_up else np.array([], dtype=int)

    # i_left_range_inter = range(min(i_left_down) + intersection_margin, max(i_left_down))
    if left_down_keys.size > 0:
        start = int(left_down_keys.min()) + intersection_margin
        stop = int(left_down_keys.max())
        i_left_range_inter = np.arange(start, stop, dtype=int) if stop > start else np.array([], dtype=int)
    else:
        i_left_range_inter = np.array([], dtype=int)

    # i_left_range_simple = range(max(i_left_down), max(i_left_up))
    if left_down_keys.size > 0 and left_up_keys.size > 0:
        start = int(left_down_keys.max())
        stop = int(left_up_keys.max())
        i_left_range_simple = np.arange(start, stop, dtype=int) if stop > start else np.array([], dtype=int)
    else:
        i_left_range_simple = np.array([], dtype=int)

    # Right ranges
    right_down_keys = np.array(sorted(groups_right_down.keys()), dtype=int) if groups_right_down else np.array([], dtype=int)
    right_up_keys = np.array(sorted(groups_right_up.keys()), dtype=int) if groups_right_up else np.array([], dtype=int)

    # i_right_range_inter = range(min(i_right_down), max(i_right_down) - intersection_margin)
    if right_down_keys.size > 0:
        start = int(right_down_keys.min())
        stop = int(right_down_keys.max()) - intersection_margin
        i_right_range_inter = np.arange(start, stop, dtype=int) if stop > start else np.array([], dtype=int)
    else:
        i_right_range_inter = np.array([], dtype=int)

    # i_right_range_simple = range(min(i_right_up), min(i_right_down))
    if right_up_keys.size > 0 and right_down_keys.size > 0:
        start = int(right_up_keys.min())
        stop = int(right_down_keys.min())
        i_right_range_simple = np.arange(start, stop, dtype=int) if stop > start else np.array([], dtype=int)
    else:
        i_right_range_simple = np.array([], dtype=int)

    # Calculate extremes
    j_left_down_ext = calculate_extremes_i(i_left_range_inter, groups_left_down, is_upper=False) if i_left_range_inter.size > 0 else np.array([], dtype=float)
    j_left_up_ext   = calculate_extremes_i(i_left_range_inter, groups_left_up,   is_upper=True)  if i_left_range_inter.size > 0 else np.array([], dtype=float)

    j_left_up_simple = calculate_extremes_i(i_left_range_simple, groups_left_up, is_upper=True) if i_left_range_simple.size > 0 else np.array([], dtype=float)

    j_right_down_ext = calculate_extremes_i(i_right_range_inter, groups_right_down, is_upper=False) if i_right_range_inter.size > 0 else np.array([], dtype=float)
    j_right_up_ext   = calculate_extremes_i(i_right_range_inter, groups_right_up,   is_upper=True)  if i_right_range_inter.size > 0 else np.array([], dtype=float)

    j_right_up_simple = calculate_extremes_i(i_right_range_simple, groups_right_up, is_upper=True) if i_right_range_simple.size > 0 else np.array([], dtype=float)

    # Compute weighted means
    mean_left_inter, i_loc_left_inter, sum_left_inter, weight_left_inter = compute_weighted(i_left_range_inter, j_left_up_ext, j_left_down_ext, False)
    mean_left_simple, i_loc_left_simple, sum_left_simple, weight_left_simple = compute_weighted(i_left_range_simple, j_left_up_simple, np.zeros_like(j_left_up_simple), True)

    mean_right_inter, i_loc_right_inter, sum_right_inter, weight_right_inter = compute_weighted(i_right_range_inter, j_right_up_ext, j_right_down_ext, False)
    mean_right_simple, i_loc_right_simple, sum_right_simple, weight_right_simple = compute_weighted(i_right_range_simple, j_right_up_simple, np.zeros_like(j_right_up_simple), True)

    # Combine results (preserving original ordering of concatenation)
    sum_all = sum_left_inter + sum_left_simple + sum_right_inter + sum_right_simple
    total_weight = weight_left_inter + weight_left_simple + weight_right_inter + weight_right_simple
    vertical_center = float(sum_all / total_weight) if total_weight != 0 else 0.0

    # Concatenate i_location_list and mean_list in same order as original:
    # i_loc_right_simple + i_loc_right_inter + i_loc_left_simple + i_loc_left_inter
    i_location_list = np.concatenate([
        i_loc_right_simple if i_loc_right_simple.size > 0 else np.array([], dtype=int),
        i_loc_right_inter if i_loc_right_inter.size > 0 else np.array([], dtype=int),
        i_loc_left_simple if i_loc_left_simple.size > 0 else np.array([], dtype=int),
        i_loc_left_inter if i_loc_left_inter.size > 0 else np.array([], dtype=int),
    ]) if any(arr.size > 0 for arr in [i_loc_right_simple, i_loc_right_inter, i_loc_left_simple, i_loc_left_inter]) else np.array([], dtype=int)

    mean_list = np.concatenate([
        mean_right_simple if mean_right_simple.size > 0 else np.array([], dtype=float),
        mean_right_inter if mean_right_inter.size > 0 else np.array([], dtype=float),
        mean_left_simple if mean_left_simple.size > 0 else np.array([], dtype=float),
        mean_left_inter if mean_left_inter.size > 0 else np.array([], dtype=float),
    ]) if any(arr.size > 0 for arr in [mean_right_simple, mean_right_inter, mean_left_simple, mean_left_inter]) else np.array([], dtype=float)

    return vertical_center, i_location_list, mean_list

# def vertical_center(i_list: NDArray[np.float64], j_list: NDArray[np.float64],
#                     intersection_margin: int = 4):
#     """
#     Calculate the vertical center of a shape defined by i_list and j_list coordinates.
#     The intersection margin is a margin from the left side to prevent errors in special cases.

#     Args:
#         i_list (list): List of i-coordinates (horizontal).
#         j_list (list): List of j-coordinates (vertical).
#         intersection_margin (int): Margin to avoid edge cases.

#     Returns:
#         tuple: (vertical_center, i_location_list, mean_list)

#     Authors:
#         - Yassin Riyazi (Improved vertical center calculation)
#         - Sajjad Shumaly
#     """
#     # Convert inputs to numpy arrays
#     i_list, j_list = np.array(i_list), np.array(j_list)

#     # Split into left and right based on the vertical middle
#     i_middle_vertical = int(np.mean(i_list[j_list == j_list.max()]))
#     left_mask = i_list <= i_middle_vertical
#     i_left, j_left = i_list[left_mask], j_list[left_mask]
#     i_right, j_right = i_list[~left_mask], j_list[~left_mask]

#     # Split left into up and down
#     j_middle_left = int(np.mean(j_left[i_left == i_left.min()]))
#     left_down_mask = j_left <= j_middle_left
#     i_left_down, j_left_down = i_left[left_down_mask], j_left[left_down_mask]
#     i_left_up, j_left_up = i_left[~left_down_mask], j_left[~left_down_mask]

#     # Split right into up and down
#     j_middle_right = int(np.mean(j_right[i_right == i_right.max()]))
#     right_down_mask = j_right <= j_middle_right
#     i_right_down, j_right_down = i_right[right_down_mask], j_right[right_down_mask]
#     i_right_up, j_right_up = i_right[~right_down_mask], j_right[~right_down_mask]

#     def calculate_extremes(i_vals: range, j_vals: NDArray[np.int64],
#                            is_upper: bool, i_ref: NDArray[np.int64]) -> NDArray[np.float64]:
#         """
#         Helper function to find the extreme (uppermost/lowermost) pixel for a given i-coordinate.
#         """
#         j_out: List[np.int64] = []
#         for i in i_vals:
#             j_pixels = j_vals[i_ref == i]
#             if j_pixels.size > 0:
#                 j_value = j_pixels.min() if is_upper else j_pixels.max()
#                 # Adjust for continuous pixels
#                 for j in range(len(j_pixels)):
#                     target = j_pixels.min() + j if is_upper else j_pixels.max() - j
#                     if target not in j_pixels:
#                         j_out.append(target - 1 if is_upper else target + 1)
#                         break
#                 else:
#                     j_out.append(j_value)
#             else:
#                 j_out.append(np.nan)
#         return np.array(j_out)

#     def compute_weighted_mean(i_vals:range,
#                               j_up:NDArray[np.float64], j_down:NDArray[np.float64], 
#                               i_range:range,
#                               is_simple:bool=False
#                               )-> Tuple[List[np.float64], List[np.int64], float, float]:
#         """
#         Helper function to compute weighted mean for intersection or simple calculations.
#         """
#         mean_list, i_loc_list, sum_weighted, total_weight = [], [], 0, 0
#         for i in i_range:
#             up_pix, down_pix = j_up[list(i_vals).index(i)], j_down[list(i_vals).index(i)]
#             if not (np.isnan(up_pix) or np.isnan(down_pix)):
#                 weight = abs(up_pix - (0 if is_simple else down_pix))
#                 mean = np.mean([up_pix, 0 if is_simple else down_pix])
#                 mean_list.append(mean)
#                 i_loc_list.append(i)
#                 sum_weighted += weight * mean
#                 total_weight += weight
#         return mean_list, i_loc_list, sum_weighted, total_weight

#     # Left side calculations
#     i_left_range_inter = range(min(i_left_down) + intersection_margin, max(i_left_down))
#     i_left_range_simple = range(max(i_left_down), max(i_left_up))
    
#     j_left_down_ext = calculate_extremes(i_left_range_inter, j_left_down, False, i_left_down)
#     j_left_up_ext = calculate_extremes(i_left_range_inter, j_left_up, True, i_left_up)
#     mean_left_inter, i_loc_left_inter, sum_left_inter, weight_left_inter = compute_weighted_mean(
#         i_left_range_inter, j_left_up_ext, j_left_down_ext, i_left_range_inter
#     )
    
#     j_left_up_simple = calculate_extremes(i_left_range_simple, j_left_up, True, i_left_up)
#     mean_left_simple, i_loc_left_simple, sum_left_simple, weight_left_simple = compute_weighted_mean(
#         i_left_range_simple, j_left_up_simple, np.zeros_like(j_left_up_simple), i_left_range_simple, True
#     )

#     # Right side calculations
#     i_right_range_inter = range(min(i_right_down), max(i_right_down) - intersection_margin)
#     i_right_range_simple = range(min(i_right_up), min(i_right_down))
    
#     j_right_down_ext = calculate_extremes(i_right_range_inter, j_right_down, False, i_right_down)
#     j_right_up_ext = calculate_extremes(i_right_range_inter, j_right_up, True, i_right_up)
#     mean_right_inter, i_loc_right_inter, sum_right_inter, weight_right_inter = compute_weighted_mean(
#         i_right_range_inter, j_right_up_ext, j_right_down_ext, i_right_range_inter
#     )
    
#     j_right_up_simple = calculate_extremes(i_right_range_simple, j_right_up, True, i_right_up)
#     mean_right_simple, i_loc_right_simple, sum_right_simple, weight_right_simple = compute_weighted_mean(
#         i_right_range_simple, j_right_up_simple, np.zeros_like(j_right_up_simple), i_right_range_simple, True
#     )

#     # Combine results
#     sum_all = sum_left_inter + sum_left_simple + sum_right_inter + sum_right_simple
#     total_weight = weight_left_inter + weight_left_simple + weight_right_inter + weight_right_simple
#     vertical_center = sum_all / total_weight if total_weight != 0 else 0
    
#     i_location_list = i_loc_right_simple + i_loc_right_inter + i_loc_left_simple + i_loc_left_inter
#     mean_list = mean_right_simple + mean_right_inter + mean_left_simple + mean_left_inter

#     return vertical_center, i_location_list, mean_list

class plotDrop:
    def __init__(self, save_address:str ,
                 dpi:int=100,
                 cm_on_pixel:float=5/1280):
        """
        Class for visualizing the contact angle measurement results and saving the figure.
        """
        self.font_size=14
        upscale_factor=3
        self.conversion_factor=cm_on_pixel/upscale_factor

        self.save_address = save_address
        self.dpi = dpi

        self.fig, self.ax = plt.subplots(figsize=(15, 10),dpi=dpi)  # type: ignore
        self.ax.clear()
    
    def DropShape(self,i_list:NDArray[np.int64], j_list:NDArray[np.int64])->None:
        # Drop shape
        self.ax.plot(i_list, j_list, '.', color='black') # type: ignore

    def Contact_angle_edge(self,
                           i_left:NDArray[np.int64], j_left:NDArray[np.int64],
                           i_right:NDArray[np.int64], j_right:NDArray[np.int64])->None:
        self.ax.plot(i_left,     j_left,         '.', color='red', markersize=12) # type: ignore
        self.ax.plot(i_right,    j_right,        '.', color='red', markersize=12) # type: ignore

    def Poly_fit(self,
                 j_poly_left:NDArray[np.float64],  i_poly_left:NDArray[np.float64],
                 j_poly_right:NDArray[np.float64], i_poly_right:NDArray[np.float64])->None:
        self.ax.plot(i_poly_left,        j_poly_left, '--', color='yellow', linewidth=4) # type: ignore
        self.ax.plot(i_poly_right,       j_poly_right, '--', color='yellow', linewidth=4) # type: ignore

    def Left_angle(self,
                   left_angle_degree:float, m:float,
                   i_poly_left:NDArray[np.float64], j_poly_left:NDArray[np.float64],
                   )->None:
        ax = self.ax
        ax.plot([i_poly_left[0]+20, i_poly_left[0]], [j_poly_left[0], j_poly_left[0]], linewidth=3, color='blue') # type: ignore
        ax.plot([i_poly_left[0], i_poly_left[0] + (1/m) * j_poly_left[20]],[j_poly_left[0], j_poly_left[20]], linewidth=3, color='blue') # type: ignore
        ax.text(i_poly_left[0], j_poly_left[0] - 12, 'Advancing=' + str(round(left_angle_degree, 2)), color="blue", fontsize=self.font_size) # type: ignore

    def Right_angle(self,
                    right_angle_degree:float, m:float,
                    i_poly_right:NDArray[np.float64], j_poly_right:NDArray[np.float64],
                    )->None:
        ax = self.ax
        ax.plot([i_poly_right[0]-20, i_poly_right[0]], [j_poly_right[0], j_poly_right[0]], linewidth=3, color='blue') # type: ignore
        ax.plot([i_poly_right[0], i_poly_right[0] - (1/m) * j_poly_right[20]], [j_poly_right[0], j_poly_right[20]], linewidth=3, color='blue') # type: ignore
        ax.text(i_poly_right[0] - 65, j_poly_right[0] - 12, 'Receding=' + str(round(right_angle_degree, 2)), color="blue", fontsize=self.font_size) # type: ignore
    
    def Contact_line(self,
                     x_cropped:int,
                     j_poly_right:NDArray[np.float64], j_poly_left:NDArray[np.float64],
                     left_angle_point:float, right_angle_point:float,
                     contact_line_length:np.float64
                     )->None:
        ax = self.ax
        ax.plot([(x_cropped * 3) + np.array(left_angle_point), (x_cropped * 3) + np.array(right_angle_point)], [0, 0], '--', linewidth=1, color='red') # type: ignore
        ax.text(((x_cropped * 3) + np.array(right_angle_point) + (x_cropped * 3) + np.array(left_angle_point)) / 2 - 60, j_poly_right[0] - 12, 'Contact line length=' + str(round(contact_line_length, 3)) + ' cm', color="red", fontsize=self.font_size) # type: ignore
            
    def Center(self,
               h_center:float,
               j_list:NDArray[np.int64], i_list:NDArray[np.int64],
               drop_height:np.float64,v_center:np.float64
               )->None:
        ax = self.ax
        i_text_horizontal = (j_list[i_list == int(h_center)][0] + v_center) / 2
        ax.plot([h_center, h_center], [min(j_list), j_list[i_list == int(h_center)][0]], '--', color='green') # type: ignore
        ax.text(h_center + 5, i_text_horizontal, str(round(drop_height, 3)) + ' cm', color="green", fontsize=self.font_size) # type: ignore

    def Middle_line(self,
                    i_middle_line:NDArray[np.float64], j_middle_line:NDArray[np.float64],
                    middle_angle_degree:Tuple[np.float64,np.float64],
                    i2_middle_line:np.float64
                    )->None:
        ax = self.ax
        ax.plot([i_middle_line[-1], i2_middle_line], [0, j_middle_line[i_middle_line == i2_middle_line][0]], '-', color='black') # type: ignore
        ax.text(i2_middle_line - 35, j_middle_line[i_middle_line == i2_middle_line][0] - 20, 'Angle=' + str(round(middle_angle_degree[0], 2)), color="black", fontsize=self.font_size) # type: ignore

    def Vertical_center(self,i_list:NDArray[np.int64], j_list:NDArray[np.int64],
                        v_center:np.float64, drop_length:np.float64, i_text_vertical:np.float64
                        ) ->None:
        ax = self.ax
        ax.plot([min(i_list[j_list == int(v_center)]), max(i_list[j_list == int(v_center)])], [v_center, v_center], '--', color='green') # type: ignore
        ax.text(i_text_vertical, v_center + 5, str(round(drop_length, 3)) + ' cm', color="green", fontsize=self.font_size) # type: ignore

    def Center_point(self,
                     h_center:int, v_center:int
                     )->None:
        ax = self.ax
        ax.plot(h_center, v_center, '.', color='blue', markersize=14) # type: ignore
        ax.text(h_center + 5, v_center + 5, 'Center= [x=' + str(round(h_center, 3)) + ' mm, y=' + str(round(v_center, 3)) + ' mm]', color="blue", fontsize=self.font_size) # type: ignore

    def Save(self)->None:
        ax = self.ax
         # ax.axis('equal')
        ax.set_ylim(-30, 300)  # Set y limit as requested
        ax.tick_params(axis='both', labelsize=20) # type: ignore
        plt.tight_layout()
        self.fig.savefig(self.save_address,dpi=self.dpi) # type: ignore
        plt.close(self.fig) # type: ignore

def visualize(save_address:str , 
              i_list:NDArray[np.int64],                 j_list:NDArray[np.int64],
              i_left:NDArray[np.int64],                 j_left:NDArray[np.int64],
              i_right:NDArray[np.int64],                j_right:NDArray[np.int64],
              j_poly_left:NDArray[np.float64],          i_poly_left:NDArray[np.float64],
              j_poly_right:NDArray[np.float64],         i_poly_right:NDArray[np.float64],
              x_cropped:int,
              i_poly_left_rotated:NDArray[np.float64],  j_poly_left_rotated:NDArray[np.float64], 
              i_poly_right_rotated:NDArray[np.float64], j_poly_right_rotated:NDArray[np.float64],
              cm_on_pixel:float=5/1280,
              middle_line_switch:int=0,
              dpi:int=100,
              frame_width:int=1248):
    """
    Visualize the contact angle measurement results and save the figure.

    Author:
        - Sajjad Shumaly
    """
    _locPlotter = plotDrop(save_address=save_address, dpi=dpi, cm_on_pixel=cm_on_pixel)

    _locPlotter.DropShape(i_list, j_list)
    _locPlotter.Contact_angle_edge(i_left, j_left, i_right, j_right)
    _locPlotter.Poly_fit(j_poly_left, i_poly_left, j_poly_right, i_poly_right)


    # Left angle
    left_angle_degree, left_angle_point = left_angle(i_poly_left_rotated, j_poly_left_rotated, 1)
    left_angle_radian = np.deg2rad(left_angle_degree)
    m = np.tan(left_angle_radian)
    _locPlotter.Left_angle(left_angle_degree, m, i_poly_left, j_poly_left)

    # Right angle
    right_angle_degree, right_angle_point = right_angle(i_poly_right_rotated, j_poly_right_rotated, 1)
    right_angle_radian = np.deg2rad(right_angle_degree)
    m = np.tan(right_angle_radian)
    _locPlotter.Right_angle(right_angle_degree, m, i_poly_right, j_poly_right)

    # Contact line
    contact_line_length = (right_angle_point - left_angle_point) * _locPlotter.conversion_factor
    _locPlotter.Contact_line(x_cropped, j_poly_right, j_poly_left, left_angle_point, right_angle_point, contact_line_length)
    right_angle_point = ((frame_width) * 3 - right_angle_point - (x_cropped) * 3) * _locPlotter.conversion_factor
    left_angle_point = ((frame_width) * 3 - left_angle_point + -(x_cropped) * 3) * _locPlotter.conversion_factor
    
    # Centers
    v_center, *_ = vertical_center(i_list, j_list)
    h_center, i_mean, j_mean = horizontal_center(i_list, j_list)
    # drop_height = abs(min(j_list) - j_list[i_list == int(h_center)][0]) * _locPlotter.conversion_factor
    # --- Safe drop_height calculation ---
    mask = (i_list == int(h_center))
    if np.any(mask):
        # ✅ Exact match found
        drop_height = abs(min(j_list) - j_list[mask][0]) * _locPlotter.conversion_factor
    else:
        # ⚠️ No exact match — use nearest neighbor fallback
        idx = np.argmin(np.abs(i_list - h_center))
        drop_height = abs(min(j_list) - j_list[idx]) * _locPlotter.conversion_factor
        print(f"[WARN] No exact pixel match for h_center={h_center}. "
            f"Using nearest index {idx} (i={i_list[idx]}).")    
    _locPlotter.Center(h_center, j_list, i_list, drop_height, v_center)

    # Middle line
    i_middle_line, j_middle_line    = poly_fitting(i_mean, j_mean, polynomial_degree=1, line_space=100)
    middle_angle_degree             = middle_angle(i_middle_line, j_middle_line)
    if middle_line_switch != 0:
        i_middle_line, j_middle_line = poly_fitting(i_mean, j_mean, polynomial_degree=1, line_space=100)
        middle_angle_degree = middle_angle(i_middle_line, j_middle_line)
        i2_middle_line = min(i_middle_line[j_middle_line <= j_list[i_list == int(h_center)][0]])
        _locPlotter.Middle_line(i_middle_line, j_middle_line, middle_angle_degree, i2_middle_line)


    # Vertical center
    v_center, i_mean, j_mean = vertical_center(i_list, j_list)
    i_text_vertical = (min(i_list) + h_center) / 2
    drop_length = abs(min(i_list[j_list == int(v_center)]) - max(i_list[j_list == int(v_center)])) * _locPlotter.conversion_factor
    _locPlotter.Vertical_center(i_list=i_list, j_list=j_list, v_center=v_center, drop_length=drop_length, i_text_vertical=i_text_vertical)

    # Center point
    x_center = ((frame_width) * 3 - h_center) * _locPlotter.conversion_factor
    y_center = v_center * _locPlotter.conversion_factor
    _locPlotter.Center_point(h_center, v_center)

    
    _locPlotter.Save()

    return left_angle_degree, right_angle_degree, right_angle_point, left_angle_point, contact_line_length, x_center, y_center, middle_angle_degree[0]

if __name__ == "__main__":
    

    left_polynomial_degree  = 3
    right_polynomial_degree = 2

    cm_on_pixel_ratio           = 0.0039062
    num_px_ratio                = (0.0039062)/cm_on_pixel_ratio

    _image = cv2.imread('/media/Dont/Teflon-AVP/280/S3-SNr3.07_D/T105_11_79.813535314440/databases/frame_002781.png')
    if _image is None:
        raise ValueError("Image not found or unable to load.")
    elif len(_image.shape) > 2:
        # raise ValueError("Invalid image format.")
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)

    i_list, j_list  = edge_extraction( _image[:-5,:], thr=30)#cv2.bitwise_not(_image)


    left_number_of_pixels   = int(64*num_px_ratio)
    right_number_of_pixels  = int(65*num_px_ratio)
    i_left, j_left          = Advancing_pixel_selection_Euclidean(i_list,j_list, left_number_of_pixels=left_number_of_pixels)
    i_right, j_right        = Receding_pixel_selection_Euclidean(i_list,j_list, right_number_of_pixels=right_number_of_pixels)

    
    i_poly_left_rotated, j_poly_left_rotated    = poly_fitting(j_left,i_left,left_polynomial_degree,left_number_of_pixels)
    i_poly_right_rotated, j_poly_right_rotated  = poly_fitting(i_right,i_right,right_polynomial_degree,right_number_of_pixels)


    _horizontal_center, mean_list, j_location_list = horizontal_center(i_list, j_list, intersection_margin=4)
    v_center, i_mean, j_mean = vertical_center(i_list, j_list)

    plt.figure(figsize=(10, 6))
    # plt.plot(_horizontal_center, mean_list, label='Mean Profile', color='blue')
    plt.axhline(y=_horizontal_center, color='green', linestyle='--', label='Horizontal Center')
    plt.axvline(x=v_center, color='orange', linestyle='--', label='Vertical Center')
    # plt.scatter(_horizontal_center, j_location_list, label='J Locations', color='red')
    plt.title('Horizontal Center and Mean Profile')
    plt.xlabel('Horizontal Center')
    plt.ylabel('Mean Profile / J Locations')
    plt.legend()
    plt.grid()
    plt.show()
