"""
    Project : Automatic Video Processor (AVP)
    File    : DropGeometry/__init__.py
    License : GNU General Public License v3.0

    Description:
        Measures the silhouette of a single drop in leveled frames (`<view>/frames_rotated`, as written by BaseLine):
        height, width, base (contact) width, centroid and projected area, all in pixels with sub-pixel edges.

    Assumptions:
        - Frames are grayscale, leveled, and everything below the substrate is black (BaseLine output).
        - Exactly one drop sits on the substrate. Other dark objects (e.g. the needle) do not touch the substrate.
"""
import  os
import  sys
import  tqdm
import  cv2
import  numpy               as  np
import  pandas              as  pd
from    numpy.typing        import  NDArray
from    typing              import  Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import  BaseUtils


def find_baseline(dark: NDArray[np.bool_], full_ratio: float = 0.95) -> int:
    """Row index of the top of the black band at the bottom of a leveled frame (the substrate line)."""
    full = dark.mean(axis=1) > full_ratio
    if not full[-1]:
        raise ValueError("No black substrate band at the bottom of the frame; is the frame leveled?")
    row = len(full) - 1
    while row > 0 and full[row - 1]:
        row -= 1
    return row


def _subpixel_edge(pos_bright: NDArray[np.float64],
                   i_bright: NDArray[np.float64],
                   i_dark: NDArray[np.float64],
                   threshold: float,
                   step: int) -> NDArray[np.float64]:
    """Position where intensity crosses `threshold` between a bright pixel and its dark neighbour at `pos_bright + step`."""
    contrast = i_bright - i_dark
    frac = np.where(contrast > 0, (i_bright - threshold) / np.where(contrast > 0, contrast, 1.0), 0.5)
    return pos_bright + step * np.clip(frac, 0.0, 1.0)


def drop_mask(gray: NDArray[np.uint8], threshold: float | None = None) -> tuple[NDArray[np.bool_], int, float]:
    """
    Returns (filled drop mask above the baseline, baseline row, threshold).
    The drop is the largest dark component touching the baseline.
    """
    if threshold is None:
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = gray < threshold
    base = find_baseline(dark)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark[:base].astype(np.uint8), connectivity=8)
    on_base = [i for i in range(1, n) if stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] == base]
    if not on_base:
        raise ValueError("No drop found on the baseline.")
    drop = max(on_base, key=lambda i: stats[i, cv2.CC_STAT_AREA])

    # Close the footprint with one extra row so bright spots touching the substrate are filled as holes too
    component = (labels == drop)
    footprint = np.flatnonzero(component[base - 1])
    closed = np.zeros((base + 1, gray.shape[1]), np.uint8)
    closed[:base] = component
    closed[base, footprint.min():footprint.max() + 1] = 1
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)

    return filled[:base].astype(bool), base, float(threshold)


def measure_drop(image_path: str, threshold: float | None = None) -> dict[str, Any]:
    """
    Measures one leveled frame. All lengths are in pixels; y grows downwards (image coordinates).

    Returns:
        baseline_y          first black row under the drop
        height              apex height above the substrate line
        width               widest horizontal extent of the silhouette
        base_width          width in the last row above the substrate (projected contact width)
        centroid_x/_y       area centroid of the filled silhouette (image coordinates)
        centroid_height     centroid height above the substrate line
        area                projected area in pixels
        touches_border      True if the drop is cut by the left/right image border (width is then a lower bound)
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Image not found or unable to read: {image_path}")

    mask, base, thr = drop_mask(gray, threshold)
    g = gray.astype(np.float64)
    width_px = gray.shape[1]
    substrate_y = base - 0.5  # boundary between the last drop row and the black band

    # Left/right edges per row
    rows = np.flatnonzero(mask.any(axis=1))
    lefts = mask[rows].argmax(axis=1)
    rights = width_px - 1 - mask[rows, ::-1].argmax(axis=1)
    left_ok, right_ok = lefts > 0, rights < width_px - 1
    lb, rb = np.where(left_ok, lefts - 1, lefts), np.where(right_ok, rights + 1, rights)
    x_left = np.where(left_ok, _subpixel_edge(lb, g[rows, lb], g[rows, lefts], thr, +1), lefts - 0.5)
    x_right = np.where(right_ok, _subpixel_edge(rb, g[rows, rb], g[rows, rights], thr, -1), rights + 0.5)
    widths = x_right - x_left

    # Top edge per column; apex from a parabola through the flattest part of the top edge
    cols = np.flatnonzero(mask.any(axis=0))
    tops = mask[:, cols].argmax(axis=0)
    top_ok = tops > 0
    tb = np.where(top_ok, tops - 1, tops)
    y_top = np.where(top_ok, _subpixel_edge(tb, g[tb, cols], g[tops, cols], thr, +1), tops - 0.5)
    apex_y = float(y_top.min())
    near = y_top <= apex_y + 1.0
    if near.sum() >= 3:
        a, b, c = np.polyfit(cols[near], y_top[near], 2)
        if a > 0 and cols[near].min() <= -b / (2 * a) <= cols[near].max():
            vertex_y = float(c - b * b / (4 * a))
            # The single highest sample is biased upwards by noise; trust the fitted vertex if it is plausible
            if apex_y <= vertex_y <= apex_y + 1.0:
                apex_y = vertex_y

    ys, xs = np.nonzero(mask)
    centroid_y = float(ys.mean())

    return {
        "threshold": thr,
        "baseline_y (px)": base,
        "height (px)": substrate_y - apex_y,
        "width (px)": float(widths.max()),
        "base_width (px)": float(widths[-1]) if rows[-1] == base - 1 else float("nan"),
        "centroid_x (px)": float(xs.mean()),
        "centroid_y (px)": centroid_y,
        "centroid_height (px)": substrate_y - centroid_y,
        "area (px^2)": int(mask.sum()),
        "touches_border": bool(mask[:, 0].any() or mask[:, -1].any()),
    }


def process_folder(address: str,
                   frames_folder: str = str(BaseUtils.config["rotated_frames_folder"]),
                   output_csv: str = "result.csv",
                   cm_on_pixel: float | None = None) -> pd.DataFrame:
    """
    Measures every frame in `address/frames_folder` and writes `address/output_csv` (one row per frame, keyed by 'file number').
    Frames that cannot be measured are listed in `address/error_log.txt` and skipped.

    Pixel columns are always written. cm columns use `cm_on_pixel`, defaulting to
    config['Experiment_Parameters']['front_cm_on_pixel_ratio'] (a constant, so it ignores perspective changes).
    """
    if cm_on_pixel is None:
        cm_on_pixel = float(BaseUtils.config["Experiment_Parameters"]["front_cm_on_pixel_ratio"])  # type: ignore

    images = BaseUtils.ImageLister(address, frameAddress=frames_folder)

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for image in tqdm.tqdm(images, desc=f"Drop geometry {address}"):
        try:
            rows.append({"file number": os.path.basename(image), **measure_drop(image)})
        except Exception as e:
            errors.append(f"{os.path.basename(image)}: {type(e).__name__}: {e}")

    if errors:
        with open(os.path.join(address, "error_log.txt"), "a", encoding="utf-8") as log_file:
            log_file.write("\n".join(errors) + "\n")
    if not rows:
        raise RuntimeError(f"No frame could be measured in {os.path.join(address, frames_folder)}. Check error_log.txt")

    df = pd.DataFrame(rows)
    for name in ("height", "width", "base_width", "centroid_x", "centroid_height"):
        df[f"{name} (cm)"] = df[f"{name} (px)"] * cm_on_pixel
    df["area (cm^2)"] = df["area (px^2)"] * cm_on_pixel ** 2
    df.to_csv(os.path.join(address, output_csv), index=False)
    return df
