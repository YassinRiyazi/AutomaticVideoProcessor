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
    return measure_drop_with_geometry(image_path, threshold)[0]


def measure_drop_with_geometry(image_path: str, threshold: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Same as `measure_drop`, plus the sub-pixel boundary and feature positions (image coordinates) used by `plot_drop`."""
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
    apex_x = float(cols[y_top.argmin()])
    near = y_top <= apex_y + 1.0
    if near.sum() >= 3:
        a, b, c = np.polyfit(cols[near], y_top[near], 2)
        if a > 0 and cols[near].min() <= -b / (2 * a) <= cols[near].max():
            vertex_y = float(c - b * b / (4 * a))
            # The single highest sample is biased upwards by noise; trust the fitted vertex if it is plausible
            if apex_y <= vertex_y <= apex_y + 1.0:
                apex_y, apex_x = vertex_y, float(-b / (2 * a))

    ys, xs = np.nonzero(mask)
    centroid_y = float(ys.mean())
    widest = int(widths.argmax())
    on_substrate = rows[-1] == base - 1

    result = {
        "threshold": thr,
        "baseline_y (px)": base,
        "height (px)": substrate_y - apex_y,
        "width (px)": float(widths[widest]),
        "base_width (px)": float(widths[-1]) if on_substrate else float("nan"),
        "centroid_x (px)": float(xs.mean()),
        "centroid_y (px)": centroid_y,
        "centroid_height (px)": substrate_y - centroid_y,
        "area (px^2)": int(mask.sum()),
        "touches_border": bool(mask[:, 0].any() or mask[:, -1].any()),
    }
    geometry = {
        "substrate_y": substrate_y,
        "boundary_x": np.concatenate([x_left, x_right, cols]),
        "boundary_y": np.concatenate([rows, rows, y_top]).astype(np.float64),
        "apex": (apex_x, apex_y),
        "widest_row": (float(x_left[widest]), float(x_right[widest]), float(rows[widest])),
        "contact_points": (float(x_left[-1]), float(x_right[-1])) if on_substrate else None,
    }
    return result, geometry


def plot_limits(geometries: list[dict[str, Any]],
                aspect: float = 1.5,
                margin: float = 0.12) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis limits that fit the drop in every frame (x: image column, y: px above the substrate), with x/y range ratio `aspect`."""
    x_min = min(float(g["boundary_x"].min()) for g in geometries)
    x_max = max(float(g["boundary_x"].max()) for g in geometries)
    y_max = max(float(g["substrate_y"] - g["boundary_y"].min()) for g in geometries)
    pad = margin * max(x_max - x_min, y_max)
    xlim, ylim = [x_min - pad, x_max + pad], [-pad, y_max + pad]
    x_range, y_range = xlim[1] - xlim[0], ylim[1] - ylim[0]
    if x_range / y_range < aspect:
        extra = (aspect * y_range - x_range) / 2
        xlim = [xlim[0] - extra, xlim[1] + extra]
    else:
        ylim[1] += x_range / aspect - y_range
    return (xlim[0], xlim[1]), (ylim[0], ylim[1])


def plot_drop(ax: Any,
              result: dict[str, Any] | pd.Series,
              geometry: dict[str, Any],
              cm_on_pixel: float,
              font_size: int = 14) -> None:
    """
    Draws the drop boundary and the measured features on `ax`, in the style of the side view plots
    (CaMeasurer/visualization.py). x is the image column and y the height above the substrate line, both in px.
    """
    def up(y: Any) -> Any:
        return geometry["substrate_y"] - y

    label = dict(textcoords="offset points", fontsize=font_size)
    ax.plot(geometry["boundary_x"], up(geometry["boundary_y"]), '.', color='black')

    apex_x, _ = geometry["apex"]
    height = float(result["height (px)"])
    ax.plot([apex_x, apex_x], [0, height], '--', color='green')
    ax.annotate(f"Height={height * cm_on_pixel:.3f} cm", (apex_x, 0.75 * height), xytext=(6, 0), va='center', color='green', **label)

    centroid_x, centroid_height = float(result["centroid_x (px)"]), float(result["centroid_height (px)"])
    x_left, x_right, row = geometry["widest_row"]
    ax.plot([x_left, x_right], [up(row), up(row)], '--', color='green')
    ax.annotate(f"Width={float(result['width (px)']) * cm_on_pixel:.3f} cm", ((x_left + centroid_x) / 2, up(row)),
                xytext=(0, 6), ha='center', color='green', **label)

    if geometry["contact_points"] is not None:
        base_left, base_right = geometry["contact_points"]
        ax.plot([base_left, base_right], [0, 0], '--', linewidth=1, color='red')
        ax.annotate(f"Base width={float(result['base_width (px)']) * cm_on_pixel:.3f} cm", ((base_left + base_right) / 2, 0),
                    xytext=(0, -8), ha='center', va='top', color='red', **label)

    ax.plot(centroid_x, centroid_height, '.', color='blue', markersize=14)
    ax.annotate(f"Centroid= [x={centroid_x * cm_on_pixel:.3f} cm, y={centroid_height * cm_on_pixel:.3f} cm]",
                (centroid_x, centroid_height), xytext=(6, -6), va='top', color='blue', **label)


def save_plots(df: pd.DataFrame,
               geometries: dict[str, dict[str, Any]],
               output_dir: str,
               cm_on_pixel: float,
               dpi: int = 100) -> None:
    """Saves one plot per frame to `output_dir/<file number>`, with the same axis limits for the whole video."""
    from matplotlib.figure import Figure  # no pyplot: keeps the global backend untouched

    os.makedirs(output_dir, exist_ok=True)
    xlim, ylim = plot_limits(list(geometries.values()))
    fig = Figure(figsize=(15, 10), dpi=dpi)
    ax = fig.add_subplot()

    for i, (_, row) in enumerate(tqdm.tqdm(df.iterrows(), total=len(df), desc=f"Plotting {output_dir}")):
        ax.clear()
        plot_drop(ax, row, geometries[row["file number"]], cm_on_pixel)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both', labelsize=20)
        if i == 0:
            fig.tight_layout()  # limits and ticks are identical for every frame, so the layout is computed once
        fig.savefig(os.path.join(output_dir, str(row["file number"])), dpi=dpi)


def process_folder(address: str,
                   frames_folder: str = str(BaseUtils.config["rotated_frames_folder"]),
                   output_csv: str = "result.csv",
                   cm_on_pixel: float | None = None,
                   plot_folder: str | None = "SR_edge") -> pd.DataFrame:
    """
    Measures every frame in `address/frames_folder` and writes `address/output_csv` (one row per frame, keyed by 'file number').
    Frames that cannot be measured are listed in `address/error_log.txt` and skipped.
    If `plot_folder` is set, a plot of the boundary and features of every frame is saved to `address/plot_folder`.

    Pixel columns are always written. cm columns use `cm_on_pixel`, defaulting to
    config['Experiment_Parameters']['front_cm_on_pixel_ratio'] (a constant, so it ignores perspective changes).
    """
    if cm_on_pixel is None:
        cm_on_pixel = float(BaseUtils.config["Experiment_Parameters"]["front_cm_on_pixel_ratio"])  # type: ignore

    images = BaseUtils.ImageLister(address, frameAddress=frames_folder)

    rows: list[dict[str, Any]] = []
    geometries: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for image in tqdm.tqdm(images, desc=f"Drop geometry {address}"):
        name = os.path.basename(image)
        try:
            result, geometries[name] = measure_drop_with_geometry(image)
            rows.append({"file number": name, **result})
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")

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

    if plot_folder:
        save_plots(df, geometries, os.path.join(address, plot_folder), cm_on_pixel)
    return df
