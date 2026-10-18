# DropGeometry

Measures the drop silhouette in leveled frames: height, width, base width, centroid and projected area. It is used for the front view in [main_front.py](../main_front.py), but it works on any frames produced by BaseLine.

## Usage

```python
import DropGeometry

df = DropGeometry.process_folder("/path/to/experiment/front")   # writes front/result.csv and front/SR_edge/*.png
row = DropGeometry.measure_drop("/path/to/experiment/front/frames_rotated/frame_002262.png")
```

## How it works ([__init__.py](__init__.py))

1. `drop_mask` applies an Otsu threshold to the frame. The baseline is the top of the black band that BaseLine writes below the substrate (`find_baseline`).
2. The drop is the largest dark connected component touching the baseline. The needle and small artifacts do not touch it and are ignored. Holes such as the bright spot inside the drop are filled, including holes that touch the substrate.
3. `measure_drop` finds sub-pixel edges by interpolating where the intensity crosses the threshold:
   - **height:** distance from the substrate line to the apex. The apex is the vertex of a parabola fitted to the top edge.
   - **width:** widest row of the silhouette. **base_width:** width of the last row above the substrate.
   - **centroid:** area centroid of the filled silhouette. **centroid_height** is measured from the substrate line.
4. `process_folder` measures every frame and writes the CSV. Frames that cannot be measured go to `error_log.txt`.
5. `save_plots` saves one plot per frame to `<view>/SR_edge` (set `plot_folder=None` to skip it). The style matches the side view's `SR_edge` plots. Each plot shows only the sub-pixel boundary (black) and the features: height (green, vertical), width (green, at the widest row), base width (red, on the substrate) and centroid (blue). x is the image column and y the height above the substrate, both in px. The axis limits are fixed for the whole video, so the plots can be played as a sequence. Plotting takes about 80 ms per frame.

On synthetic drops (anti-aliased, blurred, noisy), height, width and centroid errors are about 0.1 px.

## Input / output

- **Input:** `<view>/frames_rotated/frame_*.png` (BaseLine output).
- **Output:** `<view>/result.csv` with one row per frame. `file number` matches `side/result.csv`. The other columns are `threshold`, `baseline_y`, `height`, `width`, `base_width`, `centroid_x`, `centroid_y`, `centroid_height` and `area` in px, plus cm columns and `touches_border`. One plot per frame is written to `<view>/SR_edge/frame_*.png`.

## Scale

cm values use the constant `Experiment_Parameters.front_cm_on_pixel_ratio` from `config.yaml`. The drop slides toward the front camera, so the true scale changes by about 24% along the slide. In tilt-20-rep-001/002 the front/side apex-height ratio goes from 0.77 to 0.95. The constant is therefore off by up to about ±10% at the ends of a run. Pixel columns are always written.

## Assumptions

- Exactly one drop sits on the substrate and does not touch the needle.
- Frames are leveled and everything below the substrate is black.
- If `touches_border` is True, the drop is cut by the image edge and `width` is only a lower bound.
