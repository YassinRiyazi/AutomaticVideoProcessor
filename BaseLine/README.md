# BaseLine

Levels the substrate in every frame and blacks out everything below it, so later stages see only the drop above a black base.

## Usage

```python
import BaseLine

BaseLine.BaseLine().Forward("/path/to/experiment")
```

`Forward(experiment)` calls `folderBaseLineNormalizer(experiment)`.

## How it works ([BaseNormalizer.py](BaseNormalizer.py))

1. `fit_and_rotate_image` fits a line to the edges in the bottom 30 % of the third frame with RANSAC and returns its angle. It saves `frames/Rotation.png` as a diagnostic plot.
2. `line_finder` runs on every frame in a process pool. It rotates the frame and returns the surface row: the first row (below 60 % of the height) whose median is below 50, minus `margin=3` rows.
3. The mean of these rows is used as `cropped_height`, and the frames are rotated again around `(w // 2, cropped_height + 10)`.
4. `process_image` rotates each frame, sets every row from `cropped_height` down to 0, and calls `bottom_row_unifierGRAY`.
5. `bottom_row_unifierGRAY` aligns the bottom of the content to the frame height and adds `pad_bottom=5` black rows.

Other helpers: `fit_image` (RANSAC line height), `line_finder` (surface row per frame).

## Input / output

- **Input:** `<experiment>/frames/frame_*.png` (grayscale).
- **Output:** `<experiment>/frames_rotated/frame_*.png` and `<experiment>/frames/Rotation.png`.

## Assumptions

- Frames are already extracted, grayscale, readable and sequentially numbered (see [FrameExtractor](../FrameExtractor/README.md)).
- The surface is visible in the lower part of the frame and is darker than the background.

## Notes

- The thresholds in `line_finder` (`0.6` of the height, median `< 50`, `margin=3`) were tuned on one dataset. Check `frames_rotated/` when the lighting changes.
- Any non-black row left under the drop becomes a bright band after the crop is inverted, and the contact angles come out near 0°.

## Config keys

`frame_folder`, `rotated_frames_folder`, `image_extension`.

## Dependencies

OpenCV, NumPy, scikit-image (`ransac`, `LineModelND`), Matplotlib (Agg backend).
