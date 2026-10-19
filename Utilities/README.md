# Utilities

Removes frames where the drop is missing or not fully in view, crops the drop, and post-processes the results.

## API ([__init__.py](__init__.py))

| Name | Description |
|------|-------------|
| `YoloWalker(num_workers=None)` | Persistent process pool with one YOLO model per worker. `run(image_folder, skip=90, detector_factory=BaseUtils.DropDetection_YOLO)`, `close()` |
| `singleFolderDropNormalizer(images, detector=BaseUtils.DropDetection_YOLO)` | Trims frames at the start and end of the sequence (see below) |
| `crop_Save(image_folder)` | Crops every frame in `frames_rotated/` into `databases/` and writes `databases/detections.csv` (via `BaseUtils.drop_detection.Main`) |
| `position_velocity_correction(df_or_path)` | Fixes `x_center (cm)` values that go backwards and recomputes `velocity (cm/s)`. Given a path, writes `<name>_corrected.csv` |
| `position_velocity_correctionV2(df_or_path, time_step=0.00025)` | Same, also rebuilds the time column |
| `create_video_from_images(image_folder, output_video_path, extension="jpg", fps=30)` | Builds an MP4 with ffmpeg |
| `main(image_folder)` | Older all-in-one filter + crop |

## Typical use (as in `main.py`)

```python
import Utilities

yolo = Utilities.YoloWalker(num_workers=5)
yolo.run(image_folder=folder, skip=40)
yolo.run(image_folder=folder, skip=5)

images = Utilities.BaseUtils.ImageLister(folder, "frames_rotated")
Utilities.singleFolderDropNormalizer(images, Utilities.BaseUtils.DropDetection_YOLO)
Utilities.crop_Save(image_folder=folder)
# ... CaMeasurer ...
Utilities.position_velocity_correction(f"{folder}/result.csv")
yolo.close()
```

## Frame trimming (`singleFolderDropNormalizer`)

With `WIDTH = 1000`:
- **Forward pass** (from the first frame): deletes frames with no drop or with the drop's right edge at or beyond `Left_boundary_ratio * WIDTH` (900 px), and stops at the first frame where the drop is inside.
- **Backward pass** (from the last frame): deletes frames with no drop or with the drop's left edge at or before `Right_boundary_ratio * WIDTH` (60 px), and stops at the first frame where the drop is inside.

Frames are deleted from `frames_rotated/`.

> [!NOTE]
> `YoloWalker.run` currently only runs detection on the first and last frame of each chunk; it does not delete frames.

## Files

| File | Contents |
|------|----------|
| [Detection_Sparse.py](Detection_Sparse.py) | `YoloWalker` |
| [Detection_frameNormalizer.py](Detection_frameNormalizer.py) | `singleFolderDropNormalizer` |
| [DF_postprocessing.py](DF_postprocessing.py) | `position_velocity_correction`, `position_velocity_correctionV2` |
| [Jpg2Video.py](Jpg2Video.py) | `create_video_from_images`, `process_experiment` (video from `SR_edge/`) |
| [VIdeo_Compressor.py](VIdeo_Compressor.py) | `compress_video(input_path, output_path, crf=23, preset="slow")`, `check_video_health` |
| [frameNr.py](frameNr.py) | Script: frame counts per experiment with ffprobe |

## Input / output

- **Input:** `<experiment>/frames_rotated/frame_*.png`; `<experiment>/result.csv` for post-processing.
- **Output:** trimmed `frames_rotated/`, `databases/frame_*.png`, `databases/detections.csv`, `result_corrected.csv`.

## Config keys

`rotated_frames_folder`, `databases_folder`, `image_extension`, `Video.cropping_properties.WIDTH`, `single_FolderDropNormalizer.{Left,Right}_boundary_ratio`.

## Dependencies

BaseUtils, Ultralytics (CUDA), pandas, NumPy, tqdm, colorama, ffmpeg/ffprobe.
