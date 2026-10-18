# BaseUtils

Lowest-level package: configuration, frame listing, drop detectors, edge extraction and logging. Other packages import it; it does not import them.

## API ([__init__.py](__init__.py))

| Name | Description |
|------|-------------|
| `config` | Dictionary loaded from the project's `config.yaml` ([config.py](config.py)) |
| `ImageLister(FolderAddress, frameAddress="frames", extension="*.png")` | Sorted list of `frame*` images in `FolderAddress/frameAddress`; falls back to `*.jpg` |
| `FileIndexChecker(FolderAddress, frameAddress="frames")` | Raises `ValueError` if a frame index is missing |
| `DropDetection_YOLO` | YOLO detector (see below) |
| `DropDetection_SUM` | Intensity-profile detector |
| `DropDetection_SUM_YOLO` | Combines both |

## Drop detectors ([drop_detection.py](drop_detection.py))

All detectors expose `detect_drops(frame_path) -> (bounds, found)` and `horizontal_bound_extractor(bounds) -> (x1, x2)`.

| Class | Method |
|-------|--------|
| `DropDetection_YOLO()` | Loads `Detection/Weights/<yolo_name>.<extension_yolo>` with Ultralytics, runs on CUDA with `yolo_conf`, returns the highest-confidence box |
| `DropDetection_SUM(scaleDownFactor=1, drop_width=300)` | Column-intensity profile (`detectionV2`); reports no drop if a dark region touches the bottom corners |
| `DropDetection_SUM_YOLO(scaleDownFactor=5, drop_width=300)` | Left bound from YOLO, right bound = max of both |
| `DropDetection_Difference()` | Difference between consecutive frames (OpenCV CUDA if available); needs the previous frame on disk |

## Detection/

| File | Contents |
|------|----------|
| [DropDetection_Sum.py](Detection/DropDetection_Sum.py) | `detectionV2` (intensity profile), `detectionV3` (YOLO), `Main` (crops all frames), `DetectCropSave`, `DetectEdgeSave`, `crop_and_save_image` |
| [edgeDetection.py](Detection/edgeDetection.py) | `edge_extraction`, `Advancing_pixel_selection_Euclidean`, `Receding_pixel_selection_Euclidean` (used by CaMeasurer) |
| [DropDetection_Difference.py](Detection/DropDetection_Difference.py) | `DropPreProcessor` for frame differencing |
| [DropCoordinateSystem.py](Detection/DropCoordinateSystem.py) | Sorts drop edge points along the contour (`EdgePointSorter`, `EdgePointSorter_Numba`) |
| [LightSourceReflectionRemoving.py](Detection/LightSourceReflectionRemoving.py) | `DropBoundryExtractor`, `LightSourceReflectionRemover` |

### Cropping (`Main`)

`Main(experiment, SaveAddress, SaveAddressCSV, extension=".png", Detect=DetectCropSave)`:
- runs `detectionV3` (YOLO) on every frame in `experiment`;
- saves the crop `image[:, endpoint-15 : beginning+15]`, inverted with `bitwise_not`, to `SaveAddress`;
- writes `detections.csv` (`image`, `endpoint`, `beginning`) to `SaveAddressCSV` after all frames succeed.

It deletes old crops and `detections.csv`/`detections.pkl` first, and raises if any frame fails. `Utilities.crop_Save` calls it with `frames_rotated/` → `databases/`.

### Detection/Weights/

| File | Description |
|------|-------------|
| `Gray-320-s.pt` | YOLO small model (default, `yolo_name`) |
| `Gray-320-n.pt` | YOLO nano model |
| `Gray-320-s.engine`, `Gray-320-n.engine` | TensorRT versions; rebuild with `python pre_requisites.py` after a TensorRT upgrade |
| [YOLO_tensorRT.py](Detection/Weights/YOLO_tensorRT.py) | `enginemaker(name)`: exports a `.pt` model to TensorRT |

To use an engine set `extension_yolo: "engine"` in `config.yaml`. The confidence threshold cannot be changed on an exported engine.

## Logging ([logException.py](logException.py))

`LogException(base_path)` creates `base_path/log_<timestamp>.log` and provides `log_exception(e, custom_message, Verbose)`, `warning_message`, `success_message`.

## Config keys

`frame_folder`, `image_extension`, `yolo_name`, `extension_yolo`, `yolo_conf`.

## Dependencies

Ultralytics, PyTorch (CUDA), OpenCV, NumPy, pandas, PyYAML, colorama; Numba for `EdgePointSorter_Numba`.
