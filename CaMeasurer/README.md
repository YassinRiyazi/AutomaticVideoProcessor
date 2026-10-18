# CaMeasurer

Measures the advancing and receding contact angles, contact-line length and drop centre for every cropped frame (4S-SROF method).

## Usage

```python
import CaMeasurer

s4 = CaMeasurer.processes_mp_shared(num_workers=8)
s4.run("/path/to/experiment")   # Writes result.csv and SR_edge/
s4.close()
```

`processes_mp_shared` keeps one worker pool (and one super-resolution model per worker) alive across folders.

| Entry point | Description |
|-------------|-------------|
| `processes_mp_shared(num_workers)` | `.run(folder)` returns the result DataFrame; `.close()` |
| `processes_mp(shared_address, num_workers=15)` | Creates a new pool per call |
| `processes(address)` | Single process |

## Per-frame steps

1. **Load** `databases/frame_*.png` and read `endpoint` from `databases/detections.csv` to place the crop in the full frame.
2. **Cut** `PreProcessing.buttom_Line_height` (5) rows from the bottom.
3. **Super-resolution** ×3 with `models/converted_model.pt` ([superResolution.py](superResolution.py)).
4. **Edges** — `edge_extraction` (BaseUtils) takes the first pixel above `edge_Detection_threshold` (30) from the left, right and top. The y-axis is flipped so `y = 0` is the bottom.
5. **Contact-point pixels** — the edge pixels closest to the bottom-left (advancing) and bottom-right (receding) points.
6. **Fit** — a polynomial in rotated coordinates ([processing.py](processing.py)). Degree and number of pixels are re-chosen from the first angle estimate (`polyOrderDecider` in [main.py](main.py)):

   | Angle | Pixels | Degree |
   |-------|--------|--------|
   | ≤ 60° | 60 | 2 |
   | 60–105° | 85 | 2 |
   | 105–135° | 125 | 3 |
   | > 135° | 145 | 4 |

7. **Angles** — `left_angle` / `right_angle` ([criteria_definition.py](criteria_definition.py)) from the slope between the first two points of the fit.
8. **Visualise and measure** — `visualize` ([visualization.py](visualization.py)) computes contact-line length, centre and middle angle, and saves the figure to `SR_edge/`.

After all frames, `velocity = diff(x_center) * fps_experiment`; the last frame is dropped so all columns have the same length.

## Input / output

- **Input:** `<experiment>/databases/frame_*.png` (drop white on black, bottom rows black) and `<experiment>/databases/detections.csv`.
- **Output:**
  - `<experiment>/result.csv` — `file number, time (s), x_center (cm), adv (degree), rec (degree), contact_line_length (cm), y_center (cm), middle_angle_degree (degree), velocity (cm/s)`.
  - `<experiment>/SR_edge/frame_*.png` — one diagnostic figure per frame.
  - `<experiment>/databases/error_log.txt` — traceback for each frame that failed.

`run` raises `RuntimeError` if no frame succeeds.

## Files

| File | Contents |
|------|----------|
| [__init__.py](__init__.py) | Worker pool, `process_one_file`, `processes*` |
| [main.py](main.py) | `base_function_process`, `read_csv_for_endpoint_beginning`, `polyOrderDecider` |
| [superResolution.py](superResolution.py) | `PyTorchModel`, `initiation`, `upscale_image` |
| [processing.py](processing.py) | `poly_fitting` |
| [criteria_definition.py](criteria_definition.py) | `left_angle`, `right_angle`, `middle_angle` |
| [visualization.py](visualization.py) | `visualize`, `horizontal_center`, `vertical_center` |
| [preprocessing.py](preprocessing.py) | Preprocessing helpers |
| [V1__init__.py](V1__init__.py) | Older version used by `AVPV2.py` / `AVP_Pass2.py` |
| `doc/` | Images used in the docstrings (edge extraction, pixel selection) |

### models/

| File | Description |
|------|-------------|
| `converted_model.pt` | Super-resolution weights in use (`Super_Resolution.model_path`) |
| `SuperRes_weights.h5` | Original Keras weights |
| [h5toPt.py](models/h5toPt.py) | Converts the Keras weights to PyTorch |
| [edge_superres_pytorch.py](models/edge_superres_pytorch.py) | PyTorch model definition |
| `best.pt`, `YOLO12_n.pt` | Not loaded by the pipeline |

## Troubleshooting

If `adv`/`rec` come out near 0–15° and the red points in `SR_edge/` lie along the baseline, the crops have a bright band at the bottom. Check that everything under the surface is black in `frames_rotated/` (see [BaseLine](../BaseLine/README.md)).

## Config keys

`databases_folder`, `PreProcessing.buttom_Line_height`, `Super_Resolution.{upscale_factor,model_path}`, `Processing_Parameters.edge_Detection_threshold`, `Experiment_Parameters.{fps_experiment,cm_on_pixel_ratio,num_px_ratio}`.

## Dependencies

PyTorch (CUDA), OpenCV, NumPy, pandas, Matplotlib, tqdm.
