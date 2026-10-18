# FrameExtractor

Extracts frames from the experiment video with ffmpeg, crops them, converts them to grayscale and checks them.

## Usage

```python
import FrameExtractor

fe = FrameExtractor.FrameExtractor()
fe.Forward("/path/to/experiment")
```

`Forward(FolderAddress, fps=None, out_dir=None)`:
1. `extract_frames` — runs ffmpeg with the crop from `config.yaml`. FPS is read with `ffprobe` when `fps` is `None`. Skipped if `frames/` already exists (pass `wipe=True` to `extract_frames` to overwrite).
2. `BandGMaker` — converts any non-grayscale frame to grayscale.
3. `Bottom_row_checker` — samples every 4th frame; the warning is currently disabled.
4. `HealthChecker` — deletes unreadable or zero-size frames.
5. `BaseUtils.FileIndexChecker` — raises if a frame index is missing.

If `out_dir` is given, frames are written to `out_dir/frame_%06d.png` instead.

## API

| Method | Description |
|--------|-------------|
| `FrameExtractor(InPlaceOutput=False)` | Checks that ffmpeg/ffprobe are installed |
| `resolve_video_path(path)` | Returns the single video in a folder; raises if there are zero or several |
| `ffprobe_fps(video_path)` | Video frame rate |
| `extract_frames(FolderAddress, output_frame_pattern=None, wipe=False, use_cuda=False, fps=None, grayscale=True, health_check=True)` | Runs the extraction |
| `BandGMaker`, `HealthChecker`, `Bottom_row_checker` | Post-extraction checks (take `FolderAddress`) |

[Video2Jpg.py](Video2Jpg.py) contains the ffmpeg helpers (`ffmpeg_frame_extractor`, `init`).

## Input / output

- **Input:** one video in the folder (`.mp4 .mov .avi .mkv .mpeg .mpg .m4v`).
- **Output:** `<experiment>/frames/frame_%06d.png`, grayscale, cropped to `Video.cropping_properties` (`X=750, Y=0, WIDTH=1000, HEIGHT=188`).

## Config keys

`frame_folder`, `image_extension`, `Video.cropping_properties.{X,Y,WIDTH,HEIGHT}`.

## Dependencies

`ffmpeg`, `ffprobe`, Pillow, NumPy, colorama.
