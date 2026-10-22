"""
    Project : Automatic Video Processor (AVP)
    File    : main.py
    License : GNU General Public License v3.0

    Description:
        Unified pipeline. For every video folder it processes the side view first (contact angles, CaMeasurer) and
        then the front view (drop geometry, DropGeometry). The front view keeps only the frames the side view kept.
        Videos are found under `Root_Address/Video_Addresses` and are never modified. Each one is processed in
        `Temporary_Processing.address/<relative path>` (frames and detail images, outside the Samba share); only its CSVs,
        logs and `.done` markers are mirrored to `Root_Address/Output_Address/<relative path>`, which is also the record of
        what has been processed. Once both views finish, `post_run` checks that their frames match and merges both CSVs
        into `result.csv`.

    Usage:
        python main.py                                  # one pass over all folders with a pending view
        python main.py --daemon                         # watch Video_Addresses; queue each folder once its video
                                                        # has not been written for Daemon.stability_seconds
        python main.py -C                               # delete all side and front outputs first, then process
        python main.py -C front --video-list /path/a    # redo only the front view of these video folders
"""
import  os
import  re
import  glob
import  time
import  queue
import  shutil
import  signal
import  argparse
import  json
import  threading
import  traceback
from    typing  import  Callable

import  tqdm
import  numpy   as  np
import  pandas  as  pd
import  BaseLine
import  BaseUtils
import  BaseUtils.logException  as  logException
import  Utilities
import  CaMeasurer
import  DropGeometry
import  FrameExtractor

SIDE: str = "side"
FRONT: str = "front"
VIEWS: tuple[str, ...] = (SIDE, FRONT)

INPUT_ROOT: str = os.path.normpath(os.path.join(str(BaseUtils.config['Root_Address']), str(BaseUtils.config['Video_Addresses'])))
OUTPUT_ROOT: str = os.path.normpath(os.path.join(str(BaseUtils.config['Root_Address']), str(BaseUtils.config['Output_Address'])))
VIDEO_SOURCE_GLOB: str = os.path.join(INPUT_ROOT, "*")

TMP_CONFIG: dict = BaseUtils.config.get("Temporary_Processing") or {}  # type: ignore
PROCESSING_ROOT: str = os.path.normpath(str(TMP_CONFIG.get("address", "/run/media/roboprocessing/Data/tmp_processing")))
TMP_MAX_FILES: int = int(TMP_CONFIG.get("max_files", 3_000_000))
TMP_MAX_BYTES: int = int(float(TMP_CONFIG.get("max_size_gb", 500)) * 1e9)

DONE: str = ".done"
MIRRORED_SUFFIXES: tuple[str, ...] = (".csv", ".log", ".txt")   # plus `.done` markers
FRAMES_FOLDER: str = str(BaseUtils.config["frame_folder"])
ROTATED_FRAMES_FOLDER: str = str(BaseUtils.config["rotated_frames_folder"])
DATABASES_FOLDER: str = str(BaseUtils.config["databases_folder"])
FPS_EXPERIMENT: float = float(BaseUtils.config["Experiment_Parameters"]["fps_experiment"])  # type: ignore

MERGED_CSV: str = "result.csv"                   # `<folder>/result.csv`, written by post_run
POST_RUN_ERROR: str = "post_run_error.txt"       # `<folder>/post_run_error.txt`, problems found by post_run
PARAMETERS_CSV: str = "experiment_parameters.csv"  # `<folder>/experiment_parameters.csv`, values used for the friction force
FRAME_COLUMN: str = "file number"

# Experiment constants: (key in the camera JSON and in config.yaml Experiment_Parameters, unit). All SI.
FRICTION_PARAMETERS: dict[str, str] = {"mass": "kg", "gravity": "m/s^2", "viscosity": "Pa s", "mStarmRatio": "-"}
ACCELERATION_WINDOW: int = int(BaseUtils.config["Experiment_Parameters"].get("acceleration_window", 21))  # type: ignore
TILT_PATTERN: re.Pattern[str] = re.compile(r"tilt-(\d+(?:\.\d+)?)", re.IGNORECASE)

DAEMON_CONFIG: dict = BaseUtils.config.get("Daemon") or {}  # type: ignore
STABILITY_SECONDS: float = float(DAEMON_CONFIG.get("stability_seconds", 60))
POLL_INTERVAL: float = float(DAEMON_CONFIG.get("poll_interval", 10))

# (name, size, mtime_ns, ctime_ns) of every video file in a folder
VideoSignature = tuple[tuple[str, int, int, int], ...]


# deleting empty folders and subfolders recursively
def delete_empty_folders(path: str) -> None:
    for root, dirs, files in os.walk(path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
    # Finally, check if the root itself is empty and remove it if it is
    if not os.listdir(path):
        os.rmdir(path)


def cleanUP(_folder: str, view: str) -> None:
    """Deletes every output of one view (`<folder>/<view>`), including its `.done` marker and logs, and the merged result built from it."""
    for name in (MERGED_CSV, POST_RUN_ERROR, PARAMETERS_CSV, DONE):
        path = os.path.join(_folder, name)
        if os.path.isfile(path):
            os.remove(path)

    address = os.path.join(_folder, view)
    for name in (FRAMES_FOLDER, ROTATED_FRAMES_FOLDER, DATABASES_FOLDER, "SR_edge"):
        shutil.rmtree(os.path.join(address, name), ignore_errors=True)

    for name in ('error_log.txt', 'result.csv', 'result_corrected.csv', '.done'):
        path = os.path.join(address, name)
        if os.path.isfile(path):
            os.remove(path)

    for log in glob.glob(os.path.join(address, '*.log')):
        os.remove(log)

    if os.path.isdir(address):
        delete_empty_folders(address)


def relative_folder(_folder: str) -> str:
    """Path of a video folder relative to `Root_Address/Video_Addresses`."""
    rel = os.path.relpath(os.path.abspath(_folder), INPUT_ROOT)
    if rel in (os.curdir, os.pardir) or rel.startswith(os.pardir + os.sep):
        raise ValueError(f"{_folder} is not inside {INPUT_ROOT}")
    return rel


def work_folder(_folder: str) -> str:
    """Where a video folder is processed: frames, detail images and results (`Temporary_Processing.address/<relative path>`)."""
    return os.path.join(PROCESSING_ROOT, relative_folder(_folder))


def output_folder(_folder: str) -> str:
    """Mirror of a video folder holding only its CSVs, logs and `.done` markers (`Root_Address/Output_Address/<relative path>`)."""
    return os.path.join(OUTPUT_ROOT, relative_folder(_folder))


def reset_view(_folder: str, view: str) -> None:
    """Deletes one view of a video folder in both the work folder and the output mirror, so it is processed again."""
    for tree in (work_folder(_folder), output_folder(_folder)):
        cleanUP(tree, view)


def cleanStart(Video_list: list[str], views: tuple[str, ...] = VIEWS) -> None:
    for _folder in tqdm.tqdm(Video_list, desc="Cleaning"):
        for view in views:
            reset_view(_folder, view)


def mirror_results(work: str, out: str) -> None:
    """
    Copies the CSVs, logs and `.done` markers of a work folder to the output mirror, keeping the folder structure.
    Unchanged files are skipped, each file is replaced atomically and `.done` markers are copied last, so the
    mirror never shows a view as done without its results.
    """
    selected: list[str] = []
    for root, _dirs, files in os.walk(work):
        selected.extend(os.path.join(root, name) for name in files if name == DONE or name.endswith(MIRRORED_SUFFIXES))

    for src in sorted(selected, key=lambda path: os.path.basename(path) == DONE):
        dst = os.path.join(out, os.path.relpath(src, work))
        st = os.stat(src)
        try:
            dst_st = os.stat(dst)
            if (dst_st.st_size, dst_st.st_mtime_ns) == (st.st_size, st.st_mtime_ns):
                continue
        except FileNotFoundError:
            pass
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        partial = os.path.join(os.path.dirname(dst), f".{os.path.basename(dst)}.partial")
        shutil.copy2(src, partial)
        os.replace(partial, dst)


def copy_experiment_json(_folder: str, work: str) -> None:
    """Copies the experiment's JSON files (`<folder name>*.json`, e.g. the camera JSON) into its work folder for post_run."""
    os.makedirs(work, exist_ok=True)
    folder_name = os.path.basename(os.path.normpath(_folder))
    for name in os.listdir(_folder):
        stem, extension = os.path.splitext(name)
        if extension.lower() == ".json" and (stem == folder_name or stem.startswith(folder_name + "_")):
            shutil.copy2(os.path.join(_folder, name), os.path.join(work, name))


def link_videos(_folder: str, work: str) -> list[str]:
    """Hard-links (or copies, across file systems) the videos of a video folder into its work folder."""
    os.makedirs(work, exist_ok=True)
    with os.scandir(work) as entries:
        for entry in entries:
            if FrameExtractor.video_matches_folder(work, entry.name) and not entry.is_dir(follow_symlinks=False):
                os.remove(entry.path)

    links: list[str] = []
    for name, *_ in video_signature(_folder) or ():
        src, dst = os.path.join(_folder, name), os.path.join(work, name)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        links.append(dst)
    return links


class ProcessingSpace:
    """
    Keeps `Temporary_Processing.address` below `max_files` files and `max_size_gb` by deleting whole experiment work
    folders, least recently processed first. Usage is tracked per work folder in `<address>/.avp_usage.json`, so the
    millions of files are only counted once (on first start) instead of on every check.
    """
    INDEX: str = ".avp_usage.json"

    def __init__(self, root: str = PROCESSING_ROOT, max_files: int = TMP_MAX_FILES, max_bytes: int = TMP_MAX_BYTES) -> None:
        self.root = root
        self.max_files = max_files
        self.max_bytes = max_bytes
        self.usage: dict[str, dict[str, float]] = {}
        os.makedirs(root, exist_ok=True)
        self._load()

    def _load(self) -> None:
        try:
            with open(os.path.join(self.root, self.INDEX), encoding="utf-8") as f:
                self.usage = {rel: entry for rel, entry in json.load(f).items() if os.path.isdir(os.path.join(self.root, rel))}
        except (OSError, ValueError, AttributeError):
            self.usage = {}
        # Work folders left behind without an entry (e.g. by a crash) are counted once here
        for rel in self._find_work_folders():
            if rel not in self.usage:
                path = os.path.join(self.root, rel)
                self.usage[rel] = {**self._measure(path), "last_used": os.path.getmtime(path)}
        self._save()

    def _find_work_folders(self) -> list[str]:
        found: list[str] = []
        for root, dirs, _files in os.walk(self.root):
            if root != self.root and any(view in dirs for view in VIEWS):
                found.append(os.path.relpath(root, self.root))
                dirs.clear()
        return found

    @staticmethod
    def _measure(path: str) -> dict[str, float]:
        files, size = 0, 0
        for root, _dirs, names in os.walk(path):
            for name in names:
                try:
                    size += os.lstat(os.path.join(root, name)).st_size
                    files += 1
                except OSError:
                    continue
        return {"files": files, "bytes": size}

    def _save(self) -> None:
        index = os.path.join(self.root, self.INDEX)
        with open(index + ".partial", "w", encoding="utf-8") as f:
            json.dump(self.usage, f)
        os.replace(index + ".partial", index)

    def totals(self) -> tuple[int, int]:
        return int(sum(e["files"] for e in self.usage.values())), int(sum(e["bytes"] for e in self.usage.values()))

    def update(self, work: str) -> None:
        """Records the current size of a work folder (call after processing it)."""
        rel = os.path.relpath(work, self.root)
        if os.path.isdir(work):
            self.usage[rel] = {**self._measure(work), "last_used": time.time()}
        else:
            self.usage.pop(rel, None)
        self._save()

    def enforce(self, keep: str | None = None) -> None:
        """Deletes the least recently processed work folders (never `keep`) until both limits are met."""
        keep_rel = os.path.relpath(keep, self.root) if keep else None
        files, size = self.totals()
        while files > self.max_files or size > self.max_bytes:
            candidates = sorted((e["last_used"], rel) for rel, e in self.usage.items() if rel != keep_rel)
            if not candidates:
                print(f"{self.root} is over its limit ({files} files, {size / 1e9:.1f} GB) but only the active folder is left.", flush=True)
                break
            _last_used, rel = candidates[0]
            path = os.path.join(self.root, rel)
            shutil.rmtree(path, ignore_errors=True)
            parent = os.path.dirname(path)
            while parent != self.root and parent.startswith(self.root + os.sep):
                try:
                    os.rmdir(parent)
                except OSError:
                    break
                parent = os.path.dirname(parent)
            entry = self.usage.pop(rel)
            files, size = files - int(entry["files"]), size - int(entry["bytes"])
            print(f"Deleted temporary folder {path} ({int(entry['files'])} files, {entry['bytes'] / 1e9:.2f} GB) to stay within "
                  f"{self.max_files} files / {self.max_bytes / 1e9:g} GB.", flush=True)
        self._save()


def discover_folders(pattern: str = VIDEO_SOURCE_GLOB) -> list[str]:
    """Recursively find video folders beneath the roots matched by a wildcard pattern."""
    def on_error(error: OSError) -> None:
        raise error

    folders: set[str] = set()
    for source in sorted(glob.glob(pattern, recursive=True)):
        if not os.path.isdir(source):
            continue
        for folder, dirs, files in os.walk(source, onerror=on_error):
            if any(
                FrameExtractor.video_matches_folder(folder, name)
                and os.path.isfile(os.path.join(folder, name))
                and not os.path.islink(os.path.join(folder, name))
                for name in files
            ):
                folders.add(os.path.normpath(folder))
                dirs.clear()
    return sorted(folders)


def is_pending(_folder: str, view: str) -> bool:
    """True if a view has not been processed yet and has no unresolved error/log files."""
    address = os.path.join(_folder, view)
    if os.path.isfile(os.path.join(address, '.done')):
        return False
    if os.path.isfile(os.path.join(address, 'error_log.txt')):
        return False
    return len(glob.glob(os.path.join(address, '*.log'))) == 0


def side_has_results(_folder: str) -> bool:
    """`side/frames_rotated` holds the final frame range once the side view has written `side/result.csv`."""
    return os.path.isfile(os.path.join(_folder, SIDE, 'result.csv'))


def pending_views(_folder: str) -> list[str]:
    """Views that still have to run, in processing order. The front view waits for the side view's results."""
    views: list[str] = []
    if is_pending(_folder, SIDE):
        views.append(SIDE)
    if is_pending(_folder, FRONT) and (views or side_has_results(_folder)):
        views.append(FRONT)
    return views


def needs_post_run(_folder: str) -> bool:
    """True if both views are finished with results but `post_run` has not merged them yet."""
    if pending_views(_folder):
        return False
    if not all(os.path.isfile(os.path.join(_folder, view, 'result.csv')) for view in VIEWS):
        return False
    return not any(os.path.isfile(os.path.join(_folder, name)) for name in (MERGED_CSV, POST_RUN_ERROR))


def needs_processing(_folder: str) -> bool:
    """True if a video folder still has a view or the merge to do, according to its output mirror."""
    out = output_folder(_folder)
    return bool(pending_views(out)) or needs_post_run(out)


def has_work_results(work: str, view: str) -> bool:
    """True if the work folder still holds a view's results and leveled frames (the temporary folder may have been cleaned)."""
    address = os.path.join(work, view)
    return os.path.isfile(os.path.join(address, 'result.csv')) and bool(glob.glob(os.path.join(address, ROTATED_FRAMES_FOLDER, "frame_*")))


def video_signature(_folder: str) -> VideoSignature | None:
    """Size and timestamps of the video files directly inside a folder, or None if it has none (yet)."""
    signature: list[tuple[str, int, int, int]] = []
    try:
        with os.scandir(_folder) as entries:
            for entry in entries:
                if not FrameExtractor.video_matches_folder(_folder, entry.name):
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                st = entry.stat(follow_symlinks=False)
                signature.append((entry.name, st.st_size, st.st_mtime_ns, st.st_ctime_ns))
    except OSError:
        return None
    return tuple(sorted(signature)) or None


def last_write_time(signature: VideoSignature) -> float:
    """Latest write to any of the videos. ctime is included because Samba clients can backdate mtime, but not ctime."""
    return max(max(mtime_ns, ctime_ns) for _name, _size, mtime_ns, ctime_ns in signature) / 1e9


def upload_finished(signature: VideoSignature, stability_seconds: float, now: float | None = None) -> bool:
    """True once every video is non-empty and none has been written for `stability_seconds`."""
    if any(size == 0 for _name, size, _mtime_ns, _ctime_ns in signature):
        return False
    return (time.time() if now is None else now) - last_write_time(signature) >= stability_seconds


def frame_index(path: str) -> int:
    """Frame number from a file name such as `frame_002262.png`."""
    return int(os.path.splitext(os.path.basename(path))[0].split("_")[-1])


def delete_raw_frames(address: str) -> None:
    """Deletes the extracted frames of a finished view. Leveled frames (needed by the front view and post_run) and `frames/Rotation.png` are kept."""
    for image in glob.glob(os.path.join(address, FRAMES_FOLDER, "frame_*")):
        os.remove(image)



class Pipeline:
    """Tools shared by all folders: a frame extractor and a leveler per view, and the side view's persistent worker pools."""

    def __init__(self) -> None:
        self.extractors = {view: FrameExtractor.FrameExtractor(SECONDARY_ADDRESS=view) for view in VIEWS}
        self.levelers   = {view: BaseLine.BaseLine(SECONDARY_ADDRESS=view) for view in VIEWS}
        self.yolo       = Utilities.YoloWalker(num_workers=10)
        self.s4         = CaMeasurer.processes_mp_shared(num_workers=8)
        self.space      = ProcessingSpace()

    def close(self) -> None:
        self.yolo.close()
        self.s4.close()

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def process_side(_folder: str, pipeline: Pipeline) -> None:
    """Side view: 
    frames -> leveled frames -> drop filtering -> crops -> 
    contact angles (`side/result.csv`)."""
    address = os.path.join(_folder, SIDE)

    # Phase 1-2: Frame extraction and base line detection (video -> side/frames -> side/frames_rotated)
    pipeline.extractors[SIDE].Forward(_folder)
    pipeline.levelers[SIDE].Forward(_folder)

    # Phase 3: Utilities (drop filtering and cropping)
    pipeline.yolo.run(image_folder=address, skip=40)
    pipeline.yolo.run(image_folder=address, skip=5)

    BaseUtils.FileIndexChecker(FolderAddress=address, frameAddress=ROTATED_FRAMES_FOLDER)
    images = BaseUtils.ImageLister(FolderAddress=address, frameAddress=ROTATED_FRAMES_FOLDER)
    Utilities.singleFolderDropNormalizer(images, BaseUtils.DropDetection_YOLO)  # type: ignore
    Utilities.crop_Save(image_folder=address)

    # Phase 4: 4S-SROF
    os.makedirs(os.path.join(address, 'SR_edge'), exist_ok=True)
    pipeline.s4.run(address)

    _ = Utilities.position_velocity_correction(os.path.join(address, 'result.csv'))


def process_front(_folder: str, pipeline: Pipeline) -> None:
    """Front view: frames -> leveled frames -> 
    keep the side view's frame range -> drop geometry (`front/result.csv`)."""
    side_frames = BaseUtils.ImageLister(os.path.join(_folder, SIDE), frameAddress=ROTATED_FRAMES_FOLDER)
    if not side_frames:
        raise FileNotFoundError(f"No frames in {os.path.join(_folder, SIDE, ROTATED_FRAMES_FOLDER)}; the front view needs the side view's frame range.")
    first, last = frame_index(side_frames[0]), frame_index(side_frames[-1])

    address = os.path.join(_folder, FRONT)

    # Phase 1-2: Frame extraction and base line detection (video -> front/frames -> front/frames_rotated)
    pipeline.extractors[FRONT].Forward(_folder)
    pipeline.levelers[FRONT].Forward(_folder)

    # Phase 3: Keep the frames inside the side view's range
    for image in BaseUtils.ImageLister(address, frameAddress=ROTATED_FRAMES_FOLDER):
        if not first <= frame_index(image) <= last:
            os.remove(image)

    # Phase 4: Drop geometry (front/result.csv and front/SR_edge)
    DropGeometry.process_folder(address)


def run_view(_folder: str, view: str, step: Callable[[str, Pipeline], None], pipeline: Pipeline) -> None:
    """
    Runs one view of a video folder from a clean state in its work folder, logging (not raising) on failure.
    On success writes `<view>/.done` and deletes the raw frames; on failure the log in `<view>/` stops the view from being retried.
    Either way the view's CSVs, logs and `.done` are then mirrored to the output folder.
    """
    work = work_folder(_folder)
    address = os.path.join(work, view)
    reset_view(_folder, view)  # a pending view can still hold partial outputs of an interrupted run
    try:
        print(f"Processing {view} view: {_folder} (in {work})")
        step(work, pipeline)

        if not os.path.isfile(os.path.join(address, 'error_log.txt')):
            with open(os.path.join(address, DONE), 'w') as f:
                f.write('Processing completed successfully.\n')
            delete_raw_frames(address)

    except Exception as e:
        os.makedirs(address, exist_ok=True)
        logger = logException.LogException(base_path=address)
        logger.log_exception(e, custom_message=f"Error processing {view} view of folder: {_folder}", Verbose=True)
        print(f"Error processing {view} view of {_folder}. Check the log in {output_folder(_folder)}/{view} for details.")
    finally:
        mirror_results(work, output_folder(_folder))


def rotated_frame_names(address: str) -> list[str]:
    """Names of the leveled frames of a view (`<view>/frames_rotated/frame_*`), in frame order."""
    return sorted((os.path.basename(p) for p in glob.glob(os.path.join(address, ROTATED_FRAMES_FOLDER, "frame_*"))), key=frame_index)


def _sample(frames: list[str], limit: int = 5) -> str:
    return ", ".join(frames[:limit]) + (f", ... ({len(frames)} in total)" if len(frames) > limit else "")


def experiment_jsons(_folder: str) -> list[str]:
    """JSON files of an experiment folder named after it, e.g. `tilt-20-rep-001_4000fps_camera.json`."""
    folder_name = os.path.basename(os.path.normpath(_folder))
    return sorted(p for p in glob.glob(os.path.join(_folder, "*.json"))
                  if os.path.splitext(os.path.basename(p))[0] == folder_name or os.path.basename(p).startswith(folder_name + "_"))


def tilt_from_name(*names: str) -> float | None:
    """Tilt angle in degrees from the first name containing `tilt-<angle>`, e.g. `tilt-20-rep-001_4000fps_camera.json` -> 20."""
    for name in names:
        match = TILT_PATTERN.search(os.path.basename(name))
        if match:
            return float(match.group(1))
    return None


def experiment_parameters(_folder: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Values used for the friction force, one row per parameter (value, unit, source), and any problems found.
    mass, gravity, viscosity and mStarmRatio come from the experiment JSON when it has them, otherwise from config.yaml;
    the tilt angle (degrees) comes from the JSON file name, or the folder name if there is no JSON.
    """
    issues: list[str] = []
    jsons = experiment_jsons(_folder)
    data: dict = {}
    for path in jsons:
        try:
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data = {**loaded, **data}
        except (OSError, ValueError) as e:
            issues.append(f"Could not read {os.path.basename(path)}: {e}")

    rows: list[dict] = []
    config = BaseUtils.config["Experiment_Parameters"]
    for key, unit in FRICTION_PARAMETERS.items():
        if isinstance(data.get(key), (int, float)) and not isinstance(data.get(key), bool):
            rows.append({"parameter": key, "value": float(data[key]), "unit": unit, "source": os.path.basename(jsons[0]) if len(jsons) == 1 else "JSON"})
        elif key in config:  # type: ignore
            rows.append({"parameter": key, "value": float(config[key]), "unit": unit, "source": "config.yaml"})  # type: ignore
        else:
            rows.append({"parameter": key, "value": np.nan, "unit": unit, "source": "missing"})
            issues.append(f"'{key}' is neither in the experiment JSON nor in config.yaml Experiment_Parameters; friction force not computed.")

    tilt_source = next((n for n in [*map(os.path.basename, jsons), os.path.basename(os.path.normpath(_folder))] if tilt_from_name(n) is not None), None)
    tilt = tilt_from_name(tilt_source) if tilt_source else None
    rows.append({"parameter": "tilt", "value": np.nan if tilt is None else tilt, "unit": "degree", "source": tilt_source or "missing"})
    if tilt is None:
        issues.append("No 'tilt-<angle>' in the JSON or folder name; friction force not computed.")

    if isinstance(data.get("fps"), (int, float)) and not np.isclose(float(data["fps"]), FPS_EXPERIMENT):
        issues.append(f"The camera JSON has fps = {data['fps']:g}, but config.yaml fps_experiment = {FPS_EXPERIMENT:g} was used for time, velocity and acceleration.")
    return pd.DataFrame(rows, columns=["parameter", "value", "unit", "source"]), issues


def smoothed_acceleration(frames: pd.Series, 
                          x_cm: pd.Series, 
                          window: int = ACCELERATION_WINDOW, 
                          fps: float = FPS_EXPERIMENT) -> np.ndarray:
    """
    Acceleration (m/s^2) along x: for each frame, twice the quadratic coefficient of a least-squares parabola fitted to
    the position over the `window` frames around it (shifted inward at the ends). This equals a Savitzky–Golay second
    derivative when no frame is missing, and uses the true frame times when some are.
    """
    index = frames.map(frame_index).to_numpy()
    t = (index - index[0]) / fps
    x_m = x_cm.to_numpy(dtype=float) / 100  # cm -> m
    span = max(window - 1, 0)
    acceleration = np.full(len(index), np.nan)
    for i, n in enumerate(index):
        lo = max(min(n - span // 2, index[-1] - span), index[0])
        start, stop = np.searchsorted(index, lo), np.searchsorted(index, lo + span, side="right")
        if stop - start >= 5:
            acceleration[i] = 2 * np.polyfit(t[start:stop] - t[i], x_m[start:stop], 2)[0]
    return acceleration


def post_run(_folder: str) -> pd.DataFrame:
    """
    Merges `side/result.csv` and `front/result.csv` into `<folder>/result.csv`, one row per frame (matched on 'file number').
    Side columns are prefixed with 'side ', front columns with 'front '. Two columns are added: 'a (m/s^2)', the
    smoothed acceleration down the slope, and 'friction force (uN)' = m·g·sin(tilt) − m·(m*/m)·a. The values used
    (and where each came from) are written to `<folder>/experiment_parameters.csv`.

    Checks that both views finished without errors, that both kept the same frames (`<view>/frames_rotated`), and that
    every kept frame has exactly one complete row in both CSVs, with the side time matching the frame number.
    The side view never measures the last frame (its velocity is a forward difference), so that frame is left out.
    Problems are written to `<folder>/post_run_error.txt`; `result.csv` then holds only the frames both views measured.
    """
    for name in (MERGED_CSV, POST_RUN_ERROR, PARAMETERS_CSV):
        path = os.path.join(_folder, name)
        if os.path.isfile(path):
            os.remove(path)

    issues: list[str] = []
    for view in VIEWS:
        if not os.path.isfile(os.path.join(_folder, view, '.done')):
            issues.append(f"The {view} view did not finish successfully (no {view}/.done).")
    # CaMeasurer logs failed frames next to the cropped images, DropGeometry in the view folder
    for log in (os.path.join(SIDE, 'error_log.txt'), os.path.join(SIDE, DATABASES_FOLDER, 'error_log.txt'), os.path.join(FRONT, 'error_log.txt')):
        if os.path.isfile(os.path.join(_folder, log)):
            issues.append(f"Some frames could not be processed; see {log}.")

    side_frames = rotated_frame_names(os.path.join(_folder, SIDE))
    front_frames = rotated_frame_names(os.path.join(_folder, FRONT))
    if not side_frames:
        issues.append(f"No frames in {SIDE}/{ROTATED_FRAMES_FOLDER}.")
    if set(side_frames) != set(front_frames):
        only_side = sorted(set(side_frames) - set(front_frames), key=frame_index)
        only_front = sorted(set(front_frames) - set(side_frames), key=frame_index)
        if only_side:
            issues.append(f"Frames in {SIDE}/{ROTATED_FRAMES_FOLDER} but not in {FRONT}/{ROTATED_FRAMES_FOLDER}: {_sample(only_side)}")
        if only_front:
            issues.append(f"Frames in {FRONT}/{ROTATED_FRAMES_FOLDER} but not in {SIDE}/{ROTATED_FRAMES_FOLDER}: {_sample(only_front)}")

    expected = side_frames[:-1]
    expected_set = set(expected)

    tables: dict[str, pd.DataFrame] = {}
    for view in VIEWS:
        df = pd.read_csv(os.path.join(_folder, view, 'result.csv'))
        names = df[FRAME_COLUMN].astype(str)

        duplicated = sorted(set(names[names.duplicated()]), key=frame_index)
        if duplicated:
            issues.append(f"{view}/result.csv has more than one row for: {_sample(duplicated)}")
        missing = sorted(expected_set - set(names), key=frame_index)
        if missing:
            issues.append(f"{view}/result.csv has no row for: {_sample(missing)}")
        unexpected = sorted(set(names) - set(side_frames), key=frame_index)
        if unexpected:
            issues.append(f"{view}/result.csv has rows for frames not in {SIDE}/{ROTATED_FRAMES_FOLDER}: {_sample(unexpected)}")
        incomplete = sorted(set(names[df.isna().any(axis=1)]), key=frame_index)
        if incomplete:
            issues.append(f"{view}/result.csv has empty values for: {_sample(incomplete)}")

        tables[view] = df[names.isin(expected_set)].drop_duplicates(FRAME_COLUMN)

    # The side view numbers time by row, so a skipped frame shifts the time (and velocity) of the frames after it
    side = tables[SIDE]
    if side_frames and len(side):
        frame_time = (side[FRAME_COLUMN].map(frame_index) - frame_index(side_frames[0])) / FPS_EXPERIMENT
        shifted = side.loc[~np.isclose(side["time (s)"], frame_time, rtol=0, atol=0.1 / FPS_EXPERIMENT), FRAME_COLUMN].tolist()
        if shifted:
            issues.append(f"side/result.csv 'time (s)' does not match the frame number (expected (frame - {frame_index(side_frames[0])}) / {FPS_EXPERIMENT:g}) for: {_sample(sorted(shifted, key=frame_index))}")

    merged = pd.merge(
        side.rename(columns=lambda c: c if c == FRAME_COLUMN else f"{SIDE} {c}"),
        tables[FRONT].rename(columns=lambda c: c if c == FRAME_COLUMN else f"{FRONT} {c}"),
        on=FRAME_COLUMN, how="inner", validate="one_to_one",
    )
    merged = merged.sort_values(FRAME_COLUMN, key=lambda s: s.map(frame_index)).reset_index(drop=True)

    # Friction force F = m·g·sin(tilt) − m·(m*/m)·a, with a measured along the direction of motion (down the slope)
    parameters, parameter_issues = experiment_parameters(_folder)
    issues.extend(parameter_issues)
    p = dict(zip(parameters["parameter"], parameters["value"]))
    x_column = "side x_center (cm)"
    if x_column in merged.columns and len(merged):
        direction = 1.0 if merged[x_column].iloc[-1] >= merged[x_column].iloc[0] else -1.0
        merged["a (m/s^2)"] = direction * smoothed_acceleration(merged[FRAME_COLUMN], merged[x_column])
    else:
        merged["a (m/s^2)"] = np.nan
        issues.append(f"{SIDE}/result.csv has no '{x_column}' column; acceleration and friction force not computed.")
    merged["friction force (uN)"] = (p["mass"] * p["gravity"] * np.sin(np.radians(p["tilt"]))
                                     - p["mass"] * p["mStarmRatio"] * merged["a (m/s^2)"]) * 1e6  # N -> µN
    parameters.to_csv(os.path.join(_folder, PARAMETERS_CSV), index=False)
    merged.to_csv(os.path.join(_folder, MERGED_CSV), index=False)

    if issues:
        with open(os.path.join(_folder, POST_RUN_ERROR), 'w', encoding='utf-8') as f:
            f.write(f"{len(merged)} of {len(expected)} frames were merged into {MERGED_CSV}.\n")
            f.write("\n".join(issues) + "\n")
        print(f"Merged {len(merged)} of {len(expected)} frames of {_folder} with problems; see {POST_RUN_ERROR}.")
    else:
        print(f"Merged {len(merged)} frames of {_folder} into {MERGED_CSV}.")
    return merged


def process_folder(_folder: str, pipeline: Pipeline) -> None:
    """
    Processes one video folder in its work folder: the side view, then the front view (each only if pending according
    to the output mirror), then merges their results. Afterwards the work folder's usage is recorded and the
    temporary folder is trimmed back to its limits.
    """
    work, out = work_folder(_folder), output_folder(_folder)
    space: ProcessingSpace = pipeline.space
    space.enforce(keep=work)
    links = link_videos(_folder, work)
    copy_experiment_json(_folder, work)
    try:
        # The temporary folder may have been trimmed since an interrupted run: redo the views whose work outputs are gone
        if (is_pending(out, FRONT) or needs_post_run(out)) and not is_pending(out, SIDE) and not has_work_results(work, SIDE):
            for view in VIEWS:
                reset_view(_folder, view)
        elif needs_post_run(out) and not has_work_results(work, FRONT):
            reset_view(_folder, FRONT)

        if is_pending(out, SIDE):
            run_view(_folder, SIDE, process_side, pipeline)

        if is_pending(out, FRONT):
            if side_has_results(out):
                run_view(_folder, FRONT, process_front, pipeline)
            else:
                print(f"Skipping front view of {_folder}: the side view has no results.")

        if needs_post_run(out):
            try:
                post_run(work)
                with open(os.path.join(work, DONE), 'w', encoding='utf-8') as f:
                    f.write("")
            except Exception:
                with open(os.path.join(work, POST_RUN_ERROR), 'a', encoding='utf-8') as f:
                    f.write(traceback.format_exc())
                print(f"Error merging the results of {_folder}; see {os.path.join(out, POST_RUN_ERROR)}.")
    finally:
        for link in links:
            if os.path.lexists(link):
                os.remove(link)
        mirror_results(work, out)
        space.update(work)
        space.enforce(keep=work)


class GracefulShutdown:
    """Tracks a shutdown request from SIGTERM/SIGINT, checked between folders (not mid-processing)."""

    def __init__(self) -> None:
        self.event = threading.Event()
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    @property
    def requested(self) -> bool:
        return self.event.is_set()

    def _handle(self, signum, frame) -> None:
        print(f"Received signal {signum}, will stop after the current folder finishes.", flush=True)
        self.event.set()


class UploadQueue:
    """
    FIFO of folders whose videos finished uploading. `scan()` (run by a watcher thread) queues each pending folder
    once its videos are quiet, oldest upload first; a folder is never queued twice, and is not retried after being
    processed unless its videos change.
    """

    def __init__(self, stability_seconds: float) -> None:
        self.stability_seconds = stability_seconds
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._queued: set[str] = set()
        self._attempted: dict[str, VideoSignature] = {}

    def scan(self) -> None:
        now = time.time()
        ready: list[tuple[float, str]] = []
        for _folder in discover_folders():
            with self._lock:
                if _folder in self._queued:
                    continue
                attempted = self._attempted.get(_folder)
            signature = video_signature(_folder)
            if signature is None or signature == attempted:
                continue
            if not upload_finished(signature, self.stability_seconds, now) or not needs_processing(_folder):
                continue
            ready.append((last_write_time(signature), _folder))

        for _uploaded, _folder in sorted(ready):
            with self._lock:
                self._queued.add(_folder)
            self._queue.put(_folder)
            print(f"Queued ({self._queue.qsize()} waiting): {_folder}", flush=True)

    def get(self, timeout: float) -> str:
        return self._queue.get(timeout=timeout)

    def qsize(self) -> int:
        return self._queue.qsize()

    def release(self, _folder: str, attempted: VideoSignature | None = None) -> None:
        """Called once a dequeued folder is handled; `attempted` stops it from being queued again for the same videos."""
        with self._lock:
            self._queued.discard(_folder)
            if attempted is not None:
                self._attempted[_folder] = attempted


def watch_uploads(uploads: UploadQueue, stop: threading.Event, poll_interval: float) -> None:
    """Watcher thread: rescans Root_Address/Video_Addresses every `poll_interval` seconds until `stop` is set."""
    while not stop.is_set():
        try:
            uploads.scan()
        except Exception:
            print(f"Error while scanning {VIDEO_SOURCE_GLOB}:\n{traceback.format_exc()}", flush=True)
        stop.wait(poll_interval)


def main_loop(pipeline: Pipeline, poll_interval: float = POLL_INTERVAL, stability_seconds: float = STABILITY_SECONDS) -> None:
    """Continuously watch Root_Address/Video_Addresses and process, in upload order, every folder whose videos finished uploading."""
    shutdown = GracefulShutdown()
    uploads = UploadQueue(stability_seconds)
    watcher = threading.Thread(target=watch_uploads, args=(uploads, shutdown.event, poll_interval), name="upload-watcher", daemon=True)
    watcher.start()
    print(f"Watching {VIDEO_SOURCE_GLOB} (stability: {stability_seconds:g} s, poll interval: {poll_interval:g} s); "
          f"processing in {PROCESSING_ROOT}, CSVs and logs mirrored to {OUTPUT_ROOT}", flush=True)

    while not shutdown.requested:
        try:
            _folder = uploads.get(timeout=1.0)
        except queue.Empty:
            continue

        # The video may have been rewritten while the folder waited in the queue; the watcher re-queues it once quiet.
        signature = video_signature(_folder)
        if signature is None or not upload_finished(signature, stability_seconds) or not needs_processing(_folder):
            uploads.release(_folder)
            continue

        try:
            print(f"Processing ({uploads.qsize()} waiting): {_folder}", flush=True)
            process_folder(_folder, pipeline)
        except Exception:
            print(f"Unexpected error processing {_folder}:\n{traceback.format_exc()}", flush=True)
        finally:
            uploads.release(_folder, attempted=signature)

    watcher.join(timeout=5.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Video Processor (AVP): side view, then front view, of every video folder")
    parser.add_argument("-C", "--clean", nargs="*", choices=VIEWS, metavar="VIEW", help="Delete the outputs of these views (default: side and front) before processing, so they are recomputed")
    parser.add_argument("--video-list", nargs="*", help="Video folders (inside Root_Address/Video_Addresses) to clean with -C, in the work folder and the output mirror (default: every video folder)")
    parser.add_argument("--daemon", action="store_true", help="Run continuously: watch Root_Address/Video_Addresses and process each folder, in upload order, once its video stops being written")
    parser.add_argument("--poll-interval", type=float, default=POLL_INTERVAL, help=f"Seconds between scans of Root_Address/Video_Addresses in --daemon mode (default: Daemon.poll_interval in config.yaml = {POLL_INTERVAL:g})")
    parser.add_argument("--stability-seconds", type=float, default=STABILITY_SECONDS, help=f"Seconds a video must go unwritten before its folder is queued (default: Daemon.stability_seconds in config.yaml = {STABILITY_SECONDS:g})")
    args = parser.parse_args()

    if args.clean is not None:
        cleanStart(sorted(args.video_list) if args.video_list else discover_folders(), views=tuple(args.clean) or VIEWS)
        print("cleanStart completed.")

    with Pipeline() as pipeline:
        if args.daemon:
            main_loop(pipeline, poll_interval=args.poll_interval, stability_seconds=args.stability_seconds)
        else:
            for _folder in discover_folders():
                if not needs_processing(_folder):
                    print(f"Skipping already-processed folder: {_folder}")
                    continue
                process_folder(_folder, pipeline)
