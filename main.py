"""
    Project : Automatic Video Processor (AVP)
    File    : main.py
    License : GNU General Public License v3.0

    Description:
        Unified pipeline. For every video folder it processes the side view first (contact angles, CaMeasurer) and
        then the front view (drop geometry, DropGeometry). The front view keeps only the frames the side view kept.
        The video stays in the folder; all outputs go to `<folder>/side` and `<folder>/front`.
        Once both views finish, `post_run` checks that their frames match and merges both CSVs into `<folder>/result.csv`.

    Usage:
        python main.py                                  # one pass over all folders with a pending view
        python main.py --daemon                         # watch Root_Address; queue each folder once its video
                                                        # has not been written for Daemon.stability_seconds
        python main.py -C                               # delete all side and front outputs first, then process
        python main.py -C front --video-list /path/a    # redo only the front view of these folders
"""
import  os
import  glob
import  time
import  queue
import  shutil
import  signal
import  argparse
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

VIDEO_SOURCE_GLOB: str = os.path.join(str(BaseUtils.config['Root_Address']), "*")
FRAMES_FOLDER: str = str(BaseUtils.config["frame_folder"])
ROTATED_FRAMES_FOLDER: str = str(BaseUtils.config["rotated_frames_folder"])
DATABASES_FOLDER: str = str(BaseUtils.config["databases_folder"])
FPS_EXPERIMENT: float = float(BaseUtils.config["Experiment_Parameters"]["fps_experiment"])  # type: ignore

MERGED_CSV: str = "result.csv"                   # `<folder>/result.csv`, written by post_run
POST_RUN_ERROR: str = "post_run_error.txt"       # `<folder>/post_run_error.txt`, problems found by post_run
FRAME_COLUMN: str = "file number"

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
    for name in (MERGED_CSV, POST_RUN_ERROR):
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


def cleanStart(Video_list: list[str], views: tuple[str, ...] = VIEWS) -> None:
    for _folder in tqdm.tqdm(Video_list, desc="Cleaning"):
        for view in views:
            cleanUP(_folder, view)


def discover_folders(pattern: str = VIDEO_SOURCE_GLOB) -> list[str]:
    """List candidate video folders (directories only), sorted."""
    return [folder for folder in sorted(glob.glob(pattern)) if os.path.isdir(folder)]


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
    return bool(pending_views(_folder)) or needs_post_run(_folder)


def video_signature(_folder: str) -> VideoSignature | None:
    """Size and timestamps of the video files directly inside a folder, or None if it has none (yet)."""
    signature: list[tuple[str, int, int, int]] = []
    try:
        with os.scandir(_folder) as entries:
            for entry in entries:
                # Hidden files are temporary upload files (e.g. rsync's `.name.mp4.XXXXXX`)
                if entry.name.startswith('.') or not entry.name.lower().endswith(FrameExtractor.VIDEO_EXTENSIONS):
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
    """Deletes the extracted frames of a finished view. Leveled frames and `frames/Rotation.png` are kept."""
    for image in glob.glob(os.path.join(address, FRAMES_FOLDER, "frame_*")):
        os.remove(image)


class Pipeline:
    """Tools shared by all folders: a frame extractor and a leveler per view, and the side view's persistent worker pools."""

    def __init__(self) -> None:
        self.extractors = {view: FrameExtractor.FrameExtractor(SECONDARY_ADDRESS=view) for view in VIEWS}
        self.levelers   = {view: BaseLine.BaseLine(SECONDARY_ADDRESS=view) for view in VIEWS}
        self.yolo       = Utilities.YoloWalker(num_workers=10)
        self.s4         = CaMeasurer.processes_mp_shared(num_workers=8)

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
    Runs one view from a clean state, logging (not raising) on failure.
    On success writes `<view>/.done` and deletes the raw frames; on failure the log in `<view>/` stops the view from being retried.
    """
    address = os.path.join(_folder, view)
    cleanUP(_folder, view)  # a pending view can still hold partial outputs of an interrupted run
    try:
        print(f"Processing {view} view: {_folder}")
        step(_folder, pipeline)

        if not os.path.isfile(os.path.join(address, 'error_log.txt')):
            with open(os.path.join(address, '.done'), 'w') as f:
                f.write('Processing completed successfully.\n')
            delete_raw_frames(address)

    except Exception as e:
        os.makedirs(address, exist_ok=True)
        logger = logException.LogException(base_path=address)
        logger.log_exception(e, custom_message=f"Error processing {view} view of folder: {_folder}", Verbose=True)
        print(f"Error processing {view} view of {_folder}. Check the log in {address} for details.")


def rotated_frame_names(address: str) -> list[str]:
    """Names of the leveled frames of a view (`<view>/frames_rotated/frame_*`), in frame order."""
    return sorted((os.path.basename(p) for p in glob.glob(os.path.join(address, ROTATED_FRAMES_FOLDER, "frame_*"))), key=frame_index)


def _sample(frames: list[str], limit: int = 5) -> str:
    return ", ".join(frames[:limit]) + (f", ... ({len(frames)} in total)" if len(frames) > limit else "")


def post_run(_folder: str) -> pd.DataFrame:
    """
    Merges `side/result.csv` and `front/result.csv` into `<folder>/result.csv`, one row per frame (matched on 'file number').
    Side columns are prefixed with 'side ', front columns with 'front '.

    Checks that both views finished without errors, that both kept the same frames (`<view>/frames_rotated`), and that
    every kept frame has exactly one complete row in both CSVs, with the side time matching the frame number.
    The side view never measures the last frame (its velocity is a forward difference), so that frame is left out.
    Problems are written to `<folder>/post_run_error.txt`; `result.csv` then holds only the frames both views measured.
    """
    for name in (MERGED_CSV, POST_RUN_ERROR):
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
    """Runs the side view, then the front view, of one video folder (each only if pending), then merges their results."""
    if is_pending(_folder, SIDE):
        run_view(_folder, SIDE, process_side, pipeline)

    if is_pending(_folder, FRONT):
        if side_has_results(_folder):
            run_view(_folder, FRONT, process_front, pipeline)
        else:
            print(f"Skipping front view of {_folder}: the side view has no results.")

    if needs_post_run(_folder):
        try:
            post_run(_folder)
        except Exception:
            with open(os.path.join(_folder, POST_RUN_ERROR), 'a', encoding='utf-8') as f:
                f.write(traceback.format_exc())
            print(f"Error merging the results of {_folder}; see {POST_RUN_ERROR}.")


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
    """Watcher thread: rescans Root_Address every `poll_interval` seconds until `stop` is set."""
    while not stop.is_set():
        try:
            uploads.scan()
        except Exception:
            print(f"Error while scanning {VIDEO_SOURCE_GLOB}:\n{traceback.format_exc()}", flush=True)
        stop.wait(poll_interval)


def main_loop(pipeline: Pipeline, poll_interval: float = POLL_INTERVAL, stability_seconds: float = STABILITY_SECONDS) -> None:
    """Continuously watch Root_Address and process, in upload order, every folder whose videos finished uploading."""
    shutdown = GracefulShutdown()
    uploads = UploadQueue(stability_seconds)
    watcher = threading.Thread(target=watch_uploads, args=(uploads, shutdown.event, poll_interval), name="upload-watcher", daemon=True)
    watcher.start()
    print(f"Watching {VIDEO_SOURCE_GLOB} (stability: {stability_seconds:g} s, poll interval: {poll_interval:g} s)", flush=True)

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
    parser.add_argument("--video-list", nargs="*", help="Folders to clean with -C (default: every folder under Root_Address)")
    parser.add_argument("--daemon", action="store_true", help="Run continuously: watch Root_Address and process each folder, in upload order, once its video stops being written")
    parser.add_argument("--poll-interval", type=float, default=POLL_INTERVAL, help=f"Seconds between scans of Root_Address in --daemon mode (default: Daemon.poll_interval in config.yaml = {POLL_INTERVAL:g})")
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

