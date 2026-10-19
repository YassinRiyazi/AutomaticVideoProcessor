"""
    Project : Automatic Video Processor (AVP)
    File    : main.py
    License : GNU General Public License v3.0

    Description:
        Unified pipeline. For every video folder it processes the side view first (contact angles, CaMeasurer) and
        then the front view (drop geometry, DropGeometry). The front view keeps only the frames the side view kept.
        The video stays in the folder; all outputs go to `<folder>/side` and `<folder>/front`.

    Usage:
        python main.py                                  # one pass over all folders with a pending view
        python main.py --daemon                         # keep polling for new folders
        python main.py -C                               # delete all side and front outputs first, then process
        python main.py -C front --video-list /path/a    # redo only the front view of these folders
"""
import  os
import  glob
import  time
import  shutil
import  signal
import  argparse
from    typing  import  Callable

import  tqdm
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
    """Deletes every output of one view (`<folder>/<view>`), including its `.done` marker and logs."""
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


def get_max_mtime(_folder: str) -> float:
    """Most recent modification time of the folder itself or any file under it (recursive)."""
    latest = os.path.getmtime(_folder)
    for root, _dirs, files in os.walk(_folder):
        for name in files:
            try:
                latest = max(latest, os.path.getmtime(os.path.join(root, name)))
            except OSError:
                continue
    return latest


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
        self.yolo       = Utilities.YoloWalker(num_workers=5)
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


def process_folder(_folder: str, pipeline: Pipeline) -> None:
    """Runs the side view, then the front view, of one video folder (each only if pending)."""
    if is_pending(_folder, SIDE):
        run_view(_folder, SIDE, process_side, pipeline)

    if is_pending(_folder, FRONT):
        if side_has_results(_folder):
            run_view(_folder, FRONT, process_front, pipeline)
        else:
            print(f"Skipping front view of {_folder}: the side view has no results.")


class GracefulShutdown:
    """Tracks a shutdown request from SIGTERM/SIGINT, checked between folders (not mid-processing)."""

    def __init__(self) -> None:
        self.requested = False
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame) -> None:
        print(f"Received signal {signum}, will stop after the current folder finishes.")
        self.requested = True


def main_loop(pipeline: Pipeline, poll_interval: float = 60.0, stability_seconds: float = 60.0) -> None:
    """Continuously scan for new folders and process each once its contents stop changing."""
    shutdown = GracefulShutdown()
    # folder -> (last_seen_max_mtime, first_seen_stable_timestamp)
    stability_tracker: dict[str, tuple[float, float]] = {}

    while not shutdown.requested:
        for _folder in discover_folders():
            if shutdown.requested:
                break

            if not pending_views(_folder):
                stability_tracker.pop(_folder, None)
                continue

            now = time.time()
            current_mtime = get_max_mtime(_folder)
            last_mtime, first_stable_seen = stability_tracker.get(_folder, (None, None))

            if last_mtime != current_mtime:
                stability_tracker[_folder] = (current_mtime, now)
                continue

            if now - first_stable_seen < stability_seconds:
                continue

            process_folder(_folder, pipeline)
            stability_tracker.pop(_folder, None)

        for _ in range(int(poll_interval)):
            if shutdown.requested:
                break
            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Video Processor (AVP): side view, then front view, of every video folder")
    parser.add_argument("-C", "--clean", nargs="*", choices=VIEWS, metavar="VIEW", help="Delete the outputs of these views (default: side and front) before processing, so they are recomputed")
    parser.add_argument("--video-list", nargs="*", help="Folders to clean with -C (default: every folder under Root_Address)")
    parser.add_argument("--daemon", action="store_true", help="Run continuously, polling for newly added folders instead of doing a single pass")
    parser.add_argument("--poll-interval", type=float, default=60.0, help="Seconds between folder rescans in --daemon mode (default: 60)")
    parser.add_argument("--stability-seconds", type=float, default=60.0, help="Seconds a folder must be unchanged before it's processed, to avoid mid-copy folders (default: 60)")
    args = parser.parse_args()

    if args.clean is not None:
        cleanStart(sorted(args.video_list) if args.video_list else discover_folders(), views=tuple(args.clean) or VIEWS)
        print("cleanStart completed.")

    with Pipeline() as pipeline:
        if args.daemon:
            main_loop(pipeline, poll_interval=args.poll_interval, stability_seconds=args.stability_seconds)
        else:
            for _folder in discover_folders():
                if not pending_views(_folder):
                    print(f"Skipping already-processed folder: {_folder}")
                    continue
                process_folder(_folder, pipeline)

