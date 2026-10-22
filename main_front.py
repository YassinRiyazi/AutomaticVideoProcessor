import  os
import  pathlib
import  time
import  glob
import  signal
import  tqdm
import  shutil
import  argparse
import  BaseLine
import  BaseUtils
import  Utilities
import  CaMeasurer
import  DropGeometry
import  FrameExtractor
from    cleanUp         import  create_video_from_images # type:ignore

SECONDARY_ADDRESS: str = "front"
VIDEO_SOURCE_GLOB: str = os.path.join(str(BaseUtils.config['Root_Address']),
                                      str(BaseUtils.config['Video_Addresses']), "*")


def cleanUP(_folder: str|os.PathLike[str]) -> None:
    shutil.rmtree(os.path.join(_folder, SECONDARY_ADDRESS, "frames"),          ignore_errors=True)
    shutil.rmtree(os.path.join(_folder, SECONDARY_ADDRESS, "frames_rotated"),  ignore_errors=True)
    shutil.rmtree(os.path.join(_folder, SECONDARY_ADDRESS, "databases"),       ignore_errors=True)
    shutil.rmtree(os.path.join(_folder, SECONDARY_ADDRESS, "SR_edge"),         ignore_errors=True)

    for name in ('error_log.txt', 'result.csv', 'result_corrected.csv', '.done'):
        path = os.path.join(_folder, SECONDARY_ADDRESS, name)
        if os.path.isfile(path):
            os.remove(path)

    for log in glob.glob(os.path.join(_folder, SECONDARY_ADDRESS, '*.log')):
        os.remove(log)

def cleanStart(Video_list: list[str] = sorted(glob.glob(VIDEO_SOURCE_GLOB))):  
    for _folder in tqdm.tqdm(Video_list):
        cleanUP(_folder)

def discover_folders(pattern: str = VIDEO_SOURCE_GLOB) -> list[str]:
    """List candidate video folders (directories only), sorted."""
    Video_list = sorted(glob.glob(pattern))
    return [folder for folder in Video_list if os.path.isdir(folder)]

def is_pending(_folder: str|os.PathLike[str]) -> bool:
    """True if a folder has not been processed yet and has no unresolved error/log files."""
    if os.path.isfile(os.path.join(_folder, SECONDARY_ADDRESS, '.done')):
        return False
    if os.path.isfile(os.path.join(_folder, SECONDARY_ADDRESS, 'error_log.txt')):
        return False
    if len(glob.glob(os.path.join(_folder, SECONDARY_ADDRESS, '*.log'))) > 0:
        return False
    return True

def get_max_mtime(_folder: str|os.PathLike[str]) -> float:
    """Most recent modification time of the folder itself or any file under it (recursive)."""
    latest = os.path.getmtime(_folder)
    for root, _dirs, files in os.walk(_folder):
        for name in files:
            try:
                latest = max(latest, os.path.getmtime(os.path.join(root, name)))
            except OSError:
                continue
    return latest

def process_folder(_folder: str|os.PathLike[str], fe, bld, YOLO, S4) -> None:
    """Run the full AVP pipeline on a single folder, logging (not raising) on failure.

    The video stays in `_folder`; every intermediate and result file is written to `_folder/SECONDARY_ADDRESS`.
    """
    address: str = os.path.join(_folder, SECONDARY_ADDRESS)
    try:
        print(f"Processing folder: {_folder}")

        # Phase 1: Frame Extraction (reads the video in _folder, writes to address/frames)
        if not os.path.exists(os.path.join(address, 'frames')):
            fe.Forward(_folder)
        # Phase 2: Base Line Detection (address/frames -> address/frames_rotated)
        if not os.path.exists(os.path.join(address, 'frames_rotated')):
            bld.Forward(_folder)

        # read indexes from "side/frames_rotated"
        files = glob.glob(os.path.join(_folder, "side",'frames_rotated', '*'))
        files.sort()  # Sort the files to maintain a consistent order
        def filename_ext(file_path: str) -> str:
            return int(os.path.splitext(file_path.split(os.sep)[-1].split("_")[-1])[0])
        beginning_index = filename_ext(files[0]) if files else None
        last_index = filename_ext(files[-1]) if files else None
        files = glob.glob(os.path.join(_folder, SECONDARY_ADDRESS,'frames_rotated', '*'))
        
        # delete frames inside the range from beginning_index to last_index

        if beginning_index and last_index:
            for file_path in files:
                if  (filename_ext(file_path) < beginning_index) or \
                    (filename_ext(file_path) > last_index):
                    try:
                        os.remove(file_path)
                    except OSError:
                        continue
                else:
                    pass

        # Phase 3: Drop geometry (front/frames_rotated -> front/result.csv)
        DropGeometry.process_folder(address)

        if not os.path.isfile(os.path.join(address, 'error_log.txt')):
            with open(os.path.join(address, '.done'), 'w') as f:
                f.write('Processing completed successfully.\n')

        os.rmdir(os.path.join(address, 'frames'))

    except Exception as e:
        import BaseUtils.logException as logException

        # Log inside the side folder so is_pending() sees it and the folder is not retried endlessly
        os.makedirs(address, exist_ok=True)
        logger = logException.LogException(base_path=address)
        logger.log_exception(e, custom_message=f"Error processing folder: {_folder}", Verbose=True)
        print(f"Error processing folder: {_folder}. Check the log in {address} for details.")


class GracefulShutdown:
    """Tracks a shutdown request from SIGTERM/SIGINT, checked between folders (not mid-processing)."""

    def __init__(self) -> None:
        self.requested = False
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame) -> None:
        print(f"Received signal {signum}, will stop after the current folder finishes.")
        self.requested = True


def main_loop(poll_interval: float = 60.0, stability_seconds: float = 60.0) -> None:
    """Continuously scan for new folders and process each once its contents stop changing."""
    fe = FrameExtractor.FrameExtractor(SECONDARY_ADDRESS=SECONDARY_ADDRESS)
    bld = BaseLine.BaseLine(SECONDARY_ADDRESS=SECONDARY_ADDRESS)

    YOLO = Utilities.YoloWalker(num_workers=5)
    S4 = CaMeasurer.processes_mp_shared(num_workers=8)

    shutdown = GracefulShutdown()
    # folder -> (last_seen_max_mtime, first_seen_stable_timestamp)
    stability_tracker: dict[str, tuple[float, float]] = {}

    try:
        while not shutdown.requested:
            for _folder in discover_folders():
                if shutdown.requested:
                    break

                if not is_pending(_folder):
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

                process_folder(_folder, fe, bld, YOLO, S4)
                stability_tracker.pop(_folder, None)

                break
            break

            for _ in range(int(poll_interval)):
                if shutdown.requested:
                    break
                time.sleep(1)
                
    finally:
        YOLO.close()
        S4.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Video Processor (AVP) launcher")
    parser.add_argument("-C", "--clean", action="store_true", help="Run cleanStart and exit")
    parser.add_argument("--video-list", nargs="*", help="Optional list of folders to clean (overrides default glob)")
    parser.add_argument("--daemon", action="store_true", help="Run continuously, polling for newly added folders instead of doing a single pass")
    parser.add_argument("--poll-interval", type=float, default=60.0, help="Seconds between folder rescans in --daemon mode (default: 60)")
    parser.add_argument("--stability-seconds", type=float, default=60.0, help="Seconds a folder must be unchanged before it's processed, to avoid mid-copy folders (default: 60)")
    args = parser.parse_args()

    if args.clean:
        if args.video_list:
            cleanStart(Video_list=sorted(args.video_list))
        else:
            cleanStart()
        print("cleanStart completed, exiting.")
        

    if args.daemon:
        main_loop(poll_interval=args.poll_interval, stability_seconds=args.stability_seconds)
    else:
        fe = FrameExtractor.FrameExtractor(SECONDARY_ADDRESS=SECONDARY_ADDRESS)
        bld = BaseLine.BaseLine(SECONDARY_ADDRESS=SECONDARY_ADDRESS)

        YOLO = Utilities.YoloWalker(num_workers=5)
        S4   = CaMeasurer.processes_mp_shared(num_workers=8)

        for _folder in discover_folders():
            if not is_pending(_folder):
                print(f"Skipping already-processed folder: {_folder}")
                continue
            process_folder(_folder, fe, bld, YOLO, S4)
            break

        YOLO.close()
        S4.close()
