"""
    Date    : 2025-09-30
    Author  : Yassin Riyazi
    Project : Automatic Video Processor (AVP)
    File    : AVP.py
    Version : 1.0.0
    License : GNU General Public License v3.0

    Description:
        This is the main file of the Automatic Video Processor (AVP) project. 
        It orchestrates the various components of the AVP system to process videos automatically.

    Components:
        + Frame Extractor.Extractor (Folder Address:str|os.pathlike[str]): Void []

        + Base Line detection.Forward ()

        + Utilities.Empty_frames_remover()
        + Utilities.Normalizing_drop_Position()
        + Utilities.Detect&crop()

        + SROF_4S.DF_generation(): Void []
        + SROF_4S.Result_video_maker(): Void []

        + CleanUP(): Void []
"""
import  os
import  time
import  glob
import  signal
import  tqdm
import  shutil
import  argparse
import  BaseLine
import  Utilities
import  CaMeasurer
import  FrameExtractor
from    cleanUp         import  create_video_from_images # type:ignore

VIDEO_SOURCE_GLOB = os.path.join("/run/media/d25u2/Dont/output/2026-10-16_18-12-14-317689", "*")


def cleanUP(_folder: str|os.PathLike[str]) -> None:
    shutil.rmtree(os.path.join(_folder, "frames"),          ignore_errors=True)
    shutil.rmtree(os.path.join(_folder, "frames_rotated"),  ignore_errors=True)
    shutil.rmtree(os.path.join(_folder, "databases"),       ignore_errors=True)
    shutil.rmtree(os.path.join(_folder, "SR_edge"),         ignore_errors=True)
    os.remove(os.path.join(_folder, 'error_log.txt')) if os.path.isfile(os.path.join(_folder, 'error_log.txt')) else None
    os.remove(os.path.join(_folder, 'result.csv')) if os.path.isfile(os.path.join(_folder, 'result.csv')) else None
    os.remove(os.path.join(_folder, 'result_video.mkv')) if os.path.isfile(os.path.join(_folder, 'result_video.mkv')) else None

def cleanStart(Video_list: list[str] = sorted(glob.glob("/media/Dont/Teflon-AVP/*/*/*"))):  
    for _folder in tqdm.tqdm(Video_list):
        cleanUP(_folder)

        os.remove(os.path.join(_folder, '.done')) if os.path.isfile(os.path.join(_folder, '.done')) else None
        
        logs = glob.glob(os.path.join(_folder,'*.log'))
        for log in logs:
            os.remove(log)


def discover_folders(pattern: str = VIDEO_SOURCE_GLOB) -> list[str]:
    """List candidate video folders (directories only), sorted."""
    Video_list = sorted(glob.glob(pattern))
    return [folder for folder in Video_list if os.path.isdir(folder)]


def is_pending(_folder: str|os.PathLike[str]) -> bool:
    """True if a folder has not been processed yet and has no unresolved error/log files."""
    if os.path.isfile(os.path.join(_folder, '.done')):
        return False
    if os.path.isfile(os.path.join(_folder, 'error_log.txt')):
        return False
    if len(glob.glob(os.path.join(_folder, '*.log'))) > 0:
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
    """Run the full AVP pipeline on a single folder, logging (not raising) on failure."""
    try:
        cleanUP(_folder)

        print(f"Processing folder: {_folder}")

        # Phase 1: Frame Extraction
        if not os.path.exists(os.path.join(_folder, 'frames')):
            fe.Forward(_folder)
        # Phase 2: Base Line Detection
        if not os.path.exists(os.path.join(_folder, 'frames_rotated')):
            bld.Forward(_folder)

        # Phase 3: Utilities
        YOLO.run(image_folder=_folder, skip=40)
        YOLO.run(image_folder=_folder, skip=5)

        Utilities.BaseUtils.FileIndexChecker(FolderAddress=_folder, frameAddress=str(Utilities.BaseUtils.config["rotated_frames_folder"]))
        images = Utilities.BaseUtils.ImageLister(FolderAddress=_folder, frameAddress=str(Utilities.BaseUtils.config["rotated_frames_folder"]))
        Utilities.singleFolderDropNormalizer(images, Utilities.BaseUtils.DropDetection_YOLO)  # type: ignore
        Utilities.crop_Save(image_folder=_folder)

        os.makedirs(os.path.join(_folder, 'SR_edge'), exist_ok=True)

        # Phase 4: 4S-SROF
        S4.run(_folder)

        _ = Utilities.position_velocity_correction(os.path.join(_folder, 'result.csv'))

        if not os.path.isfile(os.path.join(_folder, 'error_log.txt')):
            with open(os.path.join(_folder, '.done'), 'w') as f:
                f.write('Processing completed successfully.\n')

    except Exception as e:
        import BaseUtils.logException as logException

        logger = logException.LogException(base_path=_folder)
        logger.log_exception(e, custom_message=f"Error processing folder: {_folder}", Verbose=True)
        print(f"Error processing folder: {_folder}. Check error_log.txt for details.")


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
    fe = FrameExtractor.FrameExtractor()
    bld = BaseLine.BaseLine()

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
        fe = FrameExtractor.FrameExtractor()
        bld = BaseLine.BaseLine()

        YOLO = Utilities.YoloWalker(num_workers=5)
        S4   = CaMeasurer.processes_mp_shared(num_workers=8)

        for _folder in discover_folders():
            if not is_pending(_folder):
                print(f"Skipping already-processed folder: {_folder}")
                continue
            process_folder(_folder, fe, bld, YOLO, S4)

        YOLO.close()
        S4.close()
