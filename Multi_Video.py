"""
    Author:      Yassin Riyazi
    Date:        29-07-2025
    Description: This script opens multiple videos in a grid format, allowing for easy viewing and navigation.

    TODO:
        - [01-09-2025] Type hinting for all function arguments and return types
"""
import os
import cv2
import glob
import numpy as np
import tkinter as tk  # used to get screen resolution as a fallback


def CleanUp(caps: list[cv2.VideoCapture]) -> None:
    """
    Releases all video capture objects and closes all OpenCV windows.
    """
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


def _get_screen_resolution() -> tuple[int, int]:
    """Return screen resolution using tkinter (cross-platform)."""
    root = tk.Tk()
    
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    w = 1280 if w <= 0 else w
    h = 720 if h <= 0 else h
    return w, h


def MultiVideo(
    video_paths: list[str],
    VideoGrid: tuple[int, int] = (3, 5),
    paused: bool = False,
    show_paths: bool = False,
    show_progress: bool = False,
) -> bool:
    """
    Opens multiple videos in a grid format with adaptive per-cell sizing that covers the whole window.
    Videos are stretched to exactly fill their grid cell (aspect ratio NOT preserved).
    """
    _row, _col = VideoGrid
    num_videos = _row * _col

    # Open captures (but limit later to num_videos to avoid index errors)
    caps: list[cv2.VideoCapture] = []
    total_frames: list[int] = []
    video_labels: list[str] = []

    for path in video_paths:
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"Error: Could not open video {path}")
            # release any opened captures and exit
            for c in caps:
                c.release()
            exit(1)
        caps.append(cap)
        total_frames.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        # precompute a short label (safe even if path has fewer components)
        parts = path.split(os.sep)
        video_labels.append("/".join(parts[-4:-1]) if len(parts) >= 4 else path)

    # If more videos were provided than the grid can show, truncate:
    if len(caps) > num_videos:
        print(f"Warning: {len(caps)} videos provided but grid has {num_videos} slots. Truncating.")
        # release the extras
        for c in caps[num_videos:]:
            c.release()
        caps = caps[:num_videos]
        total_frames = total_frames[:num_videos]
        video_labels = video_labels[:num_videos]

    # Determine an initial output_size based on screen resolution (fallback)
    screen_w, screen_h = _get_screen_resolution()
    # create a window and set fullscreen to initialize the window size to screen size
    window_name = f"{_row}x{_col} Video Grid"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # ensure window stays on top if supported
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    # Query actual window size; fallback to screen dims if not available
    try:
        x, y, win_w, win_h = cv2.getWindowImageRect(window_name)
    except Exception:
        win_w, win_h = screen_w, screen_h

    if win_w <= 0 or win_h <= 0:
        win_w, win_h = screen_w, screen_h

    # initial per-cell size (width, height)
    output_size = (max(1, win_w // _col), max(1, win_h // _row))

    # prepare frames list (one per grid cell)
    blank_frame = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    frames = [blank_frame.copy() for _ in range(num_videos)]

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        # try to get current window size each loop (so resizing / fullscreen toggle is handled)
        try:
            x, y, win_w, win_h = cv2.getWindowImageRect(window_name)
        except Exception:
            # if getWindowImageRect fails, keep previous size (or screen fallback)
            win_w, win_h = win_w, win_h

        if win_w <= 0 or win_h <= 0:
            # fallback to screen resolution
            screen_w, screen_h = _get_screen_resolution()
            win_w, win_h = screen_w, screen_h

        new_output_size = (max(1, win_w // _col), max(1, win_h // _row))
        if new_output_size != output_size:
            # update blank_frame and resize every existing frame to the new size so stacking works
            output_size = new_output_size
            blank_frame = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
            for i in range(num_videos):
                # resize current contents (could be blank or previous frames)
                frames[i] = cv2.resize(frames[i], output_size, interpolation=cv2.INTER_LINEAR)

        if not paused:
            checking_end_all_video = 0
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret or frame is None:
                    # set this slot to blank
                    frames[i][:] = 0
                    continue

                # Stretch/rescale frame to exactly the cell size (aspect ratio NOT preserved)
                if frame.shape[1::-1] != output_size:
                    frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)

                if show_paths:
                    cv2.putText(frame, video_labels[i], (10, 30), font, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

                # progress calculation
                pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                total = total_frames[i] if i < len(total_frames) and total_frames[i] > 0 else 1
                percent_left = max(0, 100 - int((pos / total) * 100))
                checking_end_all_video += percent_left
                if show_progress:
                    cv2.putText(frame, f"{percent_left}% left", (10, 55), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

                frames[i] = np.asarray(frame, dtype=np.uint8)

            if checking_end_all_video == 0:
                # all videos finished
                break

        # Stack rows: ensure every slice has exactly _col elements (frames already sized)
        row_stack = [np.hstack(frames[r * _col:(r + 1) * _col]) for r in range(_row)]
        grid = np.vstack(row_stack)

        cv2.imshow(window_name, grid)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            CleanUp(caps)
            return True
        elif key == ord(" "):
            paused = not paused
        elif key in (ord("a"), ord("A")):
            show_paths = not show_paths
        elif key in (ord("v"), ord("V")):
            show_progress = not show_progress
        elif key == 81:  # Left arrow (when paused)
            if paused:
                for i, cap in enumerate(caps):
                    cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(cur - 30, 0))
        elif key == 83:  # Right arrow (when paused)
            if paused:
                for i, cap in enumerate(caps):
                    cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, min(cur + 30, total_frames[i]))
        elif key == ord("f"):  # Toggle full screen
            current = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            if current == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    CleanUp(caps)
    return True


if __name__ == "__main__":
    videos: list[str] = []
    for _idx, rep in enumerate(
        glob.glob(os.path.join("/media/Dont/Teflon-AVP/280/S2-SNr2.1_D", "*", "video.mp4"))
    ):
        videos.append(rep)

    _end = len(videos)
    VideoGrid = (8, 2)
    length_of_grid = VideoGrid[0] * VideoGrid[1]

    for lis in range(len(videos) - length_of_grid, 0, -length_of_grid):
        print(len(videos[lis:_end]))
        MultiVideo(videos[lis:_end], VideoGrid=VideoGrid, show_paths=True, show_progress=True)
