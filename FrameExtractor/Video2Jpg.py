"""
    Author: Yassin Riyazi
    Date: 03-07-2025

    Change log:
        24-08-2025
        - Added type hints for function arguments and return types.
        - Improved error handling and logging.
        - Refactored code for better readability and maintainability.

"""
import  os
import  glob
import  subprocess
from    PIL         import Image
from    typing      import Union, Optional, Tuple

def init(addressFFprobe: Union[str, os.PathLike[str], None] = None,
         addressFFmpeg: Union[str, os.PathLike[str], None] = None) -> None:
    # checking if ffprobe is available
    if addressFFprobe is None:
        addressFFprobe = 'ffprobe'

    assert subprocess.run([addressFFprobe, '-version'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE).returncode == 0, "ffprobe not found. Please install FFmpeg."
    if addressFFmpeg is None:
        addressFFmpeg = 'ffmpeg'

    assert subprocess.run([addressFFmpeg, '-version'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE).returncode == 0, "ffmpeg not found. Please install FFmpeg."

def probe_video_dimensions(video_path: Union[str, os.PathLike[str]],
                           addressFFprobe: Union[str, os.PathLike[str], None] = None) -> Tuple[int, int]:
    """
    Uses ffprobe to get the width and height of the first video stream.

    args:
        video_path (Union[str, os.PathLike[str]]): Path to the video file.
        addressFFprobe (Union[str, os.PathLike[str], None]): Path to ffprobe executable. If None, assumes it's in PATH.

    returns:
        Tuple[int, int]: Tuple of (width, height) in pixels.

    raises:
        RuntimeError if ffprobe fails or video has no streams.
    """
    # checking if ffprobe is available
    if addressFFprobe is None:
        addressFFprobe = 'ffprobe'
        

    try:
        output = subprocess.check_output([
                                            addressFFprobe, '-v',
                                            'error',
                                            '-select_streams',
                                            'v:0',
                                            '-show_entries',
                                            'stream=width,height',
                                            '-of',
                                            'csv=p=0',
                                            str(video_path)
                                        ])
        width, height = map(int, output.strip().split(b',')[:2])
        return width, height
    
    except Exception as e:
        raise RuntimeError(f"Failed to probe video dimensions: {e}")

def prepare_output_directory(output_pattern: Union[str, os.PathLike[str]],
                             wipe: bool) -> str:
    """
    Ensures output directory exists and wipes existing files if requested.
    Returns the directory Path.
    args:
        output_pattern (Path): Pattern for the output files, must include directory.
        wipe (bool): If True, delete existing files in the output directory.

    returns:
        Path: The output directory path.
    """
    out_dir = os.path.dirname(output_pattern)

    if wipe:
        for filename in os.listdir(out_dir):
            path = os.path.join(out_dir, filename)
            if os.path.isfile(path):
                os.remove(path)
    os.makedirs(out_dir, exist_ok=True)
    
    return out_dir

def build_vf_filter(fps: int,
                    height: int,
                    min_height: int = 130,
                    grayscale: bool = True) -> str:
    """
    Constructs the ffmpeg -vf filter string, adding padding if height < min_height,
    and converting to grayscale if requested.
    args:
        fps (int): Frames per second to extract.
        height (int): Height of the video stream.
        min_height (int): Minimum height for padding.
        grayscale (bool): If True, convert frames to grayscale.

    returns:
        str: The constructed filter string for ffmpeg.

    TODO:
        - Add support Rotation and maybe save the rotated video 
    """
    filters = [f'fps={fps}']

    if height < min_height:
        diff = min_height - height
        # pad format: pad=width:height+diff:x:y:color
        """
        Explanation of Your Padding (pad=iw:ih+20:0:20:black)
            iw (input width)                    → Keeps the width unchanged.
            ih+20 (input height + 20 pixels)    → Adds 20 extra pixels to the height.
            0 (x_offset)                        → The original video stays at the same horizontal position (not shifted left or right).
            20 (y_offset)                       → Moves the original video down by 20 pixels, so the extra space appears at the bottom.
            black                               → Fills the new padding area with black color.
        """
        filters.append(f'pad=iw:ih+{diff}:0:{diff}:color=white')

    if grayscale:
        filters.append('format=gray')

    return ','.join(filters)

def run_ffmpeg_extraction(
                            input_path: Union[str, os.PathLike[str]],
                            output_pattern: Union[str, os.PathLike[str]],
                            vf_filter: str,
                            use_cuda: bool,
                            grayscale: bool,
                            threads: int = 0
                        ) -> None:
    """
    Executes ffmpeg with the given filters and codec settings.
    args:
        input_path (Path): Path to the input video file.
        output_pattern (Path): Pattern for saving frames (with %06d).
        vf_filter (str): The video filter string for ffmpeg.
        use_cuda (bool): If True, use CUDA for decoding.
        grayscale (bool): If True, convert frames to grayscale.

    raises:
        FileNotFoundError: if ffmpeg or CUDA decoder is not found.
        RuntimeError: if ffmpeg extraction fails.
    """
    cmd = ['ffmpeg',
           '-y',
           '-loglevel', 'warning',
        #    '-stats', # Show progress stats,
        #    '-nostats', # Suppress detailed stats,
        #    '-hide_banner', # Suppress startup banner,
           '-threads', f'{threads}'  # Use all available threads,
           ]

    if use_cuda:
        cmd += ['-hwaccel', 'cuda',
                '-c:v', 'h264_cuvid']


    cmd += ['-i', str(input_path),
            '-vf', vf_filter]

    cmd += ['-compression_level', '9']  # Use maximum compression for grayscale
    cmd += ['-qscale:v', '2']  # Set quality scale for JPEG output
    cmd += ['-f', 'image2',
            str(output_pattern)]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    except FileNotFoundError:
        raise FileNotFoundError("ffmpeg or chosen CUDA decoder not found. Please install dependencies.")
    
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors='ignore')
        raise RuntimeError(f"FFmpeg extraction failed: {stderr}")

def health_check_frames(directory: Union[str, os.PathLike[str]],
                        ext: str) -> list[Union[str, os.PathLike[str]]]:
    """
    Attempts to open and verify each image file in directory with Pillow.
    Raises RuntimeError if any are corrupted.
    args:
        directory (Path): Directory containing the extracted frames.
        ext (str): File extension of the frames (e.g., 'jpg', 'png').

    returns:
        list: list of corrupted frame paths.

    Raises:
        RuntimeError: If Pillow is not installed or if any frame files are corrupted.
    
    Warning:
        If any frame files are corrupted, listing their paths.
    """
    corrupted: list[Union[str, os.PathLike[str]]] = []
    dir_path = glob.glob(os.path.join(directory, f'*.{ext}'))
    if not dir_path:
        raise RuntimeError(f"No frame files found in {directory} with extension {ext}")

    for frame_path in sorted(dir_path):
        if frame_path.lower().endswith(f'.{ext}'):
            try:
                with Image.open(frame_path) as img:
                    img.verify()
            except Exception:
                corrupted.append(frame_path)

    if corrupted:
        print(f"Health check passed: {len(dir_path) - len(corrupted):06d} frames verified and {len(corrupted):03d} files are corrupted.")
        raise Warning(f"Corrupted frame files: {corrupted}")
    return corrupted

def ffmpeg_frame_extractor(
                            input_video_path: Union[str, os.PathLike[str]],
                            output_frame_pattern: Optional[Union[str, os.PathLike[str]]] = None,
                            fps: int = 1,
                            wipe: bool = True,
                            use_cuda: bool = False,
                            grayscale: bool = True,
                            health_check: bool = False,
                            min_height: int = 130,
                            threads: int|None = None
                        ) -> None:
    """
    High-level frame extractor that orchestrates probing, directory setup,
    filter construction, FFmpeg extraction, and optional post-check.

    Args:
        input_video_path (Union[str, os.PathLike[str]]): Path to the input video file.
        output_frame_pattern (Optional[Union[str, os.PathLike[str]]]): Pattern for saving frames (with %06d).
        fps (int): Frames per second to extract.
        wipe (bool): Clear existing frames.
        use_cuda (bool): Enable CUDA decoding.
        grayscale (bool): Convert frames to grayscale.
        health_check (bool): Verify each frame after extraction.
        min_height (int): Minimum frame height; pad if below.

    Returns:
        None: Frames are saved to the specified output pattern.

    Raises:
        FileNotFoundError: If input video does not exist.
        RuntimeError: If probing fails or extraction fails.
    """
    if not os.path.isfile(str(input_video_path)):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    ext = 'png' if grayscale else 'jpg'
    if output_frame_pattern is None:
        video_dir = os.path.dirname(str(input_video_path))
        out_dir = os.path.join(video_dir, 'frames')
        output_pattern = os.path.join(out_dir, f'frame_%06d.{ext}')

    else:
        output_pattern = str(output_frame_pattern)
        out_dir = os.path.dirname(output_pattern)
        if not output_pattern.lower().endswith(f'.{ext}'):
            base, _ = os.path.splitext(output_pattern)
            output_pattern = base + f'.{ext}'

    prepare_output_directory(output_pattern, wipe)

    width, height = probe_video_dimensions(input_video_path)
    del width  # Width is not used further
    min_height = max(height, min_height)  # Ensure min_height is at least 1
    assert min_height >= 1, "min_height must be at least 1 pixel."

    vf = build_vf_filter(fps, height, min_height, grayscale)

    if threads is None:
        threads = 0
        
    run_ffmpeg_extraction(input_video_path, output_pattern, vf, use_cuda, grayscale, threads)
    # print(f"Frames extracted to {out_dir} at {fps} fps ({'grayscale' if grayscale else 'color'}).")

    if health_check:
        corrupted = health_check_frames(out_dir, ext)
        fix_corrupted_frames(corrupted, grayscale)

def get_mp4_files(root_dir: Union[str, os.PathLike[str]],
                  max_depth: int = 2) -> list[Union[str, os.PathLike[str]]]:
    """
        Recursively scans a directory for .mp4 files up to a specified depth.
        Args:
            root_dir ( Union[str, os.PathLike[str]]): The root directory to start scanning.
            max_depth (int): Maximum depth to scan (default is 2).
        Returns:
            list: A sorted list of paths to .mp4 files found within the specified depth.
        Raises:
            ValueError: If root_dir is not a directory.
            Exception: If an error occurs while scanning.
    """
    mp4_files: list[Union[str, os.PathLike[str]]] = []

    if not os.path.isdir(root_dir):
        raise ValueError(f"Invalid directory: {root_dir}")

    def scan_directory(path: Union[str, os.PathLike[str]],
                        current_depth: int = 0):
        """
        Recursively scans a directory for .mp4 files up to a specified depth.
        Args:
            root_dir (Union[str, os.PathLike[str]]): The root directory to start scanning.
            max_depth (int): Maximum depth to scan (default is 2).
        Returns:
            list: A sorted list of paths to .mp4 files found within the specified depth.
        Raises:
            ValueError: If root_dir is not a directory.
            Exception: If an error occurs while scanning.
        """
        if current_depth >= max_depth:
            return
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    try:
                        if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith('.mp4'):
                            mp4_files.append(entry.path)  # Store full path
                        elif entry.is_dir(follow_symlinks=False) and current_depth + 1 < max_depth:
                            scan_directory(entry.path, current_depth + 1)
                    except (PermissionError, OSError):
                        continue  # Skip entries with access issues
        except (PermissionError, OSError) as e:
            print(f"Error accessing {path}: {e}")

    scan_directory(root_dir)
    return sorted(mp4_files) # type: ignore

def check_AllVideosHaveSameDimensions(video_root: str = '/media/d2u25/Dont/S4S-ROF/Teflon/',
                                      max_depth: int = 5
                                      ) -> bool:
    """
    Checks if all videos in the provided list have the same dimensions.
    Returns True if they do, False otherwise.
    args:
        video_root (str): Root directory to search for video files.
        max_depth (int): Maximum depth to search for video files.
    returns:
        bool: True if all videos have the same dimensions, False otherwise.
    raises:
        RuntimeError if no video files are found or if probing fails.
    """
    import tqdm
    AllVideos = get_mp4_files(video_root, max_depth=5)
    if not AllVideos:
        print("No video files found.")
        return False
    
    for video in tqdm.tqdm(AllVideos):
        _temp = probe_video_dimensions(video)
        if _temp != (1280, 120):
            print(video, _temp)
    return True

def fix_corrupted_frames(corrupted: list[Union[str, os.PathLike[str]]], grey: bool) -> None:
    """
    Detects corrupted frames via health_check_frames and interpolates replacements
    using the nearest valid neighboring frames.
    args:
        corrupted (list[Union[str, os.PathLike[str]]]): List of paths to corrupted frame files.
        grey (bool): If True, convert the interpolated frames to grayscale.

    returns:
        None: Replaces corrupted frames with interpolated images..

    raises:
        RuntimeError: If Pillow is not installed or if interpolation fails.
        RuntimeError: If neighboring frames for interpolation are missing.
    """
    try:
        from PIL import Image
    except ImportError:
        Image = None
        raise RuntimeError("Pillow is required for interpolation but not installed.")

    def interpolate_frame(prev_path: Union[str, os.PathLike[str]],
                          next_path: Union[str, os.PathLike[str]],
                          output_path: Union[str, os.PathLike[str]],
                          grey: bool) -> None:
        """
        Generates an in-between frame by blending the previous and next images.
        Saves result to output_path.
        args:
            prev_path (os.PathLike): Path to the previous frame.
            next_path (os.PathLike): Path to the next frame.
            output_path (os.PathLike): Path to save the interpolated frame.
            grey (bool): If True, convert the blended image to grayscale.
        returns:
            None: Saves the blended image to output_path.
        """
        with Image.open(prev_path) as img1, Image.open(next_path) as img2:
            img1 = img1.convert('RGB')
            img2 = img2.convert('RGB')
            blended = Image.blend(img1, img2, alpha=0.5)
            if grey:
                blended = blended.convert('L')
            else:
                blended.save(output_path)

    for corrupt_path in corrupted:
        filename = os.path.basename(corrupt_path)
        idx = int(filename.split('_')[-1].split('.')[0])
        corrupt_path_str = str(corrupt_path)
        prev_frame = corrupt_path_str.replace(f'frame_{idx:06d}', f'frame_{idx-1:06d}')
        next_frame = corrupt_path_str.replace(f'frame_{idx:06d}', f'frame_{idx+1:06d}')
        if os.path.exists(prev_frame) and os.path.exists(next_frame):
            print(f"Interpolating {filename} between {os.path.basename(prev_frame)} and {os.path.basename(next_frame)}")
            interpolate_frame(prev_frame, next_frame, corrupt_path, grey)
        else:
            raise RuntimeError(f"Cannot interpolate for {filename}: neighbor frames missing.")

if __name__ == '__main__':
    init()  # Ensure ffmpeg and ffprobe are available

    import glob
    video_path = glob.glob(r"D:\Videos\S1_30per_T1_C001H001S0001\*.mp4")[0]
    ffmpeg_frame_extractor(
        video_path,
        output_frame_pattern=None,
        fps=30,
        wipe=False,
        use_cuda=False,
        grayscale=True,
        health_check=True
    )