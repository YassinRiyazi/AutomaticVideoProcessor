"""Video files from the software is normally very large. 
This utility compresses them using ffmpeg HEVC codec
for smaller size while maintaining high quality."""

import os
import subprocess
import glob

def compress_video(input_path, output_path, crf=23, preset="slow"):
    """Compress an MP4 video using HEVC (H.265) codec for smallest size and high quality.

    Args:
        input_path (str): Path to the input .mp4 video.
        output_path (str): Path where the compressed video will be saved.
        crf (int): Constant Rate Factor (lower = higher quality, typical range 18–28).
        preset (str): Compression speed vs efficiency (slower = smaller file).

    Returns:
        bool: True if compression succeeded, False otherwise.
    """
    assert os.path.isfile(input_path), f"Input file not found: {input_path}"

    command = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "libx264",          # Use H.265 (HEVC)
        "-preset", preset,          # Tradeoff between encoding speed and compression
        "-crf", str(crf),           # Quality level (18-28, lower = better)
        "-an",                      # Disable audio
        '-pix_fmt', 'yuv444p',      # full color fidelity (use yuv420p for max compatibility)
        '-threads', '0',            # Use all available CPU cores
        # "-movflags", "+faststart",  # Optimize for web playback
        "-y",                       # Overwrite output if exists
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Compressed: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg compression failed: {e}")
        return False

def check_video_health(video_path):
    """Check if a video file is healthy (not corrupted).

    Args:
        video_path (str): Path to the video file.

    Returns:
        bool: True if the video is healthy, False otherwise.
    """
    command = ["ffmpeg", "-v", "error", "-i", video_path, "-f", "null", "-"]
    try:
        subprocess.run(command, check=True, stderr=subprocess.PIPE)
        print(f"✅ Video is healthy: {video_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Video is corrupted: {video_path}")
        return False

if __name__ == "__main__":
    folder = "/media/d25u2/Dont/Viscosity/OriginalVideos"  # Change to your folder path
    mp4_files = sorted(glob.glob(os.path.join(folder, "*/*/*.mp4")))
    for video_path in mp4_files:
        output_path = os.path.join(os.path.dirname(video_path), f"compressed_{os.path.basename(video_path)}")
        result = compress_video(video_path, output_path, crf=18, preset="veryslow")
        # if result:
        #     os.remove(video_path)  # Delete original file after compression
        #     os.rename(output_path, video_path)  # Rename compressed file to original name
