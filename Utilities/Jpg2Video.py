import os
import subprocess
from multiprocessing import Pool, cpu_count
import glob

import os
import subprocess
import glob

def create_video_from_images(image_folder, output_video_path, extension="jpg", fps=30):
    """Create a video from .jpg images in a folder using ffmpeg.

    Args:
        image_folder (str): Path to the folder containing images.
        output_video_path (str): Path where the output video will be saved.
        fps (int): Frames per second for the video.

    Returns:
        bool: True if video was created successfully, False otherwise.

    Update:
        Now returns success
    """
    assert extension in ["jpg", "png"], "Extension must be 'jpg' or 'png'"
    # Change working directory to image folder
    original_cwd = os.getcwd()
    os.chdir(image_folder)

    command = [
        'ffmpeg',
        '-loglevel', 'error',
        '-framerate', str(fps),
        '-pattern_type', 'glob', '-i', f'*.{extension}',
        '-c:v', 'libx264',
        '-preset', 'veryslow',  # prioritize quality over speed
        '-pix_fmt', 'yuv444p',  # full color fidelity (use yuv420p for max compatibility)
        '-threads', '16',
        '-y',
        output_video_path
         # '-pix_fmt',     'yuv420p',              # Pixel format for compatibility
         # '-crf',         '18',                   # Set Constant Rate Factor for high quality (lower values = higher quality, 18-23 is typical range)
         # '-tune',        'film',                 # Tune the encoding for film (preserves quality)
    ]

    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg failed: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def process_experiment(_adress):
    """Process one experiment directory: create video from images and delete them if successful.

    Args:
        _adress (str): Path to the experiment directory.
    """
    image_folder = os.path.join(_adress, "SR_edge")
    outputpath = os.path.join(image_folder, "result.mp4")
    result_csv = os.path.join(_adress, 'SR_result', 'result.csv')

    # Skip if video already exists and is non-zero size
    if os.path.isfile(outputpath) and os.path.getsize(outputpath) > 0:
        return

    # Check that required files exist
    imgs = glob.glob(os.path.join(image_folder, "*.jpg"))
    if len(imgs) == 0 or not os.path.isfile(result_csv):
        return

    # Create video
    success = create_video_from_images(image_folder, outputpath)

    # Delete images only if ffmpeg succeeded
    if success:
        for img_path in imgs:
            try:
                os.remove(img_path)
            except Exception as e:
                print(f"⚠️ Error deleting {img_path}: {e}")




def process_experiment_frames(_adress):
    outputpath = os.path.join(_adress, "result.mp4")

    # Skip if result.mp4 exists and is non-zero in size
    if os.path.isfile(outputpath) and os.path.getsize(outputpath) > 0:
        return
    
    # Check for images and CSV
    imgs = glob.glob(os.path.join(_adress, "*.jpg"))
    if len(imgs) == 0:
        return

    create_video_from_images(_adress, outputpath)

    # Remove .jpg files after video is created
    for img_path in imgs:
        try:
            os.remove(img_path)
        except Exception as e:
            print(f"⚠️ Error deleting {img_path}: {e}")


if __name__ == "__main__":
    adress = "/media/ubun25/DONT/MPI/S4S-ROF/drop/"
    experiments = sorted(get_mp4_files(adress, max_depth=5))
    experiments = [experiment.replace("drop","frame_Extracted") for experiment in experiments]
    experiments = [experiment.replace(".mp4","") for experiment in experiments]
    experiments = [experiment.replace("frames","frame_Extracted") for experiment in experiments]

    experiments = list(reversed(experiments))
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_experiment, experiments)

    # ## Frames
    # experiments = sorted(get_mp4_files(adress, max_depth=5))
    # experiments = [experiment.replace("drop","frames") for experiment in experiments]
    # experiments = [experiment.replace(".mp4","") for experiment in experiments]
    # 
    # print(experiments[0])
    # with Pool(processes=cpu_count()//2) as pool:
    #     pool.map(process_experiment, experiments)

# /media/ubun25/DONT/MPI/S4S-ROF/frames/280/S2-SNr2.14_D/frames20250620_222000_DropNumber_01
# /media/ubun25/DONT/MPI/S4S-ROF/frames/280/S2-SNr2.14_D/frames20250620_222000_DropNumber_01
