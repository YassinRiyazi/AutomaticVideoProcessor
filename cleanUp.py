import os
import glob
import shutil
import subprocess
from typing         import List, Tuple # type: ignore


def create_video_from_images(image_folder: str, output_video_path: str, extension:str=".png", fps:int=30):
    if not shutil.which("ffmpeg"):
        print("❌ ffmpeg not found in PATH.")
        return False

    # Collect and sort images
    images = sorted(glob.glob(os.path.join(image_folder, f"*{extension}")))

    if not images:
        print("❌ No images found!")
        return False

    list_file = os.path.join(image_folder, "file_list.txt")
    with open(list_file, "w") as f:
        for img in images:
            f.write(f"file '{img}'\n")

    command = [
        "ffmpeg",
        "-loglevel", "error",
        "-f", "concat",
        "-safe", "0",
        "-r", str(fps),
        "-i", list_file,
        "-c:v", "libx264",
        "-preset", "fast",
        "-threads", "16",
        "-y", output_video_path
    ]

    try:
        subprocess.run(command, check=True)
        # Clean up
        os.remove(list_file)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg failed: {e}")
        return False

if __name__ == "__main__":
    # address = r"D:\Videos\S1_30per_T1_C001H001S0001\SR_edge"
    # output_video_path = os.path.join(address, "result_video.mp4")

    # create_video_from_images(image_folder=address, 
    #                          output_video_path=output_video_path,
    #                          extension=".png",
    #                          fps=30)
    import tqdm

    Video_list = sorted(glob.glob("/media/Dont/Teflon-AVP/*/*/*"))
    for _folder in tqdm.tqdm(Video_list):
        shutil.rmtree(os.path.join(_folder, "frames"),          ignore_errors=True)
        shutil.rmtree(os.path.join(_folder, "frames_rotated"),  ignore_errors=True)
        shutil.rmtree(os.path.join(_folder, "databases"),       ignore_errors=True)
        shutil.rmtree(os.path.join(_folder, "SR_edge"),         ignore_errors=True)
        os.remove(os.path.join(_folder, 'error_log.txt')) if os.path.isfile(os.path.join(_folder, 'error_log.txt')) else None
        os.remove(os.path.join(_folder, 'result.csv')) if os.path.isfile(os.path.join(_folder, 'result.csv')) else None
        os.remove(os.path.join(_folder, 'result_video.mkv')) if os.path.isfile(os.path.join(_folder, 'result_video.mkv')) else None
        
        logs = glob.glob(os.path.join(_folder,'*.log'))
        for log in logs:
            os.remove(log)