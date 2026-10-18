import subprocess
import glob
import os
import re
import pandas as pd
import tqdm

def get_video_frame_count(video_path: str) -> int:
    """Return the total number of frames in a video file using ffprobe."""
    command = [
        "ffprobe",
        "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path,
    ]
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        frame_count_str = result.stdout.strip()
        return int(frame_count_str) if frame_count_str.isdigit() else -1
    except Exception as e:
        print(f"Warning: Error reading frame count for {video_path}: {e}")
        return -1


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # 1. Build the nested dictionary
    # ------------------------------------------------------------------ #
    Fluids: dict[str, dict[int, float | str]] = {}

    # ---- initialise every fluid with all tilt values (default = 'NA') ---- #
    tilt_dirs = glob.glob("/media/d25u2/Dont/Viscosity/*")
    for tilt in tilt_dirs:
        folder_name = os.path.basename(tilt)
        if not re.fullmatch(r"\d{3}", folder_name):
            continue
        for fluid_dir in sorted(glob.glob(os.path.join(tilt, "*"))):
            fluid_name = os.path.basename(fluid_dir)
            fluid_name = fluid_name[3:].split("_")[0]

            # create entry if not yet present
            if fluid_name not in Fluids:
                Fluids[fluid_name] = {t: "NA" for t in range(280, 345, 5)}

    # ---- fill the dictionary with averages ---- #
    for tilt in tqdm.tqdm(sorted(tilt_dirs)):
        folder_name = os.path.basename(tilt)
        if not re.fullmatch(r"\d{3}", folder_name):
            continue

        tilt_value = int(folder_name)

        for fluid_dir in sorted(glob.glob(os.path.join(tilt, "*"))):
            video_frames: list[int] = []
            # for video_file in glob.glob(os.path.join(fluid_dir, "*.mp4")):
                # fc = get_video_frame_count(video_file)
            for video_file in sorted(glob.glob(os.path.join(fluid_dir, "*","frames_rotated"))):
                fc = len(glob.glob(os.path.join(video_file, "*.png")))
                if fc > 0:          # ignore -1 errors
                    video_frames.append(fc)

            fluid_name = os.path.basename(fluid_dir)
            fluid_name = fluid_name[3:].split("_")[0]

            if video_frames:
                Fluids[fluid_name][tilt_value] = sum(video_frames) / len(video_frames)
            else:
                Fluids[fluid_name][tilt_value] = "NA"

    # ------------------------------------------------------------------ #
    # 2. Convert to DataFrame (rows = fluids, columns = tilt values)
    # ------------------------------------------------------------------ #
    # All possible tilt columns (in order)
    tilt_columns = list(range(280, 345, 5))

    # Build a list of rows: [fluid_name, val280, val285, ...]
    rows = []
    for fluid, tilt_dict in Fluids.items():
        row = [fluid]  # first column = fluid name
        for t in tilt_columns:
            val = tilt_dict.get(t, "NA")
            # keep numbers as float, keep "NA" as string
            row.append(val)
        rows.append(row)

    # Column names
    columns = ["Fluid"] + [str(t) for t in tilt_columns]

    df = pd.DataFrame(rows, columns=columns)

    # ------------------------------------------------------------------ #
    # 3. Save to CSV
    # ------------------------------------------------------------------ #
    output_path = "fluids_tilt_summary.csv"
    df.to_csv(output_path, index=False)
    print(f"\nCSV written to: {os.path.abspath(output_path)}")
    print("\nPreview:")
    print(df.head())