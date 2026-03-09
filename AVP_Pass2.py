"""
    Date    : 2025-09-30
    Author  : Yassin Riyazi
    Project : Automatic Video Processor (AVP)
    File    : AVP_Pass2.py
    Version : 1.0.0
    License : GNU General Public License v3.0
"""
import  re
import  os
import  glob
import  tqdm
import  Utilities
import  CaMeasurer.V1__init__ as V1__init__
import  pandas      as      pd
import  numpy       as      np
from    cleanUp     import  create_video_from_images # type:ignore
from    typing      import  Any, Dict, Set, List

pattern = re.compile(r"Exception: ValueError: Image '(frame_\d+\.png)' not found in the CSV\.")
def Reg(path_to_logfile: str, pattern: re.Pattern[str] = pattern) -> Set[str]:
    # Regex pattern to match the ValueError line and extract frame index
    if not os.path.isfile(error_log_path):
            return set()
    # Read the log file
    with open(path_to_logfile, "r") as f:
        log_data = f.read()

    # Find all frame indices in the log
    frame_numbers = {(m.group(1)) for m in pattern.finditer(log_data)}

    return frame_numbers

def append_measurements_to_df(df_result: pd.DataFrame,
                              result_dict: Dict[str, Any],
                              time_step: float = 0.00025
                              ) -> pd.DataFrame:
    """
    Append measurement results from a dictionary to the result DataFrame.

    Args:
        df_result (pd.DataFrame): Existing result DataFrame with required columns.
        result_dict (dict): Dictionary mapping file paths to measurement results.
        time_step (float): Time increment between frames (default: 0.00025 s).

    Returns:
        pd.DataFrame: Updated DataFrame with new rows appended.
    """
    new_rows:List[Dict[str, Any]] = []

    for path, data in result_dict.items():
        file_name = os.path.basename(path)
        frame_number = int(file_name.replace("frame_", "").replace(".png", ""))

        # Build a new row
        row:Dict[str, Any] = {
                                "file number": file_name,
                                "time (s)": frame_number * time_step,
                                "x_center (cm)": data["x_center"],
                                "adv (degree)": float(data["adv"]),
                                "rec (degree)": float(data["rec"]),
                                "contact_line_length (cm)": float(data["contact_line_length"]),
                                "y_center (cm)": data["y_center"],
                                "middle_angle_degree (degree)": float(data["middle_angle_degree"]),
                                "velocity (cm/s)": np.nan  # You can fill this later
                            }
        new_rows.append(row)

    # Convert new rows to DataFrame and append
    new_df = pd.DataFrame(new_rows)
    df_result = pd.concat([df_result, new_df], ignore_index=True)

    df_result = df_result.sort_values(by="file number",).reset_index(drop=True)

    return df_result

if __name__ == "__main__":
    for folder_Address in tqdm.tqdm(sorted(glob.glob("/media/d25u2/Dont/Viscosity/*/*/*"))):
        folder_Address = '/media/d25u2/Dont/Viscosity/280/S5-SDS99_S20/D155328_09_1.06'


        error_log_path      = os.path.join(folder_Address, "databases", "error_log.txt")
        

        print(f"Processing folder: {folder_Address}")
        
        result_csv_path     = os.path.join(folder_Address, "result.csv")
        detections_csv_path = os.path.join(folder_Address, "databases", "detections.csv")
        if not os.path.isfile(result_csv_path) or not os.path.isfile(detections_csv_path):
            # print(f"Missing result.csv or detections.csv in {folder_Address}, skipping.")
            continue

        df              = pd.read_csv(detections_csv_path)
        df_result       = pd.read_csv(result_csv_path)
        AllImages       = glob.glob(os.path.join(folder_Address, "frames_rotated", "*.png"))    
        AllImages_names = {os.path.basename(img) for img in AllImages}
        ErrorMissing    = Reg(error_log_path)
        AllMissing      = (set(df_result['file number']) | AllImages_names) - (set(df_result['file number']) & AllImages_names)
        # print("Total frames in frames_rotated:", len(AllMissing),AllMissing)
        vv = ErrorMissing | AllMissing
        # print("Missing frames:", vv)

        # Ensure every entry in vv exists in df['image']
        missing_not_in_df = vv - set(df['image'].astype(str))
        if missing_not_in_df:
            raise ValueError(f"The following frames from vv are not present in detections.csv: {sorted(missing_not_in_df)}")
        
        ## Step. CA measurement for missing frames
        vv = [os.path.join(folder_Address, "databases", img) for img in vv]
        data = V1__init__.single(vv)

        df_result = append_measurements_to_df(df_result, data)
        df_result.to_csv(os.path.join(folder_Address, "result_2Pass.csv"), index=False)

        ## Step. DF post processing
        _ = Utilities.position_velocity_correctionV2(os.path.join(folder_Address, 'result_2Pass.csv'))
        break