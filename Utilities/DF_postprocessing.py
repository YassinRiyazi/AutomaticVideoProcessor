"""
    When detecting position of drop sometimes it mismatches the position by one pixel causing issues in dataframe and velocity calculations.
"""

import pandas as pd
import os
import glob
import numpy as np

def position_velocity_correction(df: pd.DataFrame|str) -> pd.DataFrame:
    save = None
    if isinstance(df, str):
        save = df.replace('.csv', '_corrected.csv')
        df = pd.read_csv(df)

    col = "x_center (cm)"
    df[col] = pd.to_numeric(df[col], errors="coerce")

    n = len(df)
    fixed = 0

    # interpolate any interior point that is less than the previous point
    for i in range(1, n - 1):
        prev, cur, nxt = df.at[i - 1, col], df.at[i, col], df.at[i + 1, col]

        if pd.notna(prev) and pd.notna(cur) and pd.notna(nxt) and cur < prev:
            df.at[i, col] = (prev + nxt) / 2.0
            fixed += 1

    # first row: if decreasing compared to second, copy second
    if n >= 2 and pd.notna(df.at[0, col]) and pd.notna(df.at[1, col]) and df.at[0, col] < df.at[1, col]:
        df.at[0, col] = df.at[1, col]
        fixed += 1

    # last row: if decreasing compared to previous, copy previous
    if n >= 2 and pd.notna(df.at[n - 2, col]) and pd.notna(df.at[n - 1, col]) and df.at[n - 1, col] < df.at[n - 2, col]:
        df.at[n - 1, col] = df.at[n - 2, col]
        fixed += 1


    vel = np.diff(df['x_center (cm)']) / np.diff(df['time (s)'])
    vel = np.insert(vel, 0, vel[0])  # Maintain same length

    # update the velocity column in the dataframe
    df['velocity (cm/s)'] = vel
    
    if save:
        df.to_csv(save, index=False)

    return df


if __name__ == "__main__":
    df = pd.read_csv("/media/Dont/Teflon-AVP/280/S2-SNr2.1_D/T528_12_4.460000000000/result.csv")
    df.to_csv("/media/Dont/Teflon-AVP/280/S2-SNr2.1_D/T528_12_4.460000000000/result_corrected.csv", index=False)