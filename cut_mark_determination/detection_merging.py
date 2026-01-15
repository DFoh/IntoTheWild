# 3. Script: Merge
from pathlib import Path

import numpy as np
import pandas as pd

from cut_mark_determination.common import load_final_frames_dataframe_from_excel
from util import get_participant_data, get_heat_trials


def get_average_running_speed_mps(start_number: int, df_participants: pd.DataFrame) -> float | None:
    df_participants["start_number"] = df_participants["start_number"].dropna().astype(int)
    df_participant = df_participants[df_participants["start_number"] == start_number]
    finish_time = df_participant["actual_finish_time"].values[0]
    if finish_time == "dnf":
        return None
    minutes = int(finish_time.split(":")[0])
    seconds = int(finish_time.split(":")[1].split(",")[0])
    time_s = minutes * 60 + seconds
    distance_m = 5000
    if time_s > 0:
        return distance_m / time_s


def main(heat: int):
    frame_rate = 85  # frames per second
    distance_after_final_frame_m = 2  # meters
    distance_before_final_frame_m = 15  # meters (capture volume length + a little bit extra)
    trial_names = get_heat_trials(heat)
    df_all = pd.DataFrame(
        columns=["start_number",
                 "heat",
                 "lap",
                 "first_frame",
                 "final_frame",
                 "last_frame",
                 "trial_name"]
    )
    df_participants = get_participant_data()
    for trial_name in trial_names:
        df_trial = load_final_frames_dataframe_from_excel(heat=heat, trial_name=trial_name)
        trial_number = int(trial_name.split(" ")[-1])
        for start_number in df_trial["number"].unique():
            df_sn = df_trial[df_trial["number"] == start_number]
            final_frame = df_sn["final_frame"].values[0]  # last frame runner was detected in the front cameras
            if pd.isna(final_frame):
                print(f"Heat {heat} Trial {trial_name} Start number {start_number} Final frame: {final_frame}")
                continue
            final_frame = int(final_frame)
            # Now, we want to add enough frames at the end, so they are approx. 2m BEHIND the camera based on their average running pace
            # The same for the FIRST frame. We want to add enough frames at the start, so they are approx. 12m (capture volume) BEFORE the position of the final_frame
            avg_speed_mps = get_average_running_speed_mps(start_number=start_number,
                                                          df_participants=df_participants)
            if avg_speed_mps is None:
                print(f"Could not determine average speed for start number {start_number} in trial {trial_name}")
                continue
            n_frames_after = frame_rate * distance_after_final_frame_m / avg_speed_mps
            n_frames_before = frame_rate * distance_before_final_frame_m / avg_speed_mps
            first_frame = max(0, final_frame - int(n_frames_before))
            last_frame = final_frame + int(n_frames_after)
            df_all = pd.concat([
                df_all,
                pd.DataFrame([{
                    "start_number": start_number,
                    "heat": heat,
                    "lap": trial_number,
                    "first_frame": first_frame,
                    "final_frame": final_frame,
                    "last_frame": last_frame,
                    "trial_name": trial_name
                }])
            ], ignore_index=True)
    filename_out = f"cut_marks_heat_{heat}.xlsx"
    path_out = Path("cut_marks")
    path_out.mkdir(parents=True, exist_ok=True)
    path_out = path_out / filename_out

    df_all.to_excel(path_out, index=False)


if __name__ == '__main__':
    main(heat=3)
