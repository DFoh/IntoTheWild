import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from analysis.util import PATH_ROOT, load_events_from_excel, make_file_path, \
    safe_result_dataframe
from gait_events import get_running_events


def get_valid_frame_range(data, side: str):
    # ISSUE: the data will have NaN values at the start and end because the skeleton is not solved in the first and last frames.
    # SOLUTION: check the data for the first and last valid frame and only analyze this window.
    # also: save the first frame to later offset the detected events to the actual frame number in the original file.
    # This is important for later comparison with the force platform data.

    heel_pos_data = data[f"{side}_Heel_Pos"][0][0]
    toe_pos_data = data[f"{side}_Toe_Pos"][0][0]
    mask = ~np.isnan(heel_pos_data[:, 0]) & ~np.isnan(toe_pos_data[:, 0])
    valid_indices = np.where(mask)[0]
    first_valid_frame = valid_indices[0]
    last_valid_frame = valid_indices[-1]

    return first_valid_frame, last_valid_frame


def get_events(mat_file) -> dict | None:
    data = loadmat(mat_file)
    try:
        events = get_running_events(data)
    except Exception as e:
        warnings.warn(f"Exception {e} in {mat_file}")
        events = None
    if (events is not None) and (len(events) == 0):
        warnings.warn(f"No events detected in {mat_file}")
        events = None
    return events


def calc_events(path_data_root: Path, heats: list, laps: list) -> pd.DataFrame:
    rows = []
    for heat in heats:
        path_mat = path_data_root / heat
        bib_numbers = [d.name for d in path_mat.iterdir() if d.is_dir()]
        bib_numbers.sort()
        print(bib_numbers)
        for bib_number in bib_numbers:
            print(f"Processing Heat {heat}, Bib {bib_number}...")
            mat_files = list((path_mat / bib_number).glob("*filt.mat"))
            lap_file_dict = {int(f.stem.split("_")[2]): f for f in mat_files}
            for lap_no, mat_file in lap_file_dict.items():
                print(f"Processing lap {lap_no} from file {mat_file.name}...")
                events = get_events(mat_file)
                rows.append({"Heat": heat, "Bib": bib_number, "Lap": lap_no, "Events": events})
    df_events = pd.DataFrame(rows)
    # sort by heat, bib, lap
    df_events.sort_values(by=["Heat", "Bib", "Lap"], inplace=True)
    # reindex the dataframe
    df_events.reset_index(drop=True, inplace=True)
    # set the dtype of "Bib" to int
    df_events["Bib"] = df_events["Bib"].astype(int)
    return df_events


def calc_running_speed(data) -> np.float64 | None:
    frame_rate_hz = data['FRAME_RATE'][0][0][0][0]
    try:
        pelvis_com_pos = data['Pelvis_COM_Position'][0][0]
    except KeyError as e:
        warnings.warn(f"Pelvis_COM_Position not found")
        return None
    # anterior-posterior position of the pelvis COM
    pelvis_com_ap_pos = pelvis_com_pos[:, 0]
    # anterior-posterior velocity of the pelvis COM in m/s:
    pelvis_com_ap_vel_ms = np.gradient(pelvis_com_ap_pos) * frame_rate_hz
    # take the median of the velocity (to be robust against outliers) as the running speed for this lap
    running_speed_ms = np.nanmedian(pelvis_com_ap_vel_ms)
    return running_speed_ms


def time_normalize_signal(signal: np.ndarray, new_length: int) -> np.ndarray:
    """
    Time-normalize a signal to a new length using linear interpolation.
    signal: 1D array of shape (N,)
    new_length: int, the desired length of the time-normalized signal
    returns: 1D array of shape (new_length,)
    """
    original_length = len(signal)
    if original_length == 0:
        return np.full(new_length, np.nan)  # return NaNs if the input signal is empty
    original_time = np.linspace(0, 1, original_length)
    new_time = np.linspace(0, 1, new_length)
    normalized_signal = np.interp(new_time, original_time, signal)
    return normalized_signal


def calc_vertical_pelvis_movement_sided(data, events, side) -> float:
    pelvis_com_vert_pos = data['Pelvis_COM_Position'][0][0][:, 2]
    if events is None or side not in events:
        warnings.warn(f"No events found.")
        return np.nan
    ics = events.get(side).get("IC", [])
    ics_contralateral = events.get("Right" if side == "Left" else "Left").get("IC", []).copy()
    if len(ics) == 0 or len(ics_contralateral) == 0:
        warnings.warn(f"No initial contact events found for {side} side or contralateral side.")
        return np.nan
    # print(ics_contralateral)
    while ics[0] > ics_contralateral[0]:
        ics_contralateral.pop(0)  # remove the first contralateral IC if it occurs before the first ipsilateral IC
    # print(ics_contralateral)

    amplitude = []
    for ic, ic_contralateral in zip(ics, ics_contralateral):
        if ic_contralateral - ic < 20:  # if the contralateral IC is too close to the ipsilateral IC, skip this step (probably a detection error)
            warnings.warn(
                f"Contralateral IC at frame {ic_contralateral} is too close to ipsilateral IC at frame {ic}, skipping this step.")
            continue
        step_pelv_motion = pelvis_com_vert_pos[ic:ic_contralateral]
        # plt.plot(step_pelv_motion)
        amplitude.append(np.ptp(step_pelv_motion))
    # return in cm just for convenience
    return np.mean(amplitude) * 100 if len(amplitude) > 0 else np.nan


def calc_max_knee_flexion(data, events, side: str) -> float:
    knee_flexion = data[f'{side}_Knee_Angles'][0][0][:, 0]
    if events is None or side not in events:
        warnings.warn(f"No events found.")
        return np.nan
    evts = events.get(side)
    max_flex = []
    for ic, to in zip(evts["IC"], evts["TO"]):
        stride_knee_flexion = knee_flexion[ic:to]
        max_flex.append(np.max(stride_knee_flexion))

    return np.mean(max_flex) if len(max_flex) > 0 else np.nan


def calc_knee_flexion_rom(data, events, side: str) -> float:
    knee_flexion = data[f'{side}_Knee_Angles'][0][0][:, 0]
    if events is None or side not in events:
        warnings.warn(f"No events found.")
        return np.nan
    evts = events.get(side)
    knee_flex_rom = []
    for ic, to in zip(evts["IC"], evts["TO"]):
        stride_knee_flexion = knee_flexion[ic:to]
        knee_flex_rom.append(np.ptp(stride_knee_flexion))

    return np.mean(knee_flex_rom) if len(knee_flex_rom) > 0 else np.nan


def calc_knee_flexion_at_ic(data, events, side: str) -> float:
    knee_flexion = data[f'{side}_Knee_Angles'][0][0][:, 0]
    if events is None or side not in events:
        warnings.warn(f"No events found.")
        return np.nan
    evts = events.get(side)
    knee_flex_at_ic_list = []
    for ic in evts["IC"]:
        knee_flexion_at_ic = knee_flexion[ic]
        knee_flex_at_ic_list.append(knee_flexion_at_ic)

    return np.mean(knee_flex_at_ic_list) if len(knee_flex_at_ic_list) > 0 else np.nan


def calc_overstriding(data, events, side: str, parameter: str) -> float:
    if events is None or side not in events:
        warnings.warn(f"No events found.")
        return np.nan
    hip_center_traj = data[f'{side}_Hip_Center'][0][0]
    knee_center_traj = data[f'{side}_Knee_Center'][0][0]
    ankle_center_traj = data[f'{side}_Ankle_Center'][0][0]
    # choose positive values
    hip_ankle_ap_diff = ankle_center_traj[:, 0] - hip_center_traj[:, 0]
    knee_ankle_ap_diff = ankle_center_traj[:, 0] - knee_center_traj[:, 0]
    evts = events.get(side)
    # variables according to Lieberman et al., 2015 "Effects of stride frequency..." doi:10.1242/jeb.125500
    overstriding_oh = []
    overstriding_ok = []
    for ic in evts["IC"]:
        overstriding_oh.append(hip_ankle_ap_diff[ic])
        overstriding_ok.append(hip_ankle_ap_diff[ic])

    if parameter == "hip":
        overstriding = overstriding_oh * 100  # convert to cm just for convenience
    elif parameter == "knee":
        overstriding = overstriding_ok * 100  # convert to cm just for convenience
    else:
        raise ValueError(f"Invalid parameter {parameter} for overstriding calculation. Use 'hip' or 'knee'.")
    return np.mean(overstriding) if len(overstriding) > 0 else np.nan


def calc_step_rate(events, framerate) -> float:
    if events is None:
        warnings.warn(f"No events found.")
        return np.nan
    ics = events.get("Left", {}).get("IC", []) + events.get("Right", {}).get("IC", [])
    ics = np.array(sorted(ics))
    step_rates = 60 / np.diff(ics) * framerate  # convert to stepsper minute
    step_rate = np.mean(step_rates) if len(step_rates) > 0 else np.nan
    if step_rate > 300:  # if the step rate is higher than 300 steps per minute, it's probably a detection error, so we set it to NaN
        warnings.warn(f"Step rate of {step_rate} steps per minute is too high, setting to NaN.")
        step_rate = np.nan
    elif step_rate < 60:  # if the step rate is lower than 60 steps per minute, it's probably a detection error, so we set it to NaN
        warnings.warn(f"Step rate of {step_rate} steps per minute is too low, setting to NaN.")
        step_rate = np.nan
    else:
        step_rate = step_rate
    return step_rate


def calc_contact_time(events_side, frame_rate) -> float:
    cts = []
    for ic, to in zip(events_side.get("IC", []), events_side.get("TO", [])):
        ct = (to - ic) / frame_rate * 1000  # convert to ms
        cts.append(ct)
    return np.mean(cts) if len(cts) > 0 else np.nan


def calc_flight_time(events_side, frame_rate) -> float:
    fts = []
    for to, ic in zip(events_side.get("TO", []), events_side.get("IC", [])[1:]):
        ft = (ic - to) / frame_rate * 1000
        fts.append(ft)
    return np.mean(fts) if len(fts) > 0 else np.nan


def calc_step_length(data, events, side) -> float:
    # calculate the ap distance between consecutive ipsi-/contralateral foot center positions during stance
    # at the moment where the foot COM velocity is minimal (mid-stance proxy)
    foot_pos = data[f"{side}_Foot_COM_Position"][0][0]
    foot_pos_contralateral = data[f"{'Right' if side == 'Left' else 'Left'}_Foot_COM_Position"][0][0]
    mid_stance_evt = events.get(side).get("MS", [])
    mid_stance_evt_contralateral = events.get("Right" if side == "Left" else "Left").get("MS", []).copy()
    while mid_stance_evt[0] > mid_stance_evt_contralateral[0]:
        mid_stance_evt_contralateral.pop(0)
    step_lengths = []
    for ms, ms_ctl in zip(mid_stance_evt, mid_stance_evt_contralateral):
        if ms_ctl - ms < 20:
            warnings.warn(
                f"Contralateral MS at frame {ms_ctl} is too close to ipsilateral MS at frame {ms}, skipping this step.")
            continue
        step_length = foot_pos_contralateral[ms_ctl, 0] - foot_pos[ms, 0]
        step_lengths.append(step_length)
    return np.mean(step_lengths) if len(step_lengths) > 0 else np.nan


def calc_trunk_flexion(data, events, side) -> float:
    # Global CS is defined as:
    # x: anterior direction (running direction)
    # y: left
    # z: up
    trunk_flexion = data[f"Thorax_Angles"][0][0][:, 1]
    if events is None or side not in events:
        warnings.warn(f"No events found.")
        return np.nan
    evts = events.get(side)
    flexions = []
    for ic, to in zip(evts["IC"], evts["TO"]):
        stance_trunk_flexion = trunk_flexion[ic:to]
        flexions.append(np.max(stance_trunk_flexion))
    return np.mean(flexions) if len(flexions) > 0 else np.nan


import matplotlib.pyplot as plt


def calc_peak_pelvis_ap_tilt(data, events, side) -> float:
    # TODO: Check out if this makes sense. The pelvis anterior tilt is the rotation around the ml axis in the global CS,
    # TODO: ... but the actual "peak" is shortly after the TO and not during the stance
    # TODO: ...(which is not what Maas et al. 2018 report).
    # TODO: ... so the question is, if we should look for the negative peak instead (which is markedly in during the stance).
    # TODO: ... need to check this in the literature before continuing with the implementation.
    return np.nan
    pelvis_ap_tilt = data[f"Pelvis_Angles"][0][0][:, 1]
    if events is None or side not in events:
        warnings.warn(f"No events found.")
        return np.nan
    evts = events.get(side)
    plt.close()
    plt.plot(pelvis_ap_tilt, label="Pelvis AP Tilt")
    for ic, tc in zip(evts["IC"], evts["TO"]):
        plt.axvline(x=ic, color='g', linestyle='--', label="IC")
        plt.axvline(x=tc, color='r', linestyle='--', label="TO")
    plt.show()
    tilts = []
    for ic, to in zip(evts["IC"], evts["TO"]):
        stance_pelvis_ap_tilt = pelvis_ap_tilt[ic:to]
        tilts.append(np.max(stance_pelvis_ap_tilt))
    return np.mean(tilts) if len(tilts) > 0 else np.nan


def calc_peak_pelvis_obliquity(data, events, side) -> float:
    # Global CS is defined as:
    # x: forward direction
    # y: left
    # z: up
    pelvis_obliquity = data[f"Pelvis_Angles"][0][0][:, 0]  # rotation around ap-axis
    # invert the signal for the left side to make it comparable to the right side.
    # -> negative values reflect a contralateral drop of the pelvis
    if side == "Left":
        pelvis_obliquity = -pelvis_obliquity

    if events is None or side not in events:
        warnings.warn(f"No events found.")
        return np.nan
    evts = events.get(side)
    obliquities = []
    for ic, to in zip(evts["IC"], evts["TO"]):
        stance_pelvis_obliquity = pelvis_obliquity[ic:to]
        # we take the minimum because a contralateral drop of the pelvis is reflected in negative values
        obliquities.append(np.min(stance_pelvis_obliquity))
    return np.mean(obliquities) if len(obliquities) > 0 else np.nan


def calc_pelvis_rotation_rom(data, events, side) -> float:
    # TODO: PHEW... check the signal quality of the pelvis...
    # TODO: Currently, it doesn't seem like we'll get meaningful results for this parameter.
    return np.nan
    # Global CS is defined as:
    # x: forward direction
    # y: left
    # z: up
    pelvis_vertical_rotation = data[f"Pelvis_Angles"][0][0][:, 2]  # rotation around up-axis
    # offset the signal, since the global/lab CS is defined rotated around -90 def
    pelvis_vertical_rotation = pelvis_vertical_rotation + 90
    # invert the signal for the left side to make it comparable to the right side.
    # -> negative values reflect

    if side == "Left":
        pelvis_vertical_rotation = -pelvis_vertical_rotation

    if events is None or side not in events:
        warnings.warn(f"No events found.")
        return np.nan
    ctrltrl_events = events.get("Right" if side == "Left" else "Left")
    plt.close()
    plt.plot(pelvis_vertical_rotation, label="Pelvis Vertical Rotation")
    plt.axhline(y=0, color='k', linestyle='--', label="90 degrees")
    for ic, tc in zip(events.get(side).get("IC", []), events.get(side).get("TO", [])):
        plt.axvline(x=ic, color='g', linestyle='-', label="IC")
        plt.axvline(x=tc, color='r', linestyle='-', label="TO")
    for ic, tc in zip(ctrltrl_events.get("IC", []), ctrltrl_events.get("TO", [])):
        plt.axvline(x=ic, color='g', linestyle='--', label="IC")
        plt.axvline(x=tc, color='r', linestyle='--', label="TO")
    plt.show()


def calc_hip_flexion_rom(data, events, side) -> float:
    # TODO: Consider if this makes sense.
    # TODO: Peak hip flexion
    hip_flexion = data[f'{side}_Hip_Angles'][0][0][:, 0]
    if events is None or side not in events:
        warnings.warn(f"No events found.")
        return np.nan
    evts = events.get(side)
    flexions = []
    for ic, to in zip(evts["IC"], evts["TO"]):
        stance_hip_flexion = hip_flexion[ic:to]
        flexions.append(np.ptp(stance_hip_flexion))
    return np.mean(flexions) if len(flexions) > 0 else np.nan


def calc_kinematic_params(df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate biomechanical outcome parameters for each lap based on the detected events and the kinematic data from the .mat files. The parameters include:
    - Running speed (m/s) ✅
    - Step rate (steps per minute) ✅
    - Contact time (ms) ✅
    - Flight time (ms)  ✅
    - Step length (cm) ✅
    - Peak trunk flexion (forward lean) during stance (degrees)  ✅
    - Vertical pelvis movement (cm) ✅
    - Vertical pelvis movement for left and right side separately (cm) ✅
    - Peak pelvis anterior-posterior tilt during stance (degrees) ❌ needs further investigation
    - Pelvis obliquity range of motion during stance (degrees) ✅
    - Pelvis rotation range of motion during stance (degrees) ❌ needs further investigation
    - Hip flexion range of motion during stance (degrees) ✅
    - Max knee flexion during stance (degrees) ✅
    - Knee flexion at initial contact (degrees) (not implemented yet)
    - Knee flexion range of motion during stance (degrees) (not implemented yet)
    - Ankle plantarflexion range of motion during stance (degrees) (not implemented yet)
    - Overstriding (cm) ✅
    """
    path_mat_root = Path(PATH_ROOT) / "kinematics" / "mat"
    rows = []
    for index, row in df_events.iterrows():
        heat = row["Heat"]
        bib = row["Bib"]
        lap_no = row["Lap"]
        events = row["Events"]
        if events is None or not isinstance(events, dict):
            continue
        file_path = make_file_path(path_mat_root, heat, bib, lap_no)
        print(f"Processing lap {lap_no} from file {file_path.name}...")
        if not file_path.exists():
            warnings.warn(f"File {file_path} does not exist, skipping...")
            continue
        data = loadmat(str(file_path))
        framerate = data['FRAME_RATE'][0][0][0][0]
        #
        #
        # Single value params
        #
        #
        running_speed_ms = calc_running_speed(data)
        step_rate = calc_step_rate(events, framerate)
        #
        #
        # Sided params:
        #
        #
        sides = ["Left", "Right"]
        for side in sides:
            events_side = events.get(side)
            contact_time = calc_contact_time(events_side, framerate)
            flight_time = calc_flight_time(events_side, framerate)
            step_length = calc_step_length(data, events, side)
            peak_trunk_flexion = calc_trunk_flexion(data, events, side)
            # Pelvis Parameters
            vertical_pelvis_movement = calc_vertical_pelvis_movement_sided(data, events, side)
            peak_pelvis_ap_tilt = calc_peak_pelvis_ap_tilt(data, events, side)
            neg_peak_pelvis_obliquity = calc_peak_pelvis_obliquity(data, events, side)
            pelvis_rotation_rom = calc_pelvis_rotation_rom(data, events, side)
            # Hip
            hip_flexion_rom = calc_hip_flexion_rom(data, events, side)
            # Knee
            peak_knee_flex_stance = calc_max_knee_flexion(data, events, side)
            knee_flexion_at_ic = calc_knee_flexion_at_ic(data, events, side)
            knee_flexion_rom = calc_knee_flexion_rom(data, events, side)

            overstriding = calc_overstriding(data, events, side, parameter="hip")

            row_data = {"Heat": heat, "Bib": bib, "Lap": lap_no, "Side": side}

            rows.append({**row_data,
                         "running_speed_ms": running_speed_ms,
                         # just duplicate the running speed for both sides for easier analysis later, even though it's not a sided parameter
                         "step_rate_spm": step_rate,  # same here
                         "contact_time_ms": contact_time,
                         "flight_time_ms": flight_time,
                         "step_length_m": step_length,
                         "trunk_flexion_deg": peak_trunk_flexion,
                         "vertical_pelvis_movement_cm": vertical_pelvis_movement,
                         "peak_pelvis_ap_tilt_deg": peak_pelvis_ap_tilt,
                         "neg_peak_pelvis_obliquity_deg": neg_peak_pelvis_obliquity,
                         "hip_flexion_rom_deg": hip_flexion_rom,
                         "peak_knee_flex_stance_deg": peak_knee_flex_stance,
                         "knee_flexion_at_ic_deg": knee_flexion_at_ic,
                         "knee_flexion_rom_deg": knee_flexion_rom,
                         "overstriding_cm": overstriding,
                         })

    df_kinematic_params = pd.DataFrame(rows)
    df_kinematic_params.sort_values(by=["Heat", "Bib", "Lap", "Side"], inplace=True)
    df_kinematic_params.reset_index(drop=True, inplace=True)
    return df_kinematic_params


if __name__ == '__main__':
    path_mat_root = Path(PATH_ROOT) / "kinematics" / "mat"
    heat_directories = [d for d in path_mat_root.iterdir() if d.is_dir()]
    heat_directories.sort()
    heats = [d.name for d in heat_directories]
    laps = list(range(1, 14))

    # df_events = calc_events(path_data_root=path_mat_root, heats=heats, laps=laps)
    # safe_event_dataframe(df_events)
    #
    df_events = load_events_from_excel()

    #
    #
    # Segment length based data checks
    #
    #

    # data_check(events)
    # df_limb_lenghts = load_result_dataframe("limb_lenghts.xlsx")
    # plot_limb_lengths_over_laps(df_limb_lenghts)
    # print(events.head())

    df_kinematic_params = calc_kinematic_params(df_events)
    safe_result_dataframe(df_kinematic_params, "kinematic_params.xlsx")
