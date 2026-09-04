import ast
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from labtools.batch_processor import BatchProcessor
from scipy.io import loadmat

from analysis.util import get_participant_data, load_cleaned_demographics_data, load_path_root, load_result_dataframe, safe_event_dataframe
from analysis.util import load_events_from_excel, safe_result_dataframe
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


def get_events(mat_file, frame_range_start: int = 0, frame_range_end: int = -1) -> dict | None:
    data = loadmat(mat_file)
    try:
        events = get_running_events(data, frame_range_start, frame_range_end)
    except Exception as e:
        warnings.warn(f"Exception {e} in {mat_file}")
        events = None
    if (events is not None) and (len(events) == 0):
        warnings.warn(f"No events detected in {mat_file}")
        events = None
    return events


# def calc_events(path_data_root: Path, heats: list, laps: list) -> pd.DataFrame:
#     rows = []
#     for heat in heats:
#         path_mat = path_data_root / heat
#         bib_numbers = [d.name for d in path_mat.iterdir() if d.is_dir()]
#         bib_numbers.sort()
#         print(bib_numbers)
#         for bib_number in bib_numbers:
#             print(f"Processing Heat {heat}, Bib {bib_number}...")
#             mat_files = list((path_mat / bib_number).glob("*filt.mat"))
#             lap_file_dict = {int(f.stem.split("_")[2]): f for f in mat_files}
#             for lap_no, mat_file in lap_file_dict.items():
#                 print(f"Processing lap {lap_no} from file {mat_file.name}...")
#                 events = get_events(mat_file)
#                 rows.append({"Heat": heat, "Bib": bib_number, "Lap": lap_no, "Events": events})
#     df_events = pd.DataFrame(rows)
#     # sort by heat, bib, lap
#     df_events.sort_values(by=["Heat", "Bib", "Lap"], inplace=True)
#     # reindex the dataframe
#     df_events.reset_index(drop=True, inplace=True)
#     # set the dtype of "Bib" to int
#     df_events["Bib"] = df_events["Bib"].astype(int)
#     return df_events


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


def calc_knee_flexion_metrics(data, events, side: str) -> dict:
    knee_flexion = data[f'{side}_Knee_Angles'][0][0][:, 0]
    if events is None or side not in events:
        warnings.warn(f"No events found.")
        return {"knee_flex_at_ic": np.nan, "knee_flex_max": np.nan, "knee_flex_rom": np.nan}

    evts = events.get(side)
    at_ic, max_flex, rom = [], [], []

    for ic, to in zip(evts["IC"], evts["TO"]):
        stance = knee_flexion[ic:to]
        val_at_ic = knee_flexion[ic]
        val_max = np.max(stance)

        at_ic.append(val_at_ic)
        max_flex.append(val_max)
        rom.append(val_max - val_at_ic)

    return {
        "knee_flex_at_ic": np.mean(at_ic) if at_ic else np.nan,
        "knee_flex_max": np.mean(max_flex) if max_flex else np.nan,
        "knee_flex_rom": np.mean(rom) if rom else np.nan,
    }


def calc_ankle_flexion_metrics(data, events, side: str) -> dict:
    ankle_flexion = data[f'{side}_Ankle_Angles'][0][0][:, 0]
    if events is None or side not in events:
        warnings.warn(f"No events found.")
        return {"ankle_flex_at_ic": np.nan, "ankle_flex_max": np.nan, "ankle_flex_rom": np.nan}

    evts = events.get(side)
    at_ic, max_flex, rom = [], [], []

    for ic, to in zip(evts["IC"], evts["TO"]):
        stance = ankle_flexion[ic:to]
        val_at_ic = ankle_flexion[ic]
        val_max = np.max(stance)

        at_ic.append(val_at_ic)
        max_flex.append(val_max)
        rom.append(val_max - val_at_ic)

    return {
        "ankle_flex_at_ic": np.mean(at_ic) if at_ic else np.nan,
        "ankle_flex_max": np.mean(max_flex) if max_flex else np.nan,
        "ankle_flex_rom": np.mean(rom) if rom else np.nan,
    }


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
        overstriding_oh.append(hip_ankle_ap_diff[ic] * 100)  # convert to cm just for convenience
        overstriding_ok.append(hip_ankle_ap_diff[ic] + 100)  # convert to cm just for convenience

    if parameter == "hip":
        overstriding = overstriding_oh
    elif parameter == "knee":
        overstriding = overstriding_ok
    else:
        raise ValueError(f"Invalid parameter {parameter} for overstriding calculation. Use 'hip' or 'knee'.")
    return np.mean(overstriding) if len(overstriding) > 0 else np.nan


def calc_step_rate(events, framerate) -> float:
    if events is None:
        raise ValueError("No gait events found.")
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


def sided_events_to_sequential(sided_events: dict) -> list[tuple]:
    sequential_events = []
    for side in sided_events.keys():
        for event, event_frames in sided_events[side].items():
            for frame in event_frames:
                sequential_events.append((frame, side, event))
    return sorted(sequential_events)


def calc_leg_stiffness(data, events, side, bodymass, leg_length) -> float:
    """
    Calulate the leg srping stiffness according to Morin et al. (2005): "A Simple Method for Measuring Stiffness during Running"
    doi:10.1123/jab.21.2.167
    """
    # todo: see if per-step speed calculation changes outcome
    speed_meters_per_second = calc_running_speed(data)
    sample_rate = data['FRAME_RATE'].item().item()

    sequential_events = sided_events_to_sequential(events)
    # filter mid-stance events
    sequential_events = [s for s in sequential_events if s[2] != "MS"]
    contralateral_side = "Right" if side == "Left" else "Left"

    # to loop over steps, we need the first contralateral toe-off event and the last contralateral initial contact
    first_contralateral_to = [c_ic for c_ic in sequential_events if (c_ic[1] == contralateral_side) & (c_ic[2] == "TO")][0]
    last_contralateral_ic = [c_ic for c_ic in sequential_events if (c_ic[1] == contralateral_side) & (c_ic[2] == "IC")][-1]
    i_first_contralateral_to = sequential_events.index(first_contralateral_to)
    i_last_contralateral_ic = sequential_events.index(last_contralateral_ic)
    sequential_events_reduced = sequential_events[i_first_contralateral_to:i_last_contralateral_ic + 1]

    ics = [ic for ic in sequential_events_reduced if (ic[1] == side) & (ic[2] == "IC")]
    tos = [to for to in sequential_events_reduced if (to[1] == side) & (to[2] == "TO")]
    k_leg_list = []
    for ic, to in zip(ics, tos):
        c_to = sequential_events_reduced[sequential_events_reduced.index(ic) - 1]
        c_ic = sequential_events_reduced[sequential_events_reduced.index(to) + 1]
        contact_time = (to[0] - ic[0]) / sample_rate
        flight_time_previous = (ic[0] - c_to[0]) / sample_rate
        flight_time_consecutive = (c_ic[0] - to[0]) / sample_rate
        flight_time = (flight_time_previous + flight_time_consecutive) / 2
        # modelled maximum ground reaction force
        # Formula (6) from Morin et al. 2005
        F_max_modelled = bodymass * 9.81 * np.pi * 0.5 * (flight_time / contact_time + 1)
        # modelled vertical center of mass displacement
        # Formula (7) from Morin et al. 2005
        delta_y_c_modelled = ((F_max_modelled * contact_time ** 2) / (bodymass * np.pi ** 2)) + 9.81 * (contact_time ** 2 / 8)
        # modelled leg length variation
        # Formula (9) from Morin et al. 2005
        delta_L_modelled = leg_length - np.sqrt(leg_length ** 2 - (speed_meters_per_second * contact_time * 0.5) ** 2) + delta_y_c_modelled
        # ... and the modelled leg stiffness
        # Formula (8) in Morin
        k_leg = F_max_modelled / delta_L_modelled  # N/m
        k_leg_list.append(k_leg)
    out = np.array(k_leg_list)

    return np.mean(out)


def calc_avg_leg_length(df_leg_length, bib) -> dict:
    means = df_leg_length.loc[df_leg_length["Bib"] == bib, :].mean(numeric_only=True)
    left_shank = means["avg_shank_length_left"]
    right_shank = means["avg_shank_length_right"]
    left_thigh = means["avg_thigh_length_left"]
    right_thigh = means["avg_thigh_length_right"]
    return {
        "Left": left_shank + left_thigh,
        "Right": right_shank + right_thigh,
    }


def calc_kinematic_params(
        row: pd.Series,
        df_events: pd.DataFrame,
        df_demographics: pd.DataFrame,
        df_leg_length: pd.DataFrame) -> dict | None:
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
    - Knee flexion at initial contact (degrees) ✅
    - Knee flexion range of motion during stance (degrees) ✅
    - Ankle plantarflexion at initial contact (degrees)  ✅
    - Ankle flexion range of motion during stance (degrees) ✅
    - Overstriding (cm) ✅
    """
    bib = int(row["Bib"])
    lap_no = int(row["Lap"].split("_")[2])
    df_events["Bib"] = df_events["Bib"].astype(int)
    df_events["Lap"] = df_events["Lap"].astype(int)
    events = event_dict_from_row(row, bib, lap_no)

    data = loadmat(row.path)
    framerate = data['FRAME_RATE'][0][0][0][0]
    #
    #
    # Single value params
    #
    #
    running_speed_ms = calc_running_speed(data)
    try:
        step_rate = calc_step_rate(events, framerate)
    except ValueError as e:
        warnings.warn(f"Value Error in step rate calculation for lap {lap_no} bib {bib}:  {e}")
        step_rate = None
    #
    #
    # KNEE FLEXION PLOTS STEP WISE
    #
    #
    plt.close()

    path_plot = load_path_root() / "TrackGrandPrix" / "kinematics" / "plots" / "knee_flexion_angles_raw"
    fig, ax = plt.subplots()
    for side, col in zip(["Left", "Right"], ["red", "blue"]):
        knee_flexion = data[f'{side}_Knee_Angles'][0][0][:, 0]
        events_side = events.get(side)
        for ic, to in zip(events_side["IC"], events_side["TO"]):
            ax.plot(knee_flexion[ic:to], color=col)
    fig.suptitle(f"{bib} - {lap_no}")
    f_name = f"knee_flexion_{bib}_{lap_no}.png"
    plt.show()
    # fig.savefig((path_plot / f_name), bbox_inches="tight")

    #
    #
    # Sided params:
    #
    #
    out = dict()
    sides = ["Left", "Right"]
    leg_lengths = calc_avg_leg_length(df_leg_length, bib)
    bodymass = df_demographics.loc[df_demographics["Bib"] == bib, "body_mass_kg"].item()
    for side in sides:
        out.update({side: dict()})
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
        knee_metrics = calc_knee_flexion_metrics(data, events, side)
        peak_knee_flex_stance = knee_metrics.get("knee_flex_max")
        knee_flexion_at_ic = knee_metrics.get("knee_flex_at_ic")
        knee_flexion_rom = knee_metrics.get("knee_flex_rom")
        # Ankle
        ankle_metrics = calc_ankle_flexion_metrics(data, events, side)
        ankle_flexion_at_ic = ankle_metrics.get("ankle_flex_at_ic")
        ankle_flexion_rom = ankle_metrics.get("ankle_flex_rom")
        ankle_flexion_max = ankle_metrics.get("ankle_flex_max")

        overstriding = calc_overstriding(data, events, side, parameter="hip")

        # leg spring stiffness
        if not pd.isna(bodymass):
            leg_spring_stiffness = calc_leg_stiffness(data, events, side, bodymass, leg_lengths[side])
        else:
            leg_spring_stiffness = None

        out[side].update({
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
            "ankle_flexion_at_ic_deg": ankle_flexion_at_ic,
            "ankle_flexion_rom_deg": ankle_flexion_rom,
            "ankle_dorsiflexion_max_deg": ankle_flexion_max,
            "overstriding_cm": overstriding,
            'leg_spring_stiffness': leg_spring_stiffness,
        })
    # invert hierarchy (side->param to param->side)
    d = defaultdict(dict)
    for a, b in out.items():
        for c, _d in b.items():
            d[c][a] = _d
    out = dict(d)
    return out


def event_dict_from_row(row: pd.Series, bib: int, lap_no: int) -> dict | None:
    lap_events = df_events[(df_events["Bib"] == bib) & (df_events["Lap"] == lap_no)]
    if pd.isna(lap_events["Left.IC"].values):
        return None

    row = lap_events.iloc[0]
    events = defaultdict(lambda: defaultdict(list))
    for col in ["Left.IC", "Left.MS", "Left.TO", "Right.IC", "Right.MS", "Right.TO"]:
        side, evt = col.split(".")
        events[side][evt] = ast.literal_eval(row[col])
    events = {side: dict(evts) for side, evts in events.items()}
    return events


def events_processor(row: pd.Series, df_demographics: pd.DataFrame) -> dict | None:
    if not {"Bib", "Lap", "path"}.issubset(row.index):
        return {}
    if "_filt" not in row.Lap:
        return None
    bib = int(row.Bib)
    lap = int(row.Lap.split("lap_")[-1].split("_")[0])
    df_sub = df_demographics[df_demographics["start_number"] == bib]
    if df_sub.empty:
        return None

    matches = [item for item in frame_range_adjustments if (item[0] == bib) & (item[1] == lap)]
    if matches:
        events = get_events(row.path, frame_range_start=matches[0][2], frame_range_end=matches[0][3])
        data = loadmat(row.path)
        pelvis_com = data['Pelvis_COM_Position'][0][0]
        plt.close()
        plt.plot(pelvis_com)
        for side, col in zip(["Left", "Right"], ["r", "b"]):

            for ics, tos in zip(events[side]["IC"], events[side]["TO"]):
                plt.axvline(x=ics, color=col)
                plt.axvline(x=tos, color=col, linestyle="--")
        plt.title(f"{bib} - {lap}")
        plt.show()

        foo = 1
    else:
        events = get_events(row.path)

    return events


frame_range_adjustments = [
    # (bib_no, lap, start_frame, end_frame)
    (183, 11, 0, 200),
    (186, 3, 20, -1),
    (213, 12, 0, 200),
    (219, 8, 0, 250),
    (222, 1, 0, 145),
    (225, 6, 65, -1),
    (245, 1, 0, 140),  # check if it's the correct person or if the second part is the correct one
    (277, 6, 0, 240),
    (280, 11, 0, 210),
    (360, 12, 5, -1)
]


def remove_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers from the dataframe based on visual inspection of the data.

    """
    outliers = [
        (183, 11, "faulty gait events"),  # solution: cut file to the first 200 frames!
        (186, 3, "unsteady trajectories"),  # solution: cut first 20 frames off
        (222, 1, "unsteady pelvis ap trajectory mid file"),  # MAYBE: could be cut to the first 120 frames (short period then...)
        (225, 6, "gap in trajectories"),  # solution:cut first 65 frames off
        (245, 1, "gap in trajectories"),  # NO CHANCE: Person tracking ID must have swapped.
        (277, 6, "gap in trajectories"),  # solution:cut to the first 229 frames
        (280, 11, "gap in trajectories"),  # solution:cut to the first 205 frames
        (360, 12, "unsteady trajectories"),  # solution: cut first 10 frames off
    ]

    for bib, lap, reason in outliers:
        print(f"Removing outlier for Bib {bib}, Lap {lap} due to {reason}")
        df = df[~((df["Bib"] == bib) & (df["Lap"] == lap))]

    return df


if __name__ == '__main__':
    recalc_events = False
    recalc_kinematics = True
    path_root = load_path_root()
    path_kinematics = Path(path_root) / "TrackGrandPrix" / "kinematics"
    path_mat_root = path_kinematics / "mat"
    bp = BatchProcessor(path_mat_root,
                        level_names=["Bib", "Lap"],
                        file_pattern="_filt.mat")

    df_demo = get_participant_data()

    # calculate or load events:
    if recalc_events:
        res = bp.apply(events_processor,
                       multiprocess=False,
                       df_demographics=df_demo)
        df_events = pd.json_normalize(res)
        df_events = pd.concat([bp.index.reset_index(drop=True), df_events], axis=1)
        df_events["Lap"] = df_events["Lap"].apply(lambda x: x.split("_")[2])
        df_events["Lap"] = df_events["Lap"].astype(int)

        df_step_count = df_events.copy()
        df_step_count["count_left"] = df_events.apply(lambda x: len(x["Left.MS"]) if isinstance(x["Left.MS"], list) else 0, axis=1)
        df_step_count["count_right"] = df_events.apply(lambda x: len(x["Right.MS"]) if isinstance(x["Right.MS"], list) else 0, axis=1)
        df_step_count = df_step_count[["Bib", "Lap", "count_left", "count_right"]]

        df_step_count_left = df_step_count.pivot(index='Bib', columns='Lap', values="count_left").fillna(0).astype(int)

        df_step_count_left.to_excel((path_kinematics / "step_count_left.xlsx"))

        safe_event_dataframe(df_events)
    else:
        df_events = load_events_from_excel()

    # get the limb lengths and mass for leg spring stiffness calculations
    df_limb_lenghts = load_result_dataframe("limb_lenghts.xlsx")
    df_demographics = load_cleaned_demographics_data()

    if recalc_kinematics:
        res_kin = bp.apply(calc_kinematic_params,
                           df_events=df_events,
                           df_demographics=df_demographics,
                           df_leg_length=df_limb_lenghts,
                           multiprocess=False)

        df_kinematic_params = pd.json_normalize(res_kin)
        df_kinematic_params = pd.concat([bp.index.reset_index(drop=True), df_kinematic_params], axis=1)
        df_kinematic_params["Lap"] = df_kinematic_params["Lap"].apply(lambda x: int(x.split("_")[2]))
        df_kinematic_params.sort_values(["Bib", "Lap"], inplace=True)
    else:
        df_kinematic_params = load_result_dataframe("kinematic_params.xlsx")
    # df_kinematic_params = remove_outliers(df_kinematic_params)

    #
    #
    # Segment length based data checks
    #
    #
    #
    # # data_check(events)
    # # plot_limb_lengths_over_laps(df_limb_lenghts)
    # # print(events.head())
    #
    # if recalc_kinematics:
    #     df_kinematic_params.drop(["path"], axis=1, inplace=True)
    #     safe_result_dataframe(df_kinematic_params, "kinematic_params.xlsx")
    # # reformat dataframe so there are no sided-columns, but a column "side" instead (long format)
    #
    # # Make wide format with lap_side_param, side_lap_param, param_side_lap, separat für l/r und einmal gemittelt
    # # Und dann die Finish Time hinzufügen
    #
    # params = ["running_speed_ms", "step_rate_spm", "contact_time_ms", "flight_time_ms", "vertical_pelvis_movement_cm", "leg_spring_stiffness"]
    # cols = [c for c in df_kinematic_params.columns if c.split(".")[0] in params]
    # wide = df_kinematic_params.pivot(index='Bib', columns='Lap', values=cols)
    #
    # df_finish_time = pd.DataFrame(df_demographics.set_index("Bib")["finish_time_s"])
    # # df_finish_time.dropna(subset=["finish_time_s"], inplace=True)
    # df_finish_time.columns = pd.MultiIndex.from_tuples([("finish_time_s", "")])
    #
    # wide = wide.join(df_finish_time)
    # wide.columns = [f"{c[0]}.lap{c[1]}" for c in wide.columns]
    # wide.reset_index(inplace=True, drop=False)
    # safe_result_dataframe(wide, "leg_stiffness_params_wide.xlsx")
    # wide.to_excel("")
    #
    # id_cols = ["Heat", "Bib", "Lap"]
    #
    # long = df_kinematic_params.set_index(id_cols)
    # # Spalten in MultiIndex (param, side) zerlegen am Punkt
    # long.columns = long.columns.str.rsplit(".", n=1, expand=True)
    # long.columns.names = ["param", "side"]
    # # side aus den Spalten in den Index stacken
    # long = long.stack("side").reset_index()
    # long.columns.name = None
    # safe_result_dataframe(long, "kinematic_params_long.xlsx")
