import ast
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
from labtools.batch_processor import BatchProcessor
from labtools.analyses.kinetics.event_detection import get_force_events_treadmill
from labtools.systems.zebris.utils import get_force
from labtools.utils.c3d import load_c3d
from matplotlib import pyplot as plt
from scipy.io import loadmat

from analysis.gait_events import get_running_events
from analysis.main import calc_ankle_flexion_metrics, calc_avg_leg_length, calc_contact_time, calc_flight_time, calc_hip_flexion_rom, calc_knee_flexion_metrics, calc_overstriding, \
    calc_peak_pelvis_ap_tilt, \
    calc_peak_pelvis_obliquity, \
    calc_pelvis_rotation_rom, \
    calc_running_speed, \
    calc_step_length, calc_step_rate, \
    calc_trunk_flexion, \
    calc_vertical_pelvis_movement_sided
from analysis.util import get_participant_data, load_path_lab_data, safe_event_dataframe


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


def events_processor(row: pd.Series, df_demographics: pd.DataFrame) -> dict | None:
    if not {"participant_id", "trial_no", "path"}.issubset(row.index):
        return {}
    p_id = row.participant_id
    df_demo_sub = df_demographics[df_demographics["participant_id"] == p_id]
    if df_demo_sub.empty:
        return None
    else:
        events = get_events(row.path)
    return events


def event_dict_from_row(row: pd.Series, df_events, p_id: str, trial_no: str) -> dict | None:
    lap_events = df_events[(df_events["participant_id"] == p_id) & (df_events["trial_no"] == trial_no)]
    if pd.isna(lap_events["Left.IC"].values):
        return None

    row = lap_events.iloc[0]
    events = defaultdict(lambda: defaultdict(list))
    for col in ["Left.IC", "Left.MS", "Left.TO", "Right.IC", "Right.MS", "Right.TO"]:
        if col not in df_events.columns:
            continue
        side, evt = col.split(".")
        events[side][evt] = ast.literal_eval(row[col]) if isinstance(row[col], str) else row[col]
    events = {side: dict(evts) for side, evts in events.items()}
    return events


def calc_kinematic_params(
        row: pd.Series,
        df_events: pd.DataFrame,
        df_demographics: pd.DataFrame) -> dict | None:
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
    p_id = row["participant_id"]
    trial_no = row["trial_no"]
    events = event_dict_from_row(row, df_events, p_id, trial_no)

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
        warnings.warn(f"Value Error in step rate calculation for trial {trial_no} p_id {p_id}:  {e}")
        step_rate = None
    #
    #
    # KNEE FLEXION PLOTS STEP WISE
    #
    #
    plt.close()

    path_plot = load_path_lab_data() / "kinematics" / "plots" / "knee_flexion_angles_raw"
    path_plot.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots()
    for side, col in zip(["Left", "Right"], ["red", "blue"]):
        knee_flexion = data[f'{side}_Knee_Angles'][0][0][:, 0]
        events_side = events.get(side)
        for ic, to in zip(events_side["IC"], events_side["TO"]):
            ax.plot(knee_flexion[ic:to], color=col)
    fig.suptitle(f"{p_id} - {trial_no}")
    f_name = f"knee_flexion_{p_id}_{trial_no}.png"
    # plt.show()
    fig.savefig((path_plot / f_name), bbox_inches="tight")

    #
    #
    # Sided params:
    #
    #
    out = dict()
    sides = ["Left", "Right"]
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
        })
    # invert hierarchy (side->param to param->side)
    d = defaultdict(dict)
    for a, b in out.items():
        for c, _d in b.items():
            d[c][a] = _d
    out = dict(d)
    return out


def analyze_kinematics(path_root):
    path_kinematics = Path(path_root) / "kinematics"
    path_mat_root = path_kinematics / "mat"
    bp = BatchProcessor(path_mat_root,
                        level_names=["participant_id", "trial_no", "trial_name"],
                        file_pattern="kmh_filt.mat")

    df_demo = get_participant_data()

    res = bp.apply(events_processor,
                   multiprocess=False,
                   df_demographics=df_demo)
    df_events = pd.json_normalize(res)
    df_events = pd.concat([bp.index.reset_index(drop=True), df_events], axis=1)

    df_events.to_excel(path_kinematics / "events_kinematics.xlsx")

    # convert events from frames to milliseconds
    cols = df_events.filter(regex=r"IC|TO").columns
    df_events[cols] = df_events[cols].map(lambda x: [ev / 85 * 1000 for ev in x])
    df_events.to_excel(path_kinematics / "events_kinematics_ms.xlsx")

    res_kin = bp.apply(calc_kinematic_params,
                       df_events=df_events,
                       df_demographics=df_demo,
                       multiprocess=False)
    for err in bp.errors:
        print(f"Error in {err[0]}: {err[1]}")

    df_kinematic_params = pd.json_normalize(res_kin)
    df_kinematic_params = pd.concat([bp.index.reset_index(drop=True), df_kinematic_params], axis=1)

    df_kinematic_params.to_excel(path_kinematics / "results_kinematics.xlsx")


def events_processor_pressure(row: pd.Series, df_demographics: pd.DataFrame) -> dict | None:
    if not {"participant_id", "trial", "path"}.issubset(row.index):
        return {}
    p_id = row.participant_id
    df_demo_sub = df_demographics[df_demographics["participant_id"] == p_id]
    if df_demo_sub.empty:
        return None
    else:
        data, meta = load_c3d(row.path)
        events = dict()
        sample_rate = data['analog_rate']
        f_z_r, f_z_l = get_force(data, True)
        events_r = get_force_events_treadmill(f_z=f_z_r,
                                              sample_rate=sample_rate)
        events_l = get_force_events_treadmill(f_z=f_z_l,
                                              sample_rate=sample_rate)
        # write and convert from numpy type
        events["Right"] = {k: [int(x) for x in v] for k, v in events_r.items()}
        events["Left"] = {k: [int(x) for x in v] for k, v in events_l.items()}
    return events


def calc_spatio_temporal_params(
        row: pd.Series,
        df_events: pd.DataFrame,
        df_demographics: pd.DataFrame) -> dict | None:
    p_id = row["participant_id"]
    trial = row["trial"]

    df_events.rename(columns={"Left.ic": "Left.IC",
                              "Left.tc": "Left.TO",
                              "Right.ic": "Right.IC",
                              "Right.tc": "Right.TO"
                              }, inplace=True)
    events = event_dict_from_row(row, df_events, p_id, trial)

    data, meta = load_c3d(row.path)
    framerate = data['analog_rate']

    try:
        step_rate = calc_step_rate(events, framerate)
    except ValueError as e:
        warnings.warn(f"Value Error in step rate calculation for trial {trial} p_id {p_id}:  {e}")
        step_rate = None

    out = dict()

    for side in ["Left", "Right"]:
        out.update({side: dict()})
        events_side = events.get(side)
        contact_time = calc_contact_time(events_side, framerate)
        flight_time = calc_flight_time(events_side, framerate)

        out[side].update({
            "step_rate_spm": step_rate,  # same here
            "contact_time_ms": contact_time,
            "flight_time_ms": flight_time
        })
    return out


def analyze_pressure(path_root):
    mp = False
    path_pressure = Path(path_root) / "pressure"
    path_c3d_root = path_pressure / "c3d"
    bp = BatchProcessor(path_c3d_root,
                        level_names=["participant_id", "trial"],
                        file_pattern=".c3d")

    df_demo = get_participant_data()

    res = bp.apply(events_processor_pressure,
                   multiprocess=mp,
                   df_demographics=df_demo)

    df_events = pd.json_normalize(res)
    df_events = pd.concat([bp.index.reset_index(drop=True), df_events], axis=1)

    # rename columns
    df_events.rename(columns={"Right.ic": "Right.IC",
                                   "Right.tc": "Right.TO",
                                   "Left.ic": "Left.IC",
                                   "Left.tc": "Left.TO"
                                   }, inplace=True)

    df_events.to_excel(path_pressure / "events_pressure.xlsx")

    # convert events from frames to milliseconds
    cols = df_events.filter(regex=r"IC|TO").columns
    df_events[cols] = df_events[cols].map(lambda x: [ev / 300 * 1000 for ev in x])
    df_events.to_excel(path_pressure / "events_pressure_ms.xlsx")

    # spatio-temporals right away:
    res_pressure = bp.apply(calc_spatio_temporal_params,
                            df_events=df_events,
                            multiprocess=mp,
                            df_demographics=df_demo)

    for err in bp.errors:
        print(f"Error in {err[0]}: {err[1]}")

    df_pressure_params = pd.json_normalize(res_pressure)
    df_pressure_params = pd.concat([bp.index.reset_index(drop=True), df_pressure_params], axis=1)

    df_pressure_params.to_excel(path_pressure / "results_pressure.xlsx")


if __name__ == '__main__':
    path_root = load_path_lab_data()
    # analyze_kinematics(path_root)
    analyze_pressure(path_root)
