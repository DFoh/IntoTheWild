import sys
import warnings
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns

PATH_ROOT_WIN = r"C:\Users\dominik.fohrmann\OneDrive - MSH Medical School Hamburg - University of Applied Sciences and Medical University\Dokumente\Projects\IntoTheWild\data\TrackGrandPrix"
# for macOS:
PATH_ROOT_MAC = r"/Users/dominikfohrmann/OneDrive - MSH Medical School Hamburg - University of Applied Sciences and Medical University/Dokumente/Projects/IntoTheWild/data/TrackGrandPrix"

if sys.platform.startswith('win'):
    PATH_ROOT = PATH_ROOT_WIN
elif sys.platform.startswith('darwin'):
    PATH_ROOT = PATH_ROOT_MAC
else:
    raise OSError("Unsupported operating system. Please use Windows or macOS.")


def get_result_frame_path(filename: str) -> Path:
    path_safe = Path(PATH_ROOT) / "kinematics" / filename
    return path_safe


def safe_result_dataframe(df_result: pd.DataFrame, filename: str):
    path_safe = get_result_frame_path(filename)
    try:
        df_result.to_excel(path_safe, index=False)
    except Exception as e:
        warnings.warn(f"Could not save to Excel file: {e}")


def get_event_frame_path() -> Path:
    path_safe = get_result_frame_path("events.xlsx")
    return path_safe



def safe_event_dataframe(df_events: pd.DataFrame):
    safe_result_dataframe(df_events, "events.xlsx")

def safe_running_speed_dataframe(df_running_speed: pd.DataFrame):
    safe_result_dataframe(df_running_speed, "running_speed.xlsx")

def load_result_dataframe(filename: str) -> pd.DataFrame:
    path_safe = get_result_frame_path(filename)
    try:
        df_result = pd.read_excel(path_safe)
        print(f"Result loaded from {path_safe}")
        return df_result
    except Exception as e:
        warnings.warn(f"Could not load result from Excel file: {e}")
        return pd.DataFrame()

def load_events_from_excel() -> pd.DataFrame:
    path_safe = get_event_frame_path()
    try:
        df_events = pd.read_excel(path_safe)
        # parse the "Events" column from string back to dict
        df_events["Events"] = df_events["Events"].apply(lambda x: eval(x) if isinstance(x, str) else x)
        print(f"Events loaded from {path_safe}")
        return df_events
    except Exception as e:
        warnings.warn(f"Could not load events from Excel file: {e}")
        return pd.DataFrame(columns=["Heat", "Bib", "Lap", "Events"])


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


def get_running_events(data):
    # assuming position data in meters and FRAME RATE in Hz
    """
    Identify running events (initial contact and toe-off) based on vertical position of heel and toe markers in
    accordance to Mailwald et al. 2009 "Detecting foot-to-ground contact from kinematic data in running"
    Footwear Sci., DOI: 10.1080/19424280903133938
    """
    events = {}
    sample_rate = data['FRAME_RATE'][0][0][0][0]  # Assuming frame rate is stored in this field
    path_file = PureWindowsPath(data['FILE_NAME'][0][0][0])
    filename = path_file.stem
    bib_no = filename.split("_")[0]
    lap = filename.split("_")[2]
    print(f"Processing {filename}")

    for side in ["Left", "Right"]:
        ics = []
        tos = []
        # print(f"Processing {side} side...")
        valid_start, valid_end = get_valid_frame_range(data, side)

        # Check if the valid frame range is sufficient for analysis (1.5 seconds at 85 Hz = 128 frames)
        if (valid_end - valid_start) < 128:
            warnings.warn(
                f"Valid frame range for {side} side is too short for reliable event detection. Skipping this side.")
            continue

        heel_vert_pos = data[f"{side}_Heel_Pos"][0][0][valid_start:valid_end,
                        2]  # Assuming vertical position is the 3rd column
        heel_vert_vel = np.gradient(heel_vert_pos) * sample_rate  # Vertical velocity of the heel
        heel_vert_acc = np.gradient(heel_vert_vel) * sample_rate  # Vertical acceleration of the heel
        toe_vert_pos = data[f"{side}_Toe_Pos"][0][0][valid_start:valid_end,
                       2]  # Assuming vertical position is the 3rd column
        toe_vert_vel = np.gradient(toe_vert_pos) * sample_rate
        toe_vert_acc = np.gradient(toe_vert_vel) * sample_rate
        # for robustness, we can set an expected minimum step rate to define "distance" between peaks.
        min_step_rate = 120  # steps per minute
        min_step_interval = (60 / min_step_rate) * sample_rate  # minimum interval in frames

        heel_pks, _ = find_peaks(heel_vert_pos, height=0.2, distance=min_step_interval)

        # see if the first swing event (heel_pks) has sufficient frames before start of the file to capture the preceding initial contact event
        avg_swing_evt_frames = np.diff(heel_pks).mean()

        search_window_ratio = 0.66  # adjust from experience
        if heel_pks[0] > search_window_ratio * avg_swing_evt_frames:
            heel_pks = np.insert(heel_pks, 0,
                                 0)  # additional start frame for the minimum search up to the first swing event

        # same for the last event
        if heel_pks[-1] < len(heel_vert_pos) - search_window_ratio * avg_swing_evt_frames:
            heel_pks = np.append(heel_pks, len(heel_vert_pos) - 1)

        # print(f"heel_pks type: {type(heel_pks)}, shape: {heel_pks.shape}, values: {heel_pks}")
        for pk, next_pk in zip(heel_pks, heel_pks[1:]):
            # "Prior to analyzing the temporal aspect of vertical acceleration of either marker,
            # the time frames of the minimum vertical position of HEEL and MTH5 were compared.
            # In order to accommodate the algorithm for possible midfoot and forefoot strikers,
            # the earlier event and thus the target marker was chosen as an indicator of the
            # type of foot strike and approximate time of touch down (TDapprox)."
            heel_min = np.argmin(heel_vert_pos[pk:next_pk])
            toe_min = np.argmin(toe_vert_pos[pk:next_pk])
            if heel_min < toe_min:
                td_approx = pk + heel_min
                acc = heel_vert_acc
            else:
                td_approx = pk + toe_min
                acc = toe_vert_acc

            # "Subsequently, a sufficiently narrow time interval around this minimal position was defined
            # [TDapprox - 50 ms, TDapprox + 100 ms]. This ensured that FCA was able to deal with various
            # types of foot strike and only detected those acceleration peaks that were associated with
            # touch down rather than those associated with take off or any vertical braking that occurs
            # during foot swing and pre touch down phases. To estimate touch down, FCA uses a characteristic
            # maximum in the vertical acceleration curve of the target marker in the given approximation interval."
            window_start = int(max(0, td_approx - 0.05 * sample_rate))
            window_end = int(min(len(heel_vert_acc), td_approx + 0.1 * sample_rate))
            # for debugging:
            frames_acc = np.arange(window_start, window_end)
            acc_plot = acc[window_start:window_end]
            acc_pks, _ = find_peaks(acc_plot)
            # if no peaks are found, something went wrong, so raise a warning. Use the maximum of the window as fallback for the initial contact event
            if len(acc_pks) == 0:
                warnings.warn(f"No acceleration peaks found in window for {side} side, using maximum as fallback")
                acc_pk = np.argmax(acc_plot)
            else:
                # choose the highest peak in the window as the initial contact event
                i_acc_pk = np.argmax(acc_plot[acc_pks])
                acc_pk = acc_pks[i_acc_pk]
                # TODO: TRY TAKING THE FIRST PEAK INSTEAD OF THE HIGHEST
                acc_pk = acc_pks[0]
            ic = window_start + acc_pk

            # TAKE OFF
            # "For take off, a local maximum in the vertical acceleration of TIP is detected and compared to the
            # minimal vertical position of TIP. A logical operation selects the event that occurs earlier in time,
            # which is then used to estimate take off. The two events usually occur almost simultaneously.
            # For some running styles, e.g. with only limited vertical ground clearance of the foot after take off,
            # we observed a major gap between those events. Choosing the earlier event, which was the minimal vertical
            # position in most cases, yielded a better match to the force platform event, the gold standard.
            # Similar to the detection of touch down, additional constraints were necessary in order to select the
            # correct acceleration peak. This was accomplished by constraining the search for peaks to a suitable
            # time window for ground contact times observed during running at 3.5 m/s [TDapprox + 100 ms, TDapprox + 400 ms]."
            to_window_start = ic + int(0.1 * sample_rate)
            # well, it can't be after the next swing event, so we need to check that as well
            to_window_end = min(next_pk, ic + int(0.5 * sample_rate))

            # make sure we get at least 100ms window to search for the to event
            if (to_window_end - to_window_start) < 0.1 * sample_rate:
                # if not, skip the take off detection for this step and raise a warning
                warnings.warn(
                    f"Take off search window is too short for {side} side, skipping take off detection for this step")
                continue

            # for debugging:
            frames_to = np.arange(to_window_start, to_window_end)
            acc_to_plot = toe_vert_acc[frames_to]
            to_acc_pks, _ = find_peaks(acc_to_plot)
            # for pk in to_acc_pks[0]:
            #     ax_bottom.scatter(frames_to[pk], acc_to_plot[pk], color='blue', marker='x')
            if len(to_acc_pks) == 0:
                warnings.warn(
                    f"No acceleration peaks found in take off window for {side} side, using maximum as fallback")
                to_acc_pk = np.argmax(acc_to_plot)
            else:
                i_to_acc_pk = np.argmax(acc_to_plot[to_acc_pks])
                to_acc_pk = to_acc_pks[i_to_acc_pk]
            to = frames_to[to_acc_pk]
            # all done. write to the lists
            ics.append(ic)
            tos.append(to)

        # correct event frames back for the offset of the valid data window
        ics = [ic + valid_start for ic in ics]
        tos = [to + valid_start for to in tos]
        assert len(ics) == len(tos)
        events[side] = {"IC": ics, "TO": tos}
    return events


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
    safe_event_dataframe(df_events)
    return df_events


def make_file_path(path_mat_root: Path, heat: str, bib_number: int, lap_no: int) -> Path:
    """
    create the file path for each heat, bib number and lap number based on the root path
    heat: str, e.g. "heat_1"
    bib_number: int, e.g. 100
    lap_no: int, e.g. 1
    Heat_1/100/100_lap_2_filt.mat
    """

    return path_mat_root / heat / str(bib_number) / f"{bib_number}_lap_{lap_no}_filt.mat"


def calc_running_speed(events: pd.DataFrame) -> pd.DataFrame:
    path_mat_root = Path(PATH_ROOT) / "kinematics" / "mat"

    rows = []
    for index, row in events.iterrows():
        heat = row["Heat"]
        bib = row["Bib"]
        lap_no = row["Lap"]
        file_path = make_file_path(path_mat_root, heat, bib, lap_no)
        print(f"Processing lap {lap_no} from file {file_path.name}...")
        if not file_path.exists():
            warnings.warn(f"File {file_path} does not exist, skipping...")
            continue
        data = loadmat(file_path)
        frame_rate_hz = data['FRAME_RATE'][0][0][0][0]
        try:
            pelvis_com_pos = data['Pelvis_COM_Position'][0][0]
        except KeyError as e:
            warnings.warn(f"Pelvis_COM_Position not found in {file_path}: {e}")
            continue
        # anterior-posterior position of the pelvis COM
        pelvis_com_ap_pos = pelvis_com_pos[:, 0]
        # anterior-posterior velocity of the pelvis COM in m/s:
        pelvis_com_ap_vel_ms = np.gradient(pelvis_com_ap_pos) * frame_rate_hz

        # take the median of the velocity (to be robust against outliers) as the running speed for this lap
        running_speed_ms = np.nanmedian(pelvis_com_ap_vel_ms)
        rows.append({"Heat": heat, "Bib": bib, "Lap": lap_no, "running_speed_ms": running_speed_ms})
    df_running_speed = pd.DataFrame(rows)
    df_running_speed.sort_values(by=["Heat", "Bib", "Lap"], inplace=True)
    df_running_speed.reset_index(drop=True, inplace=True)
    safe_running_speed_dataframe(df_running_speed)
    return df_running_speed

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

def calc_kinematic_params(df_events: pd.DataFrame) -> pd.DataFrame:
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
        data = loadmat(file_path)
        frame_rate_hz = data['FRAME_RATE'][0][0][0][0]
        # Start with something simple like plotting the knee flexion angle for each stride normalized to 100% of the stride.
        colors = ['blue', 'red']
        side = ["Left", "Right"]
        max_flex_sided = {}
        for color, side in zip(colors, side):
            knee_flexion = data[f'{side}_Knee_Angles'][0][0][:, 0]
            if events is None or side not in events:
                warnings.warn(f"No events found for {side} side in {file_path}, skipping plotting for this side.")
                continue
            evts = events.get(side)
            max_flex = []
            for ic, to in zip(evts["IC"], evts["TO"]):
                stride_knee_flexion = knee_flexion[ic:to]
                max_flex.append(np.max(stride_knee_flexion))
                # plt.plot(stride_knee_flexion, color=color)

                # stride_knee_flexion_time_norm = time_normalize_signal(stride_knee_flexion, new_length=101)
                # t_norm = np.linspace(0, 100, 101)
                # plt.plot(t_norm, stride_knee_flexion_time_norm)
            max_flex_sided[side] = np.mean(max_flex)
        max_flex_combined = np.mean(list(max_flex_sided.values()))
        # path_plot = Path(PATH_ROOT) / "plots" / "knee_flexion_angles_raw"
        # path_plot.mkdir(parents=True, exist_ok=True)
        # path_file = path_plot / f"{bib}_lap_{lap_no}_knee_flexion.png"
        # plt.title(f"Bib {bib}, Lap {lap_no}")
        # plt.savefig(path_file)
        # plt.close()
        rows.append({"Heat": heat, "Bib": bib, "Lap": lap_no, "max_knee_flexion_combined": max_flex_combined, "max_knee_flexion_left": max_flex_sided.get("Left", np.nan), "max_knee_flexion_right": max_flex_sided.get("Right", np.nan)})
    df_kinematic_params = pd.DataFrame(rows)
    df_kinematic_params.sort_values(by=["Heat", "Bib", "Lap"], inplace=True)
    df_kinematic_params.reset_index(drop=True, inplace=True)
    safe_result_dataframe(df_kinematic_params, "knee_flexion_angles.xlsx")
    return df_kinematic_params

def plot_param_over_laps(df: pd.DataFrame, column_name: str):
    # sns.lineplot(data=df, x="Lap", y=column_name, hue="Heat", markers=False)
    # sns.lineplot(data=df, x="Lap", y=column_name, hue="Heat", units="Bib", estimator=None)
    # plt.show()
    path_plot = Path(PATH_ROOT) / "plots" / f"{column_name}_over_laps"
    path_plot.mkdir(parents=True, exist_ok=True)
    bibs = df["Bib"].unique()
    for bib in bibs:
        df_bib = df[df["Bib"] == bib]
        plt.plot(df_bib["Lap"], df_bib[column_name], marker='o', label=f'Bib {bib}')
        plt.title(f"{bib} - {column_name}")
        plt.savefig(path_plot / f"{bib} - {column_name}.png")
        plt.close()

def plot_knee_flexion_over_laps(df: pd.DataFrame):
    # sns.lineplot(data=df, x="Lap", y=column_name, hue="Heat", markers=False)
    # sns.lineplot(data=df, x="Lap", y=column_name, hue="Heat", units="Bib", estimator=None)
    # plt.show()
    path_plot = Path(PATH_ROOT) / "plots" / "max_knee_flexion_over_laps"
    path_plot.mkdir(parents=True, exist_ok=True)
    bibs = df["Bib"].unique()
    for bib in bibs:
        df_bib = df[df["Bib"] == bib]
        plt.plot(df_bib["Lap"], df_bib["max_knee_flexion_combined"], marker='o', color="k")
        plt.plot(df_bib["Lap"], df_bib["max_knee_flexion_left"], marker='o', color="b")
        plt.plot(df_bib["Lap"], df_bib["max_knee_flexion_right"], marker='o', color="r")

        plt.title(f"{bib} - max_knee_flexion")
        plt.savefig(path_plot / f"{bib} - max_knee_flexion.png")
        plt.close()


def data_check(df_events: pd.DataFrame):
    path_mat_root = Path(PATH_ROOT) / "kinematics" / "mat"

    rows = []
    for index, row in events.iterrows():
        heat = row["Heat"]
        bib = row["Bib"]
        lap_no = row["Lap"]
        file_path = make_file_path(path_mat_root, heat, bib, lap_no)
        print(f"Processing lap {lap_no} from file {file_path.name}...")
        if not file_path.exists():
            warnings.warn(f"File {file_path} does not exist, skipping...")
            continue
        data = loadmat(file_path)

        limb_lenghts = {}
        for side in ["Left", "Right"]:
            hip_center_traj = data[f'{side}_Hip_Center'][0][0]
            knee_center_traj = data[f'{side}_Knee_Center'][0][0]
            try:
                ankle_center_traj = data[f'{side}_Ankle_Center'][0][0]
            except KeyError:
                ankle_center_traj = data[f'{side}t_Ankle_Center'][0][0]  # stupid typo in the original data, but we have to deal with it
            thigh_length = np.linalg.norm(hip_center_traj - knee_center_traj, axis=1)
            shank_length = np.linalg.norm(knee_center_traj - ankle_center_traj, axis=1)
            avg_thigh_length = np.nanmean(thigh_length)
            avg_shank_length = np.nanmean(shank_length)
            limb_lenghts[f'avg_thigh_length_{side.lower()}'] = avg_thigh_length
            limb_lenghts[f'avg_shank_length_{side.lower()}'] = avg_shank_length
        row_data = {"Heat": heat, "Bib": bib, "Lap": lap_no, **limb_lenghts}
        rows.append(row_data)
    df_limb_lenghts = pd.DataFrame(rows)
    df_limb_lenghts.sort_values(by=["Heat", "Bib", "Lap"], inplace=True)
    df_limb_lenghts.reset_index(drop=True, inplace=True)
    safe_result_dataframe(df_limb_lenghts, "limb_lenghts.xlsx")
    return df_limb_lenghts

def plot_limb_lengths_over_laps(df):
    path_plot = Path(PATH_ROOT) / "plots" / "limb_lengths"
    path_plot.mkdir(parents=True, exist_ok=True)
    bibs = df["Bib"].unique()
    for bib in bibs:
        df_bib = df[df["Bib"] == bib]
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        params = ["avg_thigh_length_left", "avg_thigh_length_right", "avg_shank_length_left", "avg_shank_length_right"]
        for p, param in enumerate(params):
            plot_data = df_bib[param]
            ax =axs.flatten()[p]
            ax.plot(df_bib["Lap"], plot_data, marker='o', color="k")
            # plot the average avg_shank_length_left over all laps and the +/- 0.5cm range as dashed lines
            avg = plot_data.median()
            ax.axhline(avg, color="k", linestyle="--")
            ax.axhspan(avg - 0.005, avg + 0.005, color="k", alpha=0.2)
            # plt.axhline(avg + 0.005, color="k", linestyle="--")
            # plt.axhline(avg - 0.005, color="k", linestyle="--")
            ax.set_title(param)
        fig.suptitle(f"Bib {bib} - Limb Lengths over Laps")
        plt.savefig(path_plot / f"{bib} - limb_lengths.png")
        plt.close()


if __name__ == '__main__':
    path_mat_root = Path(PATH_ROOT) / "kinematics" / "mat"
    heat_directories = [d for d in path_mat_root.iterdir() if d.is_dir()]
    heat_directories.sort()
    heats = [d.name for d in heat_directories]
    laps = list(range(1, 14))

    # events = calc_events(path_data_root=path_mat_root, heats=heats, laps=laps)
    events = load_events_from_excel()
    # data_check(events)
    df_limb_lenghts = load_result_dataframe("limb_lenghts.xlsx")
    plot_limb_lengths_over_laps(df_limb_lenghts)
    # print(events.head())

    # running_speeds = calc_running_speed(events)
    # df_kinematic_params = calc_kinematic_params(events)
    # df_kinematic_params = load_result_dataframe("knee_flexion_angles.xlsx")
    # plot_knee_flexion_over_laps(df_kinematic_params)


