import sys
import warnings
from pathlib import Path, PureWindowsPath

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks

PATH_ROOT_WIN = r"C:\Users\dominik.fohrmann\OneDrive - MSH Medical School Hamburg - University of Applied Sciences and Medical University\Dokumente\Projects\IntoTheWild\data\TrackGrandPrix"
# for macOS:
PATH_ROOT_MAC = r"/Users/dominikfohrmann/OneDrive - MSH Medical School Hamburg - University of Applied Sciences and Medical University/Dokumente/Projects/IntoTheWild/data/TrackGrandPrix"


if sys.platform.startswith('win'):
    PATH_ROOT = PATH_ROOT_WIN
    matplotlib.use(
        'Qt5Agg')  # maybe unnecessary with matplotlib 3.7.1, but explicitly set for PyCharm debugging on Windows to avoid backend issues
elif sys.platform.startswith('darwin'):
    if matplotlib.__version__ != '3.7.1':
        raise ImportError(
            f"matplotlib version 3.7.1 is required for proper debugging in PyCharm, but found {matplotlib._version.version}. Please install the correct version.")

    PATH_ROOT = PATH_ROOT_MAC
else:
    raise OSError("Unsupported operating system. Please use Windows or macOS.")


def safe_event_dataframe(df_events: pd.DataFrame):
    path_safe = Path(PATH_ROOT_MAC) / "kinematics" / "events.xlsx"
    try:
        df_events.to_excel(path_safe, index=False)
        print(f"Events saved to {path_safe}")
    except Exception as e:
        warnings.warn(f"Could not save events to Excel file: {e}")


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
        size = [2560.5, 1282.5]  # MSH screen
        # size = [3456, 2030]
        fig, axs = plt.subplots(3, 1,
                                figsize=(size[0] / 100, size[1] / 100),
                                dpi=100,
                                sharex=True)
        ax_top = axs[0]
        ax_mid = axs[1]
        ax_bottom = axs[2]
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
        ax_top.plot(heel_vert_pos, label=f"{side} Heel Pos")
        ax_top.plot(toe_vert_pos, label=f"{side} Toe Pos")
        # plot vertical velocity
        ax_mid.plot(heel_vert_vel, label=f"{side} Heel Vel")
        ax_mid.plot(toe_vert_vel, label=f"{side} Toe Vel")
        ax_mid.set_ylabel("Vertical Velocity (m/s)")
        ax_mid.axhline(0, color='gray', linestyle='--', alpha=0.5)

        # plot vertical acceleration
        ax_bottom.plot(heel_vert_acc, label=f"{side} Heel Acc")
        ax_bottom.plot(toe_vert_acc, label=f"{side} Toe Acc")
        ax_bottom.set_xlabel("Frame (85 Hz)")
        ax_bottom.set_ylabel("Vertical Acceleration (m/s²)")
        ax_bottom.axhline(0, color='gray', linestyle='--', alpha=0.5)
        # plt.legend()
        ax_top.set_ylabel("Vertical Position (m)")
        # make a single legend for the whole figure in the middle right to avoid overlapping with the data points
        fig.legend(["Heel", "Toe"], loc="center right", bbox_to_anchor=(0.99, 0.5))

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

        for pk in heel_pks:
            ax_top.scatter(pk, heel_vert_pos[pk], color='red', marker='x')
            for ax in axs:
                ax.axvline(pk, color='red', linestyle='--', alpha=0.5)

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
                ax_top.scatter(pk + heel_min, heel_vert_pos[pk + heel_min], color='green', marker='o')
                ax_bottom.scatter(pk + heel_min, heel_vert_acc[pk + heel_min], color='green', marker='o')
                td_approx = pk + heel_min
                acc = heel_vert_acc
            else:
                ax_top.scatter(pk + toe_min, toe_vert_pos[pk + toe_min], color='green', marker='o')
                ax_bottom.scatter(pk + toe_min, toe_vert_acc[pk + toe_min], color='green', marker='o')
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
            ax_bottom.plot(frames_acc, acc_plot, color='red', label=f"{side} Acc Window")
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
            ax_top.scatter(ic, heel_vert_pos[ic] if acc is heel_vert_acc else toe_vert_pos[ic], color='magenta',
                           marker='D')
            ax_bottom.scatter(ic, acc[ic], color='magenta', marker='D')

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
            ax_bottom.plot(frames_to, acc_to_plot, color='blue', label=f"{side} TO Acc Window")
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
                ax_bottom.scatter(frames_to[to_acc_pk], acc_to_plot[to_acc_pk], color='cyan', marker='x')
            to = frames_to[to_acc_pk]
            # all done. write to the lists
            ics.append(ic)
            tos.append(to)

        # # overall title
        fig.suptitle(f"Bib {bib_no}, Lap {lap} - {side} Heel and Toe Vertical Position and Velocity", fontsize=16)
        fig.tight_layout()
        #
        path_plot = Path(PATH_ROOT) / "kinematics" / "plots" / "gait_events"
        path_plot.mkdir(exist_ok=True)
        for ic, to in zip(ics, tos):
            for ax in axs:
                ax.axvspan(ic, to, color='k', linestyle='--', alpha=0.1)
        fig.savefig(path_plot / f"{filename}_{side}_gait_events.png", bbox_inches="tight")
        # plt.show()
        plt.close(fig)
        # correct event frames back for the offset of the valid data window
        ics = [ic + valid_start for ic in ics]
        tos = [to + valid_start for to in tos]
        assert len(ics) == len(tos)
        events[side] = {"IC": ics, "TO": tos}
    return events


def get_events(mat_file) -> dict | None:
    matplotlib.use('Agg')
    data = loadmat(mat_file)
    try:
        events = get_running_events(data)
    except Exception as e:
        warnings.warn(f"Exception {e} in {mat_file}")
        events = None
    return events


if __name__ == '__main__':
    path_mat_root = Path(PATH_ROOT) / "kinematics" / "mat"
    path_plot = Path(PATH_ROOT) / "kinematics" / "plots"
    path_plot.mkdir(exist_ok=True)
    heat_directories = [d for d in path_mat_root.iterdir() if d.is_dir()]
    heat_directories.sort()
    heats = [d.name for d in heat_directories]
    print(heats)

    laps = list(range(1, 14))
    df_events = pd.DataFrame(columns=["Heat", "Bib", "Lap", "Events"])

    for heat in heats:
        path_mat = path_mat_root / heat
        bib_numbers = [d.name for d in path_mat.iterdir() if d.is_dir()]
        bib_numbers.sort()
        print(bib_numbers)
        for bib_number in bib_numbers:
            print(f"Processing Heat {heat}, Bib {bib_number}...")
            mat_files = list((path_mat / bib_number).glob("*filt.mat"))
            lap_file_dict = {int(f.stem.split("_")[2]): f for f in mat_files}
            # # MULTPROC
            # with ProcessPoolExecutor() as executor:
            #     results = list(executor.map(process_file, mat_files))

            # DEBUGGING WITHOUT MULTIPROC TO CHECK THE PLOTS
            for lap_no, mat_file in lap_file_dict.items():
                print(f"Processing lap {lap_no} from file {mat_file.name}...")
                events = get_events(mat_file)
                df_ = pd.DataFrame({
                    "Heat": heat,
                    "Bib": bib_number,
                    "Lap": lap_no,
                    "Events": [events]
                }, index=[0])
                df_heat = pd.concat([df_events, df_], ignore_index=True)
    safe_event_dataframe(df_events)
