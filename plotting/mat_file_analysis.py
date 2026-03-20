import warnings

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.signal import find_peaks

backend = 'Qt5Agg'

matplotlib.use(backend)

from pathlib import Path
from scipy.io import loadmat

PATH_ROOT = r"C:\Users\dominik.fohrmann\OneDrive - MSH Medical School Hamburg - University of Applied Sciences and Medical University\Dokumente\Projects\IntoTheWild\data\TrackGrandPrix"


def get_running_events(data):
    # assuming position data in meters and FRAME RATE in Hz
    events = {}
    sample_rate = data['FRAME_RATE'][0][0][0][0]  # Assuming frame rate is stored in this field
    path_file = Path(data['FILE_NAME'][0][0][0])
    filename = path_file.stem
    bib_no = filename.split("_")[0]
    lap = filename.split("_")[2]
    for side in ["Left", "Right"]:
        size = [2560.5, 1282.5]
        fig, axs = plt.subplots(3, 1, figsize=(size[0] / 100, size[1] / 100), dpi=100, sharex=True)
        ax_top = axs[0]
        ax_mid = axs[1]
        ax_bottom = axs[2]
        # print(f"Processing {side} side...")
        heel_vert_pos = data[f"{side}_Heel_Pos"][0][0][:, 2]  # Assuming vertical position is the 3rd column
        heel_vert_vel = np.gradient(heel_vert_pos) * sample_rate  # Vertical velocity of the heel
        heel_vert_acc = np.gradient(heel_vert_vel) * sample_rate  # Vertical acceleration of the heel
        toe_vert_pos = data[f"{side}_Toe_Pos"][0][0][:, 2]  # Assuming vertical position is the 3rd column
        toe_vert_vel = np.gradient(toe_vert_pos) * sample_rate
        toe_vert_acc = np.gradient(toe_vert_vel) * sample_rate
        ax_top.plot(heel_vert_pos, label=f"{side} Heel Pos")
        ax_top.plot(toe_vert_pos, label=f"{side} Toe Pos")
        # plt.legend()
        ax_top.set_ylabel("Vertical Position (m)")

        # for robustness, we can set an expected minimum step rate to define "distance" between peaks.
        min_step_rate = 120  # steps per minute
        min_step_interval = (60 / min_step_rate) * sample_rate  # minimum interval in frames

        heel_pks, _ = find_peaks(heel_vert_pos, height=0.2, distance=min_step_interval)
        toe_pks = find_peaks(toe_vert_pos, height=0.2, distance=min_step_interval)[0]

        # see if the first swing event (heel_pks) has sufficient frames before start of the file to capture the preceding initial contact event
        avg_swing_evt_frames = np.diff(heel_pks).mean()
        search_window_ratio = 0.66  # adjust from experience
        if heel_pks[0] > search_window_ratio * avg_swing_evt_frames:
            heel_pks = np.insert(heel_pks, 0, 0)  # additional start frame for the minimum search up to the first swing event



        heel_mins, _ = find_peaks(-heel_vert_pos, height=-0.1, distance=min_step_interval)
        toe_mins = find_peaks(-toe_vert_pos, height=-0.1, distance=min_step_interval)[0]


        for pk in heel_pks:
            ax_top.scatter(pk, heel_vert_pos[pk], color='red', marker='x')
            for ax in axs:
                ax.axvline(pk, color='red', linestyle='--', alpha=0.5)
        for mn in heel_mins:
            ax_top.scatter(mn, heel_vert_pos[mn], color='blue', marker='o')
            for ax in axs:
                ax.axvline(mn, color='blue', linestyle='--', alpha=0.5)
        for pk in toe_pks:
            ax_top.scatter(pk, toe_vert_pos[pk], color='orange', marker='x')
            for ax in axs:
                ax.axvline(pk, color='orange', linestyle='--', alpha=0.5)
        for mn in toe_mins:
            ax_top.scatter(mn, toe_vert_pos[mn], color='cyan', marker='o')
            for ax in axs:
                ax.axvline(mn, color='k', linestyle='--', alpha=0.5)

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

        # overall title
        fig.suptitle(f"Bib {bib_no}, Lap {lap} - {side} Heel and Toe Vertical Position and Velocity", fontsize=16)
        fig.tight_layout()

        path_plot = Path(PATH_ROOT) / "kinematics" / "plots" / "foot_events"
        path_plot.mkdir(exist_ok=True)
        # fig.savefig(path_plot / f"{filename}_{side}_heel_events.png", bbox_inches="tight")
        plt.show()
        plt.close(fig)
    return events


if __name__ == '__main__':
    path_mat_root = Path(PATH_ROOT) / "kinematics" / "mat"
    path_plot = Path(PATH_ROOT) / "kinematics" / "plots"
    path_plot.mkdir(exist_ok=True)
    heat_directories = [d for d in path_mat_root.iterdir() if d.is_dir()]
    heat_directories.sort()
    heats = [d.name for d in heat_directories]
    print(heats)
    for heat in heats:
        path_mat = path_mat_root / heat
        bib_numbers = [d.name for d in path_mat.iterdir() if d.is_dir()]
        bib_numbers.sort()
        print(bib_numbers)
        for bib_number in bib_numbers:
            print(f"Processing Heat {heat}, Bib {bib_number}...")
            mat_files = list((path_mat / bib_number).glob("*filt.mat"))
            size = [2560.5, 1282.5]
            fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
            for m, mat_file in enumerate(mat_files):
                #
                data = loadmat(mat_file)
                try:
                    events = get_running_events(data)
                except KeyError as key_error:
                    warnings.warn(f"{key_error} in {mat_file}")

                continue
                #
                pelvis_com_pos = data['Pelvis_COM_Position'][0][0]
                left_heel_pos = data['Left_Heel_Pos'][0][0]
                right_heel_pos = data['Right_Heel_Pos'][0][0]
                left_toe_pos = data['Left_Toe_Pos'][0][0]
                right_toe_pos = data['Right_Toe_Pos'][0][0]
                ax.plot(pelvis_com_pos[:, 0], pelvis_com_pos[:, 2], label=mat_file.stem)
                color = ax.get_lines()[-1].get_color()
                ax.plot(left_heel_pos[:, 0], left_heel_pos[:, 2], label=mat_file.stem, color=color, linestyle="dashed")
                ax.plot(left_toe_pos[:, 0], left_toe_pos[:, 2], label=mat_file.stem, color=color, linestyle="dotted")
                if m > 0:
                    break

                pelvis_com_pos_ap = pelvis_com_pos[:, 0]
                frame_rate_hz = 85  # Hz
                pelvis_com_velocity_ap = np.gradient(pelvis_com_pos_ap) * frame_rate_hz
                avg_running_speed_kmh = np.nanmean(pelvis_com_velocity_ap) * 3.6
                avg_running_speed_kmh = np.nanmedian(pelvis_com_velocity_ap) * 3.6

            # ax.set_ylim(-0.1, 1.1)
            # title = f"Heat {heat}, Bib {bib_number} - laps 1-{m + 1}"
            # ax.set_title(title)
            # # plt.legend()
            # path_plot_out = path_plot / "pelvis_heel_plots"
            # path_plot_out.mkdir(exist_ok=True)
            # # plt.savefig(path_plot_out / f"{heat}_{bib_number}_pelvis_heel.png", bbox_inches="tight")
            # plt.show()
            # plt.close()
