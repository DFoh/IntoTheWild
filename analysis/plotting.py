import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat

from analysis.util import PATH_ROOT, make_file_path, safe_result_dataframe, load_result_dataframe

# For debugging in PyCharm:
if sys.platform.startswith('win'):
    matplotlib.use('Qt5Agg')
elif sys.platform.startswith('darwin'):
    if matplotlib.__version__ != '3.7.1':
        raise ImportError(
            f"matplotlib version 3.7.1 is required for proper debugging in PyCharm, but found {matplotlib.__version__}. Please install the correct version.")
else:
    raise OSError("Unsupported operating system. Please use Windows or macOS.")


def plot_knee_flexion_over_laps_subject_wise(df: pd.DataFrame):
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
    for index, row in df_events.iterrows():
        heat = row["Heat"]
        bib = row["Bib"]
        lap_no = row["Lap"]
        file_path = make_file_path(path_mat_root, heat, bib, lap_no)
        print(f"Processing lap {lap_no} from file {file_path.name}...")
        if not file_path.exists():
            warnings.warn(f"File {file_path} does not exist, skipping...")
            continue
        data = loadmat(str(file_path))

        limb_lenghts = {}
        for side in ["Left", "Right"]:
            hip_center_traj = data[f'{side}_Hip_Center'][0][0]
            knee_center_traj = data[f'{side}_Knee_Center'][0][0]
            try:
                ankle_center_traj = data[f'{side}_Ankle_Center'][0][0]
            except KeyError:
                ankle_center_traj = data[f'{side}t_Ankle_Center'][0][
                    0]  # stupid typo in the original data, but we have to deal with it
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
            ax = axs.flatten()[p]
            ax.plot(df_bib["Lap"], plot_data, marker='o', color="k")
            # plot the average avg_shank_length_left over all laps and the +/- 0.5cm range as dashed lines
            avg = plot_data.median()
            ax.axhline(avg, color="k", linestyle="--")
            ax.axhspan(avg - 0.005, avg + 0.005, color="k", alpha=0.2)
            # plt.axhline(avg + 0.005, color="k", linestyle="--")
            # plt.axhline(avg - 0.005, color="k", linestyle="--")
            ax.set_title(param)
        fig.suptitle(f"Bib {bib} - Limb Lengths over Laps")
        # plt.savefig(path_plot / f"{bib} - limb_lengths.png")
        plt.show()
        plt.close()


def plot_parameter_over_laps(df, parameter_name, y_label):
    path_plot = Path(PATH_ROOT) / "kinematics" / "plots" / "parameters_over_laps"
    path_plot.mkdir(parents=True, exist_ok=True)
    # average for each lap and bib for left and right side

    df = df.groupby(["Heat", "Bib", "Lap"]).mean(numeric_only=True).reset_index()
    sns.lineplot(data=df, x="Lap", y=parameter_name, hue="Heat", units="Bib", estimator=None, marker='o')
    plt.title(f"{y_label} over Laps")
    plt.savefig(path_plot / f"{parameter_name}_vs_laps_indiv.png")
    plt.close()
    sns.lineplot(data=df, x="Lap", y=parameter_name, hue="Heat", marker='o')
    plt.title(f"{y_label} over Laps (average)")
    plt.savefig(path_plot / f"{parameter_name}_vs_laps_avg.png")
    plt.close()


def main():
    df_kinematic_params = load_result_dataframe("kinematic_params.xlsx")
    param_names = [
        ("running_speed_ms", "Running Speed (m/s)"),
        ("step_rate_spm", "Step Rate (steps/min)"),
        ("contact_time_ms", "Contact Time (ms)"),
        ("flight_time_ms", "Flight Time (ms)"),
        ("step_length_m", "Step Length (m)"),
        ("trunk_flexion_deg", "Peak Trunk Flexion (degrees)"),
        ("vertical_pelvis_movement_cm", "Vertical Pelvis Movement (cm)"),
        ("neg_peak_pelvis_obliquity_deg", "Negative Peak Pelvis Obliquity (degrees)"),
        ("hip_flexion_rom_deg", "Hip Flexion ROM (degrees)"),
        ("max_knee_flex_stance_deg", "Max Knee Flexion in Stance (degrees)"),
        ("overstriding_cm", "Overstriding (cm)"),
    ]
    for param_name, y_label in param_names:
        plot_parameter_over_laps(df_kinematic_params, param_name, y_label)


if __name__ == '__main__':
    main()
