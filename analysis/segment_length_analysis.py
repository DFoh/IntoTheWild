import warnings

import numpy as np
import pandas as pd
from labtools.batch_processor import BatchProcessor
from labtools.utils.c3d import load_c3d
from pathlib import Path

from analysis.util import get_participant_data, load_path_track_grand_prix, safe_result_dataframe
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['axes.formatter.useoffset'] = False



def plot_length(length, title) -> [plt.Figure, bool]:
    plt.close()
    mean, median = np.mean(length), np.median(length)
    has_diff = False
    if (median - mean) > 0.0001:
        warnings.warn(f"Difference in Median - Mean in {title}: {median-mean:.3f}")
        has_diff = True
    fig, (ax_l, ax_r) = plt.subplots(1, 2, sharey=True)
    ax_l.plot(length)
    sns.histplot(y=length, kde=True, ax=ax_r)
    for ax in (ax_l, ax_r):
        ax.axhline(mean, color='C1', ls='--', label=f'Mean {mean:.3f}')
        ax.axhline(median, color='C2', ls=':', label=f'Median {median:.3f}')
        ax.axhline(median, color='C2', ls=':', label=f'Diff {(median-mean):.3f}')
    ax_r.legend()
    ax_l.yaxis.set_major_formatter('{x:.2f}')
    ax_l.set_ylabel("cm")
    fig.suptitle(title)
    return fig, has_diff

def get_length(v1, v2):
    vec_length = v2 - v1
    # vc = vec_length[~np.isnan(vec_length).any(axis=1)]
    length = np.linalg.norm(vec_length[~np.isnan(vec_length).any(axis=1)], axis=1)
    return length

def segment_length_processor(row: pd.Series) -> dict | None:
    """ Calculates hip width, left and right thigh, and shank lengths
    from a motion file based on joint trajectories"""
    path_file = row.path
    bib = row.Bib
    lap = row.Lap.split("_")[2]
    path_temp = Path(r"C:\tmp\plots")
    # path_plot = load_path_track_grand_prix() / "kinematics" / "plots" / "segment_lengths"
    path_plot = path_temp / "segment_lengths"

    data = loadmat(path_file)
    # hip joint centers
    r_hjc = data.get("Right_Hip_Center")[0][0]
    l_hjc = data.get("Left_Hip_Center")[0][0]
    # knee joint centers
    r_kjc = data.get("Right_Knee_Center")[0][0]
    l_kjc = data.get("Left_Knee_Center")[0][0]
    # ankle joint centers
    r_ajc = data.get("Right_Ankle_Center")[0][0]
    l_ajc = data.get("Left_Ankle_Center")[0][0]

    hip_width = get_length(r_hjc, l_hjc)
    hip_width *= 100  # in cm
    fig_hip_width, has_diff = plot_length(hip_width, f"Hip Width - Bib: {bib} - Lap: {lap}")
    if has_diff:
        fname = f"hip_width_{bib}_{lap}.png"
        fig_hip_width.savefig(path_plot.joinpath(fname))

    # left side
    left_thigh_length = get_length(l_hjc, l_kjc)
    left_thigh_length *= 100
    fig_left_thigh_length, has_diff = plot_length(left_thigh_length, f"Left Thigh Length - Bib: {bib} - Lap: {lap}")
    if has_diff:
        fname = f"left_thigh_length{bib}_{lap}.png"
        fig_left_thigh_length.savefig(path_plot.joinpath(fname))

    left_shank_length = get_length(l_kjc, l_ajc)
    left_shank_length *= 100
    fig_left_shank_length, has_diff = plot_length(left_shank_length, f"Left Shank Length - Bib: {bib} - Lap: {lap}")
    if has_diff:
        fname = f"left_shank_length{bib}_{lap}.png"
        fig_left_shank_length.savefig(path_plot.joinpath(fname))

    # right side
    right_thigh_length = get_length(r_hjc, r_kjc)
    right_thigh_length *= 100
    fig_right_thigh_length, has_diff = plot_length(right_thigh_length, f"Right Thigh Length - Bib: {bib} - Lap: {lap}")
    if has_diff:
        fname = f"right_thigh_length{bib}_{lap}.png"
        fig_right_thigh_length.savefig(path_plot.joinpath(fname))

    right_shank_length = get_length(r_kjc, r_ajc)
    right_shank_length *= 100
    fig_right_shank_length, has_diff = plot_length(right_shank_length, f"Right Shank Length - Bib: {bib} - Lap: {lap}")
    if has_diff:
        fname = f"right_shank_length{bib}_{lap}.png"
        fig_right_shank_length.savefig(path_plot.joinpath(fname))

    return {
        "hip_width": np.median(hip_width),
        "left_thigh_length": np.median(left_thigh_length),
        "left_shank_length": np.median(left_shank_length),
        "right_thigh_length": np.median(right_thigh_length),
        "right_shank_length": np.median(right_shank_length),
    }





if __name__ == '__main__':
    recalc_events = True
    recalc_kinematics = False
    path_root = load_path_track_grand_prix()
    path_kinematics = Path(path_root) / "kinematics"

    path_c3d_root = path_kinematics / "mat"
    bp = BatchProcessor(path_c3d_root,
                        level_names=["Bib", "Lap"],
                        file_pattern="_filt.mat")

    res = bp.apply(segment_length_processor,
                   multiprocess=True)

    df_seg_lengths = pd.json_normalize(res)
    df_seg_lengths = pd.concat([bp.index.reset_index(drop=True), df_seg_lengths], axis=1)
    df_seg_lengths["Lap"] = df_seg_lengths["Lap"].apply(lambda x: x.split("_")[2])
    df_seg_lengths["Lap"] = df_seg_lengths["Lap"].astype(int)

    safe_result_dataframe(df_seg_lengths, "segment_lengths.xlsx")
    print(bp.index.head())
