from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil

import cv2
import pandas as pd

start_numbers_heat_1 = [100, 166, 177, 185, 186, 213, 214, 218, 222, 224, 225, 231, 245, 311, 313, 318, 363]  # M1
start_numbers_heat_2 = [15, 22, 182, 183, 187, 200, 211, 230, 258, 274, 277, 306, 315]  # F1
start_numbers_heat_3 = [107, 170, 215, 221, 223, 227, 251, 280, 308, 309, 310, 360]  # M2
start_numbers_heat_4 = [98, 219, 246, 247, 248, 252, 283, 288, 289, 295, 362]  # MIXED

start_numbers = {
    "heat_1": start_numbers_heat_1,
    "heat_2": start_numbers_heat_2,
    "heat_3": start_numbers_heat_3,
    "heat_4": start_numbers_heat_4,
}

path_video_data_root = Path("E:\ITW_Backup\HHTGP_HHTGP")
path_heat_1 = path_video_data_root.joinpath("Running - Markerless")
path_heat_2 = path_video_data_root.joinpath("Running - Markerless_3")
path_heat_3 = path_video_data_root.joinpath("Running - Markerless_4")
path_heat_4 = path_video_data_root.joinpath("Running - Markerless_5")

paths_videos = {
    "heat_1": path_heat_1,
    "heat_2": path_heat_2,
    "heat_3": path_heat_3,
    "heat_4": path_heat_4,
}


def get_participant_data() -> pd.DataFrame:
    path_participant_data = Path("../data/into_the_wild_participant_data.xlsx")
    return pd.read_excel(path_participant_data)


def get_video_path_by_trial_name_and_camera_id(heat: int, trialname: str, cam_id: str) -> Path:
    path_heat_root = paths_videos[f"heat_{heat}"]
    path_video_file = list(path_heat_root.glob(f"{trialname}_Miqus_*_{cam_id}.avi"))
    if len(path_video_file) != 1:
        raise ValueError(f"Could not find unique video file for trial {trialname} and camera id {cam_id}")
    return path_video_file[0]


def get_heat_trials(heat: int) -> list[str]:
    path_heat_root = paths_videos[f"heat_{heat}"]
    qtm_files = list(path_heat_root.glob("*trial Markerless*.qtm"))
    # strip to only get the trial names
    trial_names = [f.stem for f in qtm_files]
    return trial_names


def get_trial_videos_by_trial_name_and_path(trialname: str, path: Path) -> list[Path]:
    path_heat_root = path
    video_files = list(path_heat_root.glob(f"{trialname}_Miqus_*.avi"))
    return video_files


def get_video_file_frame_count(path_video_file: Path) -> int:
    cap = cv2.VideoCapture(str(path_video_file))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video file {path_video_file.name} has {length} frames.")
    cap.release()
    return length


def count_readable_frames(p: str) -> int:
    cap = cv2.VideoCapture(p)
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def header_frame_count(p: str) -> int:
    cap = cv2.VideoCapture(p)
    hdr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return hdr


def analyze_one(p: str) -> tuple[str, int, int, int]:
    hdr = header_frame_count(p)
    real = count_readable_frames(p)
    return (Path(p).name, hdr, real, real - hdr)


def report_mp(trialname: str, root: Path, workers: int | None = None) -> None:
    files = sorted(root.glob(f"{trialname}_Miqus_*.avi"))
    if not files:
        print("No files found.")
        return

    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    paths = [str(f) for f in files]

    results: dict[str, tuple[int, int, int]] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(analyze_one, p): p for p in paths}
        for fut in as_completed(futs):
            name, hdr, real, diff = fut.result()
            results[name] = (hdr, real, diff)

    for f in files:
        name = f.name
        hdr, real, diff = results[name]
        print(f"{name}: header={hdr}, readable={real}, diff={diff}")

    diffs = {results[f.name][1] for f in files}
    if len(diffs) > 1:
        print(f"\nWARNING: readable frame counts differ: {sorted(diffs)}")
    else:
        print(f"\nOK: all readable frame counts identical: {next(iter(diffs))}")

def probe_one(p: str):
    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        return (Path(p).name, "OPEN_FAIL")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = (n / fps) if fps and fps > 0 else None

    fourcc_i = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join([chr((fourcc_i >> (8*i)) & 0xFF) for i in range(4)])

    cap.release()
    return (Path(p).name, n, fps, dur, w, h, fourcc)

def report(trialname: str, root: Path, workers: int | None = None):
    files = sorted(root.glob(f"{trialname}_Miqus_*.avi"))
    workers = workers or max(1, (os.cpu_count() or 4) - 1)

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(probe_one, str(f)) for f in files]
        for fut in as_completed(futs):
            rows.append(fut.result())

    rows.sort(key=lambda x: x[0])
    for r in rows:
        print(r)

    # quick uniqueness check
    if rows and len(rows[0]) > 2:
        fps_set = {round(r[2], 6) for r in rows}
        dur_set = {None if r[3] is None else round(r[3], 6) for r in rows}
        size_set = {(r[4], r[5]) for r in rows}
        fourcc_set = {r[6] for r in rows}
        print("\nUnique FPS:", sorted(fps_set))
        print("Unique DUR:", sorted(dur_set))
        print("Unique SIZE:", sorted(size_set))
        print("Unique FOURCC:", sorted(fourcc_set))


def make_theia_format_data(path: Path, trial_name: str):
    # Path to the Master calibration file (with cameras 2 and 8 removed):
    path_calibration_file = Path("E:/QTM_Projects/IntoTheWild/Calibrations/20250524_094747_cams_removed.qca.txt")
    path_out = path / "TheiaFormatData" / trial_name
    path_out.mkdir(parents=True, exist_ok=True)
    # copy the master calibration file to the output folder
    shutil.copyfile(path_calibration_file, path_out / "cal.txt")
    video_files = get_trial_videos_by_trial_name_and_path(trial_name, path)
    for video_file in video_files:
        # Skip cameras "26076" and "26072"
        if video_file.stem.endswith("26076") or video_file.stem.endswith("26072"):
            continue
        # make folder from the camera SID name
        parts = video_file.stem.split("_")
        cam_id = parts[-1]
        path_cam = path_out / cam_id
        path_cam.mkdir(parents=True, exist_ok=True)
        # copy video file to the new folder and make video file name "{came_id}.avi"
        path_video_out = path_cam / f"{cam_id}.avi"
        shutil.copyfile(video_file, path_video_out)


if __name__ == '__main__':
    root = Path(r"E:\QTM_Projects\IntoTheWild\Data\Heat 1_Heat 1\Running - Markerless")
    # report_mp("Running trial Markerless 7", root, workers=None)
    trial_nos = list(range(21, 24))
    for trial_no in trial_nos:
        trial_name = f"Running trial Markerless {trial_no}"
        make_theia_format_data(root, f"Running trial Markerless {trial_no}")


    #
    # path = Path("E:\QTM_Projects\IntoTheWild\Data\Heat 1_Heat 1\Running - Markerless")
    # trial_name = "Running trial Markerless 7"
    # video_files = get_trial_videos_by_trial_name_and_path(trial_name, path)
    # for video_file in video_files:
    #     path = video_file
    #     get_video_file_frame_count(Path(path))
