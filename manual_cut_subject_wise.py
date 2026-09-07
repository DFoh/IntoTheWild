import shutil
import subprocess
from pathlib import Path

import pandas as pd

from analysis.util import load_path_root


def get_video_fps(path_video: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=nw=1:nk=1",
        str(path_video),
    ]
    rate = subprocess.check_output(cmd, text=True).strip()  # e.g. "85/1"
    num, den = rate.split("/")
    return float(num) / float(den)


def main(heat: int, bib_no: int, overwrite_existing: bool = False):
    sheet_name = f"Heat {heat}"

    omit_camera_ids = ["26076", "26072"]

    path_out = Path(r"E:\QTM_Projects\IntoTheWild\manual_cut_subject_wise")
    path_out.mkdir(parents=True, exist_ok=True)

    # path_calibration_file = Path("E:/QTM_Projects/IntoTheWild/Calibrations/20250524_094747_cams_removed.qca.txt")
    path_calibration_file = Path("E:/QTM_Projects/IntoTheWild/Calibrations/20250524_094747.qca.txt")
    path_prefs_file = Path("E:/QTM_Projects/IntoTheWild/prefs.pxt")

    # path_matching_excel = Path(r"E:\QTM_Projects\IntoTheWild\Bib_Frame_Matching.xlsx")
    path_matching_excel = load_path_root() / "Bib_Frame_Matching.xlsx"


    df_matching = pd.read_excel(path_matching_excel, sheet_name=sheet_name)
    bib_nos = df_matching["bib_no"].dropna().unique()
    if bib_no not in bib_nos:
        print(f"Bib number {bib_no} not found in sheet {sheet_name}")
        exit(1)
    bib = bib_no

    path_bib_out = path_out / sheet_name / str(int(bib))
    path_bib_out.mkdir(parents=True, exist_ok=True)

    df_bib = df_matching[df_matching["bib_no"] == bib]
    print(f"Bib number: {bib}")

    for lap, df_lap in df_bib.groupby("lap", sort=False):
        lap = int(lap)
        print(f"  Lap: {lap}")
        lap_path = path_bib_out / f"lap_{lap}"
        lap_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path_calibration_file, lap_path / "cal.txt")
        shutil.copyfile(path_prefs_file, lap_path / "prefs.pxt")

        start_frame = int(df_lap["frame_start"].iloc[0])
        end_frame = int(df_lap["frame_end"].iloc[0])

        n_frames = end_frame - start_frame + 1
        if n_frames <= 0:
            print(f"    Invalid frame range: {start_frame} to {end_frame}")
            continue

        print(f"    Frames: {start_frame} to {end_frame} (total {n_frames})")

        path_video = Path(df_lap["file_path"].iloc[0])
        if path_video.stem != "Running - Markerless":
            path_video = path_video / "Running - Markerless"

        if not path_video.exists():
            print(f"    Video folder does not exist: {path_video}")
            continue

        trial_name = str(df_lap["file_name"].iloc[0]).strip()
        video_files = sorted(path_video.glob(f"{trial_name}_Miqus_*.avi"))

        if (len(video_files) != 12) and (len(video_files) != 14):
            print(f"    Expected 12 or 14 video files for trial {trial_name}, found {len(video_files)}")
            continue

        for video_file in video_files:
            camera_id = video_file.stem.split("_")[-1]
            if camera_id in omit_camera_ids:
                print(f"    Skipping camera {camera_id} as it is in the omit list")
                continue
            path_camera_out = lap_path / camera_id
            path_camera_out.mkdir(parents=True, exist_ok=True)

            fps = get_video_fps(video_file)
            t0 = start_frame / fps
            dur = (end_frame - start_frame + 1) / fps  # exact inclusive range

            output_file_path = path_camera_out / f"{camera_id}.avi"

            if not overwrite_existing and output_file_path.exists():
                print(f"    Output file {output_file_path} already exists, skipping cut")
                continue

            cmd = [
                "ffmpeg",
                "-hide_banner", "-loglevel", "error",
                "-ss", f"{t0:.7f}",
                "-i", str(video_file),
                "-t", f"{dur:.7f}",
                "-an",
                "-c:v", "copy",
                "-y",
                str(output_file_path),
            ]

            subprocess.run(cmd, check=True)
            print(f"    Saved cut video to {output_file_path}")
    print("Done.")


def rename_output(heat: int, bib_no: int):
    """ Rename the c3d output from the parent directory (bib number) and the folder (lap number)
    e.g. "../c3d/Heat1/100/lap_1/pose_2.c3d" -> "../c3d/Heat1/100/100_lap_1.c3d"
    """
    path_c3d = Path(r"D:\IntoTheWild\c3d")
    heat_path = path_c3d / f"Heat_{heat}"
    bib_path = heat_path / str(bib_no)
    for lap_folder in bib_path.glob("lap_*"):
        lap_number = lap_folder.name.split("_")[1]
        c3d_files = list(lap_folder.glob("pose_*.c3d"))
        if len(c3d_files) == 0:
            print(f"  No c3d file found in {lap_folder}")
            # remove folder
            shutil.rmtree(lap_folder)
            continue
        for c3d_file in c3d_files:
            if "filt" in c3d_file.stem:
                new_c3d_name = f"{bib_no}_lap_{lap_number}_filt.c3d"
            else:
                new_c3d_name = f"{bib_no}_lap_{lap_number}.c3d"
            new_c3d_path = lap_folder.parent / new_c3d_name
            print(f"  Moved {c3d_file} to {new_c3d_path}")
            shutil.move(c3d_file, new_c3d_path)
            # delete folder
        print(f"  Deleting folder {lap_folder}")
        shutil.rmtree(lap_folder)


if __name__ == "__main__":
    # bibs = [107, 170, 221, 223, 280, 215,  310, 360, 227, 308, 251, 309]
    bibs_heat_1 = [100, 166, 177, 185, 186, 213, 214, 218, 222, 224, 225, 231, 245, 311, 313, 318, 363]  # M1
    bibs_heat_2 = [15, 22, 182, 183, 187, 200, 211, 230, 258, 274, 277, 306, 315]  # F1
    bibs_heat_2 = [274]  # F1
    bibs_heat_3 = [107, 170, 215, 221, 223, 227, 251, 280, 308, 309, 310, 360]  # M2
    bibs_heat_4 = [98, 219, 246, 247, 248, 252, 283, 288, 289, 295, 362]  # MIXED
    for bib in bibs_heat_2:
        print(f"Processing bib {bib}...")
        main(heat=2, bib_no=bib)
        # rename_output(heat=4, bib_no=bib)
