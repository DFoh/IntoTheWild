import shutil
import subprocess

from moviepy import VideoFileClip

from util import get_heat_trials, paths_videos

FFMPEG = shutil.which("ffmpeg")  # versucht ffmpeg aus PATH zu finden
if FFMPEG is None:
    # FALLBACK: HIER DEIN ABSOLUTER PFAD ZU ffmpeg.exe
    # z.B.: r"C:\ffmpeg\bin\ffmpeg.exe"
    FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"  # <- anpassen

cut_times_heat_1 = [
    # ["3:25", "3:45"], ["4:30", "4:55"], ["5:45", "6:10"], ["6:55", "7:25"], ["8:05", "8:40"],
    # ["9:15", "9:55"], ["10:25", "11:15"], ["11:35", "12:30"], ["12:42", "13:25"], ["13:32", "13:43"],
    # ["13:52", "14:03"], ["14:05", "14:40"], ["14:51", "15:13"], ["15:19", "15:34"], ["15:37", "15:52"],
    # ["16:06", "16:25"], ["16:30", "16:49"], ["16:52", "17:09"], ["17:20", "17:34"], ["17:36", "17:49"],
    # ["17:51", "17:58"],
    ["18:05", "18:23"],
    # ["18:37", "18:49"]
]

cut_times_heat_2 = [["0:27", "0:40"], ["1:46", "2:09"], ["3:07", "3:27"], ["3:28", "3:38"], ["4:24", "4:39"],
                    ["4:42", "4:51"], ["4:55", "5:05"], ["5:45", "6:01"], ["6:06", "6:18"], ["6:25", "6:36"],
                    ["7:05", "7:22"], ["7:31", "7:49"], ["7:54", "8:08"], ["8:39", "8:45"], ["8:56", "9:06"],
                    ["9:08", "9:29"], ["9:34", "9:55"], ["9:59", "10:11"], ["10:21", "10:34"], ["10:39", "10:45"],
                    ["10:47", "10:57"], ["11:05", "11:17"], ["11:21", "11:37"], ["11:48", "12:04"], ["12:09", "12:38"],
                    ["12:43", "12:54"], ["12:58", "13:04"], ["13:13", "13:33"], ["13:41", "14:02"], ["14:05", "14:19"],
                    ["14:22", "14:32"], ["14:22", "14:32"], ["14:43", "14:53"], ["15:07", "15:25"], ["15:27", "15:41"],
                    ["15:52", "16:04"], ["16:10", "16:19"], ["16:24", "16:34"], ["16:37", "16:43"], ["16:45", "16:54"],
                    ["16:55", "17:07"], ["17:18", "17:24"], ["17:33", "17:44"], ["17:57", "18:05"], ["18:11", "18:20"],
                    ["18:23", "18:29"], ["19:05", "19:10"]]

cut_times_heat_3 = {
    "Running trial Markerless 1": [["0:29", "0:41"]],
    "Running trial Markerless 2": [["0:10", "0:29"]],
    "Running trial Markerless 3": [["0:08", "0:34"]],
    "Running trial Markerless 4": [["0:09", "0:21"],
                                   ["0:28", "0:41"]],
    "Running trial Markerless 5": [["0:09", "0:16"],
                                   ["0:17", "0:24"],
                                   ["0:34", "0:50"]],
    "Running trial Markerless 6": [["0:09", "0:16"],
                                   ["0:19", "0:27"],
                                   ["0:39", "1:00"]],
    "Running trial Markerless 7": [["0:08", "0:16"],
                                   ["0:21", "0:29"]],
    "Running trial Markerless 8": [["0:05", "0:21"],
                                   ["0:22", "0:28"]],
    "Running trial Markerless 9": [["0:00", "0:07"],
                                   ["0:12", "0:20"]],
    "Running trial Markerless 10": [["0:06", "0:21"],
                                    ["0:27", "0:45"],
                                    ["0:49", "0:59"]],
    "Running trial Markerless 11": [["0:11", "0:17"],
                                    ["0:18", "0:38"],
                                    ["0:39", "0:47"],
                                    ["0:48", "0:53"],
                                    ["0:53", "0:57"]],
    "Running trial Markerless 12": [["0:10", "0:17"],
                                    ["0:18", "0:34"],
                                    ["0:39", "0:52"]],
    "Running trial Markerless 13": [["0:11", "0:27"],
                                    ["0:31", "0:47"],
                                    ["0:49", "0:54"]],
    "Running trial Markerless 14": [["0:06", "0:16"],
                                    ["0:21", "0:27"],
                                    ["0:28", "0:41"],
                                    ["0:44", "0:54"]],
    "Running trial Markerless 15": [["0:00", "0:04"]],
    "Running trial Markerless 16": [["0:05", "0:10"],
                                    ["0:10", "0:17"],
                                    ["0:25", "0:30"],
                                    ["0:34", "0:44"],
                                    ["0:57", "1:01"]]
}

cut_times_heat_4 = {
    "Running trial Markerless 1": [["0:11", "0:38"]],
    "Running trial Markerless 2": [["0:10", "0:21"],
                                   ["0:22", "0:33"],
                                   ["0:38", "0:46"]],
    # "Running trial Markerless 3": [["0:08", "0:34"]],  # no participant in this trial
    "Running trial Markerless 4": [["0:05", "0:16"],
                                   ["0:25", "0:42"],
                                   ["0:54", "1:02"]],
    "Running trial Markerless 5": [["0:02", "0:12"],
                                   ["0:18", "0:25"],
                                   ["0:34", "0:47"],
                                   ["0:54", "1:02"],
                                   ["1:19", "1:26"],
                                   ["1:28", "1:40"],
                                   ["1:56", "2:03"],
                                   ["2:09", "2:19"],
                                   ["2:20", "2:28"],
                                   ["2:49", "2:56"],
                                   ["2:56", "3:07"],
                                   ["3:11", "3:18"],
                                   ["3:33", "3:40"],
                                   ["3:45", "3:54"],
                                   ["4:02", "4:09"],
                                   ["4:26", "4:37"],
                                   ["4:41", "4:48"],
                                   ["5:06", "5:14"],
                                   ["5:15", "5:22"],
                                   ["5:22", "5:33"],
                                   ["5:42", "5:52"],
                                   ["5:56", "6:03"],
                                   ["6:03", "6:10"],
                                   ["6:36", "6:43"],
                                   ["6:55", "7:15"],
                                   ["7:21", "7:32"],
                                   ["7:37", "7:45"],
                                   ["8:28", "8:35"],
                                   ["8:37", "8:44"],
                                   ["8:47", "9:04"],
                                   ["9:05", "9:19"]],
    "Running trial Markerless 6": [["0:09", "0:18"],
                                   ["0:18", "0:25"],
                                   ["0:29", "0:36"],
                                   ["0:43", "0:58"],
                                   ["1:37", "1:43"],
                                   ["1:43", "1:52"],
                                   ["1:53", "1:59"],
                                   ["2:06", "2:19"],
                                   ["2:19", "2:29"],
                                   ["2:28", "2:36"],
                                   ["2:48", "2:56"],
                                   ["2:59", "3:05"],
                                   ["3:07", "3:13"],
                                   ["3:17", "3:24"],
                                   ["3:35", "3:42"],
                                   ["3:42", "3:49"],
                                   ["3:48", "3:53"],
                                   ["3:53", "3:59"],
                                   ["4:04", "4:19"],
                                   ],
    "Running trial Markerless 7": [["0:05", "0:14"],
                                   ["0:32", "0:39"],
                                   ["0:40", "0:48"],
                                   ["0:48", "0:56"],
                                   ["1:38", "1:47"]],
    "Running trial Markerless 8": [["0:12", "0:19"]],
    "Running trial Markerless 9": [["0:05", "0:13"]]
}
cut_times = {
    "heat_1": cut_times_heat_1,
    "heat_2": cut_times_heat_2,
}


def cut_video(path_video_in, path_video_out, start_time_str, end_time_str):
    # "mm:ss" → Seconds
    m, sec = map(int, start_time_str.split(":"))
    start_time = m * 60 + sec
    m, sec = map(int, end_time_str.split(":"))
    end_time = m * 60 + sec

    with VideoFileClip(str(path_video_in)) as clip:
        try:
            sub = clip.subclipped(start_time, end_time)
            sub.write_videofile(
                str(path_video_out),
                codec="copy",
                audio=False,
                fps=clip.fps
            )
        except ValueError as e:
            print(f"Error cutting video {path_video_in} from {start_time_str} to {end_time_str}: {e}")


def cut_video_lossless(path_video_in, path_video_out, start_time_str, end_time_str):
    m, sec = map(int, start_time_str.split(":"))
    start_time = m * 60 + sec
    m, sec = map(int, end_time_str.split(":"))
    end_time = m * 60 + sec
    duration = end_time - start_time
    if duration <= 0:
        print(f"Skip {path_video_in}: invalid duration {start_time_str}–{end_time_str}")
        return

    cmd = [
        FFMPEG,
        "-y",
        "-ss", str(start_time),
        "-i", str(path_video_in),
        "-t", str(duration),
        "-c", "copy",
        str(path_video_out),
    ]
    # print(" ".join(cmd))
    subprocess.run(cmd, check=False)


def process_trial(path_in, path_out, trial_name, cut_marks):
    path_videos = list(path_in.glob(f"{trial_name}_Miqus_*.avi"))
    assert len(path_videos) == 14, f"Expected 14 videos, found {len(path_videos)}"
    for new_trial, (start, end) in enumerate(cut_marks):
        print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #")
        print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #")
        print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #")
        print(f"Trial {new_trial + 1} of {len(cut_marks)}: from {start} to {end}")
        print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #")
        print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #")
        print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #")
        for path_video in path_videos:
            # filename_video_out = path_video.stem.split("_") + f"_cut_{new_trial + 1}.avi"
            out = path_video.stem.split("_")
            out.insert(1, f"cut_{str(new_trial + 1).zfill(2)}")
            filename_video_out = "_".join(out) + ".avi"
            path_video_out = path_out / filename_video_out
            # cut_video(path_video, path_video_out, start, end)
            cut_video_lossless(path_video, path_video_out, start, end)


def main(heat: int):
    path_in = paths_videos[f"heat_{heat}"]
    path_out = path_in.with_name(path_in.stem + "_cut")
    path_out.mkdir(parents=True, exist_ok=True)
    if heat > 2:
        raise NotImplementedError("Only heat 1 and 2 are supported in this script.")
    heat_trials = get_heat_trials(heat)
    assert len(heat_trials) == 1, f"Expected 1 trial, found {len(heat_trials)}"
    trial_name = heat_trials[0]
    cut_marks = cut_times[f"heat_{heat}"]

    process_trial(path_in, path_out, trial_name, cut_marks)


def process_heat_3():
    # set paths, trialname and cut_marks manually
    path_in = paths_videos["heat_3"]
    path_out = path_in.with_name(path_in.stem + "_cut")
    path_out.mkdir(parents=True, exist_ok=True)
    for trial_number in range(1, 17, 1):
        trial_name = f"Running trial Markerless {trial_number}"
        print(f"Processing {trial_name}")
        cut_marks = cut_times_heat_3[trial_name]
        process_trial(path_in, path_out, trial_name, cut_marks)


def process_heat_4():
    # set paths, trialname and cut_marks manually
    path_in = paths_videos["heat_4"]
    path_out = path_in.with_name(path_in.stem + "_cut")
    path_out.mkdir(parents=True, exist_ok=True)
    for trial_number in range(1, 10, 1):
        # skip running trial Markerless 3 (no participant)
        if trial_number == 3:
            continue
        trial_name = f"Running trial Markerless {trial_number}"
        print(f"Processing {trial_name}")
        cut_marks = cut_times_heat_4[trial_name]
        process_trial(path_in, path_out, trial_name, cut_marks)


if __name__ == '__main__':
    # main(heat=1)
    # process_heat_3()
    process_heat_4()
