from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from util import start_numbers, get_video_path_by_trial_name_and_camera_id


def get_cut_mark(heat: int) -> pd.DataFrame:
    path_cut_mark = Path("../cut_mark_determination/cut_marks")
    return pd.read_excel(path_cut_mark.joinpath(f"cut_marks_heat_{heat}.xlsx"))


def get_still_image_at_frame(video_path: Path, frame_number: int):
    cap = cv2.VideoCapture(video_path.as_posix())
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame


def main(heat: int):
    cam_id_front_right = "26075"
    cam_id_front_left = "26071"
    cam_id_rear_left = "26079"
    cam_id_rear_right = "26080"
    start_numbers_heat = start_numbers[f"heat_{heat}"]
    df_cut_marks = get_cut_mark(heat)
    #
    path_img_out = Path(f"cut_mark_determination/cut_marks/heat_{heat}_cut_mark_images")
    path_img_out.mkdir(parents=True, exist_ok=True)
    for t, trial in df_cut_marks.iterrows():
        print(f"{t}/{len(df_cut_marks)} Heat {heat} Trial {trial.trial_name} Start number {trial.start_number}")
        start_numer = trial.start_number
        trial_name = trial.trial_name

        video_path_front_right = get_video_path_by_trial_name_and_camera_id(heat, trial_name, cam_id_front_right)
        video_path_front_left = get_video_path_by_trial_name_and_camera_id(heat, trial_name, cam_id_front_left)
        video_path_rear_left = get_video_path_by_trial_name_and_camera_id(heat, trial_name, cam_id_rear_left)
        video_path_rear_right = get_video_path_by_trial_name_and_camera_id(heat, trial_name, cam_id_rear_right)

        img_first_frame_left = get_still_image_at_frame(video_path=video_path_rear_left,
                                                        frame_number=trial.first_frame -12)
        img_first_frame_right = get_still_image_at_frame(video_path=video_path_rear_right,
                                                         frame_number=trial.first_frame -12)

        img_final_frame_left = get_still_image_at_frame(video_path=video_path_front_left,
                                                        frame_number=trial.final_frame -12)
        img_final_frame_right = get_still_image_at_frame(video_path=video_path_front_right,
                                                         frame_number=trial.final_frame -12)

        img_last_frame_left = get_still_image_at_frame(video_path=video_path_front_left,
                                                       frame_number=trial.last_frame -12)
        img_last_frame_right = get_still_image_at_frame(video_path=video_path_front_right,
                                                        frame_number=trial.last_frame -12)

        plt.close('all')
        fig, ax = plt.subplots(2, 3, figsize=(32, 12))
        ax[0, 0].imshow(cv2.cvtColor(img_first_frame_left, cv2.COLOR_BGR2RGB))
        ax[0, 0].set_title(f"First frame {trial.first_frame} Rear Left")
        ax[0, 1].imshow(cv2.cvtColor(img_final_frame_left, cv2.COLOR_BGR2RGB))
        ax[0, 1].set_title(f"Final frame {trial.final_frame} Front Left")
        ax[0, 2].imshow(cv2.cvtColor(img_last_frame_left, cv2.COLOR_BGR2RGB))
        ax[0, 2].set_title(f"Last frame {trial.last_frame} Front Left")
        ax[1, 0].imshow(cv2.cvtColor(img_first_frame_right, cv2.COLOR_BGR2RGB))
        ax[1, 0].set_title(f"First frame {trial.first_frame} Rear Right")
        ax[1, 1].imshow(cv2.cvtColor(img_final_frame_right, cv2.COLOR_BGR2RGB))
        ax[1, 1].set_title(f"Final frame {trial.final_frame} Front Right")
        ax[1, 2].imshow(cv2.cvtColor(img_last_frame_right, cv2.COLOR_BGR2RGB))
        ax[1, 2].set_title(f"Last frame {trial.last_frame} Front Right")
        plt.suptitle(f"Heat {heat} Trial {trial_name} Start number {start_numer}")
        plt.tight_layout()



        filename = f"heat_{heat}_trial_{trial_name.replace(' ', '_')}_start_number_{start_numer}.png"
        path_out = path_img_out / str(start_numer) / filename
        path_out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_out.as_posix())
        plt.close(fig)


if __name__ == '__main__':
    main(heat=3)
