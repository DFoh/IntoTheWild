import csv
import threading
from pathlib import Path

import cv2
import easyocr
import numpy as np
import pandas as pd
from ultralytics import YOLO

from cut_mark_determination.common import (make_empty_raw_number_dataframe,
                                           save_numbers_dataframe_to_excel)
from util import start_numbers_heat_3, paths_videos

# ssl._create_default_https_context = ssl._create_unverified_context

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICK = 2

camera_id_conf_thresh_lut = {
    '3': 0.6,
    '5': 0.75,
    '6': 0.6,
    '12': 0.6,
}


def get_human_detection_model():
    return YOLO("../yolov8n.pt")


def setup():
    model = get_human_detection_model()
    reader = easyocr.Reader(
        ['en'],
        gpu=True,
        download_enabled=True,
        model_storage_directory='./easyocr'
    )
    return model, reader


# --- Helper function to process a single frame ---
def get_person_bbs_on_frame(frame, model):
    """
    Processes a single video frame for person detection and OCR.

    Args:
        frame (np.array): The input video frame.
        model (YOLO): The loaded YOLO model for object detection.
    Returns:
        np.array: The frame with detection bounding boxes and OCR text overlaid.
    """
    processed_frame = frame.copy()
    results = model(processed_frame, verbose=False)
    res = model.track(frame, persist=True)

    return results.boxes


def hamming_distance(s1: str, s2: str) -> int:
    """Calculate the Hamming distance between two strings of equal length."""
    if len(s1) != len(s2):
        return max(len(s1), len(s2))  # or raise an exception
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))


# --- Helper function to process a single frame ---
def process_frame(frame, model, reader, frame_number, cam_id) -> tuple[np.array, pd.DataFrame | None]:
    """
    Processes a single video frame for person detection and OCR.

    Args:
        frame (np.array): The input video frame.
        model (YOLO): The loaded YOLO model for object detection.
        reader (easyocr.Reader): The loaded EasyOCR reader.

    Returns:
        np.array: The frame with detection bounding boxes and OCR text overlaid.
        pd.DataFrame: DataFrame containing detected numbers and their details.
    """
    processed_frame = frame.copy()
    results = model.predict(processed_frame,
                            classes=[0],  # only person class
                            conf=0.75,
                            verbose=False)[0]

    if len(results) == 0:
        return processed_frame, None

    min_height_human_px = 300
    min_width_human_px = 40
    min_height_number_px = 20
    min_width_number_px = 30
    min_area_px = 0  # todo: check if this is helpful or not
    center_frame_x_px = processed_frame.shape[1] // 2
    min_text_conf = 0.4

    df_numbers = make_empty_raw_number_dataframe()

    for box in results.boxes:
        if int(box.cls) == 0:  # person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            height_human_px = y2 - y1
            width_human_px = x2 - x1
            area_px = height_human_px * width_human_px
            center_x_px = (x1 + x2) // 2
            if height_human_px < min_height_human_px or width_human_px < min_width_human_px or area_px < min_area_px:
                continue
            # if center_x_px > center_frame_x_px:
            #     continue  # only process persons on the left half of the frame

            person_crop = processed_frame[y1:y2, x1:x2]
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            ocr_result = reader.readtext(person_crop, detail=1, allowlist='0123456789')
            for (bbox, text, conf) in ocr_result:
                if conf < min_text_conf:
                    continue
                ttext = text.strip()
                if not ttext:
                    continue
                if not ttext.isdigit():
                    continue
                hamming_distances = [hamming_distance(ttext, str(n)) for n in start_numbers_heat_3]
                if min(hamming_distances) > 0:
                    continue
                # print(f"Hamming distances for OCR text '{ttext}': {hamming_distances}")

                bbox = [(int(x + x1), int(y + y1)) for (x, y) in bbox]
                pt1, pt2 = bbox[0], bbox[2]
                # add 10px to each side of the bbox
                extra_pixels = 15
                pt1 = (max(0, pt1[0] - extra_pixels), max(0, pt1[1] - extra_pixels))
                pt2 = (min(processed_frame.shape[1], pt2[0] + extra_pixels),
                       min(processed_frame.shape[0], pt2[1] + extra_pixels))
                height_number_px = pt2[1] - pt1[1]
                width_number_px = pt2[0] - pt1[0]
                cv2.rectangle(processed_frame, pt1, pt2, (0, 0, 255), 2)
                cv2.putText(processed_frame, text, (pt1[0], pt1[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # save the bboxes for a quick check:
                path_out_bboxes = Path("../output_bboxes").joinpath(ttext)
                path_out_bboxes.mkdir(parents=True, exist_ok=True)
                crop_save = processed_frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                crop_filename = Path(
                    path_out_bboxes) / f"crop_{frame_number}_{cam_id}_{ttext}_{height_number_px}_{width_number_px}_conf_{conf:.2f}.png"
                cv2.imwrite(str(crop_filename), crop_save)
                # add to dataframe
                new_df = pd.DataFrame([{
                    'frame_number': frame_number,
                    'cam_id': cam_id,
                    'text': ttext,
                    'conf': conf,
                    'x1': pt1[0],
                    'y1': pt1[1],
                    'x2': pt2[0],
                    'y2': pt2[1],
                    'height_number_px': height_number_px,
                    'width_number_px': width_number_px
                }])
                df_numbers = pd.concat([df_numbers, new_df], ignore_index=True)
    # # finally add the current time to the frame
    # current_time_msec = cv2.getTickCount() / cv2.getTickFrequency() * 1000
    # current_time_sec = int(current_time_msec // 1000)
    # cv2.putText(processed_frame, f'Time: {current_time_sec}s', (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return processed_frame, df_numbers


def process_frame_ocr_only(frame, reader):
    processed_frame = frame.copy()

    # bounding box for the area of interest (official timing clock) in the frame of camera 4
    p1 = (50, 350)
    p2 = (150, 400)
    cv2.rectangle(processed_frame, p1, p2, (255, 0, 0), 2)

    # Crop the area of interest
    clock_crop = processed_frame[p1[1]:p2[1], p1[0]:p2[0]]
    # enhanced_crop = enhance_clock_crop(clock_crop)

    # cv2.imshow("Enhanced Clock Crop", enhanced_crop)

    ocr_result = reader.readtext(clock_crop, detail=1, allowlist='0123456789')

    for (bbox, text, conf) in ocr_result:
        if text.strip():
            # bbox = [(int(x), int(y)) for (x, y) in bbox]
            bbox = [(int(x + p1[0]), int(y + p1[1])) for (x, y) in bbox]
            pt1, pt2 = bbox[0], bbox[2]
            cv2.rectangle(processed_frame, pt1, pt2, (0, 0, 255), 2)
            cv2.putText(processed_frame, text, (pt1[0], pt1[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # # finally add the current time to the frame
    # current_time_msec = cv2.getTickCount() / cv2.getTickFrequency() * 1000
    # current_time_sec = int(current_time_msec // 1000)
    # cv2.putText(processed_frame, f'Time: {current_time_sec}s', (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return processed_frame, ocr_result[0][1] if ocr_result else None


def analyse_single_video(video_path: Path, show: bool = True):
    model, reader = setup()
    cap = cv2.VideoCapture(video_path.as_posix())
    cam_id = video_path.stem.split("_")[-1]
    frame_count = 0
    df_numbers_all = make_empty_raw_number_dataframe()
    while cap.isOpened():
        ret_frame = cap.read()
        if not ret_frame[0]:
            break
        frame = ret_frame[1]
        processed_frame, df_numbers_frame = process_frame(frame,
                                                          model,
                                                          reader,
                                                          frame_count,
                                                          cam_id)
        if df_numbers_frame is not None:
            df_numbers_all = pd.concat([df_numbers_all, df_numbers_frame], ignore_index=True)
        if show:
            # Display and write to output video
            title = f"Detection + OCR - Video: {video_path.stem}"
            cv2.imshow(title, processed_frame)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    print(f"Finished processing {frame_count} frames.")
    cap.release()
    cv2.destroyAllWindows()
    return df_numbers_all


def run_tracker_in_thread(model, video_path, conf, out_dir):
    """Run YOLO tracker in its own thread for concurrent processing.

    Args:
        model_name (str): The YOLO11 model object.
        filename (str): The path to the video file or the identifier for the webcam/external camera source.
    """
    # results = model.track(filename, save=True, stream=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    csv_path = Path(out_dir) / f"{stem}_tracks.csv"
    results = model.track(source=video_path,
                          save=True,
                          conf=conf,
                          stream=True,
                          iou=0.5,
                          show=False,
                          classes=[0],
                          persist=True)
    # for r in results:
    #     pass
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "track_id", "x1", "y1", "x2", "y2", "conf", "cls"])
        frame_idx = 0
        for r in results:
            b = r.boxes
            if b is not None and len(b) > 0:
                xyxy = b.xyxy.cpu().numpy()
                confs = b.conf.cpu().numpy() if b.conf is not None else np.full(len(xyxy), np.nan)
                clss = b.cls.cpu().numpy().astype(int) if b.cls is not None else np.full(len(xyxy), -1, dtype=int)
                ids = b.id.cpu().numpy().astype(int) if getattr(b, "id", None) is not None else np.full(len(xyxy), -1,
                                                                                                        dtype=int)
                for (x1, y1, x2, y2), conf_v, cls_v, tid in zip(xyxy, confs, clss, ids):
                    w.writerow(
                        [stem, frame_idx, tid, float(x1), float(y1), float(x2), float(y2), float(conf_v), int(cls_v)])
            frame_idx += 1


def multithreaded_processing(video_paths: list[Path], out_dir: str = "output_tracks"):
    # Create and start tracker threads using a for loop
    tracker_threads = []
    for video_path in video_paths:
        camera_id = video_path.stem.split('_Miqus_')[1].split('_')[0]  # 1, 2, 3... not the actual serial number
        conf = camera_id_conf_thresh_lut.get(camera_id, 0.6)
        model = get_human_detection_model()
        thread = threading.Thread(target=run_tracker_in_thread, args=(model, video_path, conf, out_dir), daemon=True)
        tracker_threads.append(thread)
        thread.start()

    # Wait for all tracker threads to finish
    for thread in tracker_threads:
        thread.join()

    # Clean up and close windows
    cv2.destroyAllWindows()


def _color_for_id(tid: int) -> tuple[int, int, int]:
    h = (tid * 2654435761) & 0xFFFFFFFF
    return (int(h & 255), int((h >> 8) & 255), int((h >> 16) & 255))


def _draw_tracks(frame, rows):
    for _, r in rows.iterrows():
        x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
        tid = int(r["track_id"]) if "track_id" in r else -1
        color = _color_for_id(tid if tid >= 0 else 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICK)
        label = f"ID {tid}" if tid >= 0 else "ID -"
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        y_text = max(0, y1 - 6)
        cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw + 6, y_text), color, -1)
        cv2.putText(frame, label, (x1 + 3, y_text - 3), FONT, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)


# ---------- single-video worker ----------
def overlay_single_video(video_path: str, tracks_csv: str, out_path: str) -> dict:
    df = pd.read_csv(tracks_csv)
    # Expect columns: video,frame,track_id,x1,y1,x2,y2,conf,cls
    # Build fast lookup: {frame_idx: DataFrame}
    groups = {int(k): v for k, v in df.groupby("frame")}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"video": video_path, "frames": 0, "written": 0, "status": "open_failed"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:  # NaN or 0
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # Prefer mp4v; use avi only if needed
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frames = 0
    written = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rows = groups.get(frame_idx, None)
        if rows is not None and len(rows) > 0:
            _draw_tracks(frame, rows)
        writer.write(frame)
        frames += 1
        written += 1
        frame_idx += 1

    writer.release()
    cap.release()
    return {"video": video_path, "frames": frames, "written": written, "status": "ok"}


def get_heat_trials(heat: int):
    path_heat_root = paths_videos[f"heat_{heat}"]
    qtm_files = list(path_heat_root.glob("*trial Markerless*.qtm"))
    # strip to only get the trial names
    trial_names = [f.stem for f in qtm_files]
    return trial_names


def get_video_path_by_trial_name_and_camera_id(heat: int, trialname: str, cam_id: str) -> Path:
    path_heat_root = paths_videos[f"heat_{heat}"]
    path_video_file = list(path_heat_root.glob(f"{trialname}_Miqus_*_{cam_id}.avi"))
    if len(path_video_file) != 1:
        raise ValueError(f"Could not find unique video file for trial {trialname} and camera id {cam_id}")
    return path_video_file[0]


def main(heat: int):
    cam_id_front_right = "26075"
    cam_id_front_left = "26071"
    trial_names = get_heat_trials(heat)

    for trial_name in trial_names:
        video_path_front_right = get_video_path_by_trial_name_and_camera_id(heat, trial_name, cam_id_front_right)
        video_path_front_left = get_video_path_by_trial_name_and_camera_id(heat, trial_name, cam_id_front_left)
        df_start_number_detection_front_right = analyse_single_video(video_path=video_path_front_right)
        df_start_number_detection_front_left = analyse_single_video(video_path=video_path_front_left)
        # merge dataframes
        df_numbers_all = pd.concat([df_start_number_detection_front_right,
                                    df_start_number_detection_front_left], ignore_index=True)
        save_numbers_dataframe_to_excel(df_numbers_all, heat=heat, trial_name=trial_name)


if __name__ == '__main__':
    main(heat=3)
