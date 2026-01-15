import csv
import ssl
import threading
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

import easyocr

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
    return YOLO("yolov8n.pt")


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


# --- Helper function to process a single frame ---
def process_frame(frame, model, reader):
    """
    Processes a single video frame for person detection and OCR.

    Args:
        frame (np.array): The input video frame.
        model (YOLO): The loaded YOLO model for object detection.
        reader (easyocr.Reader): The loaded EasyOCR reader.

    Returns:
        np.array: The frame with detection bounding boxes and OCR text overlaid.
    """
    processed_frame = frame.copy()
    results = model(processed_frame, verbose=False)[0]

    for box in results.boxes:
        if int(box.cls) == 0:  # person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_crop = processed_frame[y1:y2, x1:x2]
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            ocr_result = reader.readtext(person_crop, detail=1, allowlist='0123456789')
            for (bbox, text, conf) in ocr_result:
                if text.strip():
                    bbox = [(int(x + x1), int(y + y1)) for (x, y) in bbox]
                    pt1, pt2 = bbox[0], bbox[2]
                    cv2.rectangle(processed_frame, pt1, pt2, (0, 0, 255), 2)
                    cv2.putText(processed_frame, text, (pt1[0], pt1[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # # finally add the current time to the frame
    # current_time_msec = cv2.getTickCount() / cv2.getTickFrequency() * 1000
    # current_time_sec = int(current_time_msec // 1000)
    # cv2.putText(processed_frame, f'Time: {current_time_sec}s', (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return processed_frame


def enhance_clock_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (no thresholding)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Light sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened


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


def analyse_single_video(video_path: str,
                         model: any,
                         reader: any,
                         start_time_sec: int = 0,
                         analysis_duration_sec: int = -1):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)
    # --- Get Video Properties for Output ---
    # Assuming all videos have the same resolution for simplicity in 2x2 grid
    # If resolutions vary significantly, you'll need to resize each frame to a common size
    # before arranging. For this example, we'll use the first video's dimensions as reference.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ref_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ref_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    output_video_path = "output_analysis.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use 'MJPG' or 'mp4v' for other codecs
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (ref_frame_width, ref_frame_height))

    # Calculate the end time in milliseconds
    if analysis_duration_sec < 0:
        end_time_msec = (start_time_sec + analysis_duration_sec) * 1000
    else:
        end_time_msec = 10E12  # Set a very large end time if analysis_duration_sec is negative (i.e., no limit)

    print(f"Starting analysis from {start_time_sec}s for {analysis_duration_sec}s.")
    print(f"Output video will be saved to: {output_video_path}")

    frame_count = 0
    texts = []
    while cap.isOpened():
        ret_frame = cap.read()
        if not ret_frame[0]:
            break
        frame = ret_frame[1]

        # HUMAN DETECTOR:
        processed_frame = process_frame(frame, model, reader)
        # NUMBER DETECTOR:
        # processed_frame, text = process_frame_ocr_only(frame, reader)
        # texts.append(text if text else "")

        # break if the current time exceeds the end time
        current_time_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if current_time_msec > end_time_msec:
            break

        # Display and write to output video
        cv2.imshow("Detection + OCR", processed_frame)
        out.write(processed_frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Finished processing {frame_count} frames.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Texts extracted from the video:")
    for i, text in enumerate(texts):
        print(f"Frame {i}: {text}")


def analyse_multiple_videos(videos: list, start_time_sec, analysis_duration_sec, model, reader):
    # --- Initialize Video Captures ---
    caps = []
    for video_path in videos:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)
        caps.append(cap)

    # --- Get Video Properties for Output ---
    # Assuming all videos have the same resolution for simplicity in 2x2 grid
    # If resolutions vary significantly, you'll need to resize each frame to a common size
    # before arranging. For this example, we'll use the first video's dimensions as reference.
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))
    ref_frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    ref_frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine the dimensions for the 2x2 output video
    output_height = ref_frame_height * 2
    output_width = ref_frame_width * 2

    # Define the codec and create VideoWriter object
    output_video_path = "output_2x2_analysis.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use 'MJPG' or 'mp4v' for other codecs
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

    # Calculate the end time in milliseconds
    end_time_msec = (start_time_sec + analysis_duration_sec) * 1000

    print(f"Starting analysis from {start_time_sec}s for {analysis_duration_sec}s.")
    print(f"Output video will be saved to: {output_video_path}")

    frame_count = 0
    while all(cap.isOpened() for cap in caps):
        rets_frames = [(cap.read()) for cap in caps]
        rets = [r for r, _ in rets_frames]
        frames = [f for _, f in rets_frames]

        # Break if any video ends or we've passed the end time
        current_time_msec = caps[0].get(cv2.CAP_PROP_POS_MSEC)
        if not all(rets) or current_time_msec > end_time_msec:
            break

        processed_frames = []
        for i, frame in enumerate(frames):
            # Resize frame to reference dimensions if they are different (optional but good practice)
            if frame.shape[0] != ref_frame_height or frame.shape[1] != ref_frame_width:
                frame = cv2.resize(frame, (ref_frame_width, ref_frame_height))
            processed_frames.append(process_frame(frame, model, reader))

        # Arrange frames in a 2x2 grid
        top_row = np.hstack((processed_frames[0], processed_frames[1]))
        bottom_row = np.hstack((processed_frames[2], processed_frames[3]))
        combined_frame = np.vstack((top_row, bottom_row))

        # Display and write to output video
        cv2.imshow("2x2 Combined Detection + OCR", combined_frame)
        out.write(combined_frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Finished processing {frame_count} frames.")
    for cap in caps:
        cap.release()
    out.release()
    cv2.destroyAllWindows()


def human_detection_in_video(video_path: str, model: any, conf: float):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bbs_list = []

    results = model.track(source=video_path,
                          conf=0.6,
                          iou=0.5,
                          show=True, classes=[0], persist=True)
    #
    # if not cap.isOpened():
    #     print(f"Error opening video file: {video_path}")
    #     return
    #
    # while cap.isOpened():
    #     print(f"Processing frame {len(bbs_list)+1}/{total_frames}", end='\r')
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #
    #     bbs = get_person_bbs_on_frame(frame, model)
    #     bbs_list.append(bbs)
    #
    # foo = 1
    # for bb in bbs_list:
    #     print(bb)
    # cap.release()
    # cv2.destroyAllWindows()


def sequential_processing(model, reader, video_paths: list):
    # Process each video from the list sequentially,
    # get the human bounding boxes,
    # post-process the bounding boxes (filter, gap fill, etc.),
    # make overlay on video,
    # and save the output video.
    for video_path in video_paths:
        camera_id = video_path.split('_Miqus_')[1].split('_')[0]
        conf = camera_id_conf_thresh_lut.get(camera_id, 0.6)
        results = model.track(source=video_path,
                              conf=conf,
                              stream=True,
                              iou=0.5,
                              show=True,
                              classes=[0],
                              persist=True)
        for r in results:
            pass
        print(type(results))
        # pickle the results
        # with open(f'results_camera_{camera_id}.pkl', 'wb') as f:
        #     pickle.dump(results, f)

        # bbs = human_detection_in_video(video_path, model, conf=conf)
        foo = 1
    pass


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


# ---------- multithreaded driver ----------
def make_overlays_parallel(video_paths: list[str],
                           tracks_dir: str = "output_tracks",
                           out_dir: str = "overlays",
                           workers: int | None = None) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for vp in video_paths:
            stem = Path(vp).stem
            csv_path = Path(tracks_dir) / f"{stem}_tracks.csv"
            if not csv_path.exists():
                results.append({"video": vp, "frames": 0, "written": 0, "status": "tracks_missing"})
                continue
            out_path = Path(out_dir) / f"{stem}_overlay.mp4"
            futs.append(ex.submit(overlay_single_video, str(vp), str(csv_path), str(out_path)))
        for f in as_completed(futs):
            results.append(f.result())
    return results


if __name__ == '__main__':
    # model, reader = setup()

    # video_path = r"data/video/25769.avi"
    # video_path = "data/Running trial Markerless 4_Miqus_6_25769.avi"

    # video_path = "data/Running trial Markerless 4_Miqus_12_26075.avi"
    # start_time_sec = 213  # <== set your desired start time here
    start_time_sec = 12  # <== set your desired start time here
    # start_time_sec = 280  # <== set your desired start time here
    analysis_duration_sec = 2  # <== Analyze duration

    # human_detection_in_video(video_path=video_path,
    #                          model=model,
    #                          reader=reader)

    # video_path_2 = "data/Running trial Markerless 4_Miqus_3_26071.avi"
    # video_path_3 = "data/Running trial Markerless 4_Miqus_5_26153.avi"
    # video_path_1 = "data/Running trial Markerless 4_Miqus_12_26075.avi"
    # video_path_4 = "data/Running trial Markerless 4_Miqus_13_26078.avi"
    #
    # video_paths = [video_path_1, video_path_2, video_path_3, video_path_4]
    # analyse_multiple_videos(video_paths, start_time_sec, analysis_duration_sec, model, reader)

    # analyse_single_video(video_path=video_path_1,
    #                      model=model,
    #                      reader=reader,
    #                      start_time_sec=14,
    #                      analysis_duration_sec=analysis_duration_sec)

    video_paths = list(Path("data").glob("Running trial*.avi"))

    # video_paths = [video_path_1]
    # sequential_processing(model,
    #                       reader,
    #                       video_paths=video_paths)

    # multithreaded_processing(video_paths=video_paths)

    summary = make_overlays_parallel(video_paths, out_dir="overlays", workers=4)
    for s in summary:
        print(s)