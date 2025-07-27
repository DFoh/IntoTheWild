import ssl

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

ssl._create_default_https_context = ssl._create_unverified_context


def setup():
    model = YOLO("yolov8n.pt")
    reader = easyocr.Reader(
        ['en'],
        gpu=True,
        download_enabled=True,
        model_storage_directory='./easyocr'
    )
    return model, reader


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
        # processed_frame = process_frame(frame, model, reader)
        processed_frame, text = process_frame_ocr_only(frame, reader)
        texts.append(text if text else "")

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


def human_detection_in_video(video_path: str, model: any, reader: any):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame, model, reader)
        cv2.imshow("Detection + OCR", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model, reader = setup()

    video_path = r"data/video/25769.avi"
    # video_path = "data/Running trial Markerless 4_Miqus_6_25769.avi"

    # video_path = "data/Running trial Markerless 4_Miqus_12_26075.avi"
    # start_time_sec = 213  # <== set your desired start time here
    start_time_sec = 240  # <== set your desired start time here
    # start_time_sec = 280  # <== set your desired start time here
    # analysis_duration_sec = 2  # <== Analyze duration

    human_detection_in_video(video_path=video_path,
                             model=model,
                             reader=reader)

    # analyse_single_video(video_path=video_path,
    #                      model=model,
    #                      reader=reader,
    #                      start_time_sec=0,
    #                      analysis_duration_sec=analysis_duration_sec)

    # video_path_1 = "data/Running trial Markerless 4_Miqus_12_26075.avi"
    # video_path_2 = "data/Running trial Markerless 4_Miqus_3_26071.avi"
    # video_path_3 = "data/Running trial Markerless 4_Miqus_5_26153.avi"
    # video_path_4 = "data/Running trial Markerless 4_Miqus_13_26078.avi"
    #
    # videos = [video_path_1, video_path_2, video_path_3, video_path_4]
    # analyse_multiple_videos(videos, start_time_sec, analysis_duration_sec, model, reader)
