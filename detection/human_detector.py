import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from classes.camera import get_camera_information_from_cal_file
from plotting.plot_3d import plot_3d_space, add_camera_calibration, add_camera_ray, plot_point_in_3d
from three_d_math.ray_geometry import get_ray_intersections_three_d


class HumanDetectionResults:
    def __init__(self):
        self.detections = []


def detect_humans(src: str,
                  min_height=100,  # Minimum height of the bounding box to consider
                  min_width=50,  # Minimum width of the bounding box to consider
                  min_area=500  # Minimum area of the bounding box to consider
                  ):
    # Load YOLOv8 model for person detection
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(src)

    colors = np.random.randint(0, 255, size=(80, 3), dtype=int).tolist()  # Random colors for each bounding box

    out = dict()
    path_out = Path(src).parent / "results"
    file_out = path_out / f"{Path(src).stem}.json"

    frame_cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        result = model(frame, verbose=False)[0]
        frame_list = list()

        # Draw bounding boxes for class 'person' (cls=0)
        for b, box in enumerate(result.boxes):
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h = y2 - y1
                if h < min_height:
                    continue
                w = x2 - x1
                if w < min_width:
                    continue
                area = w * h
                if area < min_area:
                    continue
                frame_list.append([x1, y1, x2, y2])
                color = colors[b]
                text = f"W:{w} H:{h} A:{area}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)
        out[frame_cnt] = frame_list
        frame_cnt += 1

        # Display the result
        cv2.imshow("Human Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    # Save the results to a file
    with open(file_out, 'w') as f:
        json.dump(out, f)


def three_d_track(path_camera_calibration_file: Path, path_detection_results: Path):
    """
    Placeholder for 3D tracking logic.
    This function should implement the logic to track humans in 3D space.
    """
    cal_info = get_camera_information_from_cal_file(path_camera_calibration_file)
    # build plot
    fig = plot_3d_space()
    add_camera_calibration(fig, path_camera_calibration_file)
    rays = []
    point_cluster_list = []
    for result_file in path_detection_results.glob("*.json"):
        camera_serial = result_file.stem
        try:
            cam_cal = [cam for cam in cal_info if cam.serial == camera_serial][0]
        except Exception as e:
            print(f"Camera calibration for {camera_serial} not found: {e}")
            continue
        data = json.load(open(result_file, 'r'))

        frame_zero = data.get(str(0))
        for box in frame_zero:
            x1, y1, x2, y2 = map(int, box)
            center_point = ((x1 + x2) / 2, (y1 + y2) / 2)
            ray = cam_cal.pixel_to_camera_ray(
                x_pixel=center_point[0],
                y_pixel=center_point[1],
                ray_length=15000
            )
            rays.append(ray)
            # add_camera_ray(fig=fig, start_point=ray[0], end_point=ray[1])

    point_clusters = get_ray_intersections_three_d(rays,
                                                   min_ray_count=5,
                                                   min_ray_length=300,
                                                   max_ray_length=10000)
    for i, cluster in enumerate(point_clusters):
        print(f"cluster {i}: point {cluster['point']}, rays {cluster['rays']}")
        rays_in_cluster = [rays[i] for i in cluster['rays']]
        p = cluster['point']
        for ray_in in rays_in_cluster:
            add_camera_ray(fig=fig, start_point=ray_in[0], end_point=p)
        plot_point_in_3d(fig=fig, point=p, color='red', name=f'Cluster {i}')
    fig.show()


def detect(videos_path: Path):
    for video_path in videos_path.glob("*.avi"):
        detect_humans(src=video_path)


def track(results_path: Path, calibration_path: Path):
    three_d_track(path_camera_calibration_file=calibration_path,
                  path_detection_results=results_path)


if __name__ == '__main__':
    # Step 1: Detect humans in video files
    video_path_base = Path("../data/video")
    # detect(videos_path=video_path_base)

    # Step 2: Track humans in 3D space
    path_calib = Path("../data/cal.txt")
    path_detection_results = Path("../data/video/results")
    track(results_path=path_detection_results,
          calibration_path=path_calib)
