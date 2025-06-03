import cv2
import easyocr
from ultralytics import YOLO

# --- SETUP --- #
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'], gpu=True)
video_path = "data/Running trial Markerless 4_Miqus_12_26075.avi"
video_path_2 = "data/Running trial Markerless 4_Miqus_3_26071.avi"
start_time_sec = 215  # <== set your desired start time here

cap = cv2.VideoCapture(video_path)
car_2 = cv2.VideoCapture(video_path_2)
cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)
car_2.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    for box in results.boxes:
        if int(box.cls) == 0:  # person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_crop = frame[y1:y2, x1:x2]
            # Draw box + number
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # OCR to detect numbers
            ocr_result = reader.readtext(person_crop, detail=1, allowlist='0123456789')

            for (bbox, text, conf) in ocr_result:
                if text.strip():
                    # Convert cropped coordinates back to full frame
                    bbox = [(int(x + x1), int(y + y1)) for (x, y) in bbox]
                    pt1, pt2 = bbox[0], bbox[2]  # top-left and bottom-right
                    cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
                    cv2.putText(frame, text, (pt1[0], pt1[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Detection + OCR", frame)
    # cv2.waitKey(1)
    # cv2.namedWindow("Detection + OCR", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Detection + OCR", 1280, 720)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
