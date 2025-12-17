
from ultralytics import YOLO
import cv2
import time

# load pre-trained yolov8n-face model
model = YOLO("checkpoint/yolov11n-face.pt")

# open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

prev_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # YOLO tracking (track by default uses ByteTrack/BOTSORT)
    results = model.track(source=frame, persist=True, conf=0.7, verbose=False, tracker="botsort.yaml")
    # draw boxes and IDs
    for result in results:
        boxes = result.boxes
        ids = boxes.id.cpu().numpy().astype(int) if (hasattr(boxes, 'id') and boxes.id is not None) else None
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{ids[i]} {box.conf[0]:.2f}" if ids is not None else f"{box.conf[0]:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    # FPS calculation
    frame_count += 1
    current_time = time.time()
    elapsed = current_time - prev_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        print(f"FPS: {fps:.2f}")
        prev_time = current_time
        frame_count = 0
    # show frame
    cv2.imshow("YOLO Face Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
