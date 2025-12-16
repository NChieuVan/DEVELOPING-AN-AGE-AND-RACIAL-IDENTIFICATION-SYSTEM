import cv2
from pipline.load_yolo_onnx import load_yolo_onnx

# Đường dẫn model ONNX
onnx_path = "checkpoint/yolov11n-face.onnx"

# Load model ONNX
face_detector = load_yolo_onnx(onnx_path)

# Mở camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được camera!")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được frame từ camera!")
        break
    bboxes, scores = face_detector(frame)
    for bbox, score in zip(bboxes, scores):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imshow("ONNX Camera Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
