import cv2
import numpy as np
from pipline.load_yolo_onnx import load_yolo_onnx

# Đường dẫn model ONNX
onnx_path = "checkpoint/yolov11n-face.onnx"

# Load model ONNX
face_detector = load_yolo_onnx(onnx_path)

# Đọc ảnh test (bạn có thể thay bằng ảnh của bạn)
img = cv2.imread("test.jpg")  # Đảm bảo có file test.jpg trong thư mục làm việc
if img is None:
    print("Không tìm thấy file test.jpg!")
    exit(1)

# In thông tin ảnh gốc
print(f"Ảnh gốc shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")

# Detect
bboxes, scores = face_detector(img)
print(f"Số bbox detect được: {len(bboxes)}")
for i, (bbox, score) in enumerate(zip(bboxes, scores)):
    print(f"BBox {i}: {bbox}, score: {score}")
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

cv2.imshow("ONNX Detection Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
