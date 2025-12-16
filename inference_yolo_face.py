from ultralytics import YOLO
import cv2
import time

# load pre-trained yolov8n-face model
model = YOLO("checkpoint/yolov11n-face.pt")

model.predict(source="0", conf=0.7, show=True)
# # open camera once (outside loop)
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()

# prev_time = time.time()
# frame_count = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:q
#         break
#     # detect faces
#     results = model.predict(source=frame, conf=0.7, verbose=False)
#     # draw boxes on frame
#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"{box.conf[0]:.2f}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
#     # FPS calculation
#     frame_count += 1
#     current_time = time.time()
#     elapsed = current_time - prev_time
#     if elapsed >= 1.0:
#         fps = frame_count / elapsed
#         print(f"FPS: {fps:.2f}")
#         prev_time = current_time
#         frame_count = 0
#     # show frame
#     cv2.imshow("Face Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
