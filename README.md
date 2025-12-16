# Age & Race Identification System

## 1. Mô tả dự án
- Hệ thống nhận diện khuôn mặt, phân loại độ tuổi và chủng tộc từ camera/video.
- Tối ưu đa luồng, hỗ trợ Orange Pi 6 Plus.

## 2. Cấu trúc thư mục
```
DEVELOPING AN AGE AND RACIAL IDENTIFICATION SYSTEM/
├── checkpoint/         # Model weights (YOLO, classification)
├── classification/     # Code train/inference classification
├── pipline/            # Pipeline tích hợp detection + classification
├── data/               # Dữ liệu gốc (nếu có)
├── runs/               # Kết quả predict, log
├── requirements.txt    # Thư viện cần thiết
├── README.md           # Hướng dẫn
└── inference_yolo_face.py # Test YOLO riêng
```

## 3. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

## 4. Chuẩn bị model
- Đặt các file sau vào thư mục `checkpoint/`:
  - `yolov11n-face.pt` (YOLO face detection)
  - `model_last.pth` (weight classification)

## 5. Train model classification
```bash
cd classification
python train.py
```
- Kết quả sẽ lưu vào `checkpoint/model_last.pth`

## 6. Chạy pipeline nhận diện (camera/webcam)
```bash
cd pipline
python age_race_pipeline.py
```
- Nhấn `q` để thoát.

## 7. Test YOLO detection riêng
```bash
python inference_yolo_face.py
```

## 8. Lưu ý
- Đảm bảo đúng version Python, các package đã cài đủ.
- Nếu chạy trên Orange Pi, nên tối ưu thêm cho NPU/GPU.

## 9. Liên hệ & bản quyền
- Tác giả: [Điền tên bạn]
- Mọi thắc mắc vui lòng liên hệ qua email hoặc github.
