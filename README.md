# 🎯 Age & Race Identification System

## 📋 Tổng quan

Hệ thống nhận diện khuôn mặt, phân loại độ tuổi và chủng tộc theo thời gian thực từ camera/video với kiến trúc đa luồng tối ưu. Dự án được thiết kế để đạt hiệu suất cao trên cả PC và embedded devices như Orange Pi 6 Plus.



### ✨ Tính năng chính

- 🔍 **Face Detection & Tracking**: Sử dụng YOLOv11 với thuật toán tracking IoU
- 👤 **Age Classification**: Phân loại 7 nhóm tuổi
- 🌍 **Race Classification**: Phân loại 5 nhóm chủng tộc
- ⚡ **Real-time Processing**: Đa luồng (capture, inference, display) đạt 6-8 FPS
- 📊 **FPS Logging**: Hệ thống ghi log và phân tích FPS
- 📈 **Data Analysis**: Notebook phân tích dataset, confusion matrix, và đánh giá model
- 🎯 **Smart Caching**: Cache kết quả classification theo track ID
- 🚀 **ONNX Export**: Hỗ trợ export sang ONNX để tối ưu inference

---

## 📁 Cấu trúc thư mục

```
DEVELOPING AN AGE AND RACIAL IDENTIFICATION SYSTEM/
│
├── checkpoint/                      # Model weights và ONNX files
│   ├── yolov11n-face.pt            # YOLO face detection model
│   ├── yolov11n-face.onnx          # YOLO ONNX format
│   ├── model_last.pth              # Classification model checkpoint
│   └── age_race_multihead.onnx     # Classification ONNX format
│
├── classification/                  # Module phân loại tuổi và chủng tộc
│   ├── __init__.py
│   ├── dataset.py                  # Dataset loader
│   ├── multi_head.py               # Multi-head classifier architecture
│   ├── loss.py                     # Custom loss functions
│   ├── train.py                    # Training script
│   ├── inference.py                # Inference script
│   ├── export_onxx.py              # Export to ONNX
│   ├── utkface-processing.ipynb    # Notebook phân tích data & confusion matrix
│   └── train_log.csv               # Training logs
│
├── pipline/                        # Pipeline tích hợp đầy đủ
│   ├── __init__.py
│   ├── age_race_pipeline.py        # Main pipeline (real-time)
│   ├── load_model.py               # Model loading utilities
│   ├── load_onnx_classifier.py     # ONNX classifier loader
│   ├── load_yolo_onnx.py           # ONNX YOLO loader
│   ├── write_log.py                # FPS logging utilities
│   ├── export_all_onnx.py          # Export all models to ONNX
│   └── show.py                     # Visualization script
│
├── data/                           # Dataset (nếu có)
├── runs/                           # Kết quả predict và logs yolo
├── requirements.txt                # Dependencies
├── fps_log.txt                     # FPS log file
└── README.md                       # Documentation
```

---

## 🛠️ Cài đặt

### Yêu cầu hệ thống

- **Python**: 3.12.12
- **CUDA**: 11.x+ (nếu dùng GPU)
- **RAM**: 4GB+
- **Camera**: Webcam hoặc camera USB

### Bước 1: Clone repository

```bash
git clone <repository-url>
cd "DEVELOPING AN AGE AND RACIAL IDENTIFICATION SYSTEM"
```

### Bước 2: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Các thư viện chính:**
- `ultralytics`: YOLOv11-face framework
- `torch`, `torchvision`: Deep learning framework
- `opencv-python`: Computer vision
- `onnxruntime`: ONNX inference
- `numpy`, `tqdm`: Utilities

### Bước 3: Tải hoặc chuẩn bị models

Đặt các file model vào thư mục `checkpoint/`:
- `yolov11n-face.pt`: YOLO face detection model - download from github
- `model_last.pth`: Age & Race classification model

---

## 🚀 Sử dụng

### 1️⃣ Chạy Pipeline Real-time

Chạy hệ thống nhận diện real-time từ webcam:

```bash
python pipline/age_race_pipeline.py
```

**Điều khiển:**
- Nhấn `q` để thoát
- Hệ thống hiển thị:
  - E2E Latency (ms)
  - E2E FPS
  - Track ID, Age, Race cho mỗi khuôn mặt

### 2️⃣ Chạy với FPS Logging

**Ghi log FPS trong 5 phút (300 giây):**

```bash
python pipline/age_race_pipeline.py log
```

**Ghi log FPS với thời gian tùy chỉnh (ví dụ 120 giây):**

```bash
python pipline/age_race_pipeline.py log 120
```

**Đọc và phân tích file log:**

```bash
python pipline/age_race_pipeline.py read
```

**Hoặc bật tự động trong code:**

Sửa trong [age_race_pipeline.py](pipline/age_race_pipeline.py):
```python
ENABLE_FPS_LOG = True
FPS_LOG_DURATION = 300  # 5 phút
```

### 3️⃣ Data Analysis & Evaluation

**Notebook phân tích dữ liệu:**

Mở notebook để phân tích chi tiết dataset UTKFace và đánh giá model:

```bash
jupyter notebook classification/utkface-processing.ipynb
```

**Nội dung notebook:**

1. **📊 Data Processing & EDA:**
   - Load và parse dataset UTKFace (23,000+ ảnh)
   - Phân tích phân phối age và race
   - Phân tầng train/val với stratified split theo combo (age_group + race)
   - Visualize phân phối data với bar charts và heatmaps

2. **🎯 Dataset Balancing:**
   - Stratified split đảm bảo tỷ lệ age/race cân bằng giữa train và validation
   - So sánh phân phối train vs validation
   - Heatmap phân bố Age Group × Race

3. **🏗️ Model Architecture:**
   - Multi-Head Classifier với CBAM Attention
   - Backbone: MobileNetV2 / MobileNetV3 / ResNet50
   - Dual heads: Age (7 classes) + Race (5 classes)

4. **📉 Loss Functions:**
   - Cross-entropy loss với weighted combination
   - Focal loss để xử lý class imbalance
   - Class weights cho age và race

5. **📈 Confusion Matrix Analysis:**
   - Confusion matrix cho Age classification
   - Confusion matrix cho Race classification
   - Normalized confusion matrices (phần trăm)
   - Phân tích lỗi phân loại chi tiết

**Kết quả đánh giá:**
- **Age Accuracy**: ~60-65%
- **Race Accuracy**: ~80-85%
- **Mean Accuracy**: ~70-75%

**Visualizations:**
- Age group distribution (bar chart)
- Race distribution (bar chart)
- Age × Race heatmap
- Train vs Validation comparison
- Confusion matrices (raw + normalized)

### 4️⃣ Training Model Classification

**Chuẩn bị dataset:**
- Đặt dữ liệu vào thư mục `data/`
- Cấu trúc dataset theo format trong `classification/dataset.py`

**Train model:**

```bash
cd classification
python train.py
```

**Các tham số có thể điều chỉnh trong `train.py`:**
- `num_epochs`: Số epoch
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `num_age_classes`: Số lớp tuổi (mặc định 7)
- `num_race_classes`: Số lớp chủng tộc (mặc định 5)

**Kết quả:**
- Model checkpoint: `checkpoint/model_last.pth`
- Training log: `classification/train_log.csv`

### 5️⃣ Inference với Classification Model

Chạy inference trên ảnh đơn:

```bash
cd classification
python inference.py --image <path_to_image>
```

### 6️⃣ Export sang ONNX

**Export classification model:**

```bash
cd classification
python export_onxx.py
```

**Export tất cả models:**

```bash
cd pipline
python export_all_onnx.py
```

**Kết quả:**
- `checkpoint/age_race_multihead.onnx`
- `checkpoint/yolov11n-face.onnx`

### 7️⃣ Chạy với ONNX Models

```bash
cd pipline
python load_onnx_classifier.py  # Test ONNX classifier
python load_yolo_onnx.py        # Test ONNX YOLO
```

---

## 📊 Kiến trúc hệ thống

### Multi-threading Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   CAPTURE   │────▶│  INFERENCE  │────▶│   DISPLAY   │
│   THREAD    │     │   THREAD    │     │   THREAD    │
└─────────────┘     └─────────────┘     └─────────────┘
      │                    │                    │
      ▼                    ▼                    ▼
  latest_frame        latest_draw          cv2.imshow
  (shared state)    (shared state)       (FPS logging)
```

**1. Capture Thread:**
- Liên tục capture frame từ camera
- Lưu vào `latest_frame` với timestamp

**2. Inference Thread:**
- Đọc frame mới nhất
- Face detection + tracking (IoU)
- Classification (với caching)
- Vẽ bounding boxes và labels
- Lưu vào `latest_draw`

**3. Display Thread:**
- Hiển thị frame đã xử lý
- Tính toán FPS và latency
- Ghi log FPS (nếu bật)
- Xử lý input từ người dùng

### Model Architecture

**YOLOv11-Face:**
- Backbone: CSPDarknet
- Neck: PANet
- Head: Detection head
- Input: 640x640 RGB
- Output: Bounding boxes + confidence

**Multi-Head Classifier:**
- Backbone: MobileNetV2 (pretrained)
- Head 1: Age classification (7 classes)
- Head 2: Race classification (5 classes)
- Input: 224x224 RGB
- Output: Age logits + Race logits

---

## ⚙️ Configuration

### Trong `age_race_pipeline.py`:

```python
# Camera settings
CAMERA_ID = 0
FRAME_W, FRAME_H = 640, 360

# Classification settings
MAX_CLASSIFY = 3              # Số face tối đa để classify
CLS_REFRESH_SEC = 2.0         # Thời gian cache (giây)
FPS_HIST_LEN = 30             # Độ dài history cho FPS

# Tracking settings (IoUTracker)
iou_thres = 0.4               # IoU threshold
max_lost_sec = 0.6            # Max time before drop track

# FPS Logging
ENABLE_FPS_LOG = False        # Bật/tắt FPS logging
FPS_LOG_DURATION = 300        # Thời gian log (giây)
```

---

## 📈 Performance

### Benchmark trên PC (GPU RTX 3060)

- **Resolution**: 640x360
- **Average FPS**: 6-8 FPS
- **E2E Latency**: 150-200ms
- **GPU Usage**: ~30%
- **CPU Usage**: ~25%

### Tối ưu cho Embedded Devices

**Orange Pi 6 Plus:**
- Giảm resolution: 480x270
- Sử dụng ONNX models
- Enable NPU acceleration
- Reduce MAX_CLASSIFY to 1-2

---

## 🐛 Troubleshooting

### Lỗi import module

**Lỗi:** `ModuleNotFoundError: No module named 'pipline'`

**Giải pháp:** Thêm parent directory vào sys.path trong các file pipeline:

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### Lỗi camera

**Lỗi:** `Cannot open camera`

**Giải pháp:**
- Kiểm tra CAMERA_ID (thử 0, 1, 2...)
- Thử bỏ `cv2.CAP_DSHOW` flag
- Kiểm tra quyền truy cập camera

### Lỗi CUDA

**Lỗi:** `CUDA out of memory`

**Giải pháp:**
- Giảm FRAME_W, FRAME_H
- Giảm MAX_CLASSIFY
- Chuyển sang CPU mode: `device = "cpu"`

### FPS thấp

**Giải pháp:**
- Giảm resolution
- Tăng CLS_REFRESH_SEC (giảm frequency classification)
- Giảm MAX_CLASSIFY
- Sử dụng ONNX models
- Enable GPU acceleration

---

## 🔬 Age & Race Classes

### Age Classes (7 nhóm)

| Class | Age Range |
|-------|-----------|
| 0     | 0-2       |
| 1     | 3-9       |
| 2     | 10-19     |
| 3     | 20-29     |
| 4     | 30-39     |
| 5     | 40-69     |
| 6     | 70+       |

### Race Classes (5 nhóm)

| Class | Race          |
|-------|---------------|
| 0     | White         |
| 1     | Black         |
| 2     | Asian         |
| 3     | Indian        |
| 4     | Others        |

---

## 📝 FPS Log Format

File `fps_log.txt` chứa các giá trị FPS, mỗi dòng một giá trị:

```
6.596062150091212
6.076358968861279
7.5127693492629275
...
```

**Phân tích log:**

```bash
python pipline/age_race_pipeline.py read
```

**Output:**
```
==================================================
FPS Log Analysis: fps_log.txt
==================================================
Total samples: 49
Average FPS: 6.12
Min FPS: 0.26
Max FPS: 9.39
Median FPS: 6.35
==================================================
```

---

## 🚧 TODO

- [ ] Thêm gender classification
- [ ] Hỗ trợ video file input
- [ ] Web interface (Flask/FastAPI)
- [ ] Model quantization (INT8)
- [ ] Multi-camera support
- [ ] Database integration
- [ ] REST API
- [ ] Docker deployment

---

## 📚 References

- [YOLOv11 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.

---

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [github.com/yourusername]
- **Issues**: [GitHub Issues](https://github.com/yourusername/repo/issues)

---

## 🙏 Acknowledgments

- YOLOv11 team at Ultralytics
- PyTorch community
- OpenCV contributors
- Pre-trained model sources

---

**⭐ Nếu dự án hữu ích, hãy cho một star trên GitHub! ⭐**
