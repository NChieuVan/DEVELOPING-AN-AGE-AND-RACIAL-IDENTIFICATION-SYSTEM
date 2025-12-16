import torch
import torch.nn.functional as F
from ultralytics import YOLO
from classification.multi_head import MultiHeadClassifier
from classification.inference import predict
import cv2
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_face_detector():
	model = YOLO("checkpoint/yolov11n-face.pt")
	model.to(device)
	def detect_fn(frame):
		# frame: BGR numpy array
		results = model.predict(source=frame, conf=0.7, verbose=False)
		bboxes = []
		scores = []
		for result in results:
			for box in result.boxes:
				x1, y1, x2, y2 = map(int, box.xyxy[0])
				bboxes.append([x1, y1, x2, y2])
				scores.append(float(box.conf[0]))
		return bboxes, scores
	return detect_fn

def load_age_race_classifier():
	model = MultiHeadClassifier(num_age_classes=7, num_race_classes=5, backbone_name="mobilenet_v2")
	checkpoint = torch.load("checkpoint/model_last.pth", map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.to(device)
	model.eval()
	def classify_fn(face_img):
		img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (224, 224))
		img = img.astype(np.float32) / 255.0
		img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
		img = np.transpose(img, (2, 0, 1))
		img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
		img_tensor = img_tensor.float()  # Đảm bảo đúng kiểu float32
		with torch.no_grad():
			age_logits, race_logits = model(img_tensor)
			age_probs = F.softmax(age_logits, dim=1)
			race_probs = F.softmax(race_logits, dim=1)
			age_idx = torch.argmax(age_probs, dim=1).item()
			race_idx = torch.argmax(race_probs, dim=1).item()
			age_conf = age_probs[0, age_idx].item()
			race_conf = race_probs[0, race_idx].item()
		return age_idx, age_conf, race_idx, race_conf
	return classify_fn



