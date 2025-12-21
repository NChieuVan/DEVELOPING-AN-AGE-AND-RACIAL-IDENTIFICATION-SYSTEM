import torch
import torch.nn.functional as F
from ultralytics import YOLO
import os
import sys

# Add parent directory to path to import classification module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification.multi_head import MultiHeadClassifier
import torchvision.transforms as T

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

def load_tracking_model():
	"""
	Docstring for load_tracking_model
	- Returns a function that performs tracking on input frames.
	- The returned function takes a BGR numpy array as input and returns a list of tracks.
	Each track is represented as a dictionary with keys: 'bbox':list, 'id':int, 'conf':float.
	
	"""
	model = YOLO("checkpoint/yolov11n-face.pt")
	model.to(device)
	def track_fn(frame):
		# frame: BGR numpy array
		results = model.track(source=frame, persist=True, conf=0.7, verbose=False, tracker="botsort.yaml")
		tracks = []
		for result in results:
			boxes = result.boxes
			ids = boxes.id.cpu().numpy().astype(int) if (hasattr(boxes, 'id') and boxes.id is not None) else None
			for i, box in enumerate(boxes):
				x1, y1, x2, y2 = map(int, box.xyxy[0])
				track_id = ids[i] if ids is not None else None
				tracks.append({'bbox': [x1, y1, x2, y2], 'id': track_id, 'conf': float(box.conf[0])})
		return tracks
	return track_fn

def load_age_race_classifier():
	"""
	Docstring for load_age_race_classifier
	- Returns a function that performs age and race classification on input face images.
	- The returned function takes a BGR numpy array of a face image as input and returns:
	  age_pred:int, age_conf:float, race_pred:int, race_conf:float.
	- if multiple face in frame, returns:
		list of age_pred, list of age_conf, list of race_pred, list of race_conf
	- eg:
		age_preds, age_confs, race_preds, race_confs = classify_fn(face_img)
	 
	"""
	model = MultiHeadClassifier(num_age_classes=7, num_race_classes=5, backbone_name="mobilenet_v2")
	checkpoint = torch.load("checkpoint/model_last.pth", map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.to(device)
	model.eval()

	transfrom = T.Compose([
		T.ToPILImage(),
		T.Resize((224, 224)),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	def classify_fn(face_img):
		# face_img: BGR numpy array
		img = transfrom(face_img)  # img is now a torch.Tensor
		img_tensor = img.unsqueeze(0).to(device)  # Add batch dimension and move to device
		img_tensor = img_tensor.float()  # Ensure float32
		with torch.no_grad():
			age_logits, race_logits = model(img_tensor)
		age_probs = torch.softmax(age_logits, dim=1)
		race_probs = torch.softmax(race_logits, dim=1)
		age_pred = torch.argmax(age_probs, dim=1).item()
		race_pred = torch.argmax(race_probs, dim=1).item()
		return age_pred, float(age_probs[0, age_pred].item()), race_pred, float(race_probs[0, race_pred].item())
	return classify_fn



