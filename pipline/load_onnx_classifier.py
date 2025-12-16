import onnxruntime as ort
import numpy as np
import cv2

def load_onnx_classifier(onnx_path):
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    age_out = session.get_outputs()[0].name
    race_out = session.get_outputs()[1].name

    def classify_fn(face_img):
        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)  # Đảm bảo đúng kiểu float32 cho ONNX
        age_logits, race_logits = session.run([age_out, race_out], {input_name: img})
        age_probs = softmax(age_logits[0])
        race_probs = softmax(race_logits[0])
        age_idx = int(np.argmax(age_probs))
        race_idx = int(np.argmax(race_probs))
        age_conf = float(age_probs[age_idx])
        race_conf = float(race_probs[race_idx])
        return age_idx, age_conf, race_idx, race_conf
    return classify_fn

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)
