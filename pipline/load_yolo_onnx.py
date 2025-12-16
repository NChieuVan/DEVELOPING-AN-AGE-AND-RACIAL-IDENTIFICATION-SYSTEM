def nms_numpy(boxes, scores, iou_threshold=0.45):
    # boxes: [N, 4], scores: [N]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

import onnxruntime as ort
import numpy as np
import cv2

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # Ultralytics letterbox
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r  # width, height ratios
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, dw, dh

def load_yolo_onnx(onnx_path, conf_thres=0.01):
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def detect_fn(frame):
        img, r, dw, dh = letterbox(frame, (640, 640), auto=False, scaleFill=False, scaleup=True)
        img = img[:, :, ::-1]  # BGR -> RGB
        img_in = img.astype(np.float32) / 255.0
        img_in = np.transpose(img_in, (2, 0, 1))
        img_in = np.expand_dims(img_in, 0)
        outputs = session.run([output_name], {input_name: img_in})[0]

        # outputs có thể là (1, 5, 8400) hoặc (1, 8400, 5) tùy model/export
        pred = outputs[0]
        if pred.shape[0] == 5:            # (5, 8400) -> (8400, 5)
            pred = pred.T
        elif pred.shape[-1] == 5:         # (8400, 5) ok
            pass
        else:
            raise ValueError(f"Unexpected output shape: {outputs.shape}")

        bboxes_raw, scores_raw = [], []
        h0, w0 = frame.shape[:2]

        for det in pred:
            x, y, w, h, conf = det[:5]
            if conf < conf_thres:
                continue

            # Nếu model trả normalized [0..1] thì scale lên 640
            # (có model xuất raw pixel, có model xuất normalized; check nhanh)
            if max(x, y, w, h) <= 2.0:
                x *= 640.0; y *= 640.0; w *= 640.0; h *= 640.0

            # xywh (center) -> xyxy trên ảnh letterbox 640x640
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

            # Bỏ padding + scale ngược về ảnh gốc
            x1 = (x1 - dw) / r
            y1 = (y1 - dh) / r
            x2 = (x2 - dw) / r
            y2 = (y2 - dh) / r

            # clip theo kích thước ảnh gốc
            x1 = np.clip(x1, 0, w0 - 1)
            y1 = np.clip(y1, 0, h0 - 1)
            x2 = np.clip(x2, 0, w0 - 1)
            y2 = np.clip(y2, 0, h0 - 1)

            bboxes_raw.append([x1, y1, x2, y2])
            scores_raw.append(float(conf))

        if len(bboxes_raw) == 0:
            return [], []
        bboxes_raw = np.array(bboxes_raw)
        scores_raw = np.array(scores_raw)
        keep = nms_numpy(bboxes_raw, scores_raw, iou_threshold=0.45)
        bboxes = bboxes_raw[keep].astype(int).tolist()
        scores = scores_raw[keep].tolist()
        return bboxes, scores
    return detect_fn

