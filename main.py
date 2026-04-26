import os
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

# ---- Load model ----
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.onnx")

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # (1,3,H,W) usually
input_height, input_width = int(input_shape[2]), int(input_shape[3])

try:
    model_meta = session.get_modelmeta().custom_metadata_map
    class_names = eval(model_meta["names"])
except Exception:
    class_names = {0: "Bo_Ray", 1: "Sau_Khoang"}

print(f"Mô hình tải thành công! Size: {input_width}x{input_height}")


def preprocess(img_bgr: np.ndarray):
    h0, w0 = img_bgr.shape[:2]
    img = cv2.resize(img_bgr, (input_width, input_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, 0)        # BCHW
    return img, (w0, h0)


def postprocess(pred: np.ndarray, orig_size, conf_thres=0.25, iou_thres=0.45):
    """
    Works for YOLOv8 ONNX common output: (1, N, 4+num_classes) or (1, N, 5+num_classes)
    where boxes are xywh in resized image scale.
    """
    pred = np.squeeze(pred)

    # Make it (N, D)
    if pred.ndim == 2:
        out = pred
    else:
        # Sometimes (D, N)
        out = pred
    if out.shape[0] < out.shape[1]:
        out = out.T  # ensure rows are detections

    w0, h0 = orig_size
    sx = w0 / input_width
    sy = h0 / input_height

    boxes = []
    confidences = []
    class_ids = []

    num_cols = out.shape[1]
    # Heuristic: if there is an objectness/conf column
    has_obj = (num_cols - 4) > len(class_names)  # 5+classes

    for row in out:
        x, y, w, h = row[0:4]

        if has_obj:
            obj = float(row[4])
            cls_scores = row[5:]
            cls_id = int(np.argmax(cls_scores))
            cls_conf = float(cls_scores[cls_id])
            conf = obj * cls_conf
        else:
            cls_scores = row[4:]
            cls_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[cls_id])

        if conf < conf_thres:
            continue

        # xywh -> xyxy (on resized image)
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # scale back to original image
        x1 *= sx; x2 *= sx
        y1 *= sy; y2 *= sy

        # cv2.dnn.NMSBoxes expects [x, y, width, height]
        boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
        confidences.append(conf)
        class_ids.append(cls_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, iou_thres)

    result_counts = {"Bo_Ray": 0, "Sau_Khoang": 0}

    if len(indices) > 0:
        for i in indices.flatten():
            name = class_names.get(class_ids[i], str(class_ids[i]))

            if "Bo_Ray" in name:
                result_counts["Bo_Ray"] += 1
            elif "Sau_Khoang" in name:
                result_counts["Sau_Khoang"] += 1

    return result_counts


@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Ảnh lỗi"}

    input_tensor, orig_size = preprocess(img)
    outputs = session.run(None, {input_name: input_tensor})

    # Most YOLO ONNX exports return first output as detections
    pred = outputs[0]
    return postprocess(pred, orig_size)
