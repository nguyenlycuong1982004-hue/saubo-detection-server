import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

# 1. Tải mô hình ONNX
try:
    session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
    model_inputs = session.get_inputs()
    input_name = model_inputs[0].name
    input_shape = model_inputs[0].shape 
    input_height, input_width = input_shape[2], input_shape[3]
    
    # Lấy danh sách tên Class từ metadata của file ONNX
    # Nếu file onnx của bạn không có metadata, ta sẽ dùng mặc định 0: Bo_Ray, 1: Sau_Khoang
    try:
        model_meta = session.get_modelmeta().custom_metadata_map
        class_names = eval(model_meta['names'])
    except:
        class_names = {0: 'Bo_Ray', 1: 'Sau_Khoang'}
        
    print(f"Mô hình tải thành công! Size: {input_width}x{input_height}")
except Exception as e:
    print(f"Lỗi tải mô hình: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return {"error": "Ảnh lỗi"}
    
    # Tiền xử lý
    img_resized = cv2.resize(img, (input_width, input_height))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    input_tensor = np.transpose(img_normalized, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Chạy mô hình
    outputs = session.run(None, {input_name: input_tensor})
    output = np.squeeze(outputs[0]) 
    if output.shape[0] < output.shape[1]: output = output.T

    boxes, confidences, class_ids = [], [], []
    for row in output:
        scores = row[4:] 
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            confidences.append(float(confidence))
            class_ids.append(int(class_id))
            boxes.append([0, 0, 10, 10]) 

    # Lọc trùng lặp bằng NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.45)

    result_counts = {"Bo_Ray": 0, "Sau_Khoang": 0}

    if len(indices) > 0:
        for i in indices.flatten():
            label_idx = class_ids[i]
            label_name = class_names[label_idx]
            
            # Kiểm tra tên class và cộng dồn
            if "Bo_Ray" in label_name:
                result_counts["Bo_Ray"] += 1
            elif "Sau_Khoang" in label_name:
                result_counts["Sau_Khoang"] += 1

    return result_counts
