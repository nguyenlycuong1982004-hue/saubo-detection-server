import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import io

app = FastAPI()

# 1. Load mô hình trực tiếp từ file .pt (hoặc .onnx đều được)
# Bạn nên để file best.pt vào cùng thư mục
model = YOLO("best.pt") 

@app.get("/")
async def root():
    return {"status": "Hệ thống chuẩn đang chạy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Đọc ảnh từ ESP32
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Chạy dự đoán với cấu hình y hệt phần mềm trong ảnh
    # conf=0.5 (Chế độ CAO như trong ảnh bạn chọn)
    results = model.predict(source=img, conf=0.5, save=False)
    
    bo_ray_count = 0
    sau_khoang_count = 0
    
    # 3. Duyệt qua kết quả
    result = results[0]
    for box in result.boxes:
        class_id = int(box.cls[0])
        # Lấy tên class từ mô hình để đảm bảo không bao giờ nhầm ID
        label = model.names[class_id] 
        
        if label == "Bo_Ray": # Phải viết hoa/thường y hệt lúc train
            bo_ray_count += 1
        elif label == "Sau_Khoang":
            sau_khoang_count += 1

    return {
        "Bo_Ray": bo_ray_count,
        "Sau_Khoang": sau_khoang_count,
        "Total": bo_ray_count + sau_khoang_count
    }