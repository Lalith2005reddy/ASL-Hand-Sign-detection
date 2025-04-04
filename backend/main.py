from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model
model = YOLO("yolov11_model.pt")

# Define your custom class names
custom_class_names = {
    0: "Hello",
    1: "Thank You",
    2: "Yes",
    3: "No",
    4: "Please"
}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)
    detections = []

    for box in results[0].boxes:
        label_idx = int(box.cls[0])
        label = custom_class_names.get(label_idx, f"class_{label_idx}")
        conf = float(box.conf[0])
        bbox = box.xyxy[0].tolist()

        detections.append({
            "class": label,              # 🔁 Changed from 'label' to 'class'
            "confidence": round(conf, 2),
            "bbox": bbox
        })

    return {"detections": detections}
