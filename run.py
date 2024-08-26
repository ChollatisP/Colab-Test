from ultralytics import YOLO

# Load the YOLO model
model = YOLO("/home/chollatisp/Colab-Test/300epochdirtydetect.pt")

# Predict
results = model.predict(source="0", show=True, conf=0.3)