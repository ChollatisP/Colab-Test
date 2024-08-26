import time
from ultralytics import YOLO

# Load the model
model = YOLO("/home/chollatisp/Colab-Test/300epochdirtydetect.pt")

# Add a delay before starting the prediction
while True:

# Perform prediction on webcam (replace "0" with image/video path if needed)
    results = model.predict(source="0", show=True, conf=0.3)

# Add a delay after prediction if needed

# Loop through each detected object (if any) and print details
    if results.xyxy:  # Check if any objects were detected
        for result in results.xyxy[0]:  # Loop through each detected object
            # Access object information (class, confidence, bounding box coordinates)
            class_name = result.names[int(result.cls)]
            confidence = result.conf
            x_min, y_min, x_max, y_max = result.xyxy[0].astype(int)
            
            # Print object details
            print(f"Detected object: {class_name} (Confidence: {confidence:.2f})")
            print(f"Bounding Box: ({x_min}, {y_min}) - ({x_max}, {y_max})")
    else:
        print("No objects detected.")
    
    time.sleep(10)  # Delay in seconds
