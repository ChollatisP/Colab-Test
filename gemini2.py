from ultralytics import YOLO
import pandas as pd

# Load the YOLOv8 model
model = YOLO('/home/chollatisp/Colab-Test/300epochdirtydetect.pt')  # Replace 'yolov8n.pt' with your model's path

# Perform object detection on a video
results = model('0',show=True)  # Replace 'path/to/your/video.mp4' with your video's path

# Convert results to a Pandas DataFrame
results_df = results.pandas().xyxy

# Export results to a CSV file
results_df.to_csv('detection_results.csv', index=False)

# Print the results (optional)
print(results_df)