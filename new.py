import cv2
import numpy as np
from ultralytics import YOLO  # Import YOLOv8 from ultralytics
from deep_sort_realtime.deepsort_tracker import DeepSort  # Import DeepSort

# Load the YOLOv8 model
model = YOLO('/home/chollatisp/Colab-Test/300epochdirtydetect.pt')  # Path to your best.pt file

# Define the video capture
cap = cv2.VideoCapture(0)  # 0 for default camera, or provide a file path

# Define the output video codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Define the output video file name
out_path = 'output.mp4'

# Get the frame properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create the output video writer
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

# Initialize DeepSort tracker
tracker = DeepSort()

# Process the video frame by frame
while cap.isOpened():
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (YOLOv8 expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run inference on the frame
    results = model(source=frame_rgb)

    # Extract bounding boxes, class IDs, and confidence scores
    # detections = results.xyxy[0]  # Convert results to a pandas DataFrame
    # bboxes = detections[['xmin', 'ymin', 'xmax', 'ymax']].values
    # confs = detections['confidence'].values
    # class_ids = detections['class'].values

    # # Format detections for DeepSort
    # detections_deepsort = []
    # for bbox, conf in zip(bboxes, confs):
    #     detections_deepsort.append((bbox.tolist(), conf))

    # Run multi-object tracking using DeepSort
    #outputs = tracker.update_tracks(detections_deepsort, frame)

    # Display the results
    # for track in outputs:
    #     bbox = track[:4]
    #     track_id = int(track[4])
    #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
    #     cv2.putText(frame, f'id: {track_id}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Write the frame to the output video
    out.write(frame)
    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer
cap.release()
out.release()
# Close all windows
cv2.destroyAllWindows()
