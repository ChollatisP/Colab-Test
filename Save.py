from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("/home/chollatisp/Colab-Test/300epochdirtydetect.pt")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None  # Initialize video writer to None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform detection on the current frame
    results = model.predict(source=frame, conf=0.4)  # Perform detection on the current frame

    # Extract the results for the current frame
    detections = results[0]  # Assuming results is a list and the first element contains detections

    # Loop through each detected object (if any) and print details
    if detections.xyxy[0].shape[0] > 0:  # Check if any objects were detected
        if out is None:
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            print("Recording started!")

        for result in detections.xyxy[0]:  # Loop through each detected object
            class_name = detections.names[int(result.cls)]
            confidence = result.conf
            x_min, y_min, x_max, y_max = result.xyxy[0].astype(int)

            # Print object details
            print(f"Detected object: {class_name} (Confidence: {confidence:.2f})")
            print(f"Bounding Box: ({x_min}, {y_min}) - ({x_max}, {y_max})")
    else:
        print("No objects detected.")
        if out is not None:
            out.release()
            print("Recording stopped!")
            out = None

    if out is not None:
        out.write(frame)

    # Display the frame (optional, for debugging)
    cv2.imshow('Frame', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
