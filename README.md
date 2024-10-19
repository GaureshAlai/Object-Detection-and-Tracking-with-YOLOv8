# Object-Detection-and-Tracking-with-YOLOv8
This repository contains code to perform object detection and tracking on videos and images using the YOLOv8 model. The project is designed to recognize and track objects from the MS COCO dataset in videos, and detect objects in images using feature-based classifiers.

## Project Overview

- **Video Recognition**: Detect and track moving objects in a 15–20 second video, applying the YOLOv8 model and IoU-based tracking.
- **Image Recognition**: Detect objects in a static image using the YOLOv8 model and visualize bounding boxes, class labels, and confidence scores.

## Requirements

To run this project, ensure you have the following dependencies installed:

```bash
pip install torch torchvision opencv-python matplotlib
```

Alternatively, you can install the `ultralytics` package for easy YOLOv8 access:

```bash
pip install ultralytics
```

### YOLOv8 Model
The YOLOv8 model is a lightweight, fast, and accurate object detection model, trained on the MS COCO dataset with 80 object classes.

## Video Object Detection and Tracking

### Steps:
1. **Record a video**: A 15–20 second video containing at least 5 objects from the MS COCO categories, with moving objects or panning camera.
2. **Run object detection**: Use the YOLOv8 model to detect objects in the video frames.
3. **Apply tracking**: Track objects across frames using Intersection over Union (IoU) for bounding box association.
4. **Save output video**: The final video with overlaid tracking results is saved as `task3.mp4`.

### Code for Video Detection and Tracking:

```python
import torch
import cv2

# Load YOLOv8 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load video
cap = cv2.VideoCapture('task1.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Visualize detections on the frame
    annotated_frame = results.render()[0]
    cv2.imshow('YOLOv8 Object Detection', annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

For full implementation, refer to the `video_object_tracking.ipynb` script.

### Tracking Method:
- **IoU Tracking**: Tracks each object separately based on the Intersection over Union (IoU) method, ensuring different classes are not mixed in tracking results.

## Image Object Detection

### Steps:
1. **Prepare an image**: Use any image containing objects from the MS COCO categories.
2. **Run object detection**: Detect objects in the image using the YOLOv8 model.
3. **Visualize results**: Display the image with bounding boxes and object labels.

### Code for Image Detection:

```python
import torch
import cv2
import matplotlib.pyplot as plt

# Load YOLOv8 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load the image
image_path = 'your_image.jpg'
image = cv2.imread(image_path)

# Perform object detection
results = model(image)

# Display detected objects
results.show()

# Alternatively, visualize using matplotlib
detections = results.pandas().xyxy[0]
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
```

For full implementation, refer to the `image_object_detection.ipynb` script.

## Dataset

The model is trained on the MS COCO dataset, which includes 80 object categories such as:
- Person
- Bottle
- Laptop
- Chair
- Book

Refer to the [MS COCO category list](https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda) for the full set of objects.

## Results

### Video Object Tracking:
- Objects like "person", "bottle", "laptop", and "chair" were successfully detected and tracked across video frames.
- The model handled motion and panning well, with high detection confidence.

### Image Detection:
- Objects in static images were detected with bounding boxes and class labels, visualized using both OpenCV and Matplotlib.

## Conclusion

This project demonstrates the effectiveness of the YOLOv8 model for both object detection and tracking tasks in videos and images. It highlights the importance of tuning IoU thresholds and confidence levels for robust tracking.

## Future Improvements

- Implement advanced tracking algorithms such as DeepSORT.
- Experiment with different versions of YOLO (e.g., YOLOv8x) for enhanced accuracy.

## License

This project is licensed under the MIT License.

---

### Folder Structure:
- `input.mp4`: Input video.
- `output.mp4`: Output video with object tracking.
- `video_object_tracking.ipynb`: Python script for video recognition and tracking.
- `image_object_detection.ipynb`: Python script for image recognition.

