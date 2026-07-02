# Live Crowd AI Detection

A real-time AI-powered video analytics system that detects people in a live video stream using YOLOv8, draws bounding boxes, analyzes crowd levels, and streams the processed output through a Flask web server.

## Image Demo

![Crowd Detection Demo](https://github.com/user-attachments/assets/3e8ec75e-ce0f-4edf-9be1-71345f19e355)

## The Algorithm

This project uses Ultralytics YOLOv8 to perform real-time object detection on a live video stream. The system focuses on detecting people and builds analytics on top of those detections.

### How it works:

1. **Stream Capture**
   - A background thread continuously reads frames from an HLS video stream using OpenCV.
   - The latest frame is stored and shared with the processing loop.

2. **YOLO Object Detection**
   - Every few frames (frame skipping for performance), the frame is resized and passed into the YOLOv8 model.
   - The model detects objects and returns bounding boxes, confidence scores, and class IDs.
   - Only detections for the "person" class are used.

3. **Crowd Counting**
   - Each detected person is counted per frame.
   - A short history buffer tracks crowd changes over time.

4. **Analytics**
   - Crowd status:
     - LOW (< 5 people)
     - MEDIUM (5–14 people)
     - HIGH (15+ people)
   - Trend detection compares recent frames to detect increase/decrease in crowd size.
   - A simple risk score is calculated based on crowd size.

5. **Live Output**
   - Frames are encoded as JPEG.
   - Streamed to a Flask endpoint using multipart streaming.

## Running this project

### 1. Install dependencies

Make sure you are in the project root folder, then run:

```bash
cd cedarpoint-ai/docs
pip install -r requirements.txt
```

### 2. Run the project

Before running, make sure you are inside the correct project folder:

```bash
cd cedarpoint-web
python app.py
```

### 3. Open in browser

Open:

```
http://localhost:5000
```

Or:

```
http://YOUR_DEVICE_IP:5000
```


### 4. Video Explanation
[View a video explanation here](https://github.com/user-attachments/assets/2eabd1b6-fdbf-4f18-b6b7-2888bdc7ecae)



