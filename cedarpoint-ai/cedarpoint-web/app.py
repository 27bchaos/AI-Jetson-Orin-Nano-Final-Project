from flask import Flask, Response, render_template
from ultralytics import YOLO
import cv2
import time
import threading
import numpy as np
from collections import deque

app = Flask(__name__)

# -------------------
# MODEL
# -------------------
MODEL_PATH = "models/yolov8s.pt"
model = YOLO(MODEL_PATH)

# -------------------
# STREAM
# -------------------
URL = "https://cs5.pixelcaster.com/live/cedar3.stream/chunks.m3u8"

latest_frame = None
lock = threading.Lock()

# -------------------
# MEMORY
# -------------------
people_history = deque(maxlen=10)
heatmap = None


# -------------------
# STREAM CAPTURE THREAD
# -------------------
def capture_stream():
    global latest_frame

    cap = cv2.VideoCapture(URL)

    while True:
        try:
            ret, frame = cap.read()

            if not ret or frame is None:
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(URL)
                continue

            with lock:
                latest_frame = frame

        except Exception:
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(URL)


threading.Thread(target=capture_stream, daemon=True).start()


# -------------------
# FRAME GENERATOR
# -------------------
def generate_frames():
    global latest_frame, heatmap

    frame_count = 0
    frame_skip = 5

    last_people = 0
    prev_time = time.time()

    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        frame = cv2.resize(frame, (640, 360))

        if heatmap is None:
            heatmap = np.zeros((360, 640), dtype=np.float32)

        frame_count += 1
        people = 0

        # -------------------
        # YOLO INFERENCE
        # -------------------
        if frame_count % frame_skip == 0:
            small = cv2.resize(frame, (320, 180))
            results = model(small, verbose=False)[0]

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == 0 and conf > 0.4:
                    people += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    sx, sy = 2, 2

                    x1, x2 = x1 * sx, x2 * sx
                    y1, y2 = y1 * sy, y2 * sy

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cx = min(max((x1 + x2) // 2, 0), 639)
                    cy = min(max((y1 + y2) // 2, 0), 359)

                    heatmap[cy, cx] += 1

            last_people = people
            people_history.append(last_people)

        # -------------------
        # HEATMAP DECAY
        # -------------------
        heatmap *= 0.95

        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_norm = heatmap_norm.astype(np.uint8)

        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        heatmap_color = cv2.resize(heatmap_color, (640, 360))

        frame = cv2.addWeighted(heatmap_color, 0.35, frame, 0.65, 0)

        # -------------------
        # STATUS
        # -------------------
        if last_people < 5:
            status = "LOW"
            color = (0, 255, 0)
        elif last_people < 15:
            status = "MEDIUM"
            color = (0, 255, 255)
        else:
            status = "HIGH"
            color = (0, 0, 255)

        # -------------------
        # TREND
        # -------------------
        if len(people_history) >= 5:
            if people_history[-1] > people_history[0]:
                trend = "INCREASING"
            elif people_history[-1] < people_history[0]:
                trend = "DECREASING"
            else:
                trend = "STABLE"
        else:
            trend = "STABLE"

        # -------------------
        # RISK
        # -------------------
        risk_score = min(100, last_people * 6)

       

        # -------------------
        # TEXT
        # -------------------
        cv2.putText(frame, f"People: {last_people}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, f"Crowd: {status}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, f"Risk: {risk_score}/100", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, f"Trend: {trend}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        # -------------------
        # STREAM OUTPUT
        # -------------------
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# -------------------
# ROUTES
# -------------------
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# -------------------
# RUN
# -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
