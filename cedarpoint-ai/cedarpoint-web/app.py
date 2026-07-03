from flask import Flask, Response, render_template
from ultralytics import YOLO
import cv2
import time

app = Flask(__name__)

# ======================
# CONFIG
# ======================

MODEL_PATH = "models/yolov8n.pt"  # FIXED: relative path (portable)

STREAM_URL = "https://cs5.pixelcaster.com/live/cedar3.stream/chunks.m3u8"

# Load YOLO model safely
model = YOLO(MODEL_PATH)

# Open video stream
cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

# ======================
# FRAME
# ======================

def generate_frames():
    frame_count = 0
    frame_skip = 2

    last_people = 0
    prev_time = time.time()

    while True:
        success, frame = cap.read()

        # FIX: prevent crash if stream drops
        if not success or frame is None:
            time.sleep(0.1)
            continue

        frame_count += 1

        # Resize for speed
        frame = cv2.resize(frame, (640, 360))

        people = 0

        # Run YOLO every few frames 
        if frame_count % frame_skip == 0:
            results = model(frame, verbose=False)[0]

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == 0 and conf > 0.4:  # person class
                    people += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            last_people = people

        # ======================
        # CROWD STATUS 
        # ======================

        if last_people < 5:
            status = "LOW"
            color = (0, 255, 0)
            risk = 20
        elif last_people < 15:
            status = "MEDIUM"
            color = (0, 255, 255)
            risk = 60
        else:
            status = "HIGH"
            color = (0, 0, 255)
            risk = 90

        

        # ======================
        # OVERLAY TEXT
        # ======================

        cv2.putText(frame, f"People: {last_people}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, f"Status: {status}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, f"Risk: {risk}/100", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

      
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ======================
# ROUTES
# ======================

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ======================
# MAIN
# ======================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
