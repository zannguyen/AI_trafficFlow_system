from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import cv2
from ultralytics import YOLO
import threading
import time
import os
from collections import Counter

app = Flask(__name__)
socketio = SocketIO(app)

# Load video and model
video_path = os.path.join('static', 'traffic_video.mp4')
model = YOLO('yolov8n.pt')

latest_detections = []
detection_running = False
lane_occupancy = {f'lane{i}': 'Clear' for i in range(1, 5)}

# Function to run object detection
def detect_objects():
    global latest_detections, detection_running, lane_occupancy
    cap = cv2.VideoCapture(video_path)
    while detection_running:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video if needed
            continue
        
        # Object detection using YOLO v8
        results = model(frame)
        detections = []
        frame_objects = Counter()
        
        # Classify detected objects with a confidence threshold
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                class_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                if conf < 0.5:  # Confidence threshold, adjust as necessary
                    continue
                
                class_name = model.names[class_id]
                detections.append({
                    'class': class_name,
                    'confidence': round(conf, 2),
                    'box': [int(x1), int(y1), int(x2), int(y2)]
                })
                frame_objects[class_name] += 1
        
        # Update object counts for the current frame
        object_counts = dict(frame_objects)  # Convert to regular dict for JSON serialization
        
        # Analyze lane congestion with better lane boundaries
        frame_width = frame.shape[1]  # Get frame width for lane calculations
        lane_width = frame_width // 4  # Divide frame into 4 lanes
        
        for i in range(1, 5):
            lane_start = (i - 1) * lane_width
            lane_end = i * lane_width
            lane_objects = sum([1 for d in detections if lane_start <= (d['box'][0] + d['box'][2]) / 2 < lane_end])
            lane_occupancy[f'lane{i}'] = 'Congested' if lane_objects > 3 else 'Clear'
        
        # Emit the data to the frontend
        socketio.emit('detection_update', {
            'detections': detections,
            'object_counts': object_counts,
            'lane_occupancy': lane_occupancy
        })
        
        time.sleep(0.05)  # Reduce sleep for lower latency, adjust based on FPS
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_detection')
def handle_start_detection():
    global detection_running
    if not detection_running:
        detection_running = True
        detection_thread = threading.Thread(target=detect_objects, daemon=True)
        detection_thread.start()

@socketio.on('pause_detection')
def handle_pause_detection():
    global detection_running
    detection_running = False

if __name__ == '__main__':
    socketio.run(app, debug=True)

