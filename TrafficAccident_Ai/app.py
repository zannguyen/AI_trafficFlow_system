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

# video,model 불러오기
video_path = os.path.join('static', 'traffic_video.mp4')
model = YOLO('yolov8n.pt')

latest_detections = []
detection_running = False
object_counts = Counter()
lane_occupancy = {f'lane{i}': 'Clear' for i in range(1, 5)}  

def detect_objects():
    global latest_detections, detection_running, object_counts, lane_occupancy
    cap = cv2.VideoCapture(video_path)
    
    while detection_running:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Object detection using YOLO v.8
        results = model(frame)
        
        detections = []
        frame_objects = Counter()
        
        #class 분류
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                class_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                class_name = model.names[class_id]
                detections.append({
                    'class': class_name,
                    'confidence': round(conf, 2),
                    'box': [int(x1), int(y1), int(x2), int(y2)]
                })
                frame_objects[class_name] += 1
        
        # Update global detections and counts
        latest_detections = detections
        object_counts.update(frame_objects)
        
        # 차선별 혼잡도 분석
        for i in range(1, 5):
            lane_objects = sum([1 for d in detections if int(d['box'][0]) // 100 == i]) 
            if lane_objects > 8:  
                lane_occupancy[f'lane{i}'] = 'Congested'
            else:
                lane_occupancy[f'lane{i}'] = 'Clear'
        
        # Emit the data to the frontend
        socketio.emit('detection_update', {
            'detections': detections,
            'object_counts': dict(object_counts),
            'lane_occupancy': lane_occupancy
        })
        
        time.sleep(0.1)

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('start_detection')
def handle_start_detection():
    global detection_running
    if not detection_running:
        detection_running = True
        threading.Thread(target=detect_objects, daemon=True).start()

@socketio.on('pause_detection')
def handle_pause_detection():
    global detection_running
    detection_running = False

if __name__ == '__main__':
    socketio.run(app, debug=True)

