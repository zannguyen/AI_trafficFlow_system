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
        
        # objects class별로 분류
        for r in results:
            boxes = r.boxes #감지된 결과 가져오기
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0] #box 모서리 좌표 가져오기
                class_id = int(box.cls[0].item()) #class id 가져오기
                conf = box.conf[0].item() #신뢰도 가져오기 
                
                if conf < 0.7:  # 신뢰도 설정
                    continue
                
                class_name = model.names[class_id]
                detections.append({
                    'class': class_name,
                    'confidence': round(conf, 2),
                    'box': [int(x1), int(y1), int(x2), int(y2)]
                })
                frame_objects[class_name] += 1 # objects count!
        
        # 현재 프레임에서 감지된 객체수 저장
        object_counts = dict(frame_objects)  # JS와 호환위해 JSON data 변환
        

        # 전체 frame을 4등분하여 분석
        frame_width = frame.shape[1]  
        lane_width = frame_width // 4 
        
        # Initialize variables to calculate average object area
        total_area = 0
        num_objects = 0

        #  lane object counts 정의
        lane_objects = {f'lane{i}': 0 for i in range(1, 5)}  # 1에서 4까지의 차선 초기화

        # Count objects in each lane and calculate their areas
        for detection in detections:
            if detection['class'] == 'lane':  # Only consider lane detections
                # Calculate lane index based on x-coordinate
                center_x = (detection['box'][0] + detection['box'][2]) / 2
                lane_index = int(center_x // lane_width) + 1  # +1 for lane numbering (1 to 4)
                
                if lane_index in lane_objects:
                    lane_objects[f'lane{lane_index}'] += 1
                    
                    # Calculate the area of the current object
                    x1, y1, x2, y2 = detection['box']
                    object_area = (x2 - x1) * (y2 - y1)
                    total_area += object_area
                    num_objects += 1

        # Calculate average object area if there are detected objects
        average_object_area = total_area / num_objects if num_objects > 0 else 0

        # Update lane occupancy based on object counts as a percentage of lane capacity
        for i in range(1, 5):
            # Calculate the capacity of each lane as the number of pixels in that lane
            lane_capacity = lane_width * frame.shape[0]  # Assuming the height represents capacity
            occupied_area = lane_objects[f'lane{i}'] * average_object_area  # You need to define average_object_area
            occupancy_percentage = (occupied_area / lane_capacity) * 100
            lane_occupancy[f'lane{i}'] = 'Congested' if occupancy_percentage > 80 else 'Clear'
        
        
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

