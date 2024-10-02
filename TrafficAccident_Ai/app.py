from flask import Flask, render_template, jsonify, url_for
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# 글로벌변수 선언
traffic_data = []

# YOLO모델 불러오기
def load_yolo_model():
    model = YOLO('yolov8n.pt')  
    return model

# detected object하고 traffic light 과정
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    model = load_yolo_model()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        traffic_light_color = detect_traffic_light_color(frame)
        detected_objects = detect_objects(frame, model)

        # traffic data 추가하기
        traffic_data.append({
            'traffic_light_color': traffic_light_color,
            'detected_objects': [obj[1] for obj in detected_objects]
        })

    cap.release()

def detect_traffic_light_color(frame):
    return 'red' 

def detect_objects(frame, model):
    return [('car', 'car'), ('bus', 'bus'), ('person', 'person')]

@app.route('/data')
def data():
    print(traffic_data)  # 서버에서 전송되는 데이터를 확인
    return jsonify(traffic_data)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    video_file = 'static/traffic_video.mp4' 
    process_video(video_file)  
    app.run(debug=True)

