from flask import Flask, render_template, jsonify, url_for
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Global variable to store traffic data
traffic_data = []

# Load YOLO model
def load_yolo_model():
    model = YOLO('yolov8n.pt')  # Use the appropriate YOLO model
    return model

# Process video to detect objects and traffic light colors
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    model = load_yolo_model()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect traffic light color and objects (replace with actual detection logic)
        traffic_light_color = detect_traffic_light_color(frame)
        detected_objects = detect_objects(frame, model)

        # Append to traffic_data (you can adjust this data structure)
        traffic_data.append({
            'traffic_light_color': traffic_light_color,
            'detected_objects': [obj[1] for obj in detected_objects]
        })

    cap.release()

# Mock detection functions (replace with actual detection logic)
def detect_traffic_light_color(frame):
    return 'red'  # Simplified for example

def detect_objects(frame, model):
    return [('car', 'car'), ('bus', 'bus'), ('person', 'person')]  # Simplified for example

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    return jsonify(traffic_data)

if __name__ == '__main__':
    video_file = 'static/traffic_video.mp4'  # Ensure video is in the static folder
    process_video(video_file)  # Pre-process video (for testing)
    app.run(debug=True)

