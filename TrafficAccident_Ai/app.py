from flask import Flask, render_template, jsonify
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# 글로벌 변수 선언
traffic_data = []  # 감지된 객체를 저장하는 리스트

# YOLO 모델 불러오기
def load_yolo_model():
    return YOLO('yolov8n.pt')

# Object detection function using YOLOv8
def detect_objects(frame, model):
    results = model(frame)
    detected_objects = []

    for result in results:
        for obj in result.boxes:
            class_id = int(obj.cls[0])  # Extract class ID
            detected_objects.append(class_id)

    return detected_objects

# 비디오 분석 함수
def process_video():
    global traffic_data  # 글로벌 변수로 설정
    cap = cv2.VideoCapture('static/traffic_video.mp4')  # 비디오 파일 불러오기
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    model = load_yolo_model()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detected_objects = detect_objects(frame, model)

        # 클래스별 객체 개수 세기
        car_count = 0
        bus_count = 0
        person_count = 0

        for class_id in detected_objects:
            # 클래스 ID에 따라 개수 세기 (COCO 클래스 기준)
            if class_id == 2:  # car
                car_count += 1
            elif class_id == 5:  # bus
                bus_count += 1
            elif class_id == 0:  # person
                person_count += 1

        # traffic_data에 클래스별 개수 추가
        traffic_data.append({
            'car_count': car_count,
            'bus_count': bus_count,
            'person_count': person_count
        })

        print(f"Car: {car_count}, Bus: {bus_count}, Person: {person_count}")

    cap.release()

@app.route('/data')
def data():
    if not traffic_data:
        return jsonify(message="Processing data, please wait...")  # 데이터가 비어 있을 때 메시지 반환
    return jsonify(traffic_data)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    process_video()  # 서버 시작 전에 비디오를 백그라운드에서 처리
    app.run(debug=True)

