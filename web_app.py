from flask import Flask, request, render_template, send_from_directory, jsonify
from ultralytics import YOLO
import os
import yaml
from PIL import Image
from shutil import copyfile
import requests
import base64
import json
from datetime import datetime

app = Flask(__name__)
model = YOLO(r'C:\Users\admin\Downloads\archive\archive\IP102_YOLOv5\runs\train_20250531_175746\train\weights\best.pt')

# Đọc tên class từ ip102.yaml
with open('ip102.yaml', 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
names = data['names']

# Cấu hình ESP32 IP
ESP32_IP = "172.16.71.209"

# Thông tin Telegram
bot_token = "7692052009:AAG1IpUwhnpVLYvUTaIaSP7vWGUEgEW5aaA"
chat_id = "7048917112"

# Hàm gửi thông báo về Telegram
TELEGRAM_API_URL = f"https://api.telegram.org/bot{bot_token}"
def send_telegram_message(text, image_path=None):
    try:
        # Gửi text
        resp = requests.post(f"{TELEGRAM_API_URL}/sendMessage", data={
            'chat_id': chat_id,
            'text': text
        })
        print("Send text resp:", resp.text)
        # Gửi ảnh nếu có
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as img:
                resp_img = requests.post(f"{TELEGRAM_API_URL}/sendPhoto", data={'chat_id': chat_id}, files={'photo': img})
                print("Send photo resp:", resp_img.text)
    except Exception as e:
        print(f"[Telegram] Lỗi gửi tin nhắn: {e}")

# Hàm lưu lịch sử dự đoán
HISTORY_FILE = 'history.json'
def save_history(entry):
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
        history.append(entry)
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[History] Lỗi lưu lịch sử: {e}")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/predict_esp32', methods=['POST'])
def predict_esp32():
    try:
        esp32_url = f"http://{ESP32_IP}"
        response = requests.get(esp32_url.rstrip('/') + '/capture', timeout=5)
        if response.status_code == 200:
            filename = 'esp32_capture.jpg'
            img_path = os.path.join('uploads', filename)
            os.makedirs('uploads', exist_ok=True)
            with open(img_path, 'wb') as f:
                f.write(response.content)
            
            # Nhận diện
            results = model(img_path, conf=0.15)
            details = []
            for box in results[0].boxes:
                class_id = int(box.cls.item())
                conf = float(box.conf.item())
                details.append({
                    'name': names[class_id],
                    'confidence': f'{conf:.2f}',
                    'bbox': [round(x, 2) for x in box.xyxy[0].tolist()]
                })
            
            # Lưu ảnh kết quả
            result_filename = os.path.splitext(filename)[0] + '_result.jpg'
            result_img_path = os.path.join('static', result_filename)
            results[0].save(result_img_path)
            result_img = '/static/' + result_filename
            
            # Chuyển ảnh gốc sang base64
            image_b64 = base64.b64encode(response.content).decode('utf-8')

            # Lưu lịch sử và gửi Telegram
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            history_entry = {
                'time': now,
                'type': 'esp32',
                'image': filename,
                'result_img': result_img,
                'details': details
            }
            save_history(history_entry)
            # Gửi Telegram
            msg = f"[ESP32-CAM]\nThời gian: {now}\nKết quả: {details if details else 'Không phát hiện côn trùng!'}"
            send_telegram_message(msg, image_path=result_img_path)
            
            return jsonify({
                'success': True,
                'result_img': result_img,
                'details': details,
                'original_image': image_b64
            })
        else:
            return jsonify({'success': False, 'error': f'Không thể lấy ảnh từ ESP32-CAM (HTTP {response.status_code})'})
    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'error': 'Kết nối đến ESP32-CAM bị timeout'})
    except requests.exceptions.ConnectionError:
        return jsonify({'success': False, 'error': 'Không thể kết nối đến ESP32-CAM'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Lỗi: {str(e)}'})

@app.route('/', methods=['GET', 'POST'])
def index():
    result_img = None
    details = None
    ground_truth = None
    original_img = None
    # Đọc lịch sử dự đoán
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
    except Exception as e:
        history = []
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part', esp32_ip=ESP32_IP, history=history)
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file', esp32_ip=ESP32_IP, history=history)
        if file:
            img_path = os.path.join('uploads', file.filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(img_path)
            # Đảm bảo thư mục static tồn tại trước khi copy
            static_dir = 'static'
            os.makedirs(static_dir, exist_ok=True)
            static_img_path = os.path.join(static_dir, os.path.basename(img_path))
            copyfile(img_path, static_img_path)
            original_img = '/static/' + os.path.basename(img_path)
            results = model(img_path)
            # Lưu ảnh kết quả với hậu tố _result để tránh ghi đè ảnh gốc
            result_filename = os.path.splitext(os.path.basename(img_path))[0] + '_result.jpg'
            result_img_path = os.path.join('static', result_filename)
            results[0].save(result_img_path)
            result_img = '/static/' + result_filename
            details = []
            for box in results[0].boxes:
                class_id = int(box.cls.item())
                conf = float(box.conf.item())
                details.append({
                    'name': names[class_id],
                    'confidence': f'{conf:.2f}',
                    'bbox': [round(x, 2) for x in box.xyxy[0].tolist()]
                })
            # Tìm file nhãn thật tương ứng
            label_dir = 'labels/val'  # hoặc labels/test nếu test
            label_file = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)
            gt = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        x_center, y_center, w, h = map(float, parts[1:])
                        with Image.open(img_path) as im:
                            W, H = im.size
                        x_center_pixel = x_center * W
                        y_center_pixel = y_center * H
                        w_pixel = w * W
                        h_pixel = h * H
                        x1 = round(x_center_pixel - w_pixel / 2, 2)
                        y1 = round(y_center_pixel - h_pixel / 2, 2)
                        x2 = round(x_center_pixel + w_pixel / 2, 2)
                        y2 = round(y_center_pixel + h_pixel / 2, 2)
                        gt.append({
                            'name': names[class_id],
                            'bbox': [x1, y1, x2, y2]
                        })
                ground_truth = gt
            else:
                ground_truth = None
            # Lưu lịch sử và gửi Telegram
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            history_entry = {
                'time': now,
                'type': 'upload',
                'image': file.filename,
                'result_img': result_img,
                'details': details
            }
            save_history(history_entry)
            msg = f"[UPLOAD]\nThời gian: {now}\nKết quả: {details if details else 'Không phát hiện côn trùng!'}"
            send_telegram_message(msg, image_path=result_img_path)
            # Đọc lại lịch sử sau khi thêm mới
            try:
                if os.path.exists(HISTORY_FILE):
                    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                else:
                    history = []
            except Exception as e:
                history = []
    return render_template('index.html', result_img=result_img, details=details, ground_truth=ground_truth, original_img=original_img, esp32_ip=ESP32_IP, history=history)

@app.route('/history')
def history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
    except Exception as e:
        history = []
    return render_template('history.html', history=history)

if __name__ == '__main__':
    app.run(debug=False) 