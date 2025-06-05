# Hệ Thống Nhận Diện Côn Trùng Gây Hại Sử Dụng YOLOv5 và ESP32-CAM

## Giới thiệu
Dự án này phát triển một hệ thống nhận diện côn trùng gây hại tự động sử dụng mô hình YOLOv5 và ESP32-CAM. Hệ thống có khả năng phát hiện và phân loại các loại côn trùng gây hại trong thời gian thực, giúp nông dân có thể theo dõi và xử lý kịp thời.

## Chức năng chính
1. **Thu thập hình ảnh thời gian thực**
   - Sử dụng ESP32-CAM để chụp ảnh
   - Stream video trực tiếp
   - Chụp ảnh theo yêu cầu

2. **Nhận diện côn trùng**
   - Sử dụng mô hình YOLOv5 đã được huấn luyện
   - Phát hiện và phân loại các loại côn trùng gây hại
   - Hiển thị kết quả với độ tin cậy

3. **Giao diện Web**
   - Xem stream video trực tiếp
   - Chụp và lưu ảnh
   - Hiển thị kết quả nhận diện
   - Xem lịch sử nhận diện

4. **Thông báo**
   - Gửi cảnh báo qua Telegram khi phát hiện côn trùng
   - Lưu lịch sử nhận diện
   - Xuất báo cáo kết quả

## Yêu cầu hệ thống
- Python 3.8 trở lên
- ESP32-CAM
- USB-TTL để nạp code
- Thư viện:
  - Flask
  - PyTorch
  - OpenCV
  - Ultralytics YOLOv5

## Cài đặt
1. **Cài đặt thư viện Python**
```bash
pip install -r requirements.txt
```

2. **Cấu hình ESP32-CAM**
- Nạp code vào ESP32-CAM
- Kết nối WiFi
- Kiểm tra IP address

3. **Chạy ứng dụng**
```bash
python web_app.py
```

## Cấu trúc thư mục
```
├── web_app.py           # Ứng dụng web chính
├── requirements.txt     # Các thư viện Python cần thiết
├── ip102.yaml          # File cấu hình classes
├── static/             # Thư mục chứa ảnh kết quả
├── uploads/            # Thư mục chứa ảnh upload
└── history.json        # File lưu lịch sử nhận diện
```
## Giao diện hệ thống

### 1. Giao Diện Hệ Thống Nhận Diện Côn Trùng Gây Hại
![Giao diện chính](static/example1.jpg)
- Giao diện web cho phép người dùng:
  * Tải lên ảnh côn trùng hoặc lá cây bị bệnh
  * Sử dụng camera ESP32-CAM trực tiếp
  * Hiển thị kết quả nhận diện với độ tin cậy cao

### 2. Giao Diện Kết Quả Nhận Diện
![Kết quả nhận diện](static/example2.jpg)
- Hiển thị:
  * Ảnh gốc được tải lên
  * Khung bounding box xác định vị trí côn trùng
  * Tên loài côn trùng và độ tin cậy
  * Đối chiếu với nhãn thật (ground truth)

### 3. Giao Diện Lịch Sử Dự Đoán
![Lịch sử dự đoán](static/example3.jpg)
- Bảng lịch sử chi tiết:
  * Thời gian thực hiện dự đoán
  * Loại nguồn ảnh (Upload/ESP32-CAM)
  * Tên file ảnh
  * Kết quả nhận diện
  * Ảnh kết quả với bounding box

### 4. Thông Báo Qua Telegram
![Thông báo Telegram](static/example4.jpg)
- Tự động gửi thông báo:
  * Thời gian phát hiện
  * Loại côn trùng được phát hiện
  * Độ tin cậy của nhận diện
  * Ảnh kèm bounding box
  * Vị trí phát hiện (bbox coordinates)
