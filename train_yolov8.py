from ultralytics import YOLO
import multiprocessing
import torch
import os
from datetime import datetime

def on_train_epoch_end(trainer):
    # Lấy metrics từ epoch hiện tại
    metrics = trainer.metrics
    epoch = trainer.epoch
    
    # In kết quả với định dạng đẹp hơn
    print(f"\n{'='*20} Epoch {epoch} Results {'='*20}")
    print(f"mAP50:     {metrics.get('metrics/mAP50(B)', 0):.4f}")
    print(f"mAP50-95:  {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"Box Loss:   {metrics.get('train/box_loss', 0):.4f}")
    print(f"Class Loss: {metrics.get('train/cls_loss', 0):.4f}")
    print(f"DFL Loss:   {metrics.get('train/dfl_loss', 0):.4f}")
    print(f"Learning Rate: {metrics.get('lr', 0):.6f}")
    print(f"{'='*50}")

def train_model():
    # Tạo thư mục lưu kết quả với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'runs/train_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    # Kiểm tra GPU
    print(f"\n{'='*20} System Information {'='*20}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"{'='*50}\n")

    # Tải mô hình YOLOv8
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')  # Sử dụng YOLOv8n (nano) thay vì small để tiết kiệm bộ nhớ

    print("\nStarting training with optimized parameters for RTX 3050 4GB...")
    # Huấn luyện mô hình với các tham số tối ưu cho GPU 4GB
    results = model.train(
        data='ip102.yaml',      # File cấu hình dữ liệu
        epochs=100,             # Số epochs
        imgsz=416,             # Giảm kích thước ảnh xuống 416 để tiết kiệm bộ nhớ
        batch=8,               # Giảm batch size xuống 8 để tránh OOM
        device='0',            # GPU device
        workers=4,             # Số workers cho data loading
        amp=True,              # Sử dụng Automatic Mixed Precision để tiết kiệm bộ nhớ
        cache=True,            # Cache images để tăng tốc độ
        close_mosaic=10,       # Bật mosaic augmentation trong 10 epochs đầu
        cos_lr=True,           # Cosine learning rate scheduler
        lr0=0.001,            # Learning rate ban đầu
        lrf=0.01,             # Learning rate cuối
        momentum=0.937,        # Momentum
        weight_decay=0.0005,   # Weight decay
        warmup_epochs=5,       # Warmup epochs
        warmup_momentum=0.8,   # Warmup momentum
        warmup_bias_lr=0.1,    # Warmup bias learning rate
        box=7.5,              # Box loss gain
        cls=0.3,              # Class loss gain
        dfl=1.5,              # DFL loss gain
        verbose=True,          # Hiển thị thông tin chi tiết
        seed=42,              # Random seed cố định
        deterministic=True,    # Deterministic training
        single_cls=False,     # Single class training
        rect=False,           # Rectangular training
        resume=False,         # Resume training
        fraction=1.0,         # Dataset fraction
        overlap_mask=True,     # Overlap mask
        mask_ratio=4,         # Mask ratio
        dropout=0.1,          # Thêm dropout để tránh overfitting
        val=True,             # Validate
        plots=True,           # Plot results
        save_period=10,       # Lưu mô hình mỗi 10 epochs
        project=save_dir      # Lưu kết quả vào thư mục có timestamp
    )

    # Lưu mô hình
    model.save(os.path.join(save_dir, 'yolov8n_ip102.pt'))
    
    # In kết quả chi tiết cuối cùng
    print(f"\n{'='*20} Final Results {'='*20}")
    print(f"mAP50: {results.results_dict['metrics/mAP50(B)'][-1]:.4f}")
    print(f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)'][-1]:.4f}")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*50}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_model() 