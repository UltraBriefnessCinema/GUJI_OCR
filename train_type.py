from ultralytics import YOLO

if __name__ == '__main__':
    # 加载 YOLOv8m (中等规模模型，性能极强且只需约 3.5GB 显存)
    model = YOLO('yolov8m.pt')

    print("🚀 开始训练属于你的古籍检测模型...")
    
    # 开始训练！
    results = model.train(
        data=r'F:\03_code\OCR_Ancient_Book\train_data\text\data.yaml',    # 指向你的配置文件
        epochs=300,          # 训练100轮
        imgsz=1024,           # 图像缩放尺寸 (为了省显存设为640)
        batch=2,             # 【防爆显存核心】每次只送4张图进显卡，若报错 OOM，请改为 2！
        device='0',          # 使用你的独立显卡
        amp=True,            # 开启混合精度，可节省大量显存并加速
        plots=True           # 训练结束生成分析图表
    )
    print("✅ 训练完成！请前往 runs/detect/train/weights/ 文件夹查找 best.pt")