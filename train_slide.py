from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 换用最轻量级的 Nano 模型
    model = YOLO('yolov8n.pt')

    print("🚀 开始训练 Slide 单字切分模型...")
    
    results = model.train(
        data=r'C:\Users\GaoChenye\Desktop\OCR_Ancient_Book\train_data\slide\data.yaml',
        
        # 【轮数】因为任务简单，通常 100-150 轮就会完美收敛，不需要 300 轮
        epochs=150,          
        
        # 【尺寸】字条被切片后高度约 640，这里设为 640 即可。YOLO会自动将边缘填充为灰边，保持字不被拉伸变形。
        imgsz=1024,           
        
        # 【Batch】显卡虽然只有4G，但用 n 模型 + 640尺寸，batch 绝对可以开到 8 甚至 16！这能大幅提升切字精准度。
        batch=8,             
        
        device='0',          
        amp=True,            
        
        # ================= 以下是针对“单行字条”的特殊优化参数 =================
        
        # 【极度关键】关闭马赛克增强！
        # 默认的 mosaic 会把4张图切碎拼成十字形。如果你把4根单行字条拼在一起，会变成极为扭曲的“十”字形排版，彻底破坏古籍纵向排列的规律，必须设为 0！
        mosaic=0.0,          
        
        # 【关闭擦除】不要随机遮挡字体，因为切字需要完整的字体轮廓边界
        erasing=0.0,         
        
        # 【轻微缩放与平移】允许字条在灰底背景中上下左右轻微移动和缩放，增加鲁棒性
        scale=0.2,           
        translate=0.1,       
        
        # 【关闭左右翻转】古籍的偏旁部首是有严格左右位置的，绝不翻转
        fliplr=0.0,          
        
        plots=True           
    )
    print("✅ 单字切分模型训练完成！")