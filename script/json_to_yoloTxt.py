import json
import os
import glob

# 👇 【请修改为你的图片和 json 所在的文件夹绝对路径】
target_dir = r"C:\Users\GaoChenye\Desktop\OCR_Ancient_Book\train_data\slide\images" 

# 标签映射字典 (如果你只标了 text，那就是 0)
class_mapping = {"ch": 0}

json_files = glob.glob(os.path.join(target_dir, "*.json"))
if not json_files:
    print("❌ 没找到 JSON 文件，请检查路径是否正确！")

for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    img_w = data['imageWidth']
    img_h = data['imageHeight']
    
    txt_filename = json_file.replace(".json", ".txt")
    
    with open(txt_filename, 'w', encoding='utf-8') as txt_f:
        for shape in data['shapes']:
            label = shape['label']
            if label not in class_mapping:
                continue
            class_id = class_mapping[label]
            
            # 获取矩形框的坐标
            points = shape['points']
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            
            # 转换为 YOLO 需要的 中心点 x,y 和 宽,高
            box_w = xmax - xmin
            box_h = ymax - ymin
            cx = xmin + box_w / 2.0
            cy = ymin + box_h / 2.0
            
            # 必须进行归一化处理 (除以图片的宽高)
            cx /= img_w
            cy /= img_h
            box_w /= img_w
            box_h /= img_h
            
            # 写入单行数据
            txt_f.write(f"{class_id} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}\n")

print(f"✅ 转换完美结束！共生成了 {len(json_files)} 个 YOLO txt 文件。")