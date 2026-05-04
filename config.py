#全局配置文件，包含模型路径、保存路径、字体路径等信息

import os
import sys
from PyQt5.QtGui import QColor

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "TYPE_MODEL_PATH": r"\Volumes\谷水\03_code\GUJI_OCR\model\type\best.pt",  #type模型路径
    "SLIDE_MODEL_PATH": r"\Volumes\谷水\03_code\GUJI_OCR\model\slide\best.pt",
    "SAVE_PAGE_DIR": r"\Volumes\谷水\03_code\GUJI_OCR\train_data\slide\new_saved_imgs",
    "SAVE_CROP_DIR": r"\Volumes\谷水\03_code\GUJI_OCR\train_data\text\new_saved_imgs",
    "VARIANTS_DICT_PATH": r"\Volumes\谷水\03_code\GUJI_OCR\dic_txt\dict.txt"   #异体字字典
}

DEVICE = 'cpu'

FONT_PATHS = [  
    r"\Volumes\谷水\03_code\GUJI_OCR\fonts\Jigmo.ttf",   #字体文件路径
    r"\Volumes\谷水\03_code\GUJI_OCR\fonts\Jigmo2.ttf",  
    r"\Volumes\谷水\03_code\GUJI_OCR\fonts\Jigmo3.ttf"  
]  

os.makedirs(CONFIG["SAVE_PAGE_DIR"], exist_ok=True)
os.makedirs(CONFIG["SAVE_CROP_DIR"], exist_ok=True)

def get_class_color(cls_name):
    if cls_name == 'text':
        return QColor(0, 0, 255)
    elif cls_name == 'subText':
        return QColor(0, 150, 0)
    colors =[QColor(170, 170, 255), QColor(255, 170, 170), QColor(170, 255, 170),
             QColor(255, 215, 135), QColor(135, 206, 250), QColor(221, 160, 221)]
    hash_val = sum(ord(c) for c in str(cls_name))
    return colors[hash_val % len(colors)]