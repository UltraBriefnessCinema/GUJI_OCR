# 所有的 QThread 线程类 (YOLOWorker, BatchInferWorker 等)

import os
import cv2
import json
import numpy as np
import fitz
import multiprocessing as mp
from PyQt5.QtCore import QThread, pyqtSignal, QRectF
from config import DEVICE
from core.ai_engines import paddle_ocr_target

class YOLOWorkerType(QThread):
    finished_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(self, model, image, conf_thres, iou_thres):
        super().__init__()
        self.model = model
        self.image = image
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def run(self):
        try:
            results = self.model.predict(source=self.image, conf=self.conf_thres, iou=self.iou_thres, device=DEVICE)
            boxes = results[0].boxes
            names = self.model.names
            box_data =[]
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                box_data.append((x1, y1, x2, y2, names[cls_id], conf))
            self.finished_signal.emit(box_data)
        except Exception as e:
            self.error_signal.emit(str(e))

class YOLOWorkerSlide(QThread):
    finished_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(self, model, image, line_rects, conf_thres, iou_thres):
        super().__init__()
        self.model = model
        self.image = image
        self.line_rects = line_rects  
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def run(self):
        try:
            img_h, img_w = self.image.shape[:2]
            names = self.model.names
            char_boxes =[]
            for line_id, lx1, ly1, lx2, ly2 in self.line_rects:
                lx1, ly1 = max(0, int(lx1)), max(0, int(ly1))
                lx2, ly2 = min(img_w, int(lx2)), min(img_h, int(ly2))
                line_crop = self.image[ly1:ly2, lx1:lx2]
                if line_crop.size == 0: continue
                results = self.model.predict(source=line_crop, conf=self.conf_thres, iou=self.iou_thres, device=DEVICE)
                boxes = results[0].boxes
                for box in boxes:
                    cx1, cy1, cx2, cy2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    char_boxes.append((line_id, cx1 + lx1, cy1 + ly1, cx2 + lx1, cy2 + ly1, names[cls_id], conf))
            self.finished_signal.emit(char_boxes)
        except Exception as e:
            self.error_signal.emit(str(e))

class BatchInferWorker(QThread):
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, pdf_path, type_model, slide_model, type_conf, slide_conf):
        super().__init__()
        self.pdf_path = pdf_path
        self.type_model = type_model
        self.slide_model = slide_model
        self.type_conf = type_conf
        self.slide_conf = slide_conf

    def run(self):
        try:
            doc = fitz.open(self.pdf_path)
            total = len(doc)
            result_data = {}

            for i in range(total):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
                cv_img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_BGRA2BGR)
                img_h, img_w = cv_img.shape[:2]

                type_results = self.type_model.predict(source=cv_img, conf=self.type_conf, iou=0.45, verbose=False, device=DEVICE)
                names = self.type_model.names
                t_list, s_list = [],[]
                type_boxes_data = []
                for b in type_results[0].boxes:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                    cls_name = names[int(b.cls[0].cpu().numpy())]
                    if 'text' not in cls_name.lower(): cls_name = 'text'
                    type_boxes_data.append({
                        'left': float(x1), 'right': float(x2), 'top': float(y1), 'bottom': float(y2),
                        'width': float(x2 - x1), 'height': float(y2 - y1),
                        'cx': float((x1 + x2) / 2), 'cy': float((y1 + y2) / 2),
                        'cls_name': cls_name, 'conf': float(b.conf[0].cpu().numpy())
                    })

                type_boxes_data.sort(key=lambda b: b['top'])
                columns, global_last_text_width = [], None
                for box in type_boxes_data:
                    b_left, b_right, b_width = box['left'], box['right'], box['width']
                    best_col, max_overlap = None, 0.0
                    for col in columns:
                        ref_text_box = next((b for b in reversed(col) if b['cls_name'] != 'subText'), None)
                        if ref_text_box:
                            ref_left, ref_width = ref_text_box['left'], ref_text_box['width']
                        else:
                            ref_sub = col[-1]
                            ref_width = global_last_text_width if global_last_text_width else ref_sub['width'] * 2
                            ref_left = ref_sub['cx'] - ref_width / 2.0
                        ref_right = ref_left + ref_width
                        inter_width = max(0, min(b_right, ref_right) - max(b_left, ref_left))
                        if b_width > 0:
                            overlap_ratio = inter_width / b_width
                            if overlap_ratio > 0.4 and overlap_ratio > max_overlap:
                                max_overlap, best_col = overlap_ratio, col
                    if best_col is not None: best_col.append(box)
                    else: columns.append([box])
                    if box['cls_name'] != 'subText': global_last_text_width = box['width']
                        
                columns.sort(key=lambda col: sum(b['cx'] for b in col) / len(col), reverse=True)
                sorted_type_data =[]
                for col in columns:
                    rows =[]
                    for box in col:
                        is_sub, added = (box['cls_name'] == 'subText'), False
                        if rows and is_sub:
                            last_row = rows[-1]
                            for ref_box in last_row:
                                if ref_box['cls_name'] == 'subText':
                                    t1, b1, t2, b2 = box['top'], box['bottom'], ref_box['top'], ref_box['bottom']
                                    min_h = min(b1 - t1, b2 - t2)
                                    if min_h > 0 and (max(0, min(b1, b2) - max(t1, t2)) / min_h) > 0.3:
                                        last_row.append(box); added = True; break
                        if not added: rows.append([box])
                    for row in rows:
                        if len(row) > 1: row.sort(key=lambda b: b['cx'], reverse=True)
                        sorted_type_data.extend(row)

                for t_id, box in enumerate(sorted_type_data, start=1):
                    x1, y1, width, height = box['left'], box['top'], box['width'], box['height']
                    t_list.append([x1, y1, width, height, box['cls_name'], box['conf'], 'type', t_id])
                    lx1, ly1 = max(0, int(x1)), max(0, int(y1))
                    lx2, ly2 = min(img_w, int(x1 + width)), min(img_h, int(y1 + height))
                    crop = cv_img[ly1:ly2, lx1:lx2]
                    if crop.size == 0: continue
                    slide_results = self.slide_model.predict(source=crop, conf=self.slide_conf, iou=0.45, verbose=False, device=DEVICE)
                    temp_s_boxes =[]
                    for sb in slide_results[0].boxes:
                        cx1, cy1, cx2, cy2 = sb.xyxy[0].cpu().numpy()
                        s_conf = float(sb.conf[0].cpu().numpy())
                        temp_s_boxes.append((cx1+lx1, cy1+ly1, cx2-cx1, cy2-cy1, s_conf))
                    temp_s_boxes.sort(key=lambda b: b[1] + b[3]/2)
                    for s_id, (sx, sy, sw, sh, s_conf) in enumerate(temp_s_boxes, start=1):
                        s_list.append([float(sx), float(sy), float(sw), float(sh), 'text', s_conf, 'slide', s_id])
                result_data[str(i)] = {'type': t_list, 'slide': s_list}
                self.progress_signal.emit(i + 1, total)

            self.finished_signal.emit(result_data)
        except Exception as e:
            self.error_signal.emit(str(e))

class GlobalBatchInferWorker(QThread):
    progress_signal = pyqtSignal(int, int, str, int, int)
    finished_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)

    def __init__(self, pdf_paths, type_model, slide_model, type_conf, slide_conf):
        super().__init__()
        self.pdf_paths = pdf_paths
        self.type_model = type_model
        self.slide_model = slide_model
        self.type_conf = type_conf
        self.slide_conf = slide_conf
        self._is_running = True

    def run(self):
        processed_count = 0
        try:
            for file_idx, pdf_path in enumerate(self.pdf_paths, start=1):
                if not self._is_running: break
                json_path = os.path.splitext(pdf_path)[0] + "_project.json"
                if os.path.exists(json_path): continue
                doc = fitz.open(pdf_path)
                total_pages = len(doc)
                result_data = {}
                pdf_interrupted = False
                for i in range(total_pages):
                    if not self._is_running: pdf_interrupted = True; break
                    self.progress_signal.emit(file_idx, len(self.pdf_paths), os.path.basename(pdf_path), i + 1, total_pages)
                    page = doc.load_page(i)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
                    cv_img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_BGRA2BGR)
                    img_h, img_w = cv_img.shape[:2]
                    type_results = self.type_model.predict(source=cv_img, conf=self.type_conf, iou=0.45, verbose=False, device=DEVICE)
                    names = self.type_model.names
                    t_list, s_list = [],[]
                    type_boxes_data =[]
                    for b in type_results[0].boxes:
                        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                        cls_name = names[int(b.cls[0].cpu().numpy())]
                        if 'text' not in cls_name.lower(): cls_name = 'text'
                        type_boxes_data.append({
                            'left': float(x1), 'right': float(x2), 'top': float(y1), 'bottom': float(y2),
                            'width': float(x2 - x1), 'height': float(y2 - y1),
                            'cx': float((x1 + x2) / 2), 'cls_name': cls_name, 'conf': float(b.conf[0].cpu().numpy())
                        })
                    type_boxes_data.sort(key=lambda b: b['top'])
                    columns, global_last_text_width =[], None
                    for box in type_boxes_data:
                        b_left, b_right, b_width = box['left'], box['right'], box['width']
                        best_col, max_overlap = None, 0.0
                        for col in columns:
                            ref_text_box = next((b for b in reversed(col) if b['cls_name'] != 'subText'), None)
                            if ref_text_box:
                                ref_left, ref_width = ref_text_box['left'], ref_text_box['width']
                            else:
                                ref_sub = col[-1]
                                ref_width = global_last_text_width if global_last_text_width else ref_sub['width'] * 2
                                ref_left = ref_sub['cx'] - ref_width / 2.0
                            ref_right = ref_left + ref_width
                            inter_width = max(0, min(b_right, ref_right) - max(b_left, ref_left))
                            if b_width > 0:
                                overlap_ratio = inter_width / b_width
                                if overlap_ratio > 0.4 and overlap_ratio > max_overlap:
                                    max_overlap, best_col = overlap_ratio, col
                        if best_col is not None: best_col.append(box)
                        else: columns.append([box])
                        if box['cls_name'] != 'subText': global_last_text_width = box['width']
                    columns.sort(key=lambda col: sum(b['cx'] for b in col) / len(col), reverse=True)
                    sorted_type_data =[]
                    for col in columns:
                        rows =[]
                        for box in col:
                            is_sub, added = (box['cls_name'] == 'subText'), False
                            if rows and is_sub:
                                last_row = rows[-1]
                                for ref_box in last_row:
                                    if ref_box['cls_name'] == 'subText':
                                        t1, b1, t2, b2 = box['top'], box['bottom'], ref_box['top'], ref_box['bottom']
                                        min_h = min(b1 - t1, b2 - t2)
                                        if min_h > 0 and (max(0, min(b1, b2) - max(t1, t2)) / min_h) > 0.3:
                                            last_row.append(box); added = True; break
                            if not added: rows.append([box])
                        for row in rows:
                            if len(row) > 1: row.sort(key=lambda b: b['cx'], reverse=True)
                            sorted_type_data.extend(row)

                    for t_id, box in enumerate(sorted_type_data, start=1):
                        x1, y1, width, height = box['left'], box['top'], box['width'], box['height']
                        t_list.append([x1, y1, width, height, box['cls_name'], box['conf'], 'type', t_id])
                        lx1, ly1 = max(0, int(x1)), max(0, int(y1))
                        lx2, ly2 = min(img_w, int(x1 + width)), min(img_h, int(y1 + height))
                        crop = cv_img[ly1:ly2, lx1:lx2]
                        if crop.size == 0: continue
                        slide_results = self.slide_model.predict(source=crop, conf=self.slide_conf, iou=0.45, verbose=False, device=DEVICE)
                        temp_s_boxes =[]
                        for sb in slide_results[0].boxes:
                            cx1, cy1, cx2, cy2 = sb.xyxy[0].cpu().numpy()
                            s_conf = float(sb.conf[0].cpu().numpy())
                            temp_s_boxes.append((cx1+lx1, cy1+ly1, cx2-cx1, cy2-cy1, s_conf))
                        temp_s_boxes.sort(key=lambda b: b[1] + b[3]/2)
                        for s_id, (sx, sy, sw, sh, s_conf) in enumerate(temp_s_boxes, start=1):
                            s_list.append([float(sx), float(sy), float(sw), float(sh), 'text', s_conf, 'slide', s_id])
                    result_data[str(i)] = {'type': t_list, 'slide': s_list}
                doc.close()
                if not pdf_interrupted:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, ensure_ascii=False, indent=2)
                    processed_count += 1
            self.finished_signal.emit(processed_count)
        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self._is_running = False

class CurrentBookOCRWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, pdf_path, project_data, ocr_model_path):
        super().__init__()
        self.pdf_path = pdf_path
        self.project_data = project_data.copy()
        self.ocr_model_path = ocr_model_path
        self.is_running = True

    def run(self):
        try:
            input_q = mp.Queue()
            output_q = mp.Queue()
            p = mp.Process(target=paddle_ocr_target, args=(input_q, output_q, self.ocr_model_path))
            p.daemon = True
            p.start()
            
            status = output_q.get()
            if status != "INIT_SUCCESS":
                self.error_signal.emit(f"OCR初始化失败: {status}")
                return

            doc = fitz.open(self.pdf_path)
            total_pages = len(doc)
            
            for i in range(total_pages):
                if not self.is_running: break
                
                page_str = str(i)
                if page_str not in self.project_data: continue
                s_list = self.project_data[page_str].get('slide', [])
                if not s_list: continue
                
                self.progress_signal.emit(i + 1, total_pages, f"正在识别第 {i+1} 页...")
                
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
                cv_img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_BGRA2BGR)
                img_h, img_w = cv_img.shape[:2]
                
                for s_idx, s in enumerate(s_list):
                    if len(s) < 9: s.extend(["", 0.0])
                    lx1, ly1 = max(0, int(s[0])), max(0, int(s[1]))
                    lx2, ly2 = min(img_w, int(s[0] + s[2])), min(img_h, int(s[1] + s[3]))
                    crop = cv_img[ly1:ly2, lx1:lx2]
                    if crop.size == 0: continue
                    
                    target_h = max(48, crop.shape[0] * 2)
                    scale = target_h / crop.shape[0]
                    target_w = int(crop.shape[1] * scale)
                    resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                    padded = cv2.copyMakeBorder(resized, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255,255,255])
                    
                    input_q.put((s_idx, padded))
                    try:
                        res_id, text, conf = output_q.get(timeout=15)
                        s[8] = text
                        s[9] = conf
                    except:
                        pass
                        
            doc.close()
            input_q.put("SHUTDOWN")
            p.join()
            self.finished_signal.emit(self.project_data)
        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self.is_running = False

class GlobalBatchOCRWorker(QThread):
    progress_signal = pyqtSignal(int, int, str, int, int)
    finished_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)

    def __init__(self, folder_path, ocr_model_path):
        super().__init__()
        self.folder_path = folder_path
        self.ocr_model_path = ocr_model_path
        self._is_running = True

    def run(self):
        processed_count = 0
        try:
            input_q = mp.Queue()
            output_q = mp.Queue()
            p = mp.Process(target=paddle_ocr_target, args=(input_q, output_q, self.ocr_model_path))
            p.daemon = True
            p.start()
            
            status = output_q.get()
            if status != "INIT_SUCCESS":
                self.error_signal.emit(f"OCR初始化失败: {status}")
                return

            pdf_paths =[]
            for root, dirs, files in os.walk(self.folder_path):
                for f in files:
                    if f.lower().endswith('.pdf'):
                        pdf_paths.append(os.path.join(root, f))
            
            for file_idx, pdf_path in enumerate(pdf_paths, start=1):
                if not self._is_running: break
                json_path = os.path.splitext(pdf_path)[0] + "_project.json"
                if not os.path.exists(json_path): continue
                with open(json_path, 'r', encoding='utf-8') as f:
                    project_data = json.load(f)
                doc = fitz.open(pdf_path)
                total_pages = len(doc)
                
                for i in range(total_pages):
                    if not self._is_running: break
                    self.progress_signal.emit(file_idx, len(pdf_paths), os.path.basename(pdf_path), i + 1, total_pages)
                    page_str = str(i)
                    if page_str not in project_data: continue
                    s_list = project_data[page_str].get('slide',[])
                    if not s_list: continue
                    
                    page = doc.load_page(i)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
                    cv_img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_BGRA2BGR)
                    img_h, img_w = cv_img.shape[:2]
                    
                    for s_idx, s in enumerate(s_list):
                        if len(s) < 9: s.extend(["", 0.0])
                        lx1, ly1 = max(0, int(s[0])), max(0, int(s[1]))
                        lx2, ly2 = min(img_w, int(s[0] + s[2])), min(img_h, int(s[1] + s[3]))
                        crop = cv_img[ly1:ly2, lx1:lx2]
                        if crop.size == 0: continue
                        h_orig, w_orig = crop.shape[:2]
                        target_h = max(48, h_orig * 2)
                        scale = target_h / h_orig
                        target_w = int(w_orig * scale)
                        resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                        padded = cv2.copyMakeBorder(resized, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255,255,255])
                        
                        input_q.put((s_idx, padded))
                        try:
                            res_id, text, conf = output_q.get(timeout=15)
                            s[8] = text
                            s[9] = conf
                        except:
                            pass
                doc.close()
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(project_data, f, ensure_ascii=False, indent=2)
                    
                txt_path = os.path.splitext(pdf_path)[0] + "_ocr.txt"
                all_lines =[]
                for p_idx in range(total_pages):
                    page_str = str(p_idx)
                    if page_str not in project_data: continue
                    t_list = project_data[page_str].get('type', [])
                    s_list = project_data[page_str].get('slide',[])
                    if not t_list: continue
                    t_list_sorted = sorted(t_list, key=lambda b: b[0] + b[2]/2.0, reverse=True)
                    all_lines.append(f"[{p_idx + 1}]")
                    in_subtext_block = False
                    
                    for t in t_list_sorted:
                        t_cls = t[4]
                        t_rect = QRectF(t[0], t[1], t[2], t[3])
                        col_slides =[]
                        for s in s_list:
                            s_rect = QRectF(s[0], s[1], s[2], s[3])
                            inter = s_rect.intersected(t_rect)
                            if inter.width() * inter.height() > 0.4 * (s[2] * s[3]):
                                col_slides.append(s)
                        col_slides.sort(key=lambda b: b[1] + b[3]/2.0)
                        col_text = "".join([s[8].strip() if len(s)>8 and s[8] else "" for s in col_slides])
                        if col_text:
                            if t_cls == 'subText':
                                if not in_subtext_block:
                                    all_lines.append("(") 
                                    in_subtext_block = True
                                all_lines.append(col_text)
                            else:
                                if in_subtext_block:
                                    all_lines.append(")") 
                                    in_subtext_block = False
                                all_lines.append(col_text)
                    if in_subtext_block:
                        all_lines.append(")")
                        
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(all_lines))
                processed_count += 1
                
            input_q.put("SHUTDOWN")
            p.join()
            self.finished_signal.emit(processed_count)
        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self._is_running = False

class OCRWorker(QThread):  
    progress_signal = pyqtSignal(int)            
    result_signal = pyqtSignal(int, str, float)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, boxes_data, image, rec_model_path):  
        super().__init__()
        self.boxes_data = boxes_data   
        self.image = image
        self.rec_model_path = rec_model_path
        self.is_running = True

    def run(self):
        input_q = mp.Queue()
        output_q = mp.Queue()
        p = mp.Process(target=paddle_ocr_target, args=(input_q, output_q, self.rec_model_path))
        p.daemon = True
        p.start()
        
        status = output_q.get()
        if status != "INIT_SUCCESS":
            self.error_signal.emit(f"PaddleOCR 进程启动失败: {status}")
            return

        for i, box_data in enumerate(self.boxes_data):
            if not self.is_running: break
            rect = box_data['rect']
            box_id = box_data['id']
            x, y = int(rect.x()), int(rect.y())
            w, h = int(rect.width()), int(rect.height())
            img_h, img_w = self.image.shape[:2]
            y1, y2 = max(0, y), min(img_h, y+h)
            x1, x2 = max(0, x), min(img_w, x+w)
            crop = self.image[y1:y2, x1:x2]
            text, conf = "", 0.0
            if crop.size > 0:
                h_orig, w_orig = crop.shape[:2]
                target_h = max(48, h_orig * 2)
                scale = target_h / h_orig
                target_w = int(w_orig * scale)
                resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                padded = cv2.copyMakeBorder(resized, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255,255,255])
                input_q.put((box_id, padded))
                try: _, text, conf = output_q.get(timeout=15)
                except: text, conf = "ERROR", 0.0
            self.result_signal.emit(box_id, text, conf)
            self.progress_signal.emit(i + 1)   

        input_q.put("SHUTDOWN")
        p.join()
        self.finished_signal.emit()

    def stop(self):
        self.is_running = False