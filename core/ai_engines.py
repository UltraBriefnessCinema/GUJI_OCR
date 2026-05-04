# PaddleOCR 独立进程函数 (paddle_ocr_target, resident_ocr_target)

import os
import sys
import warnings
import cv2
import numpy as np

def paddle_ocr_target(input_queue, output_queue, model_path):  
    sys.modules['torch'] = None  
    sys.modules['ultralytics'] = None

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ["FLAGS_allocator_strategy"] = "auto_growth"
    
    try:
        import paddle
        import paddleocr
        paddle.set_device('gpu')
        from paddleocr import TextRecognition
        ocr_model = TextRecognition(model_dir=model_path)
        output_queue.put("INIT_SUCCESS")
    except Exception as e:
        output_queue.put(f"INIT_ERROR: {e}")
        return  

    while True:  
        task = input_queue.get()  
        if task == "SHUTDOWN":  
            break  
        box_id, img_array = task  
        text, confidence = "", 0.0  
        
        try:  
            results = ocr_model.predict(img_array)
            if isinstance(results, tuple) and len(results) >= 1 and isinstance(results[0], list):
                results = results[0]
            elif not isinstance(results, list):
                results = [results]
                
            for res in results:  
                if isinstance(res, (tuple, list)) and len(res) >= 2:
                    text = str(res[0])
                    confidence = float(res[1])
                elif isinstance(res, dict):
                    text = str(res.get('rec_text', res.get('text', '')))
                    confidence = float(res.get('rec_score', res.get('score', 0.0)))
                else:
                    text = str(getattr(res, 'rec_text', getattr(res, 'text', '')))
                    confidence = float(getattr(res, 'rec_score', getattr(res, 'score', 0.0)))
                
                if text.strip():
                    break
                    
            if len(text) > 1:  
                text = text[0]  
        except Exception as e:  
            print(f"OCR error box {box_id}: {e}")  
            
        output_queue.put((box_id, text, confidence))

def resident_ocr_target(req_q, res_q, model_path):
    """常驻后台服务：利用极限图像扰动，强制逼出模型的前 N 个错误猜测"""
    sys.modules['torch'] = None  
    sys.modules['ultralytics'] = None
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ["FLAGS_allocator_strategy"] = "auto_growth"
    
    try:
        import paddle
        paddle.set_device('gpu')
        from paddleocr import TextRecognition
        ocr_model = TextRecognition(model_dir=model_path)
        res_q.put("READY")
    except Exception as e:
        res_q.put(f"ERROR: {e}")
        return  

    def get_extreme_variations(img):
        yield img 
        h, w = img.shape[:2]
        if h < 5 or w < 5: return
        yield cv2.resize(img, (int(w * 0.4), h))  
        yield cv2.resize(img, (int(w * 2.5), h))  
        yield cv2.resize(img, (w, int(h * 0.4)))  
        yield cv2.resize(img, (int(w * 1.5), int(h * 1.5)))
        kernel = np.ones((3,3), np.uint8)
        yield cv2.erode(img, kernel, iterations=2) 
        yield cv2.dilate(img, kernel, iterations=2) 
        yield cv2.warpAffine(img, cv2.getRotationMatrix2D((w/2, h/2), 15, 1), (w, h), borderValue=(255,255,255))
        yield cv2.warpAffine(img, cv2.getRotationMatrix2D((w/2, h/2), -15, 1), (w, h), borderValue=(255,255,255))
        hole = img.copy()
        cv2.rectangle(hole, (int(w*0.3), int(h*0.3)), (int(w*0.7), int(h*0.7)), (255,255,255), -1)
        yield hole
        noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        yield cv2.addWeighted(img, 0.5, noise, 0.5, 0)
        yield cv2.bitwise_not(img)

    while True:
        task = req_q.get()
        if task == "SHUTDOWN": break
        req_id, img_array = task
        
        candidates = {}
        try:
            for var_img in get_extreme_variations(img_array):
                results = ocr_model.predict(var_img)
                text, conf = "", 0.0
                if isinstance(results, tuple) and len(results) >= 1 and isinstance(results[0], list):
                    results = results[0]
                elif not isinstance(results, list):
                    results = [results]
                    
                for res in results:  
                    if isinstance(res, (tuple, list)) and len(res) >= 2:
                        text, conf = str(res[0]), float(res[1])
                    elif isinstance(res, dict):
                        text = str(res.get('rec_text', res.get('text', '')))
                        conf = float(res.get('rec_score', res.get('score', 0.0)))
                    else:
                        text = str(getattr(res, 'rec_text', getattr(res, 'text', '')))
                        conf = float(getattr(res, 'rec_score', getattr(res, 'score', 0.0)))
                    if text.strip(): break
                
                if len(text) > 1: text = text[0]
                if text.strip():
                    if text not in candidates or conf > candidates[text]:
                        candidates[text] = conf
                if len(candidates) >= 6: break
            
            sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:6]
            res_q.put((req_id, sorted_cands))
        except Exception:
            res_q.put((req_id, []))