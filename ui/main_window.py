import os
import sys
import cv2
import json
import numpy as np
import fitz  # PyMuPDF
import difflib
import multiprocessing as mp
from PyQt5.QtGui import QImage, QPixmap, QFontDatabase, QPen, QColor, QFont, QBrush
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFileDialog, QLabel, QDoubleSpinBox, QSlider, QTabWidget,
                             QSizePolicy, QLineEdit, QMenu, QProgressBar, QProgressDialog,
                             QMessageBox, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
                             QGraphicsTextItem,QApplication)
from PyQt5.QtCore import Qt, QTimer, QRectF

# ======= 导入全局配置和核心组件 =======
from config import CONFIG, DEVICE, FONT_PATHS, BASE_DIR
from core.ai_engines import resident_ocr_target
from core.workers import (YOLOWorkerType, YOLOWorkerSlide, BatchInferWorker, 
                          GlobalBatchInferWorker, CurrentBookOCRWorker, GlobalBatchOCRWorker, OCRWorker)
from ui.components import BoundingBox, OCRTextItem, ImageViewer

class YOLOStageTester(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 古籍版面 & 拆字系统")
        self.setGeometry(100, 100, 1400, 900)

        self.cv_image = None       
        self.display_image = None  
        self.type_model = None
        self.slide_model = None
        
        self.pdf_doc = None
        self.current_page = 0
        self.total_pages = 0
        
        self.current_mode = 'type' 
        self.show_labels = True 
        self.sort_mode = {'type': 'auto', 'slide': 'auto'}
        self.sorted_boxes_data = []   
        
        self.ocr_model_path = r"C:\Users\GaoChenye\Desktop\OCR_Ancient_Book\model\ocr\PP-OCRv5_server_rec_infer"  
        self.font_families = []  
        for path in FONT_PATHS:  
            fid = QFontDatabase.addApplicationFont(path)  
            if fid != -1:  
                families = QFontDatabase.applicationFontFamilies(fid)  
                if families: self.font_families.append(families[0])  
                
        self.ocr_scene = QGraphicsScene()
        self.ocr_text_items = []
        
        self.project_data = {} 
        self.json_path = ""
        self.history = {'type': {'undo':[], 'redo': []}, 'slide': {'undo':[], 'redo':[]}}
        
        self.manual_sort_pending =[]
        self.manual_sort_ids =[]
        self.manual_sort_backup = {}

        # 加载异体字/繁体字字典
        self.variant_dict = {}
        dict_path = CONFIG.get("VARIANTS_DICT_PATH", os.path.join(BASE_DIR, "variants.txt"))
        if os.path.exists(dict_path):
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        chars = line.strip().split()
                        if len(chars) > 1:
                            for c in chars:
                                self.variant_dict.setdefault(c, set()).update([x for x in chars if x != c])
                print(f"✅ 成功加载异体字字典，共收录 {len(self.variant_dict)} 个字符簇。")
            except Exception as e:
                print(f"⚠️ 字典文件读取失败: {e}")
        else:
            print(f"⚠️ 未找到异体字字典，请检查路径: {dict_path}")

        self.init_models()
        self.init_ui()
        self.show_msg("系统初始化就绪。")

    def init_models(self):
        import torch
        import ultralytics           
        from ultralytics import YOLO 

        print(f"系统就绪 -> 当前 YOLO 强制推理设备: {'GPU (CUDA)' if DEVICE == '0' else 'CPU'}")
        
        try:
            self.type_model = YOLO(CONFIG["TYPE_MODEL_PATH"])
            self.slide_model = YOLO(CONFIG["SLIDE_MODEL_PATH"])

            print(f"PyTorch 版本: {torch.__version__}")
            print(f"Ultralytics 版本: {ultralytics.__version__}")
            if torch.cuda.is_available():
                print(f"CUDA 可用: True, 设备名: {torch.cuda.get_device_name(0)}")
            else:
                print("CUDA 可用: False")
        except Exception as e:
            print(f"⚠️ 模型加载异常: {e}")
        
        self.resident_req_q = mp.Queue()
        self.resident_res_q = mp.Queue()
        self.resident_process = mp.Process(target=resident_ocr_target, args=(self.resident_req_q, self.resident_res_q, self.ocr_model_path))
        self.resident_process.daemon = True
        self.resident_process.start()
        print("✅ 常驻实时 OCR 推理服务已挂载！")

    def closeEvent(self, event):
        """窗口关闭时自动清理所有后台常驻进程"""
        if hasattr(self, 'resident_req_q'): self.resident_req_q.put("SHUTDOWN")
        if hasattr(self, 'resident_process') and self.resident_process.is_alive():
            self.resident_process.join(timeout=1)
        super().closeEvent(event)

    def show_msg(self, text, timeout=6000):
        self.statusBar().showMessage(text, timeout)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        layout.addLayout(self._create_top_bar())
        
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed) 
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        self._setup_type_tab()
        self._setup_slide_tab()
        self._setup_ocr_tab()              
        self.tabs.setTabEnabled(2, False)  

        self.viewer = ImageViewer(self.get_current_mode, self.sort_and_number_type_boxes)
        layout.addWidget(self.tabs)
        layout.addWidget(self.viewer)

    def _create_top_bar(self):
        top_bar = QHBoxLayout()
        btn_load_img = QPushButton("导入图片/PDF")
        btn_load_img.clicked.connect(self.load_image)
        top_bar.addWidget(btn_load_img)
        
        self.btn_batch = QPushButton("全册批量推理 (存JSON于同目录)")
        self.btn_batch.setStyleSheet("background-color: #673AB7; color: white;")
        self.btn_batch.clicked.connect(self.run_batch_inference)
        top_bar.addWidget(self.btn_batch)
        
        self.btn_prev = QPushButton("◀ 上一页")
        self.btn_prev.setEnabled(False)
        self.btn_prev.setShortcut("n")
        self.btn_prev.clicked.connect(self.prev_page)
        
        self.input_page = QLineEdit()
        self.input_page.setFixedWidth(50)
        self.input_page.setAlignment(Qt.AlignCenter)
        self.input_page.returnPressed.connect(self.jump_page)
        
        self.lbl_total_pages = QLabel(" / 0")
        
        self.btn_next = QPushButton("下一页 ▶")
        self.btn_next.setEnabled(False)
        self.btn_next.setShortcut("m")
        self.btn_next.clicked.connect(self.next_page)

        top_bar.addWidget(self.btn_prev)
        top_bar.addWidget(self.input_page)
        top_bar.addWidget(self.lbl_total_pages)
        top_bar.addWidget(self.btn_next)
        top_bar.addSpacing(20)
        top_bar.addWidget(QLabel("底图透明度"))
        
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(1, 100)
        self.slider_opacity.setValue(80)  
        self.slider_opacity.setFixedWidth(100)
        self.slider_opacity.valueChanged.connect(self.change_image_opacity)
        top_bar.addWidget(self.slider_opacity)

        self.btn_toggle_labels = QPushButton("隐藏数字")
        self.btn_toggle_labels.clicked.connect(self.toggle_labels)
        top_bar.addWidget(self.btn_toggle_labels)

        btn_clear_all = QPushButton("清空当前层")
        btn_clear_all.clicked.connect(lambda: self.clear_boxes(self.current_mode))
        top_bar.addWidget(btn_clear_all)
        top_bar.addStretch()

        self.btn_global_batch = QPushButton("📁 扫描文件夹推理")
        self.btn_global_batch.setStyleSheet("background-color: #E91E63; color: white; font-weight: bold;")
        self.btn_global_batch.setToolTip("选中一个文件夹，自动扫描内部所有 PDF 并静默推理保存")
        self.btn_global_batch.clicked.connect(self.start_global_batch)
        top_bar.addWidget(self.btn_global_batch)

        self.btn_batch_ocr = QPushButton("📚 OCR批量处理")
        self.btn_batch_ocr.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.btn_batch_ocr.setToolTip("扫描文件夹内含有 JSON 与 PDF 的数据，全局生成 OCR 并保存 TXT")
        self.btn_batch_ocr.clicked.connect(self.start_global_batch_ocr)
        top_bar.addWidget(self.btn_batch_ocr)

        self.global_progress = QProgressBar()
        self.global_progress.setTextVisible(True)
        self.global_progress.setFixedWidth(230)
        self.global_progress.setVisible(False)
        top_bar.addWidget(self.global_progress)

        self.btn_stop_global = QPushButton("⏹ 停止扫描")
        self.btn_stop_global.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")
        self.btn_stop_global.setVisible(False)
        self.btn_stop_global.clicked.connect(self.stop_global_batch)
        top_bar.addWidget(self.btn_stop_global)
        return top_bar

    def _setup_type_tab(self):
        tab_type = QWidget()
        type_layout = QHBoxLayout(tab_type)

        btn_undo_type = QPushButton("↩")
        btn_undo_type.setFixedWidth(40)
        btn_undo_type.setShortcut("Ctrl+Z")
        btn_undo_type.clicked.connect(lambda: self.undo('type'))
        
        btn_redo_type = QPushButton("↪")
        btn_redo_type.setFixedWidth(40)
        btn_redo_type.setShortcut("Ctrl+Y")
        btn_redo_type.clicked.connect(lambda: self.redo('type'))
        
        type_layout.addWidget(btn_undo_type)
        type_layout.addWidget(btn_redo_type)

        type_layout.addWidget(QLabel(" 预测阈值 Conf:"))
        self.spin_conf_type = QDoubleSpinBox()
        self.spin_conf_type.setRange(0.01, 1.0)
        self.spin_conf_type.setValue(0.25)
        type_layout.addWidget(self.spin_conf_type)

        self.btn_run_type = QPushButton("执行版面预测")
        self.btn_run_type.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 6px;")
        self.btn_run_type.clicked.connect(self.run_type_yolo)
        type_layout.addWidget(self.btn_run_type)
        
        type_layout.addSpacing(20)
        
        btn_set_text = QPushButton("将选中设为单行")
        btn_set_text.clicked.connect(lambda: self.change_selected_labels("text"))
        
        btn_set_subtext = QPushButton("将选中设为夹注")
        btn_set_subtext.clicked.connect(lambda: self.change_selected_labels("subText"))
        
        type_layout.addWidget(btn_set_text)
        type_layout.addWidget(btn_set_subtext)
        type_layout.addStretch()
        
        btn_save_crop = QPushButton("保存选中条框裁剪")
        btn_save_crop.clicked.connect(self.save_crop_image)
        btn_save_page = QPushButton("保存当前页面")
        btn_save_page.clicked.connect(self.save_page_image)
        
        type_layout.addWidget(btn_save_crop)
        type_layout.addWidget(btn_save_page)
        self.tabs.addTab(tab_type, "拆分页 (Type)")

    def _setup_slide_tab(self):
        tab_slide = QWidget()
        slide_layout = QHBoxLayout(tab_slide)

        btn_undo_slide = QPushButton("↩")
        btn_undo_slide.setFixedWidth(40)
        btn_undo_slide.clicked.connect(lambda: self.undo('slide'))
        
        btn_redo_slide = QPushButton("↪")
        btn_redo_slide.setFixedWidth(40)
        btn_redo_slide.clicked.connect(lambda: self.redo('slide'))
        
        slide_layout.addWidget(btn_undo_slide)
        slide_layout.addWidget(btn_redo_slide)

        slide_layout.addWidget(QLabel(" 自动拆分阈值 Conf:"))
        self.spin_conf_slide = QDoubleSpinBox()
        self.spin_conf_slide.setRange(0.01, 1.0)
        self.spin_conf_slide.setValue(0.25)
        slide_layout.addWidget(self.spin_conf_slide)

        self.btn_run_slide = QPushButton("手动切字预测")
        self.btn_run_slide.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 6px;")
        self.btn_run_slide.clicked.connect(self.run_slide_yolo)
        slide_layout.addWidget(self.btn_run_slide)
        slide_layout.addStretch()
        self.tabs.addTab(tab_slide, "检测页 (Slide)")

    def _setup_ocr_tab(self):
        tab_ocr = QWidget()
        ocr_layout = QVBoxLayout(tab_ocr)
        
        ref_bar = QHBoxLayout()
        self.btn_load_ref = QPushButton("导入单页/整书校本 (TXT)")
        self.btn_load_ref.setStyleSheet("background-color: #00BCD4; color: white; font-weight: bold;")
        self.btn_load_ref.clicked.connect(self.load_reference_text)
        
        self.lbl_ref_status = QLabel("尚未导入校本")
        
        self.btn_apply_ref = QPushButton("应用校本差异")
        self.btn_apply_ref.setStyleSheet("background-color: #FFC107; font-weight: bold;")
        self.btn_apply_ref.setEnabled(False)
        self.btn_apply_ref.clicked.connect(self.apply_reference_diffs)
        
        ref_bar.addWidget(self.btn_load_ref)
        ref_bar.addWidget(self.lbl_ref_status)
        ref_bar.addStretch()
        ref_bar.addWidget(self.btn_apply_ref)
        ocr_layout.addLayout(ref_bar)

        btn_bar = QHBoxLayout()
        self.btn_start_ocr = QPushButton("手动OCR本页")
        self.btn_start_ocr.setStyleSheet("background-color: #FF5722; color: white; font-weight: bold;")
        self.btn_start_ocr.clicked.connect(self.start_ocr_recognition)
        
        self.btn_batch_current_pdf = QPushButton("批量OCR识别本PDF")
        self.btn_batch_current_pdf.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold;")
        self.btn_batch_current_pdf.clicked.connect(self.start_current_pdf_ocr_batch)

        btn_bar.addWidget(self.btn_start_ocr)
        btn_bar.addWidget(self.btn_batch_current_pdf)
        btn_bar.addStretch()
        ocr_layout.addLayout(btn_bar)
        
        self.tabs.addTab(tab_ocr, "识别校对 (OCR)")
        
        self.reference_text = ""
    
    def load_reference_text(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "导入校本TXT", "", "Text Files (*.txt)")
        if not file_name: return
        with open(file_name, 'r', encoding='utf-8') as f:
            self.reference_text = f.read().replace('\n', '').replace(' ', '').replace('\u3000', '')
        self.lbl_ref_status.setText(f"已载入校本: {os.path.basename(file_name)} ({len(self.reference_text)}字)")
        self.compare_with_reference()

    def compare_with_reference(self):
        """双轨匹配算法：最长公共子串锚点法，无视首尾错漏，只抓高重合度段落"""
        if not getattr(self, 'reference_text', '') or self.current_mode != 'ocr': return
        
        type_boxes = self.get_sorted_type_boxes()
        slide_boxes =[item for item in self.viewer.scene.items() if isinstance(item, BoundingBox) and item.box_type == 'slide']
        if not type_boxes or not slide_boxes: return
        
        buffer_slides = []
        
        def process_buffer(slides):
            if not slides: return
            base_str = "".join([getattr(s, 'ocr_text', '？') for s in slides])
            length = len(base_str)
            if length < 3: return
            
            s_matcher = difflib.SequenceMatcher(None, self.reference_text, base_str)
            match = s_matcher.find_longest_match(0, len(self.reference_text), 0, length)
            
            if match.size >= max(2, length // 4):
                start_idx = match.a - match.b
                if start_idx >= 0 and start_idx + length <= len(self.reference_text):
                    best_match = self.reference_text[start_idx : start_idx + length]
                    sim = difflib.SequenceMatcher(None, base_str, best_match).ratio()
                    if sim >= 0.60:
                        for idx, s in enumerate(slides):
                            base_c = getattr(s, 'ocr_text', '？')

                            if idx < len(best_match):
                                ref_c = best_match[idx]
                            else:
                                ref_c = base_c  
                                
                            # 只有当底本字和校本字真实存在差异时才标黄处理
                            if base_c != ref_c:
                                s.ref_diff_char = ref_c
                                for item in self.ocr_scene.items():
                                    if isinstance(item, OCRTextItem) and item.source_box == s:
                                        item.setDefaultTextColor(QColor(255, 165, 0)) 
                                        item.ref_diff_char = ref_c

        for t in type_boxes:
            t_slides = [s for s in slide_boxes if s.sceneBoundingRect().intersected(t.sceneBoundingRect()).width() > 0]
            t_slides.sort(key=lambda b: b.sceneBoundingRect().center().y())
            
            if t.cls_name == 'subText':
                buffer_slides.extend(t_slides)
            else:
                if buffer_slides:
                    process_buffer(buffer_slides)
                    buffer_slides = []
                process_buffer(t_slides)
                
        if buffer_slides:
            process_buffer(buffer_slides)
            
        self.btn_apply_ref.setEnabled(True)

    def apply_reference_diffs(self):
        modified_count = 0
        for item in self.ocr_scene.items():
            if isinstance(item, OCRTextItem) and hasattr(item, 'ref_diff_char'):
                if item.toPlainText() != item.ref_diff_char:
                    item.commit_modification(item.ref_diff_char) 
                    modified_count += 1
        self.btn_apply_ref.setEnabled(False)
        self.show_msg(f"业据校本修改{modified_count} 处")

    def on_tab_changed(self, index):
        if hasattr(self, 'current_mode') and self.current_mode == 'ocr' and index != 2:
            self.save_current_page_ocr_txt()
        if not hasattr(self, 'viewer'): return
        
        if index == 0:
            self.current_mode = 'type'
            # 先关闭滚动条，避免 fitInView 计算错误
            self.viewer.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.viewer.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.viewer.setScene(self.viewer.scene)
            self.viewer.setAlignment(Qt.AlignCenter)
            self.set_boxes_visibility('type', True)
            self.set_boxes_visibility('slide', False)

            if self.display_image is not None:
                self.viewer.resetTransform()
                self.viewer.fitInView(self.viewer.sceneRect(), Qt.KeepAspectRatio)
                # 第二次延迟适应，彻底解决布局刷新问题
                QTimer.singleShot(10, lambda: self.viewer.fitInView(
                    self.viewer.sceneRect(), Qt.KeepAspectRatio))

        elif index == 1:
            self.current_mode = 'slide'
            self.viewer.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.viewer.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.viewer.setScene(self.viewer.scene)
            self.viewer.setAlignment(Qt.AlignCenter)
            self.set_boxes_visibility('type', False)
            self.set_boxes_visibility('slide', True)

            slide_boxes = [item for item in self.viewer.scene.items()
                        if isinstance(item, BoundingBox) and item.box_type == 'slide']
            if not slide_boxes:
                self.run_slide_yolo()

            if self.display_image is not None:
                self.viewer.resetTransform()
                self.viewer.fitInView(self.viewer.sceneRect(), Qt.KeepAspectRatio)
                QTimer.singleShot(10, lambda: self.viewer.fitInView(
                    self.viewer.sceneRect(), Qt.KeepAspectRatio))
        elif index == 2:
            self.current_mode = 'ocr'
            self.viewer.setScene(self.ocr_scene)

            self.viewer.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            self.viewer.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.viewer.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

            self.sort_and_number_slide_boxes()

            # ✅ 这行必须保留，否则后面 needs_ocr 会报错
            slide_boxes = [item for item in self.viewer.scene.items()
                        if isinstance(item, BoundingBox) and item.box_type == 'slide']

            self.generate_ocr_layout()

            self.viewer.resetTransform()
            view_h = self.viewer.viewport().height()
            scene_rect = self.ocr_scene.sceneRect()

            if scene_rect.height() > 0:
                scale = (view_h - 40) / scene_rect.height()
                self.viewer.scale(scale, scale)

            # 修复滚动条：强制设置场景边界，并让视口从最左端开始显示
            self.viewer.setSceneRect(scene_rect)
            self.viewer.ensureVisible(scene_rect.right(), scene_rect.top(), 1, 1)

            QApplication.processEvents()
            self.viewer.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

            needs_ocr = (slide_boxes and
                        all(not getattr(s, 'ocr_text', '').strip() for s in slide_boxes))
            if needs_ocr:
                self.start_ocr_recognition()
            else:
                self.compare_with_reference()

    def get_current_mode(self):
        if self.current_mode == 'ocr': return 'ocr'
        return self.current_mode

    def set_boxes_visibility(self, box_type, visible):
        for item in self.viewer.scene.items():
            if isinstance(item, BoundingBox) and item.box_type == box_type: item.setVisible(visible)

    def clear_boxes(self, target_type=None):
        for item in self.viewer.scene.items():
            if isinstance(item, BoundingBox):
                if target_type is None or item.box_type == target_type: self.viewer.scene.removeItem(item)
        self.trigger_auto_sort_and_snapshot(target_type)

    def toggle_labels(self):
        self.show_labels = not self.show_labels
        self.btn_toggle_labels.setText("隐藏数字" if self.show_labels else "显示数字")
        for item in self.viewer.scene.items():
            if isinstance(item, BoundingBox): item.id_bg.setVisible(self.show_labels if item.id_num > 0 else False)

    def change_selected_labels(self, label):
        selected = self.viewer.scene.selectedItems()
        changed = False
        for item in selected:
            if isinstance(item, BoundingBox) and item.box_type == 'type':
                item.set_label(label)
                changed = True
        if changed: self.trigger_auto_sort_and_snapshot('type')

    def start_manual_sort(self, boxes, mode):
        self.sort_mode[mode] = 'manual'
        self.manual_sort_pending = boxes
        self.manual_sort_ids =[b.id_num for b in boxes]
        self.manual_sort_backup = {b: b.id_num for b in boxes}
        for b in boxes: b.set_id(0)
        self.set_ui_disabled_for_manual_sort(True)
        self.show_msg("依次点击框以排序")

    def cancel_manual_sort(self):
        if not self.manual_sort_pending: return
        for b in self.manual_sort_pending: b.set_id(self.manual_sort_backup[b])
        self.manual_sort_pending =[]
        self.manual_sort_ids =[]
        self.manual_sort_backup = {}
        self.set_ui_disabled_for_manual_sort(False)
        self.show_msg("已取消手动排序")

    def restore_auto_sort(self, mode):
        self.sort_mode[mode] = 'auto'
        self.trigger_auto_sort_and_snapshot(mode)
        self.show_msg("已恢复自动排序模式并重新整理编号")

    def set_ui_disabled_for_manual_sort(self, disabled):
        self.btn_prev.setEnabled(not disabled and self.pdf_doc is not None and self.current_page > 0)
        self.btn_next.setEnabled(not disabled and self.pdf_doc is not None and self.current_page < self.total_pages - 1)
        self.input_page.setEnabled(not disabled)
        self.btn_batch.setEnabled(not disabled)
        self.btn_run_type.setEnabled(not disabled)
        self.btn_run_slide.setEnabled(not disabled)

    def trigger_auto_sort_and_snapshot(self, mode):
        if self.sort_mode.get(mode, 'auto') == 'auto':
            if mode == 'type': self.sort_and_number_type_boxes()
            elif mode == 'slide': self.sort_and_number_slide_boxes()
        self.snapshot(mode)

    def get_sorted_type_boxes(self):
        type_boxes =[item for item in self.viewer.scene.items() if isinstance(item, BoundingBox) and item.box_type == 'type']
        if not type_boxes: return[]
        type_boxes.sort(key=lambda b: b.sceneBoundingRect().top())
        columns =[]
        global_last_text_width = None
        for box in type_boxes:
            b_rect = box.sceneBoundingRect()
            b_left, b_right, b_width = b_rect.left(), b_rect.right(), b_rect.width()
            best_col, max_overlap = None, 0.0
            for col in columns:
                ref_text_box = next((b for b in reversed(col) if b.cls_name != 'subText'), None)
                if ref_text_box:
                    ref_left = ref_text_box.sceneBoundingRect().left()
                    ref_width = ref_text_box.sceneBoundingRect().width()
                else:
                    ref_sub = col[-1]
                    ref_width = global_last_text_width if global_last_text_width else ref_sub.sceneBoundingRect().width() * 2
                    ref_left = ref_sub.sceneBoundingRect().center().x() - ref_width / 2.0
                ref_right = ref_left + ref_width
                inter_width = max(0, min(b_right, ref_right) - max(b_left, ref_left))
                if b_width > 0:
                    overlap_ratio = inter_width / b_width
                    if overlap_ratio > 0.4 and overlap_ratio > max_overlap:
                        max_overlap, best_col = overlap_ratio, col
            if best_col is not None: best_col.append(box)
            else: columns.append([box])
            if box.cls_name != 'subText': global_last_text_width = box.sceneBoundingRect().width()
        columns.sort(key=lambda col: sum(b.sceneBoundingRect().center().x() for b in col) / len(col), reverse=True)
        sorted_boxes =[]
        for col in columns:
            rows =[]
            for box in col:
                is_sub, added = (box.cls_name == 'subText'), False
                if rows and is_sub:
                    last_row = rows[-1]
                    for ref_box in last_row:
                        if ref_box.cls_name == 'subText':
                            t1, b1 = box.sceneBoundingRect().top(), box.sceneBoundingRect().bottom()
                            t2, b2 = ref_box.sceneBoundingRect().top(), ref_box.sceneBoundingRect().bottom()
                            min_h = min(b1 - t1, b2 - t2)
                            if min_h > 0 and (max(0, min(b1, b2) - max(t1, t2)) / min_h) > 0.3:
                                last_row.append(box); added = True; break
                if not added: rows.append([box])
            for row in rows:
                if len(row) > 1: row.sort(key=lambda b: b.sceneBoundingRect().center().x(), reverse=True)
                sorted_boxes.extend(row)
        return sorted_boxes

    def sort_and_number_type_boxes(self):
        sorted_boxes = self.get_sorted_type_boxes()
        for idx, box in enumerate(sorted_boxes, start=1):
            box.set_id(idx)
            box.id_bg.setVisible(self.show_labels)

    def sort_and_number_slide_boxes(self):
        type_boxes = self.get_sorted_type_boxes()
        slide_boxes =[item for item in self.viewer.scene.items() if isinstance(item, BoundingBox) and item.box_type == 'slide']
        if not type_boxes or not slide_boxes: return
        
        from collections import defaultdict
        groups = defaultdict(list)
        for s in slide_boxes:
            s_rect, best_t, max_area = s.sceneBoundingRect(), None, 0
            for t in type_boxes:
                inter = s_rect.intersected(t.sceneBoundingRect())
                area = inter.width() * inter.height()
                if area > max_area: 
                    max_area = area
                    best_t = t
            if best_t: 
                groups[best_t].append(s)
                
        global_idx = 1
        self.sorted_boxes_data = []
        for t in type_boxes:
            slide_list = groups.get(t, [])
            slide_list.sort(key=lambda b: b.sceneBoundingRect().center().y())
            
            local_idx = 1
            for s in slide_list:
                s.set_id(global_idx, local_idx)
                s.id_bg.setVisible(self.show_labels)
                self.sorted_boxes_data.append({
                    'rect': s.sceneBoundingRect(),
                    'id': global_idx
                })
                global_idx += 1
                local_idx += 1

    def snapshot(self, mode, force_init=False):
        boxes =[item for item in self.viewer.scene.items() if isinstance(item, BoundingBox) and item.box_type == mode]
        state = [b.to_dict() for b in boxes]
        if force_init:
            self.history[mode]['undo'] = [state]
            self.history[mode]['redo'] = []
        else:
            self.history[mode]['undo'].append(state)
            self.history[mode]['redo'].clear()

    def undo(self, mode):
        if len(self.history[mode]['undo']) > 1:
            curr_state = self.history[mode]['undo'].pop()
            self.history[mode]['redo'].append(curr_state)
            self.restore_state(mode, self.history[mode]['undo'][-1])

    def redo(self, mode):
        if self.history[mode]['redo']:
            next_state = self.history[mode]['redo'].pop()
            self.history[mode]['undo'].append(next_state)
            self.restore_state(mode, next_state)

    def restore_state(self, mode, state_list):
        for item in self.viewer.scene.items():
            if isinstance(item, BoundingBox) and item.box_type == mode: 
                self.viewer.scene.removeItem(item)
        img_h = self.display_image.shape[0] if self.display_image is not None else 4000
        for d in state_list:
            box = BoundingBox.from_dict(d, img_h)
            box.id_bg.setVisible(self.show_labels)
            self.viewer.scene.addItem(box)
        self.set_boxes_visibility(mode, self.current_mode == mode)

    def run_type_yolo(self):
        if self.display_image is None or self.type_model is None: 
            return self.show_msg("⚠️尚未加载图片或模型！")
        self.btn_run_type.setText("检测中...")
        self.tabs.setEnabled(False)
        self.type_thread = YOLOWorkerType(self.type_model, self.display_image, self.spin_conf_type.value(), 0.45)
        self.type_thread.finished_signal.connect(self.on_type_finished)
        self.type_thread.start()

    def on_type_finished(self, box_data):
        self.btn_run_type.setText("执行版面预测")
        self.tabs.setEnabled(True)
        self.clear_boxes('type')
        self.clear_boxes('slide')
        img_h = self.display_image.shape[0] if self.display_image is not None else 4000
        for x1, y1, x2, y2, cls_name, conf in box_data:
            if 'text' not in cls_name.lower(): cls_name = 'text' 
            box = BoundingBox(QRectF(x1, y1, x2 - x1, y2 - y1), img_height=img_h, cls_name=cls_name, conf=conf, box_type='type')
            box.id_bg.setVisible(self.show_labels)
            self.viewer.scene.addItem(box)
        self.sort_mode['type'] = 'auto'
        self.trigger_auto_sort_and_snapshot('type')
        self.show_msg("版面预测已完成，自动排序已重置")

    def run_slide_yolo(self):
        if self.display_image is None or self.slide_model is None: 
            return self.show_msg("⚠️尚未加载图片或模型！")
        line_boxes = self.get_sorted_type_boxes()
        if not line_boxes: 
            return self.show_msg("⚠️尚未找到任何条框！请先执行版面拆分。")
        line_rects =[(b.id_num, b.rect().x(), b.rect().y(), b.rect().right(), b.rect().bottom()) for b in line_boxes]
        self.btn_run_slide.setText("切字拆分中...")
        self.tabs.setEnabled(False)
        self.slide_thread = YOLOWorkerSlide(self.slide_model, self.display_image, line_rects, self.spin_conf_slide.value(), 0.45)
        self.slide_thread.finished_signal.connect(self.on_slide_finished)
        self.slide_thread.start()

    def on_slide_finished(self, box_data):
        self.btn_run_slide.setText("手动切字预测")
        self.tabs.setEnabled(True)
        self.clear_boxes('slide') 
        img_h = self.display_image.shape[0] if self.display_image is not None else 4000
        for cd in box_data:
            box = BoundingBox(QRectF(cd[1], cd[2], cd[3] - cd[1], cd[4] - cd[2]), img_height=img_h, cls_name="text", conf=cd[6], box_type='slide')
            box.id_bg.setVisible(self.show_labels)
            self.viewer.scene.addItem(box)
        self.sort_mode['slide'] = 'auto'
        self.trigger_auto_sort_and_snapshot('slide')
        self.show_msg("单字拆分已完成，自动排序已重置")
        self.tabs.setTabEnabled(2, True)

    def run_batch_inference(self):
        if not self.pdf_doc or not self.json_path:
            return self.show_msg("提示: 请先导入一份PDF文件。")
        self.set_ui_disabled_for_manual_sort(True)
        self.show_msg("开始后台批量推理PDF全册，操作已被锁定，请耐心等待...")
        self.batch_worker = BatchInferWorker(self.pdf_doc.name, self.type_model, self.slide_model, self.spin_conf_type.value(), self.spin_conf_slide.value())
        self.batch_worker.progress_signal.connect(lambda cur, tot: self.show_msg(f"后台批量推理中... 进度: {cur} / {tot} 页"))
        self.batch_worker.finished_signal.connect(self.on_batch_finished)
        self.batch_worker.start()

    def generate_ocr_layout(self):
        """精准排版引擎：动态计算场景总宽度，杜绝边界截断和无法滑动的问题"""
        self.ocr_scene.clear()
        
        type_boxes = self.get_sorted_type_boxes()
        if not type_boxes: return
            
        slide_boxes =[item for item in self.viewer.scene.items() if isinstance(item, BoundingBox) and item.box_type == 'slide']
        lines = {t:[] for t in type_boxes}
        for s in slide_boxes:
            s_rect = s.sceneBoundingRect()
            best_t, max_area = None, 0
            for t in type_boxes:
                inter = s_rect.intersected(t.sceneBoundingRect())
                area = inter.width() * inter.height()
                if area > max_area:
                    max_area = area
                    best_t = t
            if best_t:
                lines[best_t].append(s)
        for t in type_boxes:
            lines[t].sort(key=lambda b: b.sceneBoundingRect().center().y())

        max_h = 0
        slicedata =[]
        for t in type_boxes:
            t_rect = t.sceneBoundingRect()
            x1, y1, x2, y2 = int(t_rect.left()), int(t_rect.top()), int(t_rect.right()), int(t_rect.bottom())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.display_image.shape[1], x2), min(self.display_image.shape[0], y2)
            if x2 <= x1 or y2 <= y1: continue
            img_w = x2 - x1
            img_h = y2 - y1
            if img_h > max_h: max_h = img_h
                
            text_w = max(80, int(img_w * 1.5))
            slicedata.append({'t': t, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'img_w': img_w, 'img_h': img_h, 'text_w': text_w})

        if not slicedata: return

        self.viewer.max_type_height = max_h
        scale_factor = max(0.05, max_h / 4000.0)
        id_font_size = max(10, int(35 * scale_factor))

        TEXT_GAP = 15              
        MARGIN_RIGHT = 300         # ✅ 大幅增加右侧宽容度留白，防止右侧贴边无法拖拽
        LEFT_MARGIN = 300          # ✅ 大幅增加左侧留白
        COLUMN_GAP = 60            

        # 1. 严格累加计算画布总宽度
        total_width = MARGIN_RIGHT
        for sd in slicedata:
            total_width += sd['img_w'] + TEXT_GAP + sd['text_w'] + COLUMN_GAP
        total_width += LEFT_MARGIN

        scene_height = max_h + 100
        base_y = max(0, (scene_height - max_h) // 2)
        
        # ✅ 将画布原点定为 0，并且强制更新矩形边界
        self.ocr_scene.setSceneRect(0, 0, total_width, scene_height)

        current_right = total_width - MARGIN_RIGHT
        col_index = 1

        for i, sd in enumerate(slicedata):
            img_w = sd['img_w']
            text_w = sd['text_w']
            t = sd['t']
            x1, y1, x2, y2 = sd['x1'], sd['y1'], sd['x2'], sd['y2']
            
            text_x = current_right - text_w
            img_right = text_x - TEXT_GAP
            img_left = img_right - img_w

            num_item = QGraphicsTextItem(str(col_index))
            num_item.setFont(QFont("Arial", id_font_size, QFont.Bold))
            num_item.setDefaultTextColor(QColor(180, 0, 0))
            
            bg_rect = QGraphicsRectItem()
            bg_rect.setBrush(QBrush(QColor(255, 255, 255, 200)))
            bg_rect.setPen(QPen(Qt.NoPen))
            
            num_item.setPos(img_left, base_y)
            text_width = num_item.boundingRect().width()
            text_height = num_item.boundingRect().height()
            bg_rect.setRect(0, 0, text_width + 4, text_height + 2)
            bg_rect.setPos(img_left - 2, base_y - 1)
            
            self.ocr_scene.addItem(bg_rect)
            self.ocr_scene.addItem(num_item)
            col_index += 1

            crop_img = self.display_image[y1:y2, x1:x2]
            qimg = self.mat_to_qimage(crop_img)
            pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
            pixmap_item.setPos(img_left, base_y + 25)
            self.ocr_scene.addItem(pixmap_item)

            for s in lines[t]:
                s_rect = s.sceneBoundingRect()
                rel_x = s_rect.left() - x1
                rel_y = s_rect.top() - y1
                
                rect_item = QGraphicsRectItem(QRectF(0, 0, s_rect.width(), s_rect.height()), pixmap_item)
                rect_item.setPos(rel_x, rel_y)
                rect_item.setPen(QPen(QColor(255, 0, 0, 180), 1))
                
                text = getattr(s, 'ocr_text', '')
                conf = getattr(s, 'ocr_conf', 0.0)
                if not text.strip():
                    text = ""
                    conf = 0.0
                    
                ocr_item = OCRTextItem(text, conf, s_rect.height(), source_box=s, font_families=self.font_families)
                ocr_item.setPos(text_x, base_y + 25 + rel_y)
                self.ocr_scene.addItem(ocr_item)

            # 更新下一组基准边
            current_right = img_left - COLUMN_GAP

    def start_ocr_recognition(self):
        if self.display_image is None: return
            
        self.sort_and_number_slide_boxes()  
        if not self.sorted_boxes_data:
            self.show_msg("当前页面没有单字切片，请先运行Slide")
            return
        
        self.progress_dialog = QProgressDialog("正在进行单字 OCR 识别...", "取消", 0, len(self.sorted_boxes_data), self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        
        self.ocr_thread = OCRWorker(self.sorted_boxes_data, self.display_image, self.ocr_model_path)
        self.ocr_thread.progress_signal.connect(self.progress_dialog.setValue)
        self.ocr_thread.result_signal.connect(self.on_ocr_result)
        self.ocr_thread.error_signal.connect(lambda msg: QMessageBox.critical(self, "错误", msg))
        self.ocr_thread.finished_signal.connect(self.on_ocr_finished_cleanup)
        self.progress_dialog.canceled.connect(self.ocr_thread.stop)
        self.ocr_thread.start()
        self.btn_start_ocr.setEnabled(False)
    
    def start_current_pdf_ocr_batch(self):
        if not self.pdf_doc or not self.json_path:
            return self.show_msg("提示: 请先导入PDF文件。")
            
        reply = QMessageBox.question(self, "确认强制重跑", "此操作将无视现有 OCR 数据，强制从头识别当前整本古籍的所有单字。\n耗时可能较长，是否继续？", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No: return
            
        self.save_current_page_to_json()
        
        self.progress_dialog = QProgressDialog("正在强制批量识别全册单字...", "取消", 0, self.total_pages, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        
        self.btn_batch_current_pdf.setEnabled(False)
        
        self.current_book_ocr_thread = CurrentBookOCRWorker(self.pdf_doc.name, self.project_data, self.ocr_model_path)
        self.current_book_ocr_thread.progress_signal.connect(lambda cur, tot, msg: self.progress_dialog.setValue(cur))
        self.current_book_ocr_thread.finished_signal.connect(self.on_current_pdf_ocr_finished)
        self.current_book_ocr_thread.error_signal.connect(lambda msg: QMessageBox.critical(self, "错误", msg))
        self.progress_dialog.canceled.connect(self.current_book_ocr_thread.stop)
        self.current_book_ocr_thread.start()

    def on_current_pdf_ocr_finished(self, updated_data):
        self.btn_batch_current_pdf.setEnabled(True)
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
            
        self.project_data = updated_data
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.project_data, f, ensure_ascii=False, indent=2)
            
        self.save_current_page_ocr_txt() 
        self.load_pdf_page() 
        self.show_msg("✅ 本册 PDF 全页批量 OCR 识别并保存完成！")

    def on_ocr_finished_cleanup(self):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        self.on_ocr_finished()
    
    def on_ocr_result(self, box_id, text, confidence):
        for item in self.viewer.scene.items():
            if isinstance(item, BoundingBox) and item.box_type == 'slide' and item.id_num == box_id:
                item.ocr_text = text if text.strip() else ""
                item.ocr_conf = confidence if text.strip() else 0.0
                break

    def on_ocr_finished(self):
        self.btn_start_ocr.setEnabled(True)
        self.show_msg("当前页单字 OCR 识别完成")
        self.save_current_page_to_json()
        if self.current_mode == 'ocr':
            self.generate_ocr_layout()

    def start_global_batch(self):
        folder_path = QFileDialog.getExistingDirectory(self, "请选择需要批量扫描的根文件夹")
        if not folder_path: return
        pdf_paths =[]
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith('.pdf'):
                    pdf_paths.append(os.path.join(root, f))
        if not pdf_paths:
            return self.show_msg("⚠️未在选中文件夹内找到任何 PDF 文件。")
        self.btn_global_batch.setVisible(False)
        self.global_progress.setVisible(True)
        self.btn_stop_global.setVisible(True)
        self.show_msg(f"开始后台扫描并推理 {len(pdf_paths)} 个 PDF")
        self.global_worker = GlobalBatchInferWorker(pdf_paths, self.type_model, self.slide_model, self.spin_conf_type.value(), self.spin_conf_slide.value())
        self.global_worker.progress_signal.connect(self.update_global_batch_progress)
        self.global_worker.finished_signal.connect(self.on_global_batch_finished)
        self.global_worker.start()
    
    def start_global_batch_ocr(self):
        folder_path = QFileDialog.getExistingDirectory(self, "请选择需要批量OCR识别的根文件夹")
        if not folder_path: return
        
        self.btn_batch_ocr.setVisible(False)
        self.btn_global_batch.setVisible(False)
        self.global_progress.setVisible(True)
        self.btn_stop_global.setVisible(True)
        self.show_msg(f"开始后台扫描并执行批量 OCR 推理")
        
        self.global_ocr_worker = GlobalBatchOCRWorker(folder_path, self.ocr_model_path)
        self.global_ocr_worker.progress_signal.connect(self.update_global_batch_progress)
        self.global_ocr_worker.finished_signal.connect(self.on_global_batch_ocr_finished)
        self.global_ocr_worker.start()

    def on_global_batch_ocr_finished(self, processed_count):
        self.global_progress.setVisible(False)
        self.btn_stop_global.setVisible(False)
        self.btn_batch_ocr.setVisible(True)
        self.btn_global_batch.setVisible(True)
        self.show_msg(f"批量 OCR 推理结束！共更新了 {processed_count} 本古籍的 TXT 排版内容。")

    def update_global_batch_progress(self, file_idx, total_files, filename, page_idx, total_pages):
        short_name = filename if len(filename) < 10 else filename[:8] + ".."
        self.global_progress.setFormat(f"({file_idx}/{total_files}) {short_name} [{page_idx}/{total_pages}页]")
        self.global_progress.setValue(int((page_idx / total_pages) * 100))

    def stop_global_batch(self):
        if hasattr(self, 'global_worker') and self.global_worker.isRunning():
            self.global_worker.stop()
        if hasattr(self, 'global_ocr_worker') and self.global_ocr_worker.isRunning():
            self.global_ocr_worker.stop()
        self.btn_stop_global.setText("正在停止...")
        self.btn_stop_global.setEnabled(False)

    def on_global_batch_finished(self, processed_count):
        self.global_progress.setVisible(False)
        self.btn_stop_global.setVisible(False)
        self.btn_global_batch.setVisible(True)
        self.show_msg(f"后台扫描推理结束！共成功生成了 {processed_count} 个全新的 JSON 标注文件。")

    def on_batch_finished(self, result_dict):
        self.set_ui_disabled_for_manual_sort(False)
        self.project_data = result_dict
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.project_data, f, ensure_ascii=False, indent=2)
        self.show_msg(f"批量推理全部完成！JSON 数据已保存至：{self.json_path}")
        self.load_page_from_json()

    def save_current_page_to_json(self):
        if not self.json_path: return
        
        if self.current_mode == 'ocr':
            self.ocr_scene.clearFocus()

        page_data = {'type':[], 'slide':[]}
        for item in self.viewer.scene.items():
            if isinstance(item, BoundingBox): 
                page_data[item.box_type].append(item.to_dict())
                
        old_data = self.project_data.get(str(self.current_page))
        if old_data and json.dumps(old_data) == json.dumps(page_data):
            return

        self.project_data[str(self.current_page)] = page_data
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.project_data, f, ensure_ascii=False, indent=2)
        
    def auto_export_single_sample(self, s_box, text, custom_folder=None):
        if self.cv_image is None or not text.strip(): return
        try:
            dataset_dir = custom_folder if custom_folder else os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune_dataset")
            img_dir = os.path.join(dataset_dir, "images")
            os.makedirs(img_dir, exist_ok=True)
            gt_file = os.path.join(dataset_dir, "rec_gt.txt")
            
            rect = s_box.sceneBoundingRect()
            x, y, w, h = int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())
            y1, y2 = max(0, y), min(self.cv_image.shape[0], y+h)
            x1, x2 = max(0, x), min(self.cv_image.shape[1], x+w)
            
            crop_img = self.cv_image[y1:y2, x1:x2]
            if crop_img.size > 0:
                img_name = f"page{self.current_page}_box{s_box.id_num}.jpg"
                cv2.imwrite(os.path.join(img_dir, img_name), crop_img)
                with open(gt_file, 'a', encoding='utf-8') as f:
                    f.write(f"images/{img_name}\t{text}\n")
        except Exception as e:
            print(f"提取微调数据失败: {e}")

    def save_current_page_ocr_txt(self):
        if not self.json_path: return
        self.save_current_page_to_json() 
        
        txt_path = self.json_path.replace("_project.json", "_ocr.txt")
        if not txt_path.endswith("_ocr.txt"):
            txt_path = os.path.splitext(self.json_path)[0] + "_ocr.txt"
            
        all_lines =[]
        for p_idx in range(self.total_pages):
            page_str = str(p_idx)
            if page_str not in self.project_data: continue
            
            p_data = self.project_data[page_str]
            t_list = p_data.get('type',[])
            s_list = p_data.get('slide',[])
            if not t_list: continue
            
            t_list_sorted = sorted(t_list, key=lambda b: b[0] + b[2]/2.0, reverse=True)
            all_lines.append(f"[{p_idx + 1}]")
            
            in_subtext_block = False
            for t in t_list_sorted:
                t_cls = t[4] 
                t_rect = QRectF(t[0], t[1], t[2], t[3])
                col_slides = []
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

    def load_page_from_json(self):
        self.viewer.scene.clear() 
        self.viewer.current_img_height = self.display_image.shape[0]
        self.image_item = self.viewer.scene.addPixmap(QPixmap.fromImage(self.mat_to_qimage(self.display_image)))
        self.image_item.setOpacity(self.slider_opacity.value() / 100.0)
        self.viewer.setSceneRect(0, 0, self.display_image.shape[1], self.display_image.shape[0])
        self.viewer.fitInView(self.viewer.sceneRect(), Qt.KeepAspectRatio)
        page_str = str(self.current_page)
        if page_str in self.project_data:
            img_h = self.display_image.shape[0]
            for t_dict in self.project_data[page_str].get('type',[]):
                box = BoundingBox.from_dict(t_dict, img_h)
                box.id_bg.setVisible(self.show_labels)
                self.viewer.scene.addItem(box)
            for s_dict in self.project_data[page_str].get('slide',[]):
                box = BoundingBox.from_dict(s_dict, img_h)
                box.id_bg.setVisible(self.show_labels)
                self.viewer.scene.addItem(box)
        self.snapshot('type', force_init=True)
        self.snapshot('slide', force_init=True)
        self.set_boxes_visibility('type', self.current_mode == 'type')
        self.set_boxes_visibility('slide', self.current_mode == 'slide')

    def save_page_image(self):
        if self.display_image is None: return
        path = os.path.join(CONFIG["SAVE_PAGE_DIR"], f"page_{self.current_page + 1}.jpg")
        cv2.imwrite(path, self.display_image)
        self.show_msg(f"已保存当前页图片至: {path}")

    def save_crop_image(self):
        if self.display_image is None: return
        selected =[item for item in self.viewer.scene.selectedItems() if isinstance(item, BoundingBox) and item.box_type == 'type']
        if not selected: return self.show_msg("⚠️请先框选需要保存的长条！")
        for box in selected:
            r = box.rect()
            x1, y1 = max(0, int(r.x())), max(0, int(r.y()))
            x2, y2 = min(self.display_image.shape[1], int(r.right())), min(self.display_image.shape[0], int(r.bottom()))
            crop = self.display_image[y1:y2, x1:x2]
            if crop.size > 0:
                cv2.imwrite(os.path.join(CONFIG["SAVE_CROP_DIR"], f"crop_page{self.current_page + 1}_id{box.id_num}_{box.cls_name}.jpg"), crop)
        self.show_msg(f"已裁剪并保存选中的条框至: {CONFIG['SAVE_CROP_DIR']}")

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "打开文件", "", "Images/PDF (*.png *.jpg *.jpeg *.pdf)")
        if not file_name: return
        self.project_data = {}
        self.json_path = os.path.splitext(file_name)[0] + "_project.json"
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r', encoding='utf-8') as f: 
                self.project_data = json.load(f)
        if file_name.lower().endswith('.pdf'):
            self.pdf_doc = fitz.open(file_name)
            self.total_pages = len(self.pdf_doc)
            self.current_page = 0
            self.btn_prev.setEnabled(True)
            self.btn_next.setEnabled(True)
            self.load_pdf_page()
        else:
            self.pdf_doc = None
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            self.input_page.setText("1")
            self.lbl_total_pages.setText(" / 1")
            self.cv_image = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.display_image = self.cv_image.copy()
            self.load_page_from_json()
        self.show_msg(f"✅ 成功加载: {file_name}")

    def load_pdf_page(self):
        if not self.pdf_doc: return
        page = self.pdf_doc.load_page(self.current_page)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        self.cv_image = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_BGRA2BGR)
        self.display_image = self.cv_image.copy()
        self.input_page.setText(str(self.current_page + 1))
        self.lbl_total_pages.setText(f" / {self.total_pages}")
        
        self.load_page_from_json()
        
        type_boxes =[item for item in self.viewer.scene.items() if isinstance(item, BoundingBox) and item.box_type == 'type']
        slide_boxes =[item for item in self.viewer.scene.items() if isinstance(item, BoundingBox) and item.box_type == 'slide']
        
        self.tabs.setTabEnabled(2, len(slide_boxes) > 0)
        
        if self.current_mode == 'ocr':
            has_ocr_data = any(getattr(s, 'ocr_text', '').strip() for s in slide_boxes)
            if has_ocr_data:
                self.tabs.setCurrentIndex(2) 
                self.on_tab_changed(2)
            elif len(slide_boxes) > 0:
                self.tabs.setCurrentIndex(2)  
                self.on_tab_changed(2)
            elif len(type_boxes) > 0:
                self.tabs.setCurrentIndex(1)  
            else:
                self.tabs.setCurrentIndex(0)  
        else:
            if str(self.current_page) not in self.project_data:
                self.tabs.setCurrentIndex(0)

    def prev_page(self):
        if self.pdf_doc and self.current_page > 0:
            self.save_current_page_to_json()
            self.save_current_page_ocr_txt()
            self.current_page -= 1
            self.load_pdf_page()

    def next_page(self):
        if self.pdf_doc and self.current_page < self.total_pages - 1:
            self.save_current_page_to_json()
            self.save_current_page_ocr_txt()
            self.current_page += 1
            self.load_pdf_page()

    def jump_page(self):
        if not self.pdf_doc: return
        try:
            target = int(self.input_page.text()) - 1
            if 0 <= target < self.total_pages and target != self.current_page:
                self.save_current_page_to_json()
                self.save_current_page_ocr_txt()
                self.current_page = target
                self.load_pdf_page()
            else:
                self.input_page.setText(str(self.current_page + 1))
        except ValueError:
            self.input_page.setText(str(self.current_page + 1))

    def change_image_opacity(self, value):
        if hasattr(self, 'image_item') and self.image_item: 
            self.image_item.setOpacity(value / 100.0)

    def mat_to_qimage(self, mat):
        if mat is None: return QImage()
        mat_rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        h, w = mat_rgb.shape[:2]
        return QImage(mat_rgb.data, w, h, 3 * w, QImage.Format_RGB888)


if __name__ == '__main__':
    mp.freeze_support() 
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    window = YOLOStageTester()
    window.show()
    sys.exit(app.exec_())