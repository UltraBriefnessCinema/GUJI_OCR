#画布相关自定义控件 (BoundingBox, OCRTextItem, ImageViewer)

import cv2
from PyQt5.QtGui import QPen, QColor, QBrush, QFont, QPainter, QPixmap, QImage
from PyQt5.QtWidgets import (QGraphicsRectItem, QGraphicsTextItem, QGraphicsView,
                             QGraphicsScene, QGraphicsItem, QMenu, QFileDialog, QApplication, QGraphicsPixmapItem)
from PyQt5.QtCore import Qt, QRectF, QPointF, QTimer
from config import get_class_color

class BoundingBox(QGraphicsRectItem):
    def __init__(self, rect, img_height=4000, cls_name="text", conf=1.0, box_type='type', id_num=0, local_id=0):
        super().__init__(rect)
        self.img_height = img_height 
        self.cls_name = cls_name
        self.conf = conf
        self.box_type = box_type 
        self.id_num = id_num
        self.local_id = local_id if local_id > 0 else id_num
        self.ocr_text = ""  
        self.ocr_conf = 0.0
        self.parent_type_rect = None
        
        self.setFlags(QGraphicsRectItem.ItemIsSelectable | QGraphicsRectItem.ItemIsMovable | QGraphicsRectItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        
        scale_factor = max(0.05, self.img_height / 4000.0)
        pen_width = max(2, int(4 * scale_factor)) if self.box_type == 'type' else max(1, int(2 * scale_factor))
        id_font_size = max(14, int(60 * scale_factor)) if self.box_type == 'type' else max(10, int(35 * scale_factor))
        
        self.base_color = get_class_color(self.cls_name)
        self.normal_pen = QPen(self.base_color, pen_width)
        self.hover_pen = QPen(QColor(255, 0, 0), pen_width + 2)
        
        fill_color = QColor(self.base_color)
        fill_color.setAlpha(40)
        self.normal_brush = QBrush(fill_color)
        
        hover_fill = QColor(self.base_color)
        hover_fill.setAlpha(120) 
        self.hover_brush = QBrush(hover_fill)
        
        self.setPen(self.normal_pen)
        self.setBrush(self.normal_brush)
        
        self.id_bg = QGraphicsRectItem(self)
        id_color = QColor(255, 255, 255)
        id_color.setAlpha(220) 
        self.id_bg.setBrush(QBrush(id_color))
        self.id_bg.setPen(QPen(QColor(255, 0, 0), 2)) 
        
        self.id_text_item = QGraphicsTextItem(self.id_bg)
        self.id_text_item.setDefaultTextColor(QColor(255, 0, 0)) 
        self.id_text_item.setFont(QFont("Arial", id_font_size, QFont.Bold))
        
        self.margin = max(3, int(6 * scale_factor))
        self.resize_mode = None
        self.update_visuals()

    def _get_main_window(self):
        if self.scene() and self.scene().views(): return self.scene().views()[0].window()
        return None

    def to_dict(self):
        r = self.rect()
        return [float(r.x()), float(r.y()), float(r.width()), float(r.height()), 
                self.cls_name, self.conf, self.box_type, self.id_num, 
                self.ocr_text, self.ocr_conf, self.local_id]

    @classmethod
    def from_dict(cls, data, img_height):
        if len(data) == 8:
            x, y, w, h, cls_name, conf, box_type, id_num = data
            ocr_text, ocr_conf, local_id = "", 0.0, id_num
        elif len(data) == 10:
            x, y, w, h, cls_name, conf, box_type, id_num, ocr_text, ocr_conf = data
            local_id = id_num
        else:
            x, y, w, h, cls_name, conf, box_type, id_num, ocr_text, ocr_conf, local_id = data[:11]
            
        box = cls(QRectF(x, y, w, h), img_height, cls_name, conf, box_type, id_num, local_id)
        box.ocr_text = ocr_text
        box.ocr_conf = ocr_conf
        return box
    
    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            if value: 
                sel_color = QColor(255, 0, 0)
                sel_color.setAlpha(100)
                self.setBrush(QBrush(sel_color))
                self.setPen(QPen(Qt.red, self.normal_pen.width() + 2))
            else: 
                self.setBrush(self.normal_brush)
                self.setPen(self.normal_pen)
        elif change == QGraphicsItem.ItemPositionChange and self.box_type == 'slide' and getattr(self, 'parent_type_rect', None):
            new_pos = value
            rect = self.rect()
            scene_l = rect.left() + new_pos.x()
            scene_r = rect.right() + new_pos.x()
            scene_t = rect.top() + new_pos.y()
            scene_b = rect.bottom() + new_pos.y()
            dx, dy = new_pos.x(), new_pos.y()
            if scene_l < self.parent_type_rect.left(): dx = self.parent_type_rect.left() - rect.left()
            elif scene_r > self.parent_type_rect.right(): dx = self.parent_type_rect.right() - rect.right()
            if scene_t < self.parent_type_rect.top(): dy = self.parent_type_rect.top() - rect.top()
            elif scene_b > self.parent_type_rect.bottom(): dy = self.parent_type_rect.bottom() - rect.bottom()
            return QPointF(dx, dy)
        return super().itemChange(change, value)

    def set_id(self, new_id, new_local_id=None): 
        self.id_num = new_id
        self.local_id = new_local_id if new_local_id else new_id
        self.update_visuals()

    def set_label(self, new_label):
        self.cls_name = new_label
        self.base_color = get_class_color(self.cls_name)
        self.normal_pen.setColor(self.base_color)
        self.id_bg.setPen(QPen(self.base_color, 1))
        self.id_text_item.setDefaultTextColor(self.base_color)
        fill_color = QColor(self.base_color)
        fill_color.setAlpha(40)
        self.normal_brush = QBrush(fill_color)
        if not self.isSelected(): 
            self.setPen(self.normal_pen)
            self.setBrush(self.normal_brush)
        self.update_visuals()

    def update_visuals(self): 
        self.id_text_item.setPlainText(str(self.id_num) if self.id_num > 0 else "")
        self.update_text_pos()

    def update_text_pos(self): 
        rect = self.rect()
        display_num = self.local_id if hasattr(self, 'local_id') and self.local_id > 0 else self.id_num
        if display_num > 0:
            self.id_text_item.setPlainText(str(display_num))
            id_rect = self.id_text_item.boundingRect()
            cx = rect.x() + rect.width() / 2 - id_rect.width() / 2
            cy = rect.y() + rect.height() / 2 - id_rect.height() / 2
            self.id_bg.setRect(cx, cy, id_rect.width(), id_rect.height())
            self.id_text_item.setPos(cx, cy)
            main_window = self._get_main_window()
            if main_window and hasattr(main_window, 'show_labels'):
                self.id_bg.setVisible(main_window.show_labels)
            else:
                self.id_bg.setVisible(True)
        else: 
            self.id_bg.setVisible(False)
            
    def hoverEnterEvent(self, event):
        if not self.isSelected(): 
            self.setPen(self.hover_pen)
            self.setBrush(self.hover_brush)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        if not self.isSelected(): 
            self.setPen(self.normal_pen)
            self.setBrush(self.normal_brush)
        super().hoverLeaveEvent(event)

    def get_resize_mode(self, pos):
        r = self.rect()
        x, y, w, h = r.x(), r.y(), r.width(), r.height()
        px, py = pos.x(), pos.y()
        l = abs(px - x) < self.margin
        rt = abs(px - (x + w)) < self.margin
        t = abs(py - y) < self.margin
        b = abs(py - (y + h)) < self.margin
        if t and l: return 'tl'
        if t and rt: return 'tr'
        if b and l: return 'bl'
        if b and rt: return 'br'
        if l: return 'l'
        if rt: return 'r'
        if t: return 't'
        if b: return 'b'
        return None

    def mousePressEvent(self, event):
        if self.box_type == 'slide':
            self.parent_type_rect = None
            best_t, max_area = None, 0
            my_rect = self.sceneBoundingRect()
            for item in self.scene().items():
                if isinstance(item, BoundingBox) and item.box_type == 'type':
                    inter = my_rect.intersected(item.sceneBoundingRect())
                    area = inter.width() * inter.height()
                    if area > max_area: 
                        max_area = area
                        best_t = item
            if best_t: self.parent_type_rect = best_t.sceneBoundingRect()
        self.resize_mode = self.get_resize_mode(event.pos())
        if self.resize_mode:
            self.start_pos = event.scenePos()
            if self.isSelected():
                for item in self.scene().selectedItems():
                    if isinstance(item, BoundingBox): item.start_rect = item.rect()
            else:
                self.start_rect = self.rect()
        else: super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.resize_mode:
            d = event.scenePos() - self.start_pos
            items_to_resize = self.scene().selectedItems() if self.isSelected() else [self]
            for item in items_to_resize:
                if isinstance(item, BoundingBox) and hasattr(item, 'start_rect'):
                    r = QRectF(item.start_rect)
                    if 'l' in self.resize_mode: r.setLeft(min(r.left() + d.x(), r.right() - 5))
                    if 'r' in self.resize_mode: r.setRight(max(r.right() + d.x(), r.left() + 5))
                    if 't' in self.resize_mode: r.setTop(min(r.top() + d.y(), r.bottom() - 5))
                    if 'b' in self.resize_mode: r.setBottom(max(r.bottom() + d.y(), r.top() + 5))
                    if item.box_type == 'slide' and getattr(item, 'parent_type_rect', None):
                        scene_r = r.translated(item.pos()).intersected(item.parent_type_rect)
                        r = scene_r.translated(-item.pos())
                    item.setRect(r)
                    item.update_text_pos()
        else: super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.resize_mode = None
        self.parent_type_rect = None 
        main_window = self._get_main_window()
        if main_window and hasattr(main_window, 'trigger_auto_sort_and_snapshot'):
            main_window.trigger_auto_sort_and_snapshot(self.box_type)

class OCRTextItem(QGraphicsTextItem):  
    def __init__(self, text, conf, box_height, source_box=None, parent=None, font_families=None):  
        super().__init__(text, parent)  
        self.conf = conf  
        self.source_box = source_box  
        self.original_text = text.strip()  
        
        font = QFont()  
        if font_families: font.setFamilies(font_families)  
        else: font.setFamily("KaiTi")  
        font.setPixelSize(max(14, int(box_height * 0.85)))  
        font.setBold(True)  
        self.setFont(font)  

        self.update_color()
        self.setFlags(QGraphicsItem.ItemIsSelectable)  
        self.setTextInteractionFlags(Qt.NoTextInteraction)  

        self.click_timer = QTimer()
        self.click_timer.setSingleShot(True)
        self.click_timer.timeout.connect(self.show_candidates_menu)

    def update_color(self):
        if self.conf == 2.0:
            self.setDefaultTextColor(QColor(0, 102, 255)) 
        elif self.conf >= 0.80:
            self.setDefaultTextColor(QColor(0, 0, 0))     
        else:
            self.setDefaultTextColor(QColor(230, 50, 50)) 

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.click_timer.start(250) 
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.click_timer.stop() 
        self.setTextInteractionFlags(Qt.TextEditorInteraction)
        self.setFocus()
        super().mouseDoubleClickEvent(event)

    def show_candidates_menu(self):
        if self.textInteractionFlags() == Qt.TextEditorInteraction: return
        
        main_window = self.scene().views()[0].window()
        menu = QMenu()
        menu_font = QFont(self.font())
        menu_font.setPixelSize(20) 
        
        if hasattr(self, 'ref_diff_char') and self.ref_diff_char:
            menu.addAction("校本").setEnabled(False)
            action = menu.addAction(f"{self.ref_diff_char}")
            action.setFont(menu_font)
            action.setObjectName(self.ref_diff_char)
            menu.addSeparator()

        menu.addAction("🔄 正在唤醒常驻服务实时推理 Top-6...").setEnabled(False)
        menu.addSeparator()

        model_candidates = [(self.original_text, self.conf)]
        
        if self.source_box and main_window.display_image is not None:
            rect = self.source_box.sceneBoundingRect()
            x, y = int(rect.x()), int(rect.y())
            w, h = int(rect.width()), int(rect.height())
            img_h, img_w = main_window.display_image.shape[:2]
            y1, y2, x1, x2 = max(0, y), min(img_h, y+h), max(0, x), min(img_w, x+w)
            crop = main_window.display_image[y1:y2, x1:x2]
            
            if crop.size > 0:
                target_h = max(48, crop.shape[0] * 2)
                scale = target_h / crop.shape[0]
                target_w = int(crop.shape[1] * scale)
                resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                padded = cv2.copyMakeBorder(resized, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255,255,255])
                
                req_id = id(self)
                while not main_window.resident_res_q.empty():
                    try: main_window.resident_res_q.get_nowait()
                    except: break
                        
                main_window.resident_req_q.put((req_id, padded))
                QApplication.processEvents()
                
                import time
                start_time = time.time()
                while time.time() - start_time < 2.0:
                    try:
                        res_id, sorted_cands = main_window.resident_res_q.get(timeout=0.1)
                        if res_id == req_id and sorted_cands:
                            model_candidates = sorted_cands
                            break
                    except Exception:
                        QApplication.processEvents() 
                        continue

        actions = menu.actions()
        for act in actions:
            if "唤醒常驻" in act.text(): menu.removeAction(act)

        menu.addSeparator()
        
        for cand_text, cand_conf in model_candidates:
            if not cand_text.strip(): continue
            action = menu.addAction(f"{cand_text}  [{cand_conf:.2f}]")
            action.setFont(menu_font)
            action.setObjectName(cand_text)
            
        variants = []
        if hasattr(main_window, 'variant_dict') and self.original_text in main_window.variant_dict:
            variants = list(main_window.variant_dict[self.original_text])
            import random
            if len(variants) > 6: variants = random.sample(variants, 6)
            
        if variants:
            menu.addSeparator()
            menu.addAction("异形").setEnabled(False)
            menu.addSeparator()
            for v in variants:
                action = menu.addAction(v)
                action.setFont(menu_font)
                action.setObjectName(v)
        
        menu.addSeparator()
        
        basic_fallback = ["□", "■", "○","々","く"]
        for c in basic_fallback:
            if not any(c == cand[0] for cand in model_candidates):
                action = menu.addAction(c)
                action.setFont(menu_font)
                action.setObjectName(c)
            
        view = self.scene().views()[0]
        action = menu.exec_(view.mapToGlobal(view.mapFromScene(self.scenePos())))
            
        if action and action.objectName():
            self.commit_modification(action.objectName())

    def contextMenuEvent(self, event):
        if self.textInteractionFlags() == Qt.TextEditorInteraction:
            super().contextMenuEvent(event)
            return
            
        menu = QMenu()
        if hasattr(self, 'ref_diff_char'):
            menu.addAction(f"💡 校本参考字: 【{self.ref_diff_char}】").setEnabled(False)
            menu.addSeparator()
            
        save_action = menu.addAction("💾 将此单字保存至指定微调文件夹")
        action = menu.exec_(event.screenPos())
        
        if action == save_action:
            main_window = self.scene().views()[0].window()
            folder = QFileDialog.getExistingDirectory(main_window, "选择微调数据集保存目录")
            if folder and hasattr(main_window, 'auto_export_single_sample') and self.source_box:
                main_window.auto_export_single_sample(self.source_box, self.toPlainText().strip(), folder)
                main_window.show_msg("✅ 已成功将该截图及其标注加入数据集！")

    def focusOutEvent(self, event):  
        super().focusOutEvent(event)  
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        current_text = self.toPlainText().strip()  
        self.commit_modification(current_text)

    def commit_modification(self, new_text):
        if new_text != self.original_text:
            self.original_text = new_text
            self.conf = 2.0 
            self.update_color()
            self.setPlainText(new_text)
            if self.source_box:
                self.source_box.ocr_text = new_text
                self.source_box.ocr_conf = 2.0


class ImageViewer(QGraphicsView):
    def __init__(self, get_mode_callback, get_resort_callback, parent=None):
        super().__init__(parent)
        self.get_mode_callback = get_mode_callback
        self.get_resort_callback = get_resort_callback
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.RubberBandDrag) 
        self.setRubberBandSelectionMode(Qt.IntersectsItemShape)
        self.drawing = False
        self.space_pressed = False
        self.current_zoom = 1.0
        self.temp_rect = None
        self.current_img_height = 4000
        self.max_type_height = 200   

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            mode = self.get_mode_callback()
            # ✅ 取消了 OCR 模式的最大缩放限制，所有模式都可以无限放大
            factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
            self.current_zoom *= factor
            
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.scale(factor, factor)
            self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
            event.accept()
            return
            
        mode = self.get_mode_callback()
        if mode == 'ocr':
            # 正常滚动鼠标滚轮时：使横向滚动条移动
            delta = event.angleDelta().y()
            step = 50
            hbar = self.horizontalScrollBar()
            if hbar: 
                hbar.setValue(hbar.value() - step if delta > 0 else hbar.value() + step)
            event.accept()
        else:
            super().wheelEvent(event)

    def contextMenuEvent(self, event):
        main_window = self.window()
        mode = self.get_mode_callback()
        menu = QMenu(self)
        if getattr(main_window, 'manual_sort_pending', None):
            action_cancel = menu.addAction("❌ 取消当前手动排序")
            if action_cancel == menu.exec_(event.globalPos()):
                main_window.cancel_manual_sort()
            return

        selected = self.scene.selectedItems()
        boxes =[item for item in selected if isinstance(item, BoundingBox) and item.box_type == mode]
        valid_manual = False
        if len(boxes) > 1:
            boxes.sort(key=lambda b: b.id_num)
            ids =[b.id_num for b in boxes]
            if ids == list(range(ids[0], ids[-1] + 1)):
                if mode == 'type': valid_manual = True
                elif mode == 'slide':
                    type_boxes =[b for b in self.scene.items() if isinstance(b, BoundingBox) and b.box_type == 'type']
                    parent_id, same_parent = None, True
                    for b in boxes:
                        best_t, max_area = None, 0
                        s_rect = b.sceneBoundingRect()
                        for t in type_boxes:
                            inter = s_rect.intersected(t.sceneBoundingRect())
                            area = inter.width() * inter.height()
                            if area > max_area: 
                                max_area = area
                                best_t = t
                        if parent_id is None: parent_id = best_t
                        elif parent_id != best_t: same_parent = False; break
                    if same_parent: valid_manual = True

        action_manual = menu.addAction("依次点击以手动排序")
        action_manual.setEnabled(valid_manual)
        action_auto = menu.addAction("恢复当前页自动排序")
        action_auto.setEnabled(main_window.sort_mode.get(mode) == 'manual')

        action = menu.exec_(event.globalPos())
        if action == action_manual:
            main_window.start_manual_sort(boxes, mode)
            self.scene.clearSelection()
        elif action == action_auto:
            main_window.restore_auto_sort(mode)

    def mousePressEvent(self, event):
        main_window = self.window()
        if getattr(main_window, 'manual_sort_pending', None):
            item = self.itemAt(event.pos())
            if isinstance(item, BoundingBox) and item in main_window.manual_sort_pending:
                item.set_id(main_window.manual_sort_ids.pop(0))
                main_window.manual_sort_pending.remove(item)
                item.setSelected(False)
                if not main_window.manual_sort_pending:
                    main_window.set_ui_disabled_for_manual_sort(False)
                    main_window.snapshot(main_window.current_mode)
                    main_window.show_msg("手动排序完成！")
            return 

        if event.button() == Qt.LeftButton and (event.modifiers() & Qt.ShiftModifier):
            item = self.itemAt(event.pos())
            if isinstance(item, BoundingBox):
                item.setSelected(not item.isSelected())
                return

        if event.button() == Qt.LeftButton and (event.modifiers() & Qt.ControlModifier):
            self.setDragMode(QGraphicsView.NoDrag)
            self.drawing = True
            self.start_point = self.mapToScene(event.pos())
            self.temp_rect = QGraphicsRectItem(QRectF(self.start_point, self.start_point))
            self.temp_rect.setPen(QPen(QColor(0, 255, 0), 2, Qt.DashLine))
            self.scene.addItem(self.temp_rect)
        else: super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.temp_rect:
            self.temp_rect.setRect(QRectF(self.start_point, self.mapToScene(event.pos())).normalized())
        else: super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.drawing = False
            self.setDragMode(QGraphicsView.ScrollHandDrag if self.space_pressed else QGraphicsView.RubberBandDrag)
            if self.temp_rect:
                r = self.temp_rect.rect()
                self.scene.removeItem(self.temp_rect)
                self.temp_rect = None
                current_mode = self.get_mode_callback()
                main_window = self.window()
                if current_mode == 'slide':
                    best_t, max_area = None, 0
                    for item in self.scene.items():
                        if isinstance(item, BoundingBox) and item.box_type == 'type':
                            inter = r.intersected(item.sceneBoundingRect())
                            area = inter.width() * inter.height()
                            if area > max_area: 
                                max_area = area; best_t = item
                    if best_t: r = r.intersected(best_t.sceneBoundingRect()) 
                    else: return 
                if r.width() > 5 and r.height() > 5:
                    new_box = BoundingBox(r, img_height=self.current_img_height, cls_name="text", box_type=current_mode)
                    if hasattr(main_window, 'show_labels'): 
                        new_box.id_bg.setVisible(main_window.show_labels)
                    self.scene.addItem(new_box)
                    if hasattr(main_window, 'trigger_auto_sort_and_snapshot'):
                        main_window.trigger_auto_sort_and_snapshot(current_mode)
        else: super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self.space_pressed = True
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        elif event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            mode = self.get_mode_callback()
            if mode == 'ocr': return 
            for item in self.scene.selectedItems():
                if isinstance(item, BoundingBox): self.scene.removeItem(item)
            main_window = self.window()
            if hasattr(main_window, 'trigger_auto_sort_and_snapshot'):
                main_window.trigger_auto_sort_and_snapshot(mode)
        else: super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self.space_pressed = False
            self.setDragMode(QGraphicsView.RubberBandDrag)
        else: super().keyReleaseEvent(event)