import os
import threading
import tempfile
import wx
import fitz          # PyMuPDF
from PIL import Image
import img2pdf
import numpy as np
import cv2           # OpenCV
import gc
import concurrent.futures

# ===== 尝试引入 GPU 计算库 =====
GPU_AVAILABLE = False
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cndi
    GPU_AVAILABLE = True
except ImportError:
    pass

class PDFBinarizerAutoFrame(wx.Frame):
    def __init__(self):
        title_suffix = "🚀 纯血 GPU 并发加速模式" if GPU_AVAILABLE else "🐢 CPU 模式 (未检测到 CuPy)"
        super().__init__(None, title=f"文档/图片 自动二值化 - {title_suffix}", size=(720, 560))
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # [界面组件定义]
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        lbl_input = wx.StaticText(panel, label="输入文件夹：")
        self.input_dir = wx.TextCtrl(panel, style=wx.TE_READONLY)
        btn_input = wx.Button(panel, label="浏览")
        btn_input.Bind(wx.EVT_BUTTON, self.on_select_input)
        hbox1.Add(lbl_input, flag=wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=8)
        hbox1.Add(self.input_dir, proportion=1, flag=wx.EXPAND)
        hbox1.Add(btn_input, flag=wx.LEFT, border=8)
        vbox.Add(hbox1, flag=wx.EXPAND | wx.ALL, border=10)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        lbl_output = wx.StaticText(panel, label="输出文件夹：")
        self.output_dir = wx.TextCtrl(panel, style=wx.TE_READONLY)
        btn_output = wx.Button(panel, label="浏览")
        btn_output.Bind(wx.EVT_BUTTON, self.on_select_output)
        hbox2.Add(lbl_output, flag=wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=8)
        hbox2.Add(self.output_dir, proportion=1, flag=wx.EXPAND)
        hbox2.Add(btn_output, flag=wx.LEFT, border=8)
        vbox.Add(hbox2, flag=wx.EXPAND | wx.ALL, border=10)

        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        lbl_dpi = wx.StaticText(panel, label="目标DPI：")
        self.dpi = wx.SpinCtrl(panel, min=50, max=600, initial=300)
        
        lbl_threads = wx.StaticText(panel, label="并发线程数：")
        self.thread_count = wx.SpinCtrl(panel, min=1, max=16, initial=4)
        self.thread_count.SetToolTip("CPU提取和处理的线程数，推荐4-8")

        hbox3.Add(lbl_dpi, flag=wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=8)
        hbox3.Add(self.dpi, flag=wx.RIGHT, border=20)
        hbox3.Add(lbl_threads, flag=wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=8)
        hbox3.Add(self.thread_count, flag=wx.RIGHT, border=20)
        vbox.Add(hbox3, flag=wx.ALL, border=10)

        self.chk_keep_pixels = wx.CheckBox(panel, label="保持原始像素（仅修改DPI元数据，不缩放图像）")
        vbox.Add(self.chk_keep_pixels, flag=wx.ALL, border=10)

        hbox4 = wx.BoxSizer(wx.HORIZONTAL)
        lbl_algo = wx.StaticText(panel, label="缩放插值：")
        self.choice_interp = wx.Choice(panel, choices=["PyMuPDF默认", "双三次 (Bicubic)", "Lanczos (锐利)"])
        self.choice_interp.SetSelection(1)
        hbox4.Add(lbl_algo, flag=wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=8)
        hbox4.Add(self.choice_interp, flag=wx.EXPAND)
        vbox.Add(hbox4, flag=wx.ALL, border=10)

        hbox5 = wx.BoxSizer(wx.HORIZONTAL)
        lbl_bin = wx.StaticText(panel, label="二值化算法：")
        self.choice_bin = wx.Choice(panel, choices=["Sauvola (自适应 - 强烈推荐)", "Otsu (全局)"])
        self.choice_bin.SetSelection(0)
        hbox5.Add(lbl_bin, flag=wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=8)
        hbox5.Add(self.choice_bin, flag=wx.EXPAND)
        
        lbl_gpu = wx.StaticText(panel, label="[NVIDIA GPU 并发就绪]" if GPU_AVAILABLE else "[依赖缺失，当前为 CPU 模式]")
        lbl_gpu.SetForegroundColour(wx.Colour(0, 150, 0) if GPU_AVAILABLE else wx.Colour(255, 0, 0))
        font = lbl_gpu.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        lbl_gpu.SetFont(font)
        hbox5.Add(lbl_gpu, flag=wx.LEFT | wx.ALIGN_CENTER_VERTICAL, border=15)
        
        vbox.Add(hbox5, flag=wx.ALL, border=10)

        hbox_height = wx.BoxSizer(wx.HORIZONTAL)
        self.chk_unify_height = wx.CheckBox(panel, label="统一页面高度（像素）：")
        self.target_height = wx.SpinCtrl(panel, min=100, max=5000, initial=4000)
        hbox_height.Add(self.chk_unify_height, flag=wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=8)
        hbox_height.Add(self.target_height, flag=wx.EXPAND)
        vbox.Add(hbox_height, flag=wx.ALL, border=10)

        hbox6 = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_start = wx.Button(panel, label="开始并发处理")
        self.btn_start.Bind(wx.EVT_BUTTON, self.on_start)
        self.gauge = wx.Gauge(panel, range=100)
        hbox6.Add(self.btn_start, flag=wx.RIGHT, border=10)
        hbox6.Add(self.gauge, proportion=1, flag=wx.EXPAND)
        vbox.Add(hbox6, flag=wx.EXPAND | wx.ALL, border=10)

        self.log = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 150))
        vbox.Add(self.log, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        panel.SetSizer(vbox)
        self.Centre()

        self.gui_lock = threading.Lock()
        self.gpu_semaphore = threading.Semaphore(3)

    def on_select_input(self, event):
        with wx.DirDialog(self, "选择输入文件夹") as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.input_dir.SetValue(dlg.GetPath())

    def on_select_output(self, event):
        with wx.DirDialog(self, "选择输出文件夹") as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.output_dir.SetValue(dlg.GetPath())

    def log_message(self, msg):
        wx.CallAfter(self.log.AppendText, msg + "\n")

    def update_progress(self, val):
        wx.CallAfter(self.gauge.SetValue, int(val))

    def on_start(self, event):
        input_path = self.input_dir.GetValue().strip()
        output_path = self.output_dir.GetValue().strip()

        if not input_path or not os.path.isdir(input_path):
            wx.MessageBox("请选择有效的输入文件夹", "错误", wx.OK | wx.ICON_ERROR)
            return
        if not output_path:
            output_path = os.path.join(input_path, "output")
            os.makedirs(output_path, exist_ok=True)
            self.output_dir.SetValue(output_path)

        self.btn_start.Enable(False)
        self.gauge.SetValue(0)
        self.log.Clear()

        # 启动主控线程
        thread = threading.Thread(target=self.process_files_master, args=(input_path, output_path))
        thread.daemon = True
        thread.start()

    # ==================== 核心算法区 ====================
    def otsu_threshold(self, img_np):
        thresh, _ = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return thresh

    def sauvola_cpu_fallback(self, img_np, k=0.2, r=128):
        win = max(15, min(img_np.shape) // 20)
        win = win if win % 2 == 1 else win + 1
        img_f = img_np.astype(np.float32)
        mean = cv2.boxFilter(img_f, cv2.CV_32F, (win, win), borderType=cv2.BORDER_REFLECT)
        mean_sq = cv2.boxFilter(img_f ** 2, cv2.CV_32F, (win, win), borderType=cv2.BORDER_REFLECT)
        variance = np.maximum(mean_sq - mean**2, 0)
        std = np.sqrt(variance)
        threshold = mean * (1 + k * ((std / r) - 1))
        return (img_np > threshold).astype(np.uint8) * 255

    def sauvola_threshold(self, img_np, k=0.2, r=128):
        if not GPU_AVAILABLE:
            return self.sauvola_cpu_fallback(img_np, k, r)

        with self.gpu_semaphore:
            try:
                win = max(15, min(img_np.shape) // 20)
                win = win if win % 2 == 1 else win + 1
                
                img_cp = cp.array(img_np, dtype=cp.float32)
                mean = cndi.uniform_filter(img_cp, size=win, mode='reflect')
                mean_sq = cndi.uniform_filter(img_cp ** 2, size=win, mode='reflect')
                std = cp.sqrt(cp.maximum(mean_sq - mean**2, 0))
                threshold = mean * (1 + k * ((std / r) - 1))
                
                result_cp = (img_cp > threshold).astype(cp.uint8) * 255
                result_np = cp.asnumpy(result_cp)
                
                del img_cp, mean, mean_sq, std, threshold, result_cp
                return result_np
                
            except Exception as e:
                with self.gui_lock:
                    self.log_message(f"      [警告] GPU 运算异常 ({str(e)})，降级 CPU。")
                try: cp.get_default_memory_pool().free_all_blocks()
                except: pass
                return self.sauvola_cpu_fallback(img_np, k, r)

    def cv2_unsharp_mask(self, img_np):
        blur = cv2.GaussianBlur(img_np, (0, 0), 2.0)
        return cv2.addWeighted(img_np, 1.5, blur, -0.5, 0)

    # ==================== 统一图像处理流水线 ====================
    def _process_image_core(self, img_np, unify_height, target_h, bin_algo):
        """无论来源是 PDF 还是图片，统一使用这段图像缩放与二值化逻辑"""
        # 1. 图像缩放处理
        if unify_height and target_h > 0:
            h, w = img_np.shape
            ratio = target_h / h
            target_w = int(round(w * ratio))
            img_np = cv2.resize(img_np, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            h, w = img_np.shape
            if h < 4000:
                n = 4000 // h
                if n > 1:
                    new_height = n * h
                    new_width = w * n
                    img_np = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        # 2. 二值化
        if bin_algo == 1:
            thresh = self.otsu_threshold(img_np)
            _, bin_np = cv2.threshold(img_np, thresh, 255, cv2.THRESH_BINARY)
        else:
            bin_np = self.sauvola_threshold(img_np)

        return bin_np

    # ==================== 提取区 (PDF & Image Worker) ====================
    def render_pdf_page(self, page, target_dpi, keep_pixels, interp_method):
        """PDF 单页提取器"""
        if keep_pixels:
            mat = fitz.Matrix(1.0, 1.0)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
            return img_np, target_dpi

        rect = page.rect
        width_pts, height_pts = rect.width, rect.height
        target_w = int(round(width_pts * target_dpi / 72))
        target_h = int(round(height_pts * target_dpi / 72))

        if interp_method == 0:  
            zoom = target_dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
            return img_np, target_dpi
        else:
            if target_dpi <= 150: intermediate_factor = 4
            elif target_dpi <= 300: intermediate_factor = 2
            else: intermediate_factor = 1.5

            intermediate_zoom = intermediate_factor * target_dpi / 72
            mat = fitz.Matrix(intermediate_zoom, intermediate_zoom)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            img_large = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

            cv2_interp = cv2.INTER_CUBIC if interp_method == 1 else cv2.INTER_LANCZOS4
            img_np = cv2.resize(img_large, (target_w, target_h), interpolation=cv2_interp)

            if target_w * target_h < 5_000_000:
                img_np = self.cv2_unsharp_mask(img_np)

            return img_np, target_dpi

    def process_single_pdf_page(self, doc, page_num, pdf_lock, dpi_val, keep_pixels, interp_idx, unify_height, target_h, bin_algo):
        """PDF 并发 Worker"""
        with pdf_lock:
            page = doc.load_page(page_num)
            img_np, actual_dpi = self.render_pdf_page(page, dpi_val, keep_pixels, interp_idx)

        # 调用统一核心算法
        bin_np = self._process_image_core(img_np, unify_height, target_h, bin_algo)

        # PDF 要求合并保存，因此写出为高保真 PNG 临时文件
        img_bin_pil = Image.fromarray(bin_np).convert('1')
        fd, temp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd) 
        img_bin_pil.save(temp_path, "PNG", dpi=(actual_dpi, actual_dpi))
        
        del img_np, bin_np, img_bin_pil
        return page_num, temp_path

    def process_single_image_file(self, input_path, output_path, dpi_val, unify_height, target_h, bin_algo):
        """独立图片 并发 Worker"""
        # 使用 np.fromfile 防止含有中文字符的路径在 OpenCV 中报错
        img_np = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img_np is None:
            return False, f"无法读取图像数据: {input_path}"

        # 调用统一核心算法
        bin_np = self._process_image_core(img_np, unify_height, target_h, bin_algo)

        # 独立图片直接保存到目标文件夹 (按要求保存为 JPG)
        # JPG 不支持 1 位图，因此转化为 'L' (8位灰度)，画质设为最高以保留二值化边缘
        img_bin_pil = Image.fromarray(bin_np).convert('L')
        img_bin_pil.save(output_path, "JPEG", dpi=(dpi_val, dpi_val), quality=95)

        del img_np, bin_np, img_bin_pil
        return True, ""

    # ==================== 主控流程 ====================
    def process_files_master(self, input_dir, output_dir):
        # 参数收集
        dpi_val = self.dpi.GetValue()
        keep_pixels = self.chk_keep_pixels.GetValue()
        interp_idx = self.choice_interp.GetSelection()
        bin_algo = self.choice_bin.GetSelection()
        unify_height = self.chk_unify_height.GetValue()
        target_h = self.target_height.GetValue() if unify_height else 0
        max_threads = self.thread_count.GetValue()

        # ========== [修改点1] 递归查找文件系统并生成映射目录 ==========
        pdf_files = []
        img_files =[]
        supported_img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        
        out_dir_abs = os.path.abspath(output_dir)

        for root, dirs, files in os.walk(input_dir):
            # 防止如果输出目录建在输入目录内部时，引发无限循环递归读取
            dirs[:] =[d for d in dirs if os.path.abspath(os.path.join(root, d)) != out_dir_abs]
            
            for f in files:
                full_path = os.path.join(root, f)
                # 计算出每个文件相对于输入根目录的层级 (例如：甲/乙/丙.pdf)
                rel_path = os.path.relpath(full_path, input_dir)
                ext = os.path.splitext(f)[1].lower()
                
                if ext == '.pdf':
                    pdf_files.append(rel_path)
                elif ext in supported_img_exts:
                    img_files.append(rel_path)

        total_files = len(pdf_files) + len(img_files)
        if total_files == 0:
            self.log_message("未在目录及其子目录中找到 PDF 或 支持的图片文件。")
            wx.CallAfter(self.btn_start.Enable, True)
            return

        self.log_message(f"🚀 启动并发引擎(线程:{max_threads}) | 发现 {len(pdf_files)}个 PDF, {len(img_files)}张 图片")
        processed_files = 0

        # ========== 阶段 1：处理所有 PDF (按文档内部页码并发) ==========
        for rel_path in pdf_files:
            self.log_message(f"▶ 正在拆解处理 PDF: {rel_path}")
            
            # [修改点2] 输入路径与复刻后的输出层级结构
            input_pdf = os.path.join(input_dir, rel_path)
            output_pdf = os.path.join(output_dir, rel_path)
            
            # 建立对应层级的输出子文件夹（如果在大文件夹直接存放，则不会创建子目录）
            os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

            temp_images_dict = {} 
            doc = None
            try:
                doc = fitz.open(input_pdf)
                total_pages = len(doc)
                pdf_lock = threading.Lock()
                processed_pages = 0

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                    futures = {
                        executor.submit(
                            self.process_single_pdf_page, doc, page_num, pdf_lock, 
                            dpi_val, keep_pixels, interp_idx, unify_height, target_h, bin_algo
                        ): page_num 
                        for page_num in range(total_pages)
                    }

                    for future in concurrent.futures.as_completed(futures):
                        page_num, temp_path = future.result()
                        temp_images_dict[page_num] = temp_path
                        processed_pages += 1
                        
                        with self.gui_lock:
                            self.log_message(f"  └─ 第 {page_num+1}/{total_pages} 页处理完毕")
                        # 细分进度条（文档级别）
                        overall_progress = ((processed_files * 100) + (processed_pages / total_pages * 100)) / total_files
                        self.update_progress(overall_progress)

                doc.close()
                doc = None

                sorted_temp_images = [temp_images_dict[i] for i in range(total_pages)]
                self.log_message(f"📦 正在合成最终 PDF: {output_pdf}")
                
                with open(output_pdf, "wb") as f:
                    f.write(img2pdf.convert(sorted_temp_images))

                for tmp in sorted_temp_images:
                    if os.path.exists(tmp): os.remove(tmp)

            except Exception as e:
                self.log_message(f"❌ 处理 {rel_path} 失败：{str(e)}")
                if doc: doc.close()
                for tmp in temp_images_dict.values():
                    if os.path.exists(tmp): os.remove(tmp)
            finally:
                processed_files += 1
                if GPU_AVAILABLE: cp.get_default_memory_pool().free_all_blocks()
                gc.collect()

        # ========== 阶段 2：处理所有图片 (多张图片跨文件并发) ==========
        if img_files:
            self.log_message(f"▶ 正在并发处理 {len(img_files)} 张独立图片...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures_img = {}
                for rel_path in img_files:
                    
                    # [修改点3] 同样为独立图片维持原文件的层级结构
                    in_path = os.path.join(input_dir, rel_path)
                    rel_dir = os.path.dirname(rel_path)
                    base_name = os.path.splitext(os.path.basename(rel_path))[0]
                    
                    out_path = os.path.join(output_dir, rel_dir, f"{base_name}.jpg")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    
                    future = executor.submit(
                        self.process_single_image_file, 
                        in_path, out_path, dpi_val, unify_height, target_h, bin_algo
                    )
                    futures_img[future] = rel_path
                
                # 回调收集
                for future in concurrent.futures.as_completed(futures_img):
                    rel_path = futures_img[future]
                    try:
                        success, err_msg = future.result()
                        with self.gui_lock:
                            if success:
                                self.log_message(f"  └─ 图像转换成功: {rel_path}")
                            else:
                                self.log_message(f"  ❌ {err_msg}")
                    except Exception as e:
                        with self.gui_lock:
                            self.log_message(f"  ❌ 处理 {rel_path} 异常: {str(e)}")

                    processed_files += 1
                    overall_progress = (processed_files / total_files) * 100
                    self.update_progress(overall_progress)

            if GPU_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        self.update_progress(100)
        self.log_message("🎉 全部任务处理完成！")
        wx.CallAfter(self.btn_start.Enable, True)

if __name__ == "__main__":
    app = wx.App(False)
    frame = PDFBinarizerAutoFrame()
    frame.Show()
    app.MainLoop()