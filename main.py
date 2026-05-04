import sys
import multiprocessing as mp
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ui.main_window import YOLOStageTester

if __name__ == '__main__':
    mp.freeze_support() 
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    window = YOLOStageTester()
    window.show()
    sys.exit(app.exec_())