from PyQt5.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QSizePolicy
from PyQt5.QtCore import pyqtSignal

class BaseTab(QWidget):
    """所有标签页的基类，提供通用功能"""
    
    # 通用信号
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)  # 减少边距
        self.layout.setSpacing(0)  # 减少间距
        
        # 创建滚动区域
        scroll = self.create_scroll_area()
        self.scroll_content = QWidget()
        scroll.setWidget(self.scroll_content)
        self.layout.addWidget(scroll)
        
    def create_scroll_area(self):
        """创建一个滚动区域"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll.setFrameShape(QScrollArea.NoFrame)  # 移除边框
        return scroll
        
    def update_status(self, message):
        """更新状态信息"""
        self.status_updated.emit(message)
        
    def update_progress(self, value):
        """更新进度条"""
        self.progress_updated.emit(value) 