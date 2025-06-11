"""
图片预处理线程
用于在独立线程中执行图片预处理任务，避免阻塞UI线程
"""

from PyQt5.QtCore import QThread, pyqtSignal, QObject
from typing import Dict
from .main_processor import ImagePreprocessor


class PreprocessingWorker(QObject):
    """图片预处理工作器类，在独立线程中运行"""
    
    # 定义信号
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    preprocessing_finished = pyqtSignal()
    preprocessing_error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.image_preprocessor = ImagePreprocessor()
        self._stop_processing = False
        
        # 连接图片预处理器的信号到工作器的信号
        self.image_preprocessor.progress_updated.connect(self.progress_updated.emit)
        self.image_preprocessor.status_updated.connect(self.status_updated.emit)
        self.image_preprocessor.preprocessing_finished.connect(self.preprocessing_finished.emit)
        self.image_preprocessor.preprocessing_error.connect(self.preprocessing_error.emit)
    
    def preprocess_images(self, params: Dict):
        """在独立线程中执行图片预处理"""
        try:
            print("PreprocessingWorker: 开始图片预处理")
            self.status_updated.emit("开始图片预处理...")
            
            # 重置停止标志
            self._stop_processing = False
            self.image_preprocessor._stop_preprocessing = False
            
            # 执行预处理
            self.image_preprocessor.preprocess_images(params)
            
        except Exception as e:
            error_msg = f"预处理过程中发生错误: {str(e)}"
            print(f"PreprocessingWorker错误: {error_msg}")
            self.preprocessing_error.emit(error_msg)
    
    def stop_preprocessing(self):
        """停止预处理过程"""
        print("PreprocessingWorker: 收到停止预处理请求")
        self._stop_processing = True
        if hasattr(self.image_preprocessor, 'stop'):
            self.image_preprocessor.stop()


class PreprocessingThread(QThread):
    """图片预处理线程类"""
    
    # 定义信号
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    preprocessing_finished = pyqtSignal()
    preprocessing_error = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.params = None
        
    def setup_preprocessing(self, params: Dict):
        """设置预处理参数"""
        self.params = params
        
        # 创建工作器并移动到线程中
        self.worker = PreprocessingWorker()
        self.worker.moveToThread(self)
        
        # 连接工作器信号到线程信号
        self.worker.progress_updated.connect(self.progress_updated.emit)
        self.worker.status_updated.connect(self.status_updated.emit)
        self.worker.preprocessing_finished.connect(self.preprocessing_finished.emit)
        self.worker.preprocessing_error.connect(self.preprocessing_error.emit)
        
        # 连接线程启动信号到工作器的预处理方法
        self.started.connect(lambda: self.worker.preprocess_images(self.params))
        
        # 连接完成信号到线程结束
        self.worker.preprocessing_finished.connect(self.quit)
        self.worker.preprocessing_error.connect(self.quit)
    
    def stop_preprocessing(self):
        """停止预处理"""
        if self.worker:
            self.worker.stop_preprocessing()
        self.quit()
        self.wait() 