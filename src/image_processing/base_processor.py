"""
基础图像处理器类
定义了通用的信号和基础功能
"""

from PyQt5.QtCore import QObject, pyqtSignal
from typing import List
import os


class BaseImageProcessor(QObject):
    """基础图像处理器，定义通用信号和基础功能"""
    
    # 定义信号
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    preprocessing_finished = pyqtSignal()
    preprocessing_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._stop_preprocessing = False

    def stop(self):
        """停止预处理过程"""
        self._stop_preprocessing = True
        self.status_updated.emit('正在停止预处理...')
        # 在停止预处理时也发出完成信号，确保UI恢复正常状态
        self.preprocessing_finished.emit()
        print("BaseImageProcessor.stop: 发出 preprocessing_finished 信号")

    def get_image_files(self, folder_path: str) -> List[str]:
        """获取文件夹中的所有图片文件"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        return [f for f in os.listdir(folder_path) 
                if os.path.isfile(os.path.join(folder_path, f)) and 
                os.path.splitext(f.lower())[1] in valid_extensions]

    def create_directories(self, *paths: str):
        """创建多个目录"""
        for path in paths:
            os.makedirs(path, exist_ok=True)

    def emit_error(self, message: str):
        """发送错误信息"""
        import traceback
        self.preprocessing_error.emit(f'{message}\n{traceback.format_exc()}')
        # 即使出错也发出完成信号，以确保UI恢复正常状态
        self.preprocessing_finished.emit()
        print(f"BaseImageProcessor: 虽然处理出错，但仍发出 preprocessing_finished 信号") 