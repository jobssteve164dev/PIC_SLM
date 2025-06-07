"""
文件管理器
专门处理文件和目录相关的操作
"""

import os
import shutil
from typing import List, Callable


class FileManager:
    """文件管理器，处理文件和目录操作"""

    @staticmethod
    def get_image_files(folder_path: str) -> List[str]:
        """获取文件夹中的所有图片文件"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        return [f for f in os.listdir(folder_path) 
                if os.path.isfile(os.path.join(folder_path, f)) and 
                os.path.splitext(f.lower())[1] in valid_extensions]

    @staticmethod
    def create_directories(*paths: str):
        """创建多个目录"""
        for path in paths:
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def copy_file(source_path: str, destination_path: str):
        """复制文件"""
        # 确保目标目录存在
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy2(source_path, destination_path)

    @staticmethod
    def file_exists(file_path: str) -> bool:
        """检查文件是否存在"""
        return os.path.exists(file_path)

    @staticmethod
    def is_directory(path: str) -> bool:
        """检查路径是否为目录"""
        return os.path.isdir(path)

    @staticmethod
    def ensure_directory_exists(path: str):
        """确保目录存在"""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def get_file_name_without_extension(file_path: str) -> str:
        """获取不带扩展名的文件名"""
        return os.path.splitext(os.path.basename(file_path))[0]

    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """获取文件扩展名"""
        return os.path.splitext(file_path)[1]

    @staticmethod
    def join_path(*args: str) -> str:
        """连接路径"""
        return os.path.join(*args)

    @staticmethod
    def create_class_folders(base_folder: str, class_names: List[str],
                           status_callback: Callable[[str], None] = None) -> None:
        """为每个类别创建文件夹"""
        try:
            for class_name in class_names:
                class_folder = os.path.join(base_folder, class_name)
                os.makedirs(class_folder, exist_ok=True)
                
            if status_callback:
                status_callback(f'已创建 {len(class_names)} 个类别文件夹')
        except Exception as e:
            if status_callback:
                status_callback(f'创建类别文件夹时出错: {str(e)}')

    @staticmethod
    def validate_source_folder(source_folder: str) -> bool:
        """验证源文件夹是否有效"""
        if not os.path.exists(source_folder):
            return False
        if not os.path.isdir(source_folder):
            return False
        return True

    @staticmethod
    def get_folder_size(folder_path: str) -> int:
        """获取文件夹大小（字节）"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
        return total_size

    @staticmethod
    def count_files_in_folder(folder_path: str, extensions: List[str] = None) -> int:
        """统计文件夹中的文件数量"""
        if not extensions:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        count = 0
        for f in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, f)):
                if any(f.lower().endswith(ext) for ext in extensions):
                    count += 1
        return count 