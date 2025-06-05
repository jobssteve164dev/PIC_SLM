"""
基础数据集分析器

定义数据集分析的通用接口和基础功能
"""

import os
import numpy as np
import cv2
from PIL import Image
from abc import ABC, abstractmethod
from PyQt5.QtCore import QObject, pyqtSignal


class BaseDatasetAnalyzer(QObject):
    """数据集分析器基类"""
    
    # 定义信号
    progress_updated = pyqtSignal(int)  # 进度更新信号
    status_updated = pyqtSignal(str)   # 状态更新信号
    
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path
        self.train_path = os.path.join(dataset_path, "train")
        self.val_path = os.path.join(dataset_path, "val")
        
    def validate_dataset_structure(self):
        """验证数据集结构"""
        if not os.path.exists(self.train_path):
            raise ValueError("数据集结构不正确，缺少train文件夹")
        if not os.path.exists(self.val_path):
            raise ValueError("数据集结构不正确，缺少val文件夹")
            
    def get_image_basic_info(self, image_path):
        """获取图像基本信息"""
        try:
            # 使用PIL获取图像尺寸
            with Image.open(image_path) as img:
                size = img.size[0] * img.size[1]
                
            # 使用OpenCV计算图像质量指标
            img_cv = cv2.imread(image_path)
            if img_cv is not None:
                # 计算清晰度（拉普拉斯方差）
                quality = cv2.Laplacian(img_cv, cv2.CV_64F).var()
                # 计算亮度
                brightness = np.mean(img_cv)
                # 计算对比度  
                contrast = np.std(img_cv)
                
                return {
                    'size': size,
                    'quality': quality,
                    'brightness': brightness,
                    'contrast': contrast
                }
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {str(e)}")
            
        return None
        
    def calculate_imbalance_ratio(self, class_counts):
        """计算类别不平衡度"""
        if not class_counts:
            return float('inf')
        values = list(class_counts.values())
        return max(values) / min(values) if min(values) > 0 else float('inf')
        
    def calculate_variation_coefficient(self, values):
        """计算变异系数（标准差/均值）"""
        if len(values) == 0 or np.mean(values) == 0:
            return 0
        return np.std(values) / np.mean(values)
        
    def is_image_file(self, filename):
        """判断是否为图像文件"""
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        
    @abstractmethod
    def analyze_data_distribution(self):
        """分析数据分布 - 子类必须实现"""
        pass
        
    @abstractmethod  
    def analyze_image_quality(self):
        """分析图像质量 - 子类必须实现"""
        pass
        
    @abstractmethod
    def analyze_annotation_quality(self):
        """分析标注质量 - 子类必须实现"""
        pass
        
    @abstractmethod
    def analyze_feature_distribution(self):
        """分析特征分布 - 子类必须实现"""
        pass 