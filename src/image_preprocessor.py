"""
图像预处理器 - 兼容性入口
为了保持向后兼容性，此文件导入新的模块化组件
所有功能已重构到 image_processing 包中的专门组件
"""

# 导入重构后的主处理器
from .image_processing import ImagePreprocessor

# 为了完全兼容，也可以直接从此文件导入
__all__ = ['ImagePreprocessor']

# 如果需要直接访问其他组件，可以从这里导入
# from .image_processing.image_transformer import ImageTransformer
# from .image_processing.augmentation_manager import AugmentationManager
# from .image_processing.dataset_creator import DatasetCreator
# from .image_processing.class_balancer import ClassBalancer
# from .image_processing.file_manager import FileManager 