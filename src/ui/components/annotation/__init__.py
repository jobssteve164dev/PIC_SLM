"""
标注组件包

提供图像分类和目标检测的标注功能组件。

主要组件:
- ClassificationWidget: 图像分类标注组件
- DetectionWidget: 目标检测标注组件  
- AnnotationCanvas: 标注画布组件
"""

from .classification_widget import ClassificationWidget
from .detection_widget import DetectionWidget
from .annotation_canvas import AnnotationCanvas

__all__ = [
    'ClassificationWidget',
    'DetectionWidget', 
    'AnnotationCanvas'
] 