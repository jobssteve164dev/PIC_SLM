"""
训练组件模块

该模块包含所有与模型训练相关的UI组件：
- ClassificationTrainingWidget: 图像分类训练组件
- DetectionTrainingWidget: 目标检测训练组件  
- TrainingControlWidget: 训练控制组件
- TrainingHelpDialog: 训练帮助对话框
- LayerConfigWidget: 层配置组件
- ModelDownloadDialog: 模型下载对话框
- WeightConfigManager: 权重配置管理器
"""

from .classification_training_widget import ClassificationTrainingWidget
from .detection_training_widget import DetectionTrainingWidget
from .training_control_widget import TrainingControlWidget
from .training_help_dialog import TrainingHelpDialog
from .layer_config_widget import LayerConfigWidget
from .model_download_dialog import ModelDownloadDialog
from .weight_config_manager import WeightConfigManager

__all__ = [
    'ClassificationTrainingWidget',
    'DetectionTrainingWidget', 
    'TrainingControlWidget',
    'TrainingHelpDialog',
    'LayerConfigWidget',
    'ModelDownloadDialog',
    'WeightConfigManager'
] 