"""
设置组件包

此包包含了设置标签页的所有子组件：
- config_manager: 配置管理器
- folder_config_widget: 文件夹配置组件
- class_weight_widget: 类别权重配置组件
- model_config_widget: 模型配置组件
- weight_strategy: 权重策略枚举
"""

from .config_manager import ConfigManager
from .folder_config_widget import FolderConfigWidget
from .class_weight_widget import ClassWeightWidget
from .model_config_widget import ModelConfigWidget
from .weight_strategy import WeightStrategy

__all__ = [
    'ConfigManager',
    'FolderConfigWidget', 
    'ClassWeightWidget',
    'ModelConfigWidget',
    'WeightStrategy'
] 