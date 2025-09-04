"""
设置组件包

此包包含了设置标签页的所有子组件：
- config_manager: 配置管理器
- folder_config_widget: 文件夹配置组件
- class_weight_widget: 类别权重配置组件
- model_config_widget: 模型配置组件
- weight_strategy: 权重策略枚举
- config_profile_selector: 配置文件选择器组件
- resource_limit_widget: 系统资源限制组件
- log_viewer_widget: 日志查看器组件
- dependency_manager_widget: 依赖管理组件
- ai_settings_widget: AI设置组件
"""

from .config_manager import ConfigManager
from .folder_config_widget import FolderConfigWidget
from .class_weight_widget import ClassWeightWidget
from .model_config_widget import ModelConfigWidget
from .weight_strategy import WeightStrategy
from .config_profile_selector import ConfigProfileSelector
from .resource_limit_widget import ResourceLimitWidget
from .log_viewer_widget import LogViewerWidget
from .dependency_manager_widget import DependencyManagerWidget
from .ai_settings_widget import AISettingsWidget
from .intelligent_training_settings_widget import IntelligentTrainingSettingsWidget

__all__ = [
    'ConfigManager',
    'FolderConfigWidget', 
    'ClassWeightWidget',
    'ModelConfigWidget',
    'WeightStrategy',
    'ConfigProfileSelector',
    'ResourceLimitWidget',
    'LogViewerWidget',
    'DependencyManagerWidget',
    'AISettingsWidget',
    'IntelligentTrainingSettingsWidget'
] 