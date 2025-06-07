# 为了保持向后兼容性，从新的模块化组件导入所有类
from .components.evaluation import (
    TrainingVisualizationWidget,
    TensorBoardWidget,
    MetricsDataManager,
    ChartRenderer,
    MetricExplanations
)

# 导出主要类供外部使用
__all__ = [
    'TrainingVisualizationWidget',
    'TensorBoardWidget', 
    'MetricsDataManager',
    'ChartRenderer',
    'MetricExplanations'
]