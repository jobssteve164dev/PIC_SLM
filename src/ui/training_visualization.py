# 为了保持向后兼容性，从新的模块化组件导入所有类
from .components.training_visualization_widget import TrainingVisualizationWidget
from .components.tensorboard_widget import TensorBoardWidget
from .components.metrics_data_manager import MetricsDataManager
from .components.chart_renderer import ChartRenderer
from .components.metric_explanations import MetricExplanations

# 导出主要类供外部使用
__all__ = [
    'TrainingVisualizationWidget',
    'TensorBoardWidget', 
    'MetricsDataManager',
    'ChartRenderer',
    'MetricExplanations'
]