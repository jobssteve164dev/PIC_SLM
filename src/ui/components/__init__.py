# 训练可视化组件包 

# 评估组件模块
from .model_evaluation_widget import ModelEvaluationWidget
from .tensorboard_manager_widget import TensorBoardManagerWidget
from .training_curve_widget import TrainingCurveWidget
from .params_comparison_widget import ParamsComparisonWidget
from .visualization_container_widget import VisualizationContainerWidget

# 现有的组件
from .training_visualization_widget import TrainingVisualizationWidget
from .tensorboard_widget import TensorBoardWidget
from .chart_manager import ChartManager
from .chart_renderer import ChartRenderer
from .metrics_data_manager import MetricsDataManager
from .result_display_manager import ResultDisplayManager
from .weight_generator import WeightGenerator
from .metric_explanations import MetricExplanations

__all__ = [
    # 评估组件
    'ModelEvaluationWidget',
    'TensorBoardManagerWidget', 
    'TrainingCurveWidget',
    'ParamsComparisonWidget',
    'VisualizationContainerWidget',
    
    # 现有组件
    'TrainingVisualizationWidget',
    'TensorBoardWidget',
    'ChartManager',
    'ChartRenderer',
    'MetricsDataManager',
    'ResultDisplayManager',
    'WeightGenerator',
    'MetricExplanations'
] 