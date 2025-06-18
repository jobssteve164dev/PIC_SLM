"""
评估标签页组件包

本包包含了模型评估相关的所有组件，包括：
- 核心Widget组件
- 可视化组件  
- 工具类和辅助模块

组织结构：
- widgets/: 核心功能组件
- visualization/: 可视化相关组件
- utils/: 工具类和辅助模块
"""

# 从widgets子模块导入核心组件
from .widgets.model_evaluation_widget import ModelEvaluationWidget
from .widgets.enhanced_model_evaluation_widget import EnhancedModelEvaluationWidget
from .widgets.tensorboard_manager_widget import TensorBoardManagerWidget
from .widgets.training_curve_widget import TrainingCurveWidget
from .widgets.params_comparison_widget import ParamsComparisonWidget
from .widgets.training_visualization_widget import TrainingVisualizationWidget
from .widgets.tensorboard_widget import TensorBoardWidget

# 从visualization子模块导入可视化组件
from .visualization.visualization_container_widget import VisualizationContainerWidget

# 从utils子模块导入工具类
from .utils.chart_renderer import ChartRenderer
from .utils.metric_explanations import MetricExplanations
from .utils.metrics_data_manager import MetricsDataManager

__all__ = [
    # 核心Widget组件
    'ModelEvaluationWidget',
    'EnhancedModelEvaluationWidget',
    'TensorBoardManagerWidget', 
    'TrainingCurveWidget',
    'ParamsComparisonWidget',
    'TrainingVisualizationWidget',
    'TensorBoardWidget',
    
    # 可视化组件
    'VisualizationContainerWidget',
    
    # 工具类
    'ChartRenderer',
    'MetricExplanations', 
    'MetricsDataManager',
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'AI Project Team'
__description__ = '模型评估组件包' 