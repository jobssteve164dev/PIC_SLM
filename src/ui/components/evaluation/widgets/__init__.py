"""
评估标签页核心Widget组件

包含所有评估功能的核心UI组件：
- EnhancedModelEvaluationWidget: 增强的模型评估组件（包含精确率、召回率、F1分数、混淆矩阵、多模型对比等专业指标）
- TensorBoardManagerWidget: TensorBoard管理
- TensorBoardParamsGuideWidget: TensorBoard参数监控说明
- TrainingCurveWidget: 训练曲线显示
- ParamsComparisonWidget: 训练参数对比
- TrainingVisualizationWidget: 训练可视化
- TensorBoardWidget: TensorBoard嵌入式组件
"""

from .enhanced_model_evaluation_widget import EnhancedModelEvaluationWidget
from .tensorboard_manager_widget import TensorBoardManagerWidget
from .tensorboard_params_guide_widget import TensorBoardParamsGuideWidget
from .training_curve_widget import TrainingCurveWidget
from .params_comparison_widget import ParamsComparisonWidget
from .training_visualization_widget import TrainingVisualizationWidget
from .tensorboard_widget import TensorBoardWidget

__all__ = [
    'EnhancedModelEvaluationWidget',
    'TensorBoardManagerWidget',
    'TensorBoardParamsGuideWidget',
    'TrainingCurveWidget', 
    'ParamsComparisonWidget',
    'TrainingVisualizationWidget',
    'TensorBoardWidget',
] 