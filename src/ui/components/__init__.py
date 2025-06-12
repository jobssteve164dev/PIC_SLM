# 训练可视化组件包 

# 从新的evaluation子包导入评估组件
from .evaluation import (
    ModelEvaluationWidget,
    TensorBoardManagerWidget,
    TrainingCurveWidget,
    ParamsComparisonWidget,
    VisualizationContainerWidget,
    TrainingVisualizationWidget,
    TensorBoardWidget,
    ChartRenderer,
    MetricsDataManager,
    MetricExplanations
)

# 数据集评估组件模块
from .dataset_evaluation import (
    ClassificationAnalyzer, 
    DetectionAnalyzer, 
    BaseDatasetAnalyzer,
    WeightGenerator,
    ChartManager,
    ResultDisplayManager
)

# 设置组件模块
from .settings import (
    ClassWeightWidget,
    ModelConfigWidget,
    FolderConfigWidget,
    ConfigManager,
    WeightStrategy
)

# 标注组件模块
from .annotation import (
    ClassificationWidget,
    DetectionWidget,
    AnnotationCanvas
)

# 训练组件模块
from .training import (
    ClassificationTrainingWidget,
    DetectionTrainingWidget,
    TrainingControlWidget,
    TrainingHelpDialog,
    LayerConfigWidget,
    ModelDownloadDialog,
    WeightConfigManager
)

# 模型分析组件
from .model_analysis_widget import ModelAnalysisWidget

__all__ = [
    # 评估组件 (从evaluation子包导入)
    'ModelEvaluationWidget',
    'TensorBoardManagerWidget', 
    'TrainingCurveWidget',
    'ParamsComparisonWidget',
    'VisualizationContainerWidget',
    'TrainingVisualizationWidget',
    'TensorBoardWidget',
    'ChartRenderer',
    'MetricsDataManager',
    'MetricExplanations',
    
    # 数据集评估组件
    'ClassificationAnalyzer',
    'DetectionAnalyzer', 
    'BaseDatasetAnalyzer',
    'WeightGenerator',
    'ChartManager',
    'ResultDisplayManager',
    
    # 设置组件
    'ClassWeightWidget',
    'ModelConfigWidget',
    'FolderConfigWidget',
    'ConfigManager',
    'WeightStrategy',
    
    # 标注组件
    'ClassificationWidget',
    'DetectionWidget',
    'AnnotationCanvas',
    
    # 训练组件
    'ClassificationTrainingWidget',
    'DetectionTrainingWidget',
    'TrainingControlWidget',
    'TrainingHelpDialog',
    'LayerConfigWidget',
    'ModelDownloadDialog',
    'WeightConfigManager',
    
    # 模型分析组件
    'ModelAnalysisWidget'
] 