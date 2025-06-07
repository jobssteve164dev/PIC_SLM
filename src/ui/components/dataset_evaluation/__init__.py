"""
DatasetEvaluation组件模块

该模块包含了数据集评估功能的所有相关组件：
- 数据集分析器（dataset_analyzers）
- 权重生成器（weight_generator）
- 图表管理器（chart_manager）
- 结果显示管理器（result_display_manager）

使用方式：
from .components.dataset_evaluation import ClassificationAnalyzer, WeightGenerator, ChartManager, ResultDisplayManager
"""

from .dataset_analyzers import ClassificationAnalyzer, DetectionAnalyzer, BaseDatasetAnalyzer
from .weight_generator import WeightGenerator
from .chart_manager import ChartManager
from .result_display_manager import ResultDisplayManager

__all__ = [
    'ClassificationAnalyzer',
    'DetectionAnalyzer', 
    'BaseDatasetAnalyzer',
    'WeightGenerator',
    'ChartManager',
    'ResultDisplayManager'
] 