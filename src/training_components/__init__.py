"""
训练组件包 - 提供模型训练相关的各种组件

主要组件：
- ModelTrainer: 主要的模型训练类
- TrainingThread: 训练线程类
- ModelFactory: 模型创建工厂
- WeightCalculator: 类别权重计算器
- ModelConfigurator: 模型配置器
- TensorBoardLogger: TensorBoard日志记录器
- TrainingValidator: 训练配置验证器
"""

from .model_trainer import ModelTrainer
from .training_thread import TrainingThread
from .model_factory import ModelFactory
from .weight_calculator import WeightCalculator
from .model_configurator import ModelConfigurator
from .tensorboard_logger import TensorBoardLogger
from .training_validator import TrainingValidator
from .optimizer_factory import OptimizerFactory, WarmupLRScheduler, LabelSmoothingCrossEntropy
from .model_ema import ModelEMA, ModelEMAManager
from .advanced_augmentation import (
    MixUpAugmentation, CutMixAugmentation, AdvancedAugmentationManager,
    MixCriterion, LabelSmoothingCrossEntropy as AdvancedLabelSmoothingCE,
    create_advanced_criterion
)

__all__ = [
    'ModelTrainer',
    'TrainingThread', 
    'ModelFactory',
    'WeightCalculator',
    'ModelConfigurator',
    'TensorBoardLogger',
    'TrainingValidator',
    'OptimizerFactory',
    'WarmupLRScheduler',
    'LabelSmoothingCrossEntropy',
    'ModelEMA',
    'ModelEMAManager',
    'MixUpAugmentation',
    'CutMixAugmentation',
    'AdvancedAugmentationManager',
    'MixCriterion',
    'create_advanced_criterion'
] 