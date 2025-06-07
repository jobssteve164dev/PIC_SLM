"""
模型训练器 - 主要的训练控制器和接口

这是拆分后的主要训练控制器，负责：
- 协调各个训练组件
- 管理训练流程
- 处理不同任务类型的训练（分类、检测）
- 提供统一的接口给UI层
"""

import os
import time
import json
from PyQt5.QtCore import QObject, pyqtSignal

from .training_thread import TrainingThread
from .training_validator import TrainingValidator


class ModelTrainer(QObject):
    """模型训练器主类"""
    
    # 定义信号
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    training_finished = pyqtSignal()
    training_error = pyqtSignal(str)
    epoch_finished = pyqtSignal(dict)
    model_download_failed = pyqtSignal(str, str)  # 模型名称，下载链接
    training_stopped = pyqtSignal()
    
    # 检测任务相关信号
    metrics_updated = pyqtSignal(dict)
    tensorboard_updated = pyqtSignal(str, float, int)

    def __init__(self):
        super().__init__()
        self.training_thread = None
        self.detection_trainer = None
        self.stop_training = False
        self.validator = TrainingValidator()
        
        # 连接验证器信号
        self.validator.status_updated.connect(self.status_updated)
        self.validator.validation_error.connect(self.training_error)

    def train_model_with_config(self, config):
        """
        使用配置训练模型
        
        Args:
            config: 训练配置字典
        """
        try:
            # 验证配置
            if not self.validator.validate_config(config):
                return
            
            # 保存配置文件
            self._save_config_file(config)
            
            # 根据任务类型选择训练方法
            task_type = config.get('task_type', 'classification')
            
            if task_type == 'classification':
                self._start_classification_training(config)
            elif task_type == 'detection':
                self._start_detection_training(config)
            else:
                self.training_error.emit(f"不支持的任务类型: {task_type}")
                
        except Exception as e:
            self.training_error.emit(f"训练初始化时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def _save_config_file(self, config):
        """保存训练配置文件"""
        try:
            # 获取时间戳和模型信息
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_name = config.get('model_name', 'Unknown')
            model_note = config.get('model_note', '')
            
            # 构建文件名
            model_filename = f"{model_name}_{timestamp}"
            if model_note:
                model_filename += f"_{model_note}"
            
            # 获取保存目录
            model_save_dir = config.get('model_save_dir', 'models/saved_models')
            param_save_dir = config.get('default_param_save_dir', model_save_dir)
            
            # 标准化路径
            model_save_dir = os.path.normpath(model_save_dir)
            param_save_dir = os.path.normpath(param_save_dir)
            
            # 确保目录存在
            os.makedirs(model_save_dir, exist_ok=True)
            os.makedirs(param_save_dir, exist_ok=True)
            
            # 在配置中添加文件名信息
            config['model_filename'] = model_filename
            config['timestamp'] = timestamp
            
            # 保存配置文件
            config_file_path = os.path.join(param_save_dir, f"{model_filename}_config.json")
            with open(config_file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            
            self.status_updated.emit(f"训练配置已保存到: {config_file_path}")
            
        except Exception as e:
            self.training_error.emit(f"保存训练配置文件时发生错误: {str(e)}")

    def _start_classification_training(self, config):
        """启动分类模型训练"""
        self.status_updated.emit("启动分类模型训练...")
        
        # 创建训练线程
        self.training_thread = TrainingThread(config)
        
        # 连接信号
        self._connect_classification_signals()
        
        # 启动训练线程
        self.training_thread.start()

    def _start_detection_training(self, config):
        """启动目标检测模型训练"""
        self.status_updated.emit("启动目标检测模型训练...")
        
        try:
            # 导入检测训练器
            from detection_trainer import DetectionTrainer
            
            # 创建检测训练器实例
            self.detection_trainer = DetectionTrainer(config)
            
            # 连接信号
            self._connect_detection_signals()
            
            # 启动训练
            self.detection_trainer.start_training(config)
            
        except Exception as e:
            self.training_error.emit(f"创建目标检测训练器时出错: {str(e)}")

    def _connect_classification_signals(self):
        """连接分类训练信号"""
        if self.training_thread:
            self.training_thread.progress_updated.connect(self.progress_updated)
            self.training_thread.status_updated.connect(self.status_updated)
            self.training_thread.training_finished.connect(self.training_finished)
            self.training_thread.training_error.connect(self.training_error)
            self.training_thread.epoch_finished.connect(self.epoch_finished)
            self.training_thread.model_download_failed.connect(self.model_download_failed)
            self.training_thread.training_stopped.connect(self.training_stopped)

    def _connect_detection_signals(self):
        """连接检测训练信号"""
        if self.detection_trainer:
            self.detection_trainer.progress_updated.connect(self.progress_updated)
            self.detection_trainer.status_updated.connect(self.status_updated)
            self.detection_trainer.training_finished.connect(self.training_finished)
            self.detection_trainer.training_error.connect(self.training_error)
            self.detection_trainer.metrics_updated.connect(self.metrics_updated)
            self.detection_trainer.tensorboard_updated.connect(self.tensorboard_updated)
            self.detection_trainer.model_download_failed.connect(self.model_download_failed)
            self.detection_trainer.training_stopped.connect(self.training_stopped)

    def stop(self):
        """停止训练过程"""
        try:
            self.stop_training = True
            
            # 停止分类训练线程
            if self.training_thread and self.training_thread.isRunning():
                self.training_thread.stop()
                self.training_thread.wait()
            
            # 停止检测训练
            if self.detection_trainer:
                self.detection_trainer.stop()
            
            self.status_updated.emit("训练已停止")
            
        except Exception as e:
            print(f"停止训练时出错: {str(e)}")
        
        # 发射训练停止信号
        self.training_stopped.emit()

    def get_supported_models(self):
        """获取支持的模型列表"""
        return [
            'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
            'MobileNetV2', 'MobileNetV3',
            'VGG16', 'VGG19',
            'DenseNet121', 'DenseNet169', 'DenseNet201',
            'InceptionV3',
            'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4',
            'Xception'
        ]

    def get_supported_activation_functions(self):
        """获取支持的激活函数列表"""
        return [
            'None', 'ReLU', 'LeakyReLU', 'PReLU', 'ELU', 'SELU', 'GELU', 'Mish', 'Swish', 'SiLU'
        ]

    def get_supported_weight_strategies(self):
        """获取支持的权重策略列表"""
        return ['balanced', 'inverse', 'log_inverse', 'custom']

    def get_supported_task_types(self):
        """获取支持的任务类型列表"""
        return ['classification', 'detection']

    def is_training_active(self):
        """检查是否有训练正在进行"""
        classification_active = (self.training_thread and 
                               self.training_thread.isRunning() and 
                               not self.training_thread.stop_training)
        
        detection_active = (self.detection_trainer and 
                          hasattr(self.detection_trainer, 'is_training') and 
                          self.detection_trainer.is_training)
        
        return classification_active or detection_active

    def get_training_info(self):
        """获取训练信息"""
        if self.training_thread:
            return getattr(self.training_thread, 'training_info', {})
        return {} 