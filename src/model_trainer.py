"""
图片分类模型训练模块 - 重构版本

这是重构后的模型训练模块，使用了组件化架构：
- 将原有的复杂逻辑拆分到独立的组件中
- 保持与原始接口的完全兼容性
- 提升代码的可维护性和可扩展性

组件架构：
- ModelTrainer: 主控制器
- TrainingThread: 训练线程
- ModelFactory: 模型工厂
- WeightCalculator: 权重计算器
- ModelConfigurator: 模型配置器
- TensorBoardLogger: 日志记录器
- TrainingValidator: 配置验证器
"""

import warnings
from PyQt5.QtCore import QObject
from src.utils.logger import get_logger, log_error, performance_monitor

# 导入新的组件化训练器
try:
    from src.training_components import ModelTrainer as NewModelTrainer
    from src.training_components import TrainingThread as NewTrainingThread
    COMPONENTS_AVAILABLE = True
    print("✅ 成功导入新的组件化训练器")
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    print(f"❌ 新的训练组件导入失败: {e}")
    warnings.warn(
        "新的训练组件不可用，将使用原始实现。"
        "请确保 src.training_components 包已正确安装。",
        ImportWarning
    )

# 为了向后兼容，保留原始的类名和接口
if COMPONENTS_AVAILABLE:
    # 使用新的组件化实现
    class ModelTrainer(NewModelTrainer):
        """
        模型训练器 - 组件化版本
        
        这是使用新组件架构的ModelTrainer，提供与原始版本相同的接口，
        但内部使用了更好的组件化设计。
        """
        
        def __init__(self):
            super().__init__()
        
        # 为了向后兼容，保留原始方法名
        def train_model(self, data_dir, model_name, num_epochs, batch_size, learning_rate, 
                       model_save_dir, task_type='classification', use_tensorboard=True):
            """
            训练模型 - 向后兼容的接口
            
            Args:
                data_dir: 数据目录
                model_name: 模型名称
                num_epochs: 训练轮数
                batch_size: 批次大小
                learning_rate: 学习率
                model_save_dir: 模型保存目录
                task_type: 任务类型
                use_tensorboard: 是否使用TensorBoard
            """
            print(f"🔄 使用新组件化训练器，参数:")
            print(f"   data_dir: {data_dir}")
            print(f"   model_name: {model_name}")
            print(f"   num_epochs: {num_epochs}")
            print(f"   batch_size: {batch_size}")
            print(f"   learning_rate: {learning_rate}")
            print(f"   task_type: {task_type}")
            
            # 构建配置字典
            config = {
                'data_dir': data_dir,
                'model_name': model_name,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'model_save_dir': model_save_dir,
                'task_type': task_type,
                'use_tensorboard': use_tensorboard,
                'use_class_weights': True,  # 默认启用类别权重
                'weight_strategy': 'balanced',  # 默认权重策略
                'activation_function': 'ReLU',  # 默认激活函数
                'dropout_rate': 0.0,  # 默认无dropout
                'model_note': ''  # 默认无备注
            }
            
            print(f"📋 完整配置字典: {config}")
            
            # 调用新的配置接口
            self.train_model_with_config(config)
        
        def configure_model(self, model, layer_config):
            """
            配置模型 - 向后兼容的接口
            
            Args:
                model: PyTorch模型
                layer_config: 层配置
                
            Returns:
                配置后的模型
            """
            try:
                from utils.model_utils import configure_model_layers
                if layer_config and layer_config.get('enabled', False):
                    return configure_model_layers(model, layer_config)
                return model
            except ImportError:
                self.status_updated.emit("警告: utils.model_utils不可用，跳过层配置")
                return model
    
    class TrainingThread(NewTrainingThread):
        """
        训练线程 - 组件化版本
        
        为了向后兼容，保留原始的TrainingThread类名
        """
        
        def __init__(self, config, parent=None):
            super().__init__(config, parent)

else:
    # 如果新组件不可用，保留原始完整实现作为后备
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import models, transforms, datasets
    import torchvision
    from PyQt5.QtCore import QObject, pyqtSignal, QThread
    import numpy as np
    from typing import Dict, Any, Optional
    import json
    import subprocess
    import sys
    from torch.utils.tensorboard import SummaryWriter
    try:
        from detection_trainer import DetectionTrainer
    except ImportError:
        DetectionTrainer = None
    import time
    try:
        from utils.model_utils import create_model, configure_model_layers
    except ImportError:
        create_model = None
        configure_model_layers = None
    from sklearn.utils.class_weight import compute_class_weight
    from collections import Counter

    # 设置matplotlib后端为Agg，解决线程安全问题
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    class TrainingThread(QThread):
        """原始的训练线程实现 - 后备版本"""
        
        # 定义信号
        progress_updated = pyqtSignal(int)
        status_updated = pyqtSignal(str)
        training_finished = pyqtSignal()
        training_error = pyqtSignal(str)
        epoch_finished = pyqtSignal(dict)
        model_download_failed = pyqtSignal(str, str)
        training_stopped = pyqtSignal()
        
        def __init__(self, config, parent=None):
            super().__init__(parent)
            self.config = config
            self.stop_training = False
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = None
            self.training_info = {}
            self.class_weights = None
            self.class_distribution = None
        
        def run(self):
            """线程运行入口"""
            try:
                self.stop_training = False
                
                # 提取参数并调用训练
                data_dir = self.config.get('data_dir', '')
                model_name = self.config.get('model_name', 'ResNet50')
                num_epochs = self.config.get('num_epochs', 20)
                batch_size = self.config.get('batch_size', 32)
                learning_rate = self.config.get('learning_rate', 0.001)
                model_save_dir = self.config.get('model_save_dir', 'models/saved_models')
                task_type = self.config.get('task_type', 'classification')
                use_tensorboard = self.config.get('use_tensorboard', True)
                
                self.train_model(
                    data_dir, model_name, num_epochs, batch_size, learning_rate, 
                    model_save_dir, task_type, use_tensorboard
                )
                
                self.training_finished.emit()
                
            except Exception as e:
                self.training_error.emit(f"训练过程中发生错误: {str(e)}")
        
        def stop(self):
            """停止训练"""
            self.stop_training = True
            self.status_updated.emit("训练线程正在停止...")
        
        def train_model(self, data_dir, model_name, num_epochs, batch_size, learning_rate, 
                       model_save_dir, task_type='classification', use_tensorboard=True):
            """执行模型训练的简化版本"""
            try:
                self.status_updated.emit("正在使用后备训练实现...")
                
                if task_type != 'classification':
                    self.training_error.emit("后备实现仅支持分类任务")
                    return
                
                # 检查数据目录
                train_dir = os.path.join(data_dir, 'train')
                val_dir = os.path.join(data_dir, 'val')
                
                if not os.path.exists(train_dir) or not os.path.exists(val_dir):
                    self.training_error.emit("训练或验证数据目录不存在")
                    return
                
                # 简化的数据加载
                data_transforms = {
                    'train': transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                    'val': transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                }
                
                image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                                for x in ['train', 'val']}
                
                dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2)
                              for x in ['train', 'val']}
                
                dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
                class_names = image_datasets['train'].classes
                num_classes = len(class_names)
                
                # 创建简单模型
                if model_name.startswith('ResNet'):
                    model = models.resnet50(pretrained=True)
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                else:
                    model = models.resnet50(pretrained=True)
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                
                model = model.to(self.device)
                
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                
                # 简化的训练循环
                best_acc = 0.0
                for epoch in range(num_epochs):
                    if self.stop_training:
                        break
                    
                    for phase in ['train', 'val']:
                        if self.stop_training:
                            break
                        
                        if phase == 'train':
                            model.train()
                        else:
                            model.eval()
                        
                        running_loss = 0.0
                        running_corrects = 0
                        
                        for i, (inputs, labels) in enumerate(dataloaders[phase]):
                            if self.stop_training:
                                break
                            
                            inputs = inputs.to(self.device)
                            labels = labels.to(self.device)
                            
                            optimizer.zero_grad()
                            
                            with torch.set_grad_enabled(phase == 'train'):
                                outputs = model(inputs)
                                _, preds = torch.max(outputs, 1)
                                loss = criterion(outputs, labels)
                                
                                if phase == 'train':
                                    loss.backward()
                                    optimizer.step()
                            
                            running_loss += loss.item() * inputs.size(0)
                            running_corrects += torch.sum(preds == labels.data)
                            
                            # 更新进度
                            progress = int(((epoch * len(dataloaders[phase]) + i + 1) /
                                          (num_epochs * len(dataloaders[phase]))) * 100)
                            self.progress_updated.emit(progress)
                        
                        epoch_loss = running_loss / dataset_sizes[phase]
                        epoch_acc = running_corrects.double() / dataset_sizes[phase]
                        
                        # 发送epoch结果
                        epoch_data = {
                            'epoch': epoch + 1,
                            'phase': phase,
                            'loss': float(epoch_loss),
                            'accuracy': float(epoch_acc),
                            'batch': len(dataloaders[phase]),
                            'total_batches': len(dataloaders[phase])
                        }
                        self.epoch_finished.emit(epoch_data)
                        
                        # 保存最佳模型
                        if phase == 'val' and epoch_acc > best_acc:
                            best_acc = epoch_acc
                            model_path = os.path.join(model_save_dir, f'{model_name}_best.pth')
                            os.makedirs(model_save_dir, exist_ok=True)
                            torch.save(model.state_dict(), model_path)
                            self.status_updated.emit(f'保存最佳模型: {model_path}')
                
                self.status_updated.emit(f'训练完成，最佳准确率: {best_acc:.4f}')
                
            except Exception as e:
                self.training_error.emit(f"后备训练实现出错: {str(e)}")

    class ModelTrainer(QObject):
        """原始的模型训练器实现 - 后备版本"""
        
        # 定义信号
        progress_updated = pyqtSignal(int)
        status_updated = pyqtSignal(str)
        training_finished = pyqtSignal()
        training_error = pyqtSignal(str)
        epoch_finished = pyqtSignal(dict)
        model_download_failed = pyqtSignal(str, str)
        training_stopped = pyqtSignal()
        metrics_updated = pyqtSignal(dict)
        tensorboard_updated = pyqtSignal(str, float, int)

        def __init__(self):
            super().__init__()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = None
            self.stop_training = False
            self.training_thread = None
            self.detection_trainer = None

        def train_model_with_config(self, config):
            """使用配置训练模型"""
            try:
                print("⚠️ 警告：正在使用后备训练实现！")
                print(f"   这表明新的组件化训练器导入失败")
                print(f"   配置参数: {config}")
                self.status_updated.emit("使用后备训练实现...")
                
                # 简单验证
                if not config.get('data_dir'):
                    self.training_error.emit("数据目录不能为空")
                    return
                
                # 启动训练线程
                self.training_thread = TrainingThread(config)
                
                # 连接信号
                self.training_thread.progress_updated.connect(self.progress_updated)
                self.training_thread.status_updated.connect(self.status_updated)
                self.training_thread.training_finished.connect(self.training_finished)
                self.training_thread.training_error.connect(self.training_error)
                self.training_thread.epoch_finished.connect(self.epoch_finished)
                self.training_thread.model_download_failed.connect(self.model_download_failed)
                self.training_thread.training_stopped.connect(self.training_stopped)
                
                self.training_thread.start()
                
            except Exception as e:
                self.training_error.emit(f"训练初始化出错: {str(e)}")

        def train_model(self, data_dir, model_name, num_epochs, batch_size, learning_rate, 
                       model_save_dir, task_type='classification', use_tensorboard=True):
            """向后兼容的训练接口"""
            config = {
                'data_dir': data_dir,
                'model_name': model_name,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'model_save_dir': model_save_dir,
                'task_type': task_type,
                'use_tensorboard': use_tensorboard
            }
            self.train_model_with_config(config)

        def stop(self):
            """停止训练"""
            try:
                if self.training_thread and self.training_thread.isRunning():
                    self.training_thread.stop()
                    self.training_thread.wait()
                
                if self.detection_trainer:
                    self.detection_trainer.stop()
                
                self.stop_training = True
                self.status_updated.emit("训练已停止")
                
            except Exception as e:
                print(f"停止训练时出错: {str(e)}")
            
            self.training_stopped.emit()

        def configure_model(self, model, layer_config):
            """配置模型"""
            if layer_config and layer_config.get('enabled', False) and configure_model_layers:
                return configure_model_layers(model, layer_config)
            return model

# 导出兼容性说明
__version__ = "2.0.0"
__architecture__ = "组件化" if COMPONENTS_AVAILABLE else "原始实现"

def get_architecture_info():
    """获取当前架构信息"""
    return {
        'version': __version__,
        'architecture': __architecture__,
        'components_available': COMPONENTS_AVAILABLE,
        'description': '重构的组件化训练架构' if COMPONENTS_AVAILABLE else '后备的原始实现'
    }