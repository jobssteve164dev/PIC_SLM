"""
训练线程 - 在单独线程中执行模型训练过程

主要功能：
- 在后台线程中执行训练，避免阻塞UI
- 处理训练过程中的各种状态更新
- 支持训练过程的停止控制
- 集成各种训练组件（模型工厂、权重计算器等）
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
from PyQt5.QtCore import QThread, pyqtSignal

from .model_factory import ModelFactory
from .weight_calculator import WeightCalculator
from .model_configurator import ModelConfigurator
from .tensorboard_logger import TensorBoardLogger
from .training_validator import TrainingValidator
from .resource_limited_trainer import ResourceLimitedTrainer, enable_resource_limited_training
from ..utils.resource_limiter import (
    initialize_resource_limiter, ResourceLimits, ResourceLimitException, get_resource_limiter
)


class TrainingThread(QThread):
    """负责在单独线程中执行训练过程的类"""
    
    # 定义信号
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    training_finished = pyqtSignal()
    training_error = pyqtSignal(str)
    epoch_finished = pyqtSignal(dict)
    model_download_failed = pyqtSignal(str, str)  # 模型名称，下载链接
    training_stopped = pyqtSignal()
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.stop_training = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.training_info = {}
        
        # 初始化各个组件
        self.model_factory = ModelFactory()
        self.weight_calculator = WeightCalculator()
        self.model_configurator = ModelConfigurator()
        self.tensorboard_logger = TensorBoardLogger()
        self.validator = TrainingValidator()
        
        # 初始化资源限制器
        self.resource_limiter = None
        self.resource_limited_trainer = None
        self._setup_resource_limiter()
        
        # 连接组件信号
        self._connect_component_signals()
    
    def _setup_resource_limiter(self):
        """设置资源限制器"""
        try:
            # 从配置中获取资源限制设置
            resource_limits_config = self.config.get('resource_limits', {})
            
            # 检查是否启用强制资源限制（从训练界面或设置界面）
            enable_from_ui = self.config.get('enable_resource_limits', False)  # 从训练界面
            enable_from_settings = resource_limits_config.get('enforce_limits_enabled', False)  # 从设置界面
            
            if enable_from_ui or enable_from_settings:
                # 创建资源限制配置
                limits = ResourceLimits(
                    max_memory_gb=resource_limits_config.get('memory_absolute_limit_gb', 8.0),
                    max_cpu_percent=resource_limits_config.get('cpu_percent_limit', 80.0),
                    max_disk_usage_gb=resource_limits_config.get('temp_files_limit_gb', 10.0),
                    max_processes=4,
                    max_threads=resource_limits_config.get('cpu_cores_limit', 8),
                    check_interval=resource_limits_config.get('check_interval', 2.0),
                    enforce_limits=True,
                    auto_cleanup=resource_limits_config.get('auto_cleanup_enabled', True)
                )
                
                # 初始化全局资源限制器
                self.resource_limiter = initialize_resource_limiter(limits)
                
                # 添加回调处理资源超限
                self.resource_limiter.add_callback('memory_limit', self._on_resource_limit_exceeded)
                self.resource_limiter.add_callback('cpu_limit', self._on_resource_limit_exceeded)
                self.resource_limiter.add_callback('disk_limit', self._on_resource_limit_exceeded)
                self.resource_limiter.add_callback('process_limit', self._on_resource_limit_exceeded)
                
                source = "训练界面" if enable_from_ui else "设置界面"
                print(f"✅ 训练进程启用强制资源限制(来源: {source}): 内存{limits.max_memory_gb}GB, CPU{limits.max_cpu_percent}%")
            else:
                print("ℹ️ 训练进程未启用强制资源限制，仅使用监控模式")
                
        except Exception as e:
            print(f"⚠️ 设置资源限制器失败: {e}")
            self.resource_limiter = None
    
    def _on_resource_limit_exceeded(self, event_type: str, current_value: float, limit_value: float):
        """处理资源限制超限"""
        resource_name = {"memory_limit": "内存", "cpu_limit": "CPU", 
                        "disk_limit": "磁盘", "process_limit": "进程"}
        resource_name = resource_name.get(event_type, event_type)
        
        error_msg = f"🚨 训练过程{resource_name}资源超限！当前: {current_value:.2f}, 限制: {limit_value:.2f}"
        print(error_msg)
        self.status_updated.emit(error_msg)
        
        # 停止训练
        self.stop_training = True
        self.training_error.emit(f"训练因{resource_name}资源超限而中断")
    
    def _connect_component_signals(self):
        """连接各个组件的信号"""
        self.model_factory.status_updated.connect(self.status_updated)
        self.model_factory.model_download_failed.connect(self.model_download_failed)
        
        self.weight_calculator.status_updated.connect(self.status_updated)
        
        self.model_configurator.status_updated.connect(self.status_updated)
        
        self.tensorboard_logger.status_updated.connect(self.status_updated)
        
        self.validator.status_updated.connect(self.status_updated)
        self.validator.validation_error.connect(self.training_error)
    
    def run(self):
        """线程运行入口，执行模型训练"""
        try:
            # 重置停止标志
            self.stop_training = False
            
            # 🔍 完整的参数接收验证
            print("=" * 60)
            print("🔍 训练线程参数接收验证")
            print("=" * 60)
            
            # 基础训练参数
            print("📋 基础训练参数:")
            basic_params = ['data_dir', 'model_name', 'num_epochs', 'batch_size', 'learning_rate', 
                          'model_save_dir', 'task_type', 'use_tensorboard']
            for param in basic_params:
                value = self.config.get(param, '未设置')
                print(f"   {param}: {value}")
            
            # 高级训练参数
            print("\n🔧 高级训练参数:")
            advanced_params = ['optimizer', 'weight_decay', 'lr_scheduler', 'use_augmentation',
                             'early_stopping', 'early_stopping_patience', 'gradient_clipping',
                             'gradient_clipping_value', 'mixed_precision', 'dropout_rate',
                             'activation_function']
            for param in advanced_params:
                value = self.config.get(param, '未设置')
                print(f"   {param}: {value}")
            
            # 预训练模型参数
            print("\n🏗️ 预训练模型参数:")
            pretrained_params = ['use_pretrained', 'pretrained_path', 'use_local_pretrained', 'pretrained_model']
            for param in pretrained_params:
                value = self.config.get(param, '未设置')
                print(f"   {param}: {value}")
            
            # 类别权重参数
            print("\n⚖️ 类别权重参数:")
            weight_params = ['use_class_weights', 'weight_strategy', 'class_weights', 'custom_class_weights']
            for param in weight_params:
                value = self.config.get(param, '未设置')
                print(f"   {param}: {value}")
            
            # 目标检测特有参数（如果是检测任务）
            if self.config.get('task_type') == 'detection':
                print("\n🎯 目标检测特有参数:")
                detection_params = ['iou_threshold', 'conf_threshold', 'resolution', 'use_mosaic',
                                  'use_multiscale', 'use_ema', 'nms_threshold', 'use_fpn']
                for param in detection_params:
                    value = self.config.get(param, '未设置')
                    print(f"   {param}: {value}")
            
            # 资源限制参数
            print("\n💾 资源与控制参数:")
            resource_params = ['enable_resource_limits', 'metrics', 'model_note', 'layer_config']
            for param in resource_params:
                value = self.config.get(param, '未设置')
                if param == 'layer_config' and isinstance(value, dict):
                    print(f"   {param}: 已配置层参数 (共{len(value)}项)")
                else:
                    print(f"   {param}: {value}")
            
            # 目录配置参数
            print("\n📁 目录配置参数:")
            dir_params = ['default_param_save_dir', 'tensorboard_log_dir']
            for param in dir_params:
                value = self.config.get(param, '未设置')
                print(f"   {param}: {value}")
            
            print("=" * 60)
            print(f"✅ 参数接收验证完成，共接收 {len(self.config)} 个参数")
            print("=" * 60)
            
            # 启动资源限制器监控
            if self.resource_limiter:
                self.resource_limiter.start_monitoring()
                self.status_updated.emit("✅ 强制资源限制已启动")
            
            # 验证配置
            if not self.validator.validate_config(self.config):
                return
            
            # 提取基本参数
            data_dir = self.config.get('data_dir', '')
            model_name = self.config.get('model_name', 'ResNet50')
            num_epochs = self.config.get('num_epochs', 20)
            batch_size = self.config.get('batch_size', 32)
            learning_rate = self.config.get('learning_rate', 0.001)
            model_save_dir = self.config.get('model_save_dir', 'models/saved_models')
            task_type = self.config.get('task_type', 'classification')
            use_tensorboard = self.config.get('use_tensorboard', True)
            
            print(f"🚀 开始执行训练流程...")
            
            # 调用训练流程
            self.train_model(
                data_dir=data_dir,
                model_name=model_name,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                model_save_dir=model_save_dir,
                task_type=task_type,
                use_tensorboard=use_tensorboard
            )
            
            # 训练完成
            self.training_finished.emit()
            
        except Exception as e:
            self.training_error.emit(f"训练过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # 确保停止资源限制器
            if self.resource_limiter:
                self.resource_limiter.stop_monitoring()
                self.status_updated.emit("🔚 资源限制器已停止")
    
    def stop(self):
        """停止训练过程"""
        self.stop_training = True
        
        # 停止资源限制器
        if self.resource_limiter:
            self.resource_limiter.request_stop()
            self.status_updated.emit("🛑 已请求资源限制器停止所有操作")
        
        self.status_updated.emit("训练线程正在停止...")
    
    def train_model(self, data_dir, model_name, num_epochs, batch_size, learning_rate, 
                   model_save_dir, task_type='classification', use_tensorboard=True):
        """执行模型训练"""
        try:
            # 标准化路径格式
            data_dir = os.path.normpath(data_dir).replace('\\', '/')
            model_save_dir = os.path.normpath(model_save_dir).replace('\\', '/')
            
            # 准备数据
            dataloaders, dataset_sizes, class_names, num_classes = self._prepare_data(
                data_dir, batch_size, task_type
            )
            
            if self.stop_training:
                return
            
            # 创建和配置模型
            self.model = self._create_and_configure_model(model_name, num_classes, task_type)
            
            if self.stop_training:
                return
            
            # 计算类别权重和设置损失函数
            criterion = self._setup_loss_function(dataloaders['train'], class_names)
            
            if self.stop_training:
                return
            
            # 设置优化器
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # 初始化TensorBoard
            tensorboard_log_dir = None
            if use_tensorboard:
                tensorboard_log_dir = self.tensorboard_logger.initialize(self.config, model_name)
                self.training_info['tensorboard_log_dir'] = tensorboard_log_dir
                
                # 记录模型图和类别信息
                self.tensorboard_logger.log_model_graph(self.model, dataloaders['train'], self.device)
                
                class_weights = getattr(self.weight_calculator, 'class_weights', None)
                class_distribution = self.weight_calculator.get_class_distribution()
                if class_weights is not None and class_distribution:
                    self.tensorboard_logger.log_class_info(
                        class_names, class_distribution, class_weights, epoch=0
                    )
            
            if self.stop_training:
                return
            
            # 执行训练循环（使用资源限制的训练器如果启用）
            if self.resource_limiter:
                # 使用资源限制的训练器
                self.resource_limited_trainer = enable_resource_limited_training(self)
                best_acc = self._resource_limited_training_loop(
                    dataloaders, dataset_sizes, class_names, num_epochs, 
                    criterion, optimizer, model_name, model_save_dir
                )
            else:
                # 使用标准训练循环
                best_acc = self._training_loop(
                    dataloaders, dataset_sizes, class_names, num_epochs, 
                    criterion, optimizer, model_name, model_save_dir
                )
            
            if self.stop_training:
                return
            
            # 保存训练信息
            self._save_training_info(model_name, num_epochs, batch_size, learning_rate, 
                                   best_acc, class_names, model_save_dir)
            
            # 关闭TensorBoard
            self.tensorboard_logger.close()
            
            self.status_updated.emit(f'训练完成，最佳准确率: {best_acc:.4f}')
            
        except Exception as e:
            self.training_error.emit(f"训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _prepare_data(self, data_dir, batch_size, task_type):
        """准备训练数据"""
        if task_type != 'classification':
            raise ValueError(f"当前仅支持分类任务，不支持: {task_type}")
        
        self.status_updated.emit("加载分类数据集...")
        
        # 数据转换
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
        
        # 加载数据集
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                data_transforms[x])
                        for x in ['train', 'val']}
        
        dataloaders = {x: DataLoader(image_datasets[x],
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=4)
                      for x in ['train', 'val']}
        
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        num_classes = len(class_names)
        
        return dataloaders, dataset_sizes, class_names, num_classes
    
    def _create_and_configure_model(self, model_name, num_classes, task_type):
        """创建和配置模型"""
        # 创建模型
        model = self.model_factory.create_model(model_name, num_classes, task_type)
        
        # 配置模型
        model = self.model_configurator.configure_model(model, self.config)
        
        # 移到设备
        model = model.to(self.device)
        
        return model
    
    def _setup_loss_function(self, train_dataset, class_names):
        """设置损失函数"""
        use_class_weights = self.config.get('use_class_weights', True)
        weight_strategy = self.config.get('weight_strategy', 'balanced')
        
        if use_class_weights:
            class_weights = self.weight_calculator.calculate_class_weights(
                train_dataset.dataset, class_names, self.config, self.device
            )
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            self.status_updated.emit(f"使用加权损失函数，权重策略: {weight_strategy}")
        else:
            criterion = nn.CrossEntropyLoss()
            self.status_updated.emit("使用标准损失函数（无类别权重）")
        
        return criterion
    
    def _training_loop(self, dataloaders, dataset_sizes, class_names, num_epochs, 
                      criterion, optimizer, model_name, model_save_dir):
        """执行训练循环"""
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            if self.stop_training:
                self.status_updated.emit("训练已停止")
                break
            
            self.status_updated.emit(f'Epoch {epoch+1}/{num_epochs}')
            
            # 训练和验证阶段
            for phase in ['train', 'val']:
                if self.stop_training:
                    break
                
                epoch_loss, epoch_acc, all_preds, all_labels = self._train_epoch(
                    phase, dataloaders, dataset_sizes, criterion, optimizer, epoch, num_epochs
                )
                
                if self.stop_training:
                    break
                
                # 记录到TensorBoard
                self.tensorboard_logger.log_epoch_metrics(epoch, phase, epoch_loss, epoch_acc)
                
                # 记录样本图像（每5个epoch一次）
                if phase == 'val' and epoch % 5 == 0:
                    self.tensorboard_logger.log_sample_images(dataloaders[phase], epoch)
                
                # 记录混淆矩阵（验证阶段）
                if phase == 'val':
                    self.tensorboard_logger.log_confusion_matrix(
                        all_labels, all_preds, class_names, epoch
                    )
                
                # 刷新TensorBoard数据
                self.tensorboard_logger.flush()
            
            if self.stop_training:
                break
            
            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                self._save_best_model(model_name, model_save_dir, epoch, best_acc)
        
        return best_acc
    
    def _resource_limited_training_loop(self, dataloaders, dataset_sizes, class_names, num_epochs, 
                                       criterion, optimizer, model_name, model_save_dir):
        """执行带资源限制的训练循环"""
        try:
            best_acc = 0.0
            
            def save_callback(epoch, model, optimizer, train_result, val_result):
                """保存回调函数"""
                nonlocal best_acc
                val_acc = val_result.get('val_accuracy', 0) / 100.0  # 转换为小数
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    self._save_best_model(model_name, model_save_dir, epoch, best_acc)
                
                # 记录到TensorBoard
                if hasattr(self, 'tensorboard_logger'):
                    self.tensorboard_logger.log_epoch_metrics(epoch-1, 'train', 
                                                            train_result['loss'], train_result['accuracy']/100.0)
                    self.tensorboard_logger.log_epoch_metrics(epoch-1, 'val', 
                                                            val_result['val_loss'], val_result['val_accuracy']/100.0)
                    self.tensorboard_logger.flush()
                
                # 发送epoch结果
                epoch_data = {
                    'epoch': epoch,
                    'phase': 'val',
                    'loss': val_result['val_loss'],
                    'accuracy': val_result['val_accuracy'],
                    'batch': len(dataloaders['val']),
                    'total_batches': len(dataloaders['val'])
                }
                self.epoch_finished.emit(epoch_data)
                
                # 更新进度
                progress = int((epoch / num_epochs) * 100)
                self.progress_updated.emit(progress)
            
            # 使用资源限制的训练器
            self.resource_limited_trainer.train_with_resource_limits(
                epochs=num_epochs,
                train_loader=dataloaders['train'],
                val_loader=dataloaders['val'],
                model=self.model,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
                save_callback=save_callback
            )
            
            return best_acc
            
        except ResourceLimitException as e:
            error_msg = f"训练因资源限制中断: {e}"
            self.status_updated.emit(error_msg)
            self.training_error.emit(error_msg)
            return 0.0
        except Exception as e:
            error_msg = f"资源限制训练循环出错: {e}"
            self.status_updated.emit(error_msg)
            self.training_error.emit(error_msg)
            return 0.0
    
    def _train_epoch(self, phase, dataloaders, dataset_sizes, criterion, optimizer, epoch, num_epochs):
        """训练一个epoch"""
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        
        # 遍历数据
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            if self.stop_training:
                break
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # 前向传播
            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # 反向传播
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            
            if self.stop_training:
                break
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # 收集预测和标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度
            progress = int(((epoch * len(dataloaders[phase]) + i + 1) /
                          (num_epochs * len(dataloaders[phase]))) * 100)
            self.progress_updated.emit(progress)
            
            # 发送训练状态更新
            if i % 10 == 0:
                current_loss = running_loss / ((i + 1) * inputs.size(0))
                current_acc = running_corrects.double() / ((i + 1) * inputs.size(0))
                epoch_data = {
                    'epoch': epoch + 1,
                    'phase': phase,
                    'loss': float(current_loss),
                    'accuracy': float(current_acc.item()),
                    'batch': i + 1,
                    'total_batches': len(dataloaders[phase])
                }
                self.epoch_finished.emit(epoch_data)
        
        if self.stop_training:
            return 0, 0, [], []
        
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
        # 发送epoch结果
        epoch_data = {
            'epoch': epoch + 1,
            'phase': phase,
            'loss': float(epoch_loss),
            'accuracy': float(epoch_acc.item()) if isinstance(epoch_acc, torch.Tensor) else float(epoch_acc),
            'batch': len(dataloaders[phase]),
            'total_batches': len(dataloaders[phase])
        }
        self.epoch_finished.emit(epoch_data)
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def _save_best_model(self, model_name, model_save_dir, epoch, best_acc):
        """保存最佳模型"""
        model_note = self.config.get('model_note', '')
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # 保存PyTorch模型
        model_save_path = os.path.join(model_save_dir, f'{model_name}_{timestamp}_{model_note}_best.pth')
        torch.save(self.model.state_dict(), model_save_path)
        self.status_updated.emit(f'保存最佳模型，Epoch {epoch+1}, Acc: {best_acc:.4f}')
        
        # 导出ONNX模型
        try:
            onnx_save_path = os.path.join(model_save_dir, f'{model_name}_{timestamp}_{model_note}_best.onnx')
            sample_input = torch.randn(1, 3, 224, 224).to(self.device)
            torch.onnx.export(
                self.model, 
                sample_input, 
                onnx_save_path,
                export_params=True,
                opset_version=11,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            self.status_updated.emit(f'导出ONNX模型: {onnx_save_path}')
        except Exception as e:
            self.status_updated.emit(f'导出ONNX模型时出错: {str(e)}')
    
    def _save_training_info(self, model_name, num_epochs, batch_size, learning_rate, 
                           best_acc, class_names, model_save_dir):
        """保存训练信息"""
        model_note = self.config.get('model_note', '')
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # 保存最终模型
        final_model_path = os.path.join(model_save_dir, f'{model_name}_{timestamp}_{model_note}_final.pth')
        torch.save(self.model.state_dict(), final_model_path)
        self.status_updated.emit(f'保存最终模型: {final_model_path}')
        
        # 保存类别信息
        class_info = {
            'class_names': class_names,
            'class_to_idx': {name: idx for idx, name in enumerate(class_names)}
        }
        
        os.makedirs(model_save_dir, exist_ok=True)
        with open(os.path.join(model_save_dir, 'class_info.json'), 'w') as f:
            json.dump(class_info, f)
        
        # 记录训练信息
        training_info = {
            'model_name': model_name,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'best_accuracy': best_acc.item() if isinstance(best_acc, torch.Tensor) else best_acc,
            'class_names': class_names,
            'model_path': final_model_path,
            'timestamp': timestamp
        }
        
        with open(os.path.join(model_save_dir, 'training_info.json'), 'w') as f:
            json.dump(training_info, f, indent=4) 