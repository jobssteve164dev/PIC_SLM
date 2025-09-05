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
from .model_ema import ModelEMAManager
from .advanced_augmentation import AdvancedAugmentationManager, create_advanced_criterion
from .real_time_metrics_collector import get_global_metrics_collector
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
    conflict_detected = pyqtSignal(list, list)  # 冲突列表，建议列表
    waiting_for_conflict_resolution = pyqtSignal()  # 等待冲突解决信号
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.stop_training = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.training_info = {}
        
        # 冲突解决相关状态
        self.conflict_resolution_result = None
        self.conflict_resolution_config = None
        self.waiting_for_resolution = False
        
        # 初始化各个组件
        self.model_factory = ModelFactory()
        self.weight_calculator = WeightCalculator()
        self.model_configurator = ModelConfigurator()
        self.tensorboard_logger = TensorBoardLogger()
        self.validator = TrainingValidator()
        self.metrics_collector = get_global_metrics_collector()
        
        # 初始化第二阶段组件
        self.ema_manager = None
        self.augmentation_manager = None
        self.advanced_criterion = None
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self._setup_stage_two_components()
        
        # 初始化资源限制器
        self.resource_limiter = None
        self.resource_limited_trainer = None
        self._setup_resource_limiter()
        
        # 连接组件信号
        self._connect_component_signals()
    
    def _setup_stage_two_components(self):
        """设置第二阶段组件（模型EMA、高级数据增强等）"""
        try:
            # 设置高级数据增强管理器
            self.augmentation_manager = AdvancedAugmentationManager(self.config)
            
            # 记录启用的高级特性
            advanced_features = []
            
            if self.config.get('model_ema', False):
                advanced_features.append("模型EMA")
            
            if self.gradient_accumulation_steps > 1:
                advanced_features.append(f"梯度累积(步数:{self.gradient_accumulation_steps})")
            
            if self.augmentation_manager.is_enabled():
                aug_methods = []
                if self.config.get('cutmix_prob', 0.0) > 0:
                    aug_methods.append("CutMix")
                if self.config.get('mixup_alpha', 0.0) > 0:
                    aug_methods.append("MixUp")
                advanced_features.append(f"高级数据增强({'+'.join(aug_methods)})")
            
            # 检查损失缩放状态
            loss_scaling_enabled = self.config.get('loss_scaling_enabled', False)
            loss_scale = self.config.get('loss_scale', 'dynamic')
            if loss_scaling_enabled and loss_scale != 'none' and loss_scale == 'static':
                advanced_features.append("静态损失缩放")
            elif loss_scaling_enabled and loss_scale != 'none' and loss_scale == 'dynamic':
                advanced_features.append("动态损失缩放")
            
            if advanced_features:
                self.status_updated.emit(f"✨ 启用第二阶段高级特性: {', '.join(advanced_features)}")
            
        except Exception as e:
            self.status_updated.emit(f"⚠️ 设置第二阶段组件失败: {e}")
            print(f"第二阶段组件设置错误: {e}")
    
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
    
    def resolve_conflict(self, user_choice, modified_config=None):
        """从主线程接收冲突解决结果"""
        self.conflict_resolution_result = user_choice
        self.conflict_resolution_config = modified_config
        self.waiting_for_resolution = False
        
    def _wait_for_conflict_resolution(self):
        """等待主线程的冲突解决结果"""
        self.waiting_for_resolution = True
        self.waiting_for_conflict_resolution.emit()
        
        # 等待主线程响应
        while self.waiting_for_resolution and not self.stop_training:
            self.msleep(100)  # 睡眠100ms避免忙等待
            
        return self.conflict_resolution_result, self.conflict_resolution_config
    
    def _validate_config_thread_safe(self, config):
        """线程安全的配置验证方法"""
        self.status_updated.emit("开始验证训练配置...")
        
        try:
            # 验证数据集路径
            if not self.validator.validate_dataset_paths(config):
                return False, config
            
            # 验证训练参数
            if not self.validator.validate_training_parameters(config):
                return False, config
            
            # 验证模型配置
            if not self.validator.validate_model_config(config):
                return False, config
            
            # 验证保存路径
            if not self.validator.validate_save_paths(config):
                return False, config
            
            # 检测超参数冲突
            conflicts, suggestions = self.validator.detect_hyperparameter_conflicts(config)
            
            if conflicts:
                self.status_updated.emit(f"检测到 {len(conflicts)} 个超参数冲突")
                
                # 通过信号通知主线程显示对话框
                self.conflict_detected.emit(conflicts, suggestions)
                
                # 等待主线程的用户选择
                user_choice, modified_config = self._wait_for_conflict_resolution()
                
                if user_choice == 'apply':
                    if modified_config:
                        self.status_updated.emit("已应用参数冲突修复")
                        return True, modified_config
                    else:
                        # 如果没有修复配置，应用自动修复
                        auto_fixed_config = self.validator.apply_conflict_fixes(config, suggestions)
                        self.status_updated.emit("已自动修复参数冲突")
                        return True, auto_fixed_config
                elif user_choice == 'ignore':
                    self.status_updated.emit("用户选择忽略冲突，继续训练")
                    return True, config
                else:
                    # 用户取消训练
                    self.status_updated.emit("用户取消训练")
                    return False, config
            
            self.status_updated.emit("配置验证通过")
            return True, config
            
        except Exception as e:
            self.training_error.emit(f"配置验证时发生错误: {str(e)}")
            return False, config
    
    def run(self):
        """线程运行入口，执行模型训练"""
        try:
            # 重置停止标志
            self.stop_training = False
            
            # 启动实时指标采集
            if self.metrics_collector:
                session_id = f"training_{int(time.time())}"
                self.metrics_collector.start_collection(session_id)
            
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
            
            # 第一阶段高级超参数
            print("\n🔧 第一阶段高级超参数:")
            stage_one_params = ['beta1', 'beta2', 'momentum', 'nesterov', 'warmup_steps', 
                              'warmup_ratio', 'warmup_method', 'min_lr', 'label_smoothing']
            for param in stage_one_params:
                value = self.config.get(param, '未设置')
                print(f"   {param}: {value}")
            
            # 第二阶段高级超参数
            print("\n⚡ 第二阶段高级超参数:")
            stage_two_params = ['model_ema', 'model_ema_decay', 'gradient_accumulation_steps',
                              'cutmix_prob', 'mixup_alpha', 'loss_scale', 'static_loss_scale']
            for param in stage_two_params:
                value = self.config.get(param, '未设置')
                print(f"   {param}: {value}")
            
            # 目录配置参数
            print("\n📁 目录配置参数:")
            dir_params = ['default_param_save_dir', 'tensorboard_log_dir']
            for param in dir_params:
                value = self.config.get(param, '未设置')
                print(f"   {param}: {value}")
            
            # 统计参数总数
            total_basic = len(basic_params)
            total_advanced = len(advanced_params)
            total_pretrained = len(pretrained_params)
            total_weight = len(weight_params)
            total_stage_one = len([p for p in stage_one_params if self.config.get(p) is not None])
            total_stage_two = len([p for p in stage_two_params if self.config.get(p) is not None])
            total_resource = len(resource_params)
            total_dir = len(dir_params)
            
            # 如果是检测任务，也统计检测参数
            total_detection = 0
            if self.config.get('task_type') == 'detection':
                detection_params = ['iou_threshold', 'conf_threshold', 'resolution', 'use_mosaic',
                                  'use_multiscale', 'use_ema', 'nms_threshold', 'use_fpn']
                total_detection = len([p for p in detection_params if self.config.get(p) is not None])
            
            print("=" * 60)
            print(f"✅ 参数接收验证完成，共接收 {len(self.config)} 个参数")
            print(f"   📋 基础参数: {total_basic}个")
            print(f"   🔧 高级参数: {total_advanced}个") 
            print(f"   🏗️ 预训练参数: {total_pretrained}个")
            print(f"   ⚖️ 权重参数: {total_weight}个")
            print(f"   🔧 第一阶段超参数: {total_stage_one}个")
            print(f"   ⚡ 第二阶段超参数: {total_stage_two}个")
            print(f"   💾 资源参数: {total_resource}个")
            print(f"   📁 目录参数: {total_dir}个")
            if total_detection > 0:
                print(f"   🎯 检测参数: {total_detection}个")
            print("=" * 60)
            
            # 启动资源限制器监控
            if self.resource_limiter:
                self.resource_limiter.start_monitoring()
                self.status_updated.emit("✅ 强制资源限制已启动")
            
            # 验证配置（线程安全版本）
            is_valid, validated_config = self._validate_config_thread_safe(self.config)
            if not is_valid:
                return
            
            # 使用验证后的配置
            self.config = validated_config
            
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
        
        # 释放数据流服务器引用
        self._release_stream_server()
        
        self.status_updated.emit("训练线程正在停止...")
    
    def _release_stream_server(self):
        """释放数据流服务器引用"""
        try:
            if hasattr(self, 'stream_server') and self.stream_server is not None:
                from ..api.stream_server_manager import release_stream_server
                release_stream_server()
                self.stream_server = None
                print("数据流服务器引用已释放")
            else:
                print("数据流服务器未启动，无需释放")
        except Exception as e:
            print(f"释放数据流服务器引用时出错: {str(e)}")
    
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
            
            # 初始化EMA管理器（第二阶段）
            if self.config.get('model_ema', False):
                self.ema_manager = ModelEMAManager(self.model, self.config)
                if self.ema_manager.is_enabled():
                    self.ema_manager.to(self.device)
                    self.status_updated.emit("🔄 EMA模型已初始化")
            
            # 计算类别权重和设置损失函数
            criterion = self._setup_loss_function(dataloaders['train'], class_names)
            
            if self.stop_training:
                return
            
            # 设置优化器（使用新的优化器工厂）
            from .optimizer_factory import OptimizerFactory
            optimizer = OptimizerFactory.create_optimizer(self.model, self.config)
            
            # 计算总训练步数（用于某些调度器）
            steps_per_epoch = len(dataloaders['train'])
            total_steps = num_epochs * steps_per_epoch
            
            # 设置学习率调度器（支持预热）
            scheduler = OptimizerFactory.create_scheduler(optimizer, self.config, total_steps)
            if scheduler:
                warmup_steps = self.config.get('warmup_steps', 0)
                warmup_ratio = self.config.get('warmup_ratio', 0.0)
                if warmup_steps > 0 or warmup_ratio > 0:
                    self.status_updated.emit(f"启用学习率预热，预热步数: {warmup_steps}")
                else:
                    self.status_updated.emit(f"使用学习率调度器: {self.config.get('lr_scheduler', 'StepLR')}")
            
            # 初始化TensorBoard和数据流服务器
            tensorboard_log_dir = None
            if use_tensorboard:
                tensorboard_log_dir = self.tensorboard_logger.initialize(self.config, model_name)
                self.training_info['tensorboard_log_dir'] = tensorboard_log_dir
                
                # 初始化数据流服务器
                self._initialize_stream_server()
                
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
                    criterion, optimizer, scheduler, model_name, model_save_dir
                )
            else:
                # 使用标准训练循环
                best_acc = self._training_loop(
                    dataloaders, dataset_sizes, class_names, num_epochs, 
                    criterion, optimizer, scheduler, model_name, model_save_dir
                )
            
            if self.stop_training:
                return
            
            # 保存训练信息
            self._save_training_info(model_name, num_epochs, batch_size, learning_rate, 
                                   best_acc, class_names, model_save_dir)
            
            # 记录超参数和最终指标
            if hasattr(self, 'tensorboard_logger') and self.tensorboard_logger:
                final_metrics = {
                    'final_accuracy': float(best_acc),
                    'final_loss': float(epoch_loss) if 'epoch_loss' in locals() else 0.0,
                    'total_epochs': num_epochs,
                    'best_epoch': epoch + 1 if 'epoch' in locals() else num_epochs
                }
                self.tensorboard_logger.log_hyperparameters(self.config, final_metrics)
            
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
        
        # 检查是否启用基础数据增强
        use_augmentation = self.config.get('use_augmentation', True)
        
        # 构建训练时的transform列表
        train_transforms = [
            transforms.Resize((224, 224)),
        ]
        
        # 基础数据增强（只有在启用时才添加）
        if use_augmentation:
            train_transforms.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
            self.status_updated.emit("✅ 启用基础数据增强（翻转、旋转、颜色抖动、仿射变换）")
        else:
            self.status_updated.emit("⚪ 基础数据增强已禁用")
        
        # 添加必要的转换
        train_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 数据转换配置
        data_transforms = {
            'train': transforms.Compose(train_transforms),
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
        
        # 根据操作系统选择合适的num_workers
        import platform
        num_workers = 0 if platform.system() == 'Windows' else 4
        
        dataloaders = {x: DataLoader(image_datasets[x],
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers)
                      for x in ['train', 'val']}
        
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        num_classes = len(class_names)
        
        # 输出数据增强配置信息
        augmentation_status = []
        if use_augmentation:
            augmentation_status.append("基础增强")
        if self.augmentation_manager and self.augmentation_manager.is_enabled():
            augmentation_status.append("高级增强")
        
        if augmentation_status:
            self.status_updated.emit(f"📊 数据增强配置: {' + '.join(augmentation_status)}")
        else:
            self.status_updated.emit("📊 数据增强配置: 无增强")
        
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
        """设置损失函数（支持标签平滑和高级数据增强）"""
        use_class_weights = self.config.get('use_class_weights', True)
        weight_strategy = self.config.get('weight_strategy', 'balanced')
        label_smoothing = self.config.get('label_smoothing', 0.0)
        
        # 计算类别权重
        class_weights = None
        if use_class_weights:
            class_weights = self.weight_calculator.calculate_class_weights(
                train_dataset.dataset, class_names, self.config, self.device
            )
        
        # 创建基础损失函数
        from .optimizer_factory import OptimizerFactory
        base_criterion = OptimizerFactory.create_criterion(self.config, class_weights)
        
        # 检查是否需要高级损失函数（第二阶段）
        if self.augmentation_manager and self.augmentation_manager.is_enabled():
            # 使用高级损失函数支持MixUp/CutMix
            criterion, aug_manager = create_advanced_criterion(self.config, base_criterion)
            self.advanced_criterion = criterion
            self.status_updated.emit("🚀 使用高级混合损失函数（支持MixUp/CutMix）")
        else:
            # 使用标准损失函数
            criterion, _ = create_advanced_criterion(self.config, base_criterion)
            self.advanced_criterion = None
        
        # 更新状态信息
        status_parts = []
        if label_smoothing > 0:
            status_parts.append(f"标签平滑(系数:{label_smoothing})")
        if use_class_weights:
            status_parts.append(f"类别权重({weight_strategy})")
        
        if status_parts:
            self.status_updated.emit(f"损失函数配置: {', '.join(status_parts)}")
        else:
            self.status_updated.emit("使用标准交叉熵损失函数")
        
        return criterion
    
    def _training_loop(self, dataloaders, dataset_sizes, class_names, num_epochs, 
                      criterion, optimizer, scheduler, model_name, model_save_dir):
        """执行训练循环（支持高级学习率调度）"""
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
                
                # 记录高级评估指标
                if len(all_labels) > 0 and len(all_preds) > 0:
                    self.tensorboard_logger.log_advanced_metrics(
                        all_labels, all_preds, epoch=epoch, phase=phase
                    )
                
                # 采集实时指标数据供智能训练使用
                if self.metrics_collector:
                    metrics_data = {
                        'loss': epoch_loss,
                        'accuracy': epoch_acc,
                        'epoch': epoch + 1,  # 转换为从1开始的epoch编号
                        'phase': phase
                    }
                    self.metrics_collector.collect_tensorboard_metrics(epoch + 1, phase, metrics_data)
                
                # 记录样本图像（每5个epoch一次）
                if phase == 'val' and epoch % 5 == 0:
                    self.tensorboard_logger.log_sample_images(dataloaders[phase], epoch)
                
                # 记录混淆矩阵（验证阶段）
                if phase == 'val':
                    self.tensorboard_logger.log_confusion_matrix(
                        all_labels, all_preds, class_names, epoch
                    )
                
                # 记录模型预测可视化（每10个epoch一次）
                if phase == 'val' and epoch % 10 == 0:
                    self.tensorboard_logger.log_model_predictions(
                        self.model, dataloaders[phase], class_names, epoch, self.device
                    )
                
                # 记录模型权重和梯度（每5个epoch一次）
                if phase == 'train' and epoch % 5 == 0:
                    self.tensorboard_logger.log_model_weights_and_gradients(self.model, epoch)
                
                # 记录学习率调度
                if phase == 'train':
                    self.tensorboard_logger.log_learning_rate_schedule(optimizer, epoch)
                
                # 记录性能指标
                self.tensorboard_logger.log_performance_metrics(
                    epoch, num_samples=dataset_sizes[phase]
                )
                
                # 刷新TensorBoard数据
                self.tensorboard_logger.flush()
            
            if self.stop_training:
                break
            
            # 更新学习率调度器
            if scheduler and phase == 'val':
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau需要传入监控的指标
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()
            
            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                self._save_best_model(model_name, model_save_dir, epoch, best_acc)
        
        return best_acc
    
    def _resource_limited_training_loop(self, dataloaders, dataset_sizes, class_names, num_epochs, 
                                       criterion, optimizer, scheduler, model_name, model_save_dir):
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
        """训练一个epoch（支持第二阶段高级特性）"""
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        
        # 梯度累积相关变量
        accumulation_steps = self.gradient_accumulation_steps if phase == 'train' else 1
        accumulated_loss = 0.0
        
        # 遍历数据
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            if self.stop_training:
                break
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # 只在累积步骤开始时清零梯度
            if phase == 'train' and i % accumulation_steps == 0:
                optimizer.zero_grad()
            
            # 前向传播
            with torch.set_grad_enabled(phase == 'train'):
                # 应用高级数据增强（第二阶段）
                if phase == 'train' and self.augmentation_manager and self.augmentation_manager.is_enabled():
                    # 使用高级数据增强（MixUp/CutMix）
                    mixed_inputs, y_a, y_b, lam, aug_method = self.augmentation_manager(inputs, labels)
                    outputs = self.model(mixed_inputs)
                    
                    # 计算混合损失
                    loss = self._calculate_mixed_loss(outputs, y_a, y_b, lam, criterion)
                    
                    # 计算混合增强的准确率
                    _, preds = torch.max(outputs, 1)
                    corrects = lam * preds.eq(y_a.data).sum().float() + (1 - lam) * preds.eq(y_b.data).sum().float()
                    
                    # 记录使用的增强方法（只在第一个batch记录）
                    if i == 0:
                        self.status_updated.emit(f"📊 使用{aug_method}高级数据增强")
                    
                else:
                    # 标准前向传播（可能包含基础数据增强）
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    # 计算标准损失
                    loss = self._calculate_standard_loss(outputs, labels, criterion)
                    corrects = torch.sum(preds == labels.data).float()
                
                # 梯度累积：缩放损失
                if phase == 'train' and accumulation_steps > 1:
                    loss = loss / accumulation_steps
                
                # 反向传播和参数更新
                if phase == 'train':
                    self._backward_and_update(loss, optimizer, i, accumulation_steps, dataloaders[phase])
            
            if self.stop_training:
                break
            
            # 累积统计信息
            if accumulation_steps > 1:
                accumulated_loss += loss.item() * inputs.size(0) * accumulation_steps  # 还原真实损失
            else:
                accumulated_loss += loss.item() * inputs.size(0)
                
            running_loss += accumulated_loss if (i + 1) % accumulation_steps == 0 else 0
            running_corrects += corrects
            
            # 收集预测和标签（用于指标计算）
            if phase == 'train' and self.augmentation_manager and self.augmentation_manager.is_enabled():
                # 混合增强时使用原始标签
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # 重置累积损失
            if (i + 1) % accumulation_steps == 0:
                accumulated_loss = 0.0
            
            # 更新进度
            progress = int(((epoch * len(dataloaders[phase]) + i + 1) /
                          (num_epochs * len(dataloaders[phase]))) * 100)
            self.progress_updated.emit(progress)
            
            # 发送训练状态更新
            if i % 10 == 0:
                current_loss = running_loss / ((i + 1) * inputs.size(0))
                current_acc = running_corrects.double() / ((i + 1) * inputs.size(0))
                
                # 添加梯度累积信息
                status_info = {
                    'epoch': epoch + 1,
                    'phase': phase,
                    'loss': float(current_loss),
                    'accuracy': float(current_acc.item()),
                    'batch': i + 1,
                    'total_batches': len(dataloaders[phase])
                }
                
                # 添加第二阶段特性信息
                if phase == 'train':
                    if accumulation_steps > 1:
                        status_info['grad_accum'] = f"{accumulation_steps}步"
                    if self.ema_manager and self.ema_manager.is_enabled():
                        status_info['ema'] = "启用"
                
                self.epoch_finished.emit(status_info)
        
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
    
    def _calculate_mixed_loss(self, outputs, y_a, y_b, lam, criterion):
        """计算混合损失（MixUp/CutMix）"""
        if self.advanced_criterion:
            return self.advanced_criterion(outputs, y_a, y_b, lam)
        else:
            return lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
    
    def _calculate_standard_loss(self, outputs, labels, criterion):
        """计算标准损失"""
        if self.advanced_criterion and hasattr(self.advanced_criterion, '__call__'):
            # 检查是否是MixCriterion类型
            if hasattr(self.advanced_criterion, 'criterion'):
                # 这是MixCriterion，但没有混合增强，直接使用基础损失函数
                return self.advanced_criterion.criterion(outputs, labels)
            else:
                # 这是其他高级损失函数（如标签平滑）
                return self.advanced_criterion(outputs, labels)
        else:
            # 使用标准损失函数
            return criterion(outputs, labels)
    
    def _backward_and_update(self, loss, optimizer, batch_idx, accumulation_steps, dataloader):
        """反向传播和参数更新"""
        loss.backward()
        
        # 只在累积步骤结束时更新参数
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            # 梯度裁剪（如果启用）
            if self.config.get('gradient_clipping', False):
                clip_value = self.config.get('gradient_clipping_value', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
            
            optimizer.step()
            
            # 更新EMA模型（第二阶段）
            if self.ema_manager and self.ema_manager.is_enabled():
                self.ema_manager.update(self.model)
    
    def _save_best_model(self, model_name, model_save_dir, epoch, best_acc):
        """保存最佳模型（支持EMA模型）"""
        model_note = self.config.get('model_note', '')
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # 保存标准模型
        model_save_path = os.path.join(model_save_dir, f'{model_name}_{timestamp}_{model_note}_best.pth')
        torch.save(self.model.state_dict(), model_save_path)
        self.status_updated.emit(f'💾 保存最佳模型，Epoch {epoch+1}, Acc: {best_acc:.4f}')
        
        # 保存EMA模型（第二阶段）
        if self.ema_manager and self.ema_manager.is_enabled():
            ema_save_path = os.path.join(model_save_dir, f'{model_name}_{timestamp}_{model_note}_best_ema.pth')
            self.ema_manager.save_ema_model(ema_save_path)
            self.status_updated.emit(f'🔄 保存最佳EMA模型: {ema_save_path}')
        
        # 导出ONNX模型（优先使用EMA模型）
        try:
            onnx_save_path = os.path.join(model_save_dir, f'{model_name}_{timestamp}_{model_note}_best.onnx')
            sample_input = torch.randn(1, 3, 224, 224).to(self.device)
            
            # 选择用于导出的模型
            export_model = self.model
            if self.ema_manager and self.ema_manager.is_enabled():
                ema_model = self.ema_manager.get_model()
                if ema_model is not None:
                    export_model = ema_model
                    self.status_updated.emit('📦 使用EMA模型导出ONNX')
            
            torch.onnx.export(
                export_model, 
                sample_input, 
                onnx_save_path,
                export_params=True,
                opset_version=11,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            self.status_updated.emit(f'📦 导出ONNX模型: {onnx_save_path}')
        except Exception as e:
            self.status_updated.emit(f'⚠️ 导出ONNX模型时出错: {str(e)}')
    
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

    def _initialize_stream_server(self):
        """初始化数据流服务器"""
        try:
            # 首先检查AI配置中的数据流服务器开关
            ai_config = self._load_ai_config()
            enable_data_stream_server = ai_config.get('general', {}).get('enable_data_stream_server', True)
            
            if not enable_data_stream_server:
                self.status_updated.emit("📊 数据流服务器已禁用（根据AI设置）")
                self.stream_server = None
                return
            
            # 导入全局数据流服务器管理器
            from ..api.stream_server_manager import get_stream_server
            
            # 创建数据流服务器配置
            stream_config = {
                'sse_host': '127.0.0.1',
                'sse_port': 8888,
                'websocket_host': '127.0.0.1',
                'websocket_port': 8889,
                'rest_api_host': '127.0.0.1',
                'rest_api_port': 8890,
                'buffer_size': 1000,
                'debug_mode': False
            }
            
            # 获取全局数据流服务器实例
            self.stream_server = get_stream_server(
                training_system=self.parent() if self.parent() else None,
                config=stream_config
            )
            
            # 设置TensorBoard日志器的数据流服务器
            self.tensorboard_logger.set_stream_server(self.stream_server)
            
            # 等待服务器启动完成
            import time
            time.sleep(2)
            
            # 获取API端点信息
            endpoints = self.stream_server.get_api_endpoints()
            self.status_updated.emit(f"数据流服务已启动:")
            self.status_updated.emit(f"• SSE: {endpoints.get('sse_stream')}")
            self.status_updated.emit(f"• WebSocket: {endpoints.get('websocket')}")
            self.status_updated.emit(f"• REST API: {endpoints.get('rest_current_metrics')}")
            
        except Exception as e:
            print(f"初始化数据流服务器失败: {str(e)}")
            self.stream_server = None
    
    def _load_ai_config(self):
        """加载AI配置文件"""
        import json
        import os
        
        config_file = "setting/ai_config.json"
        default_config = {
            'general': {
                'enable_data_stream_server': True
            }
        }
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 确保general部分存在
                    if 'general' not in config:
                        config['general'] = {}
                    # 确保enable_data_stream_server存在
                    if 'enable_data_stream_server' not in config['general']:
                        config['general']['enable_data_stream_server'] = True
                    return config
            else:
                return default_config
        except Exception as e:
            print(f"加载AI配置失败: {str(e)}")
            return default_config 