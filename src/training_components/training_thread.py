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
from ..utils.logger import get_logger, PerformanceMonitor, log_error


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
        
        # 初始化日志记录器
        self.logger = get_logger(__name__, component="TrainingThread")
        self.logger.info("训练线程初始化", extra={"operation": "init"})
        
        # 初始化各个组件
        self.model_factory = ModelFactory()
        self.weight_calculator = WeightCalculator()
        self.model_configurator = ModelConfigurator()
        self.tensorboard_logger = TensorBoardLogger()
        self.validator = TrainingValidator()
        
        # 连接组件信号
        self._connect_component_signals()
    
    def _connect_component_signals(self):
        """连接各个组件的信号"""
        self.model_factory.status_updated.connect(self.status_updated)
        self.model_factory.model_download_failed.connect(self.model_download_failed)
        
        self.weight_calculator.status_updated.connect(self.status_updated)
        
        self.model_configurator.status_updated.connect(self.status_updated)
        
        self.tensorboard_logger.status_updated.connect(self.status_updated)
        
        self.validator.status_updated.connect(self.status_updated)
        self.validator.validation_error.connect(self.training_error)
        
        self.logger.debug("组件信号连接完成", extra={"operation": "connect_signals"})
    
    def run(self):
        """线程运行入口，执行模型训练"""
        try:
            with PerformanceMonitor("模型训练完整流程", component="TrainingThread"):
                self.logger.info("开始训练流程", extra={"operation": "start_training"})
                
                # 重置停止标志
                self.stop_training = False
                
                # 验证配置
                if not self.validator.validate_config(self.config):
                    self.logger.warning("训练配置验证失败", extra={"operation": "validate_config"})
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
                
                self.logger.info(f"训练参数: 模型={model_name}, 批大小={batch_size}, 学习率={learning_rate}, 轮次={num_epochs}", 
                               extra={"operation": "training_params"})
                
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
                self.logger.info("训练流程完成", extra={"operation": "training_completed"})
                
        except Exception as e:
            error_msg = f"训练过程中发生错误: {str(e)}"
            self.training_error.emit(error_msg)
            log_error(e, context={"config": str(self.config)}, component="TrainingThread")
            import traceback
            traceback.print_exc()
    
    def stop(self):
        """停止训练过程"""
        self.stop_training = True
        self.status_updated.emit("训练线程正在停止...")
        self.logger.info("请求停止训练", extra={"operation": "stop_training"})
    
    def train_model(self, data_dir, model_name, num_epochs, batch_size, learning_rate, 
                   model_save_dir, task_type='classification', use_tensorboard=True):
        """执行模型训练"""
        try:
            # 标准化路径格式
            data_dir = os.path.normpath(data_dir).replace('\\', '/')
            model_save_dir = os.path.normpath(model_save_dir).replace('\\', '/')
            
            self.logger.info(f"开始训练模型: {model_name}", extra={
                "operation": "train_model",
                "model_name": model_name,
                "data_dir": data_dir
            })
            
            # 准备数据
            with PerformanceMonitor("数据准备", component="TrainingThread"):
                dataloaders, dataset_sizes, class_names, num_classes = self._prepare_data(
                    data_dir, batch_size, task_type
                )
            
            if self.stop_training:
                self.logger.info("训练被用户中断 - 数据准备后", extra={"operation": "training_interrupted"})
                return
            
            # 创建和配置模型
            with PerformanceMonitor("模型创建与配置", component="TrainingThread"):
                self.model = self._create_and_configure_model(model_name, num_classes, task_type)
            
            if self.stop_training:
                self.logger.info("训练被用户中断 - 模型创建后", extra={"operation": "training_interrupted"})
                return
            
            # 计算类别权重和设置损失函数
            with PerformanceMonitor("设置损失函数", component="TrainingThread"):
                criterion = self._setup_loss_function(dataloaders['train'], class_names)
            
            if self.stop_training:
                self.logger.info("训练被用户中断 - 损失函数设置后", extra={"operation": "training_interrupted"})
                return
            
            # 设置优化器
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.logger.debug(f"优化器设置完成: Adam, lr={learning_rate}", extra={"operation": "setup_optimizer"})
            
            # 初始化TensorBoard
            tensorboard_log_dir = None
            if use_tensorboard:
                with PerformanceMonitor("TensorBoard初始化", component="TrainingThread"):
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
                
                self.logger.info(f"TensorBoard初始化完成，日志目录: {tensorboard_log_dir}", 
                               extra={"operation": "tensorboard_init"})
            
            if self.stop_training:
                self.logger.info("训练被用户中断 - TensorBoard初始化后", extra={"operation": "training_interrupted"})
                return
            
            # 执行训练循环
            with PerformanceMonitor("训练循环", component="TrainingThread"):
                best_acc = self._training_loop(
                    dataloaders, dataset_sizes, class_names, num_epochs, 
                    criterion, optimizer, model_name, model_save_dir
                )
            
            if self.stop_training:
                self.logger.info("训练被用户中断 - 训练循环中", extra={"operation": "training_interrupted"})
                return
            
            # 保存训练信息
            self._save_training_info(model_name, num_epochs, batch_size, learning_rate, 
                                   best_acc, class_names, model_save_dir)
            
            # 关闭TensorBoard
            self.tensorboard_logger.close()
            
            self.status_updated.emit(f'训练完成，最佳准确率: {best_acc:.4f}')
            self.logger.info(f"模型训练完成，最佳准确率: {best_acc:.4f}", extra={
                "operation": "training_completed",
                "best_accuracy": best_acc
            })
            
        except Exception as e:
            error_msg = f"训练过程中出错: {str(e)}"
            self.training_error.emit(error_msg)
            log_error(e, context={
                "model_name": model_name,
                "data_dir": data_dir,
                "batch_size": batch_size,
                "learning_rate": learning_rate
            }, component="TrainingThread")
            import traceback
            traceback.print_exc()
    
    def _prepare_data(self, data_dir, batch_size, task_type):
        """准备训练数据"""
        if task_type != 'classification':
            self.logger.error(f"不支持的任务类型: {task_type}", extra={"operation": "prepare_data"})
            raise ValueError(f"当前仅支持分类任务，不支持: {task_type}")
        
        self.status_updated.emit("加载分类数据集...")
        self.logger.info(f"开始加载分类数据集: {data_dir}", extra={"operation": "load_dataset"})
        
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
        
        self.logger.info(f"数据集加载完成，训练集: {dataset_sizes['train']}张图片，验证集: {dataset_sizes['val']}张图片，类别数: {num_classes}", 
                       extra={"operation": "dataset_loaded"})
        
        return dataloaders, dataset_sizes, class_names, num_classes
    
    def _create_and_configure_model(self, model_name, num_classes, task_type):
        """创建和配置模型"""
        # 创建模型
        self.logger.info(f"创建模型: {model_name}, 类别数: {num_classes}", extra={"operation": "create_model"})
        model = self.model_factory.create_model(model_name, num_classes, task_type)
        
        # 配置模型
        self.logger.debug("配置模型参数", extra={"operation": "configure_model"})
        model = self.model_configurator.configure_model(model, self.config)
        
        # 移到设备
        device_name = "GPU" if self.device.type == "cuda" else "CPU"
        self.logger.info(f"模型已移至{device_name}", extra={"operation": "model_to_device", "device": self.device.type})
        model = model.to(self.device)
        
        return model
    
    def _setup_loss_function(self, train_dataset, class_names):
        """设置损失函数"""
        use_class_weights = self.config.get('use_class_weights', True)
        weight_strategy = self.config.get('weight_strategy', 'balanced')
        
        if use_class_weights:
            self.logger.info(f"使用类别权重，策略: {weight_strategy}", extra={"operation": "setup_loss_function"})
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
        self.logger.info(f"开始训练循环，计划训练 {num_epochs} 轮", 
                       extra={"operation": "training_loop_start"})
        
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            if self.stop_training:
                self.status_updated.emit("训练已停止")
                self.logger.info("训练被用户中断 - 训练循环开始", extra={"operation": "training_interrupted"})
                break
            
            self.status_updated.emit(f'Epoch {epoch+1}/{num_epochs}')
            self.logger.info(f"开始第 {epoch+1}/{num_epochs} 轮训练", 
                           extra={"operation": "epoch_start", "epoch": epoch+1, "total_epochs": num_epochs})
            
            # 训练和验证阶段
            for phase in ['train', 'val']:
                if self.stop_training:
                    self.logger.info(f"训练被用户中断 - {phase}阶段前", extra={"operation": "training_interrupted"})
                    break
                
                with PerformanceMonitor(f"Epoch {epoch+1} {phase}阶段", component="TrainingThread"):
                    self.logger.debug(f"开始 Epoch {epoch+1} {phase}阶段", 
                                    extra={"operation": f"{phase}_phase_start", "epoch": epoch+1})
                    epoch_loss, epoch_acc, all_preds, all_labels = self._train_epoch(
                        phase, dataloaders, dataset_sizes, criterion, optimizer, epoch, num_epochs
                    )
                
                if self.stop_training:
                    self.logger.info(f"训练被用户中断 - {phase}阶段后", extra={"operation": "training_interrupted"})
                    break
                
                # 记录到TensorBoard
                self.tensorboard_logger.log_epoch_metrics(epoch, phase, epoch_loss, epoch_acc)
                self.logger.debug(f"Epoch {epoch+1} {phase}阶段指标已记录到TensorBoard", 
                                extra={"operation": "tensorboard_metrics", "loss": float(epoch_loss), "accuracy": float(epoch_acc)})
                
                # 记录样本图像（每5个epoch一次）
                if phase == 'val' and epoch % 5 == 0:
                    self.tensorboard_logger.log_sample_images(dataloaders[phase], epoch)
                    self.logger.debug(f"已记录验证集样本图像到TensorBoard", extra={"operation": "tensorboard_images", "epoch": epoch+1})
                
                # 记录混淆矩阵（验证阶段）
                if phase == 'val':
                    self.tensorboard_logger.log_confusion_matrix(
                        all_labels, all_preds, class_names, epoch
                    )
                    self.logger.debug(f"已记录混淆矩阵到TensorBoard", extra={"operation": "tensorboard_confusion_matrix", "epoch": epoch+1})
                
                # 刷新TensorBoard数据
                self.tensorboard_logger.flush()
            
            if self.stop_training:
                self.logger.info("训练被用户中断 - 阶段循环后", extra={"operation": "training_interrupted"})
                break
            
            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                self._save_best_model(model_name, model_save_dir, epoch, best_acc)
                self.logger.info(f"发现新的最佳模型，准确率: {best_acc:.4f}，已保存", 
                               extra={"operation": "save_best_model", "accuracy": float(best_acc), "epoch": epoch+1})
        
        self.logger.info(f"训练循环结束，最佳准确率: {best_acc:.4f}", 
                       extra={"operation": "training_loop_end", "best_accuracy": float(best_acc)})
        return best_acc
    
    def _train_epoch(self, phase, dataloaders, dataset_sizes, criterion, optimizer, epoch, num_epochs):
        """训练一个epoch"""
        if phase == 'train':
            self.model.train()
            self.logger.debug(f"Epoch {epoch+1} 训练模式设置为训练", extra={"operation": "model_train_mode"})
        else:
            self.model.eval()
            self.logger.debug(f"Epoch {epoch+1} 训练模式设置为评估", extra={"operation": "model_eval_mode"})
        
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        
        batch_count = len(dataloaders[phase])
        self.logger.debug(f"开始处理 {phase} 数据，共 {batch_count} 批", 
                        extra={"operation": f"{phase}_batches", "batch_count": batch_count})
        
        # 遍历数据
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            if self.stop_training:
                self.logger.debug(f"批处理中断，当前批次: {i+1}/{batch_count}", 
                                extra={"operation": "batch_interrupted"})
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
                
                # 每10批记录一次日志
                self.logger.debug(f"Epoch {epoch+1} {phase} 批次 {i+1}/{batch_count}, 损失: {float(current_loss):.4f}, 准确率: {float(current_acc.item()):.4f}", 
                                extra={
                                    "operation": f"{phase}_batch_progress",
                                    "epoch": epoch+1,
                                    "batch": i+1,
                                    "loss": float(current_loss),
                                    "accuracy": float(current_acc.item())
                                })
        
        if self.stop_training:
            self.logger.info(f"Epoch {epoch+1} {phase}阶段被中断", extra={"operation": "epoch_interrupted"})
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
        
        self.logger.info(f"Epoch {epoch+1} {phase}阶段完成，损失: {float(epoch_loss):.4f}, 准确率: {float(epoch_acc.item()) if isinstance(epoch_acc, torch.Tensor) else float(epoch_acc):.4f}", 
                       extra={
                           "operation": f"{phase}_epoch_completed",
                           "epoch": epoch+1,
                           "loss": float(epoch_loss),
                           "accuracy": float(epoch_acc.item()) if isinstance(epoch_acc, torch.Tensor) else float(epoch_acc)
                       })
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def _save_best_model(self, model_name, model_save_dir, epoch, best_acc):
        """保存最佳模型"""
        with PerformanceMonitor("保存最佳模型", component="TrainingThread"):
            model_note = self.config.get('model_note', '')
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # 保存PyTorch模型
            model_save_path = os.path.join(model_save_dir, f'{model_name}_{timestamp}_{model_note}_best.pth')
            self.logger.info(f"保存最佳模型 (Epoch {epoch+1}, 准确率: {best_acc:.4f}): {model_save_path}", 
                           extra={"operation": "save_best_model_file", "path": model_save_path})
            
            torch.save(self.model.state_dict(), model_save_path)
            self.status_updated.emit(f'保存最佳模型，Epoch {epoch+1}, Acc: {best_acc:.4f}')
            
            # 导出ONNX模型
            try:
                onnx_save_path = os.path.join(model_save_dir, f'{model_name}_{timestamp}_{model_note}_best.onnx')
                sample_input = torch.randn(1, 3, 224, 224).to(self.device)
                
                self.logger.info(f"导出ONNX模型: {onnx_save_path}", 
                               extra={"operation": "export_onnx_model", "path": onnx_save_path})
                
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
                error_msg = f'导出ONNX模型时出错: {str(e)}'
                self.status_updated.emit(error_msg)
                self.logger.error(error_msg, extra={"operation": "export_onnx_error"})
                log_error(e, context={"model_name": model_name}, component="TrainingThread")
    
    def _save_training_info(self, model_name, num_epochs, batch_size, learning_rate, 
                           best_acc, class_names, model_save_dir):
        """保存训练信息"""
        with PerformanceMonitor("保存训练信息", component="TrainingThread"):
            model_note = self.config.get('model_note', '')
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            self.logger.info(f"开始保存训练信息和最终模型", 
                           extra={"operation": "save_training_info"})
            
            # 保存最终模型
            final_model_path = os.path.join(model_save_dir, f'{model_name}_{timestamp}_{model_note}_final.pth')
            torch.save(self.model.state_dict(), final_model_path)
            self.status_updated.emit(f'保存最终模型: {final_model_path}')
            self.logger.info(f"最终模型已保存: {final_model_path}", 
                           extra={"operation": "save_final_model", "path": final_model_path})
            
            # 保存类别信息
            class_info = {
                'class_names': class_names,
                'class_to_idx': {name: idx for idx, name in enumerate(class_names)}
            }
            
            os.makedirs(model_save_dir, exist_ok=True)
            class_info_path = os.path.join(model_save_dir, 'class_info.json')
            with open(class_info_path, 'w') as f:
                json.dump(class_info, f)
            
            self.logger.info(f"类别信息已保存: {class_info_path}", 
                           extra={"operation": "save_class_info", "class_count": len(class_names)})
            
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