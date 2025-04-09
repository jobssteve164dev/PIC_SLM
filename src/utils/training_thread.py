import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import subprocess
import sys
from PyQt5.QtCore import QThread, pyqtSignal

from .activation_utils import apply_activation_function, apply_dropout
from .data_utils import load_classification_datasets, save_class_info, save_training_info
from .visualization_utils import (
    create_tensorboard_writer, log_model_graph, log_batch_stats,
    log_epoch_stats, log_sample_images, log_confusion_matrix,
    log_learning_rate, log_model_parameters, close_tensorboard_writer
)
from .model_utils import create_model

class TrainingThread(QThread):
    """负责在单独线程中执行训练过程的类"""
    
    # 定义训练进度和状态信号
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
    
    def run(self):
        """线程运行入口，执行模型训练"""
        try:
            # 重置停止标志
            self.stop_training = False
            
            # 提取基本参数
            data_dir = self.config.get('data_dir', '')
            model_name = self.config.get('model_name', 'ResNet50')
            num_epochs = self.config.get('num_epochs', 20)
            batch_size = self.config.get('batch_size', 32)
            learning_rate = self.config.get('learning_rate', 0.001)
            model_save_dir = self.config.get('model_save_dir', 'models/saved_models')
            task_type = self.config.get('task_type', 'classification')
            use_tensorboard = self.config.get('use_tensorboard', True)
            
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
    
    def stop(self):
        """停止训练过程"""
        self.stop_training = True
        self.status_updated.emit("训练线程正在停止...")
    
    def train_model(self, data_dir, model_name, num_epochs, batch_size, learning_rate, 
                   model_save_dir, task_type='classification', use_tensorboard=True):
        """实现模型训练流程
        
        Args:
            data_dir: 数据目录路径
            model_name: 模型名称
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            model_save_dir: 模型保存目录
            task_type: 任务类型（'classification'或'detection'）
            use_tensorboard: 是否使用TensorBoard记录
        """
        try:
            # 标准化路径格式
            data_dir = os.path.normpath(data_dir).replace('\\', '/')
            model_save_dir = os.path.normpath(model_save_dir).replace('\\', '/')
            
            # 加载数据集
            self.status_updated.emit("加载数据集...")
            if task_type == 'classification':
                # 使用数据工具加载分类数据集
                dataloaders, dataset_sizes, class_names, num_classes, class_info = load_classification_datasets(
                    data_dir, batch_size, img_size=224, num_workers=4
                )
                
                # 保存类别信息
                save_class_info(class_info, model_save_dir)
            else:
                # 目标检测任务暂不实现
                self.training_error.emit("目标检测训练功能尚未完全实现")
                return

            # 创建模型
            self.status_updated.emit(f"创建{model_name}模型...")
            self.model = create_model(self.config)
            
            # 应用用户选择的激活函数
            activation_function = self.config.get('activation_function', 'ReLU')
            if activation_function:
                self.status_updated.emit(f"应用自定义激活函数: {activation_function}")
                self.model = apply_activation_function(self.model, activation_function)
            
            # 应用用户设置的dropout
            dropout_rate = self.config.get('dropout_rate', 0.0)
            if dropout_rate > 0:
                self.status_updated.emit(f"应用Dropout，丢弃率: {dropout_rate}")
                self.model = apply_dropout(self.model, dropout_rate)
            
            # 将模型移动到设备
            self.model = self.model.to(self.device)

            # 定义损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # 初始化TensorBoard
            writer = None
            tensorboard_run_dir = None
            if use_tensorboard:
                self.status_updated.emit("初始化TensorBoard...")
                writer, tensorboard_run_dir = create_tensorboard_writer(self.config)
                
                # 在训练信息中记录TensorBoard日志目录路径
                self.training_info['tensorboard_log_dir'] = tensorboard_run_dir
                
                # 记录模型图
                log_model_graph(writer, self.model, device=self.device)

            # 训练模型
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
                            self.status_updated.emit("训练已在batch处理时停止")
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
                                
                        # 再次检查停止状态
                        if self.stop_training:
                            self.status_updated.emit("训练已在反向传播后停止")
                            break

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        
                        # 收集预测和标签用于计算混淆矩阵
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                        # 更新进度
                        progress = int(((epoch * len(dataloaders[phase]) + i + 1) /
                                     (num_epochs * len(dataloaders[phase]))) * 100)
                        self.progress_updated.emit(progress)

                        # 每10个batch发送一次训练状态
                        if i % 10 == 0:
                            current_loss = running_loss / ((i + 1) * inputs.size(0))
                            current_acc = running_corrects.double() / ((i + 1) * inputs.size(0))
                            epoch_data = {
                                'epoch': epoch + 1,
                                'phase': phase,
                                'loss': float(current_loss),
                                'accuracy': float(current_acc.item() if hasattr(current_acc, 'item') else current_acc),
                                'batch': i + 1,
                                'total_batches': len(dataloaders[phase])
                            }
                            self.epoch_finished.emit(epoch_data)
                            
                            # 记录到TensorBoard
                            if writer and phase == 'train':
                                log_batch_stats(
                                    writer, phase, epoch, 
                                    current_loss, 
                                    current_acc.item() if hasattr(current_acc, 'item') else current_acc,
                                    i, len(dataloaders[phase])
                                )

                    if self.stop_training:
                        break

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    # 发送每个epoch的结果
                    epoch_data = {
                        'epoch': epoch + 1,
                        'phase': phase,
                        'loss': float(epoch_loss),
                        'accuracy': float(epoch_acc.item() if hasattr(epoch_acc, 'item') else epoch_acc),
                        'batch': len(dataloaders[phase]),
                        'total_batches': len(dataloaders[phase])
                    }
                    self.epoch_finished.emit(epoch_data)
                    
                    # 记录到TensorBoard
                    if writer:
                        log_epoch_stats(
                            writer, phase, epoch, 
                            epoch_loss, 
                            epoch_acc.item() if hasattr(epoch_acc, 'item') else epoch_acc
                        )
                        
                        # 记录学习率
                        if phase == 'train':
                            log_learning_rate(writer, optimizer, epoch)
                            
                            # 每5个epoch记录一次参数
                            if epoch % 5 == 0:
                                log_model_parameters(writer, self.model, epoch)
                        
                        # 每个epoch结束时记录一些样本图像
                        if phase == 'val' and epoch % 5 == 0:  # 每5个epoch记录一次
                            log_sample_images(writer, dataloaders[phase], epoch, device=self.device)
                        
                        # 如果是验证阶段，记录混淆矩阵
                        if phase == 'val':
                            log_confusion_matrix(writer, all_labels, all_preds, class_names, epoch)

                    # 保存最佳模型（验证阶段）
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        # 保存最佳模型，使用统一的命名格式
                        model_note = self.config.get('model_note', '')
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
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

            # 保存最终模型
            model_note = self.config.get('model_note', '')
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            final_model_path = os.path.join(model_save_dir, f'{model_name}_{timestamp}_{model_note}_final.pth')
            torch.save(self.model.state_dict(), final_model_path)
            self.status_updated.emit(f'保存最终模型: {final_model_path}')
            
            # 记录训练完成信息
            training_info = {
                'model_name': model_name,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'best_accuracy': best_acc.item() if hasattr(best_acc, 'item') else float(best_acc),
                'class_names': class_names,
                'model_path': final_model_path,
                'timestamp': timestamp
            }
            
            # 保存训练信息
            save_training_info(training_info, model_save_dir)
                
            self.status_updated.emit(f'训练完成，最佳准确率: {best_acc:.4f}')
            
            # 关闭TensorBoard写入器
            if writer:
                close_tensorboard_writer(writer)
                
        except Exception as e:
            self.training_error.emit(f"训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc() 