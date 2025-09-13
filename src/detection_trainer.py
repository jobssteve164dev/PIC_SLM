import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PyQt5.QtCore import QObject, pyqtSignal, QThread
import numpy as np
from typing import Dict, Any, Optional
import json
import subprocess
import sys
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision
from detection_utils import DetectionDataset, evaluate_model
import logging
import time
import shutil

class DetectionTrainer(QObject):
    """目标检测训练器主类"""
    
    # 定义信号
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    training_finished = pyqtSignal()
    training_error = pyqtSignal(str)
    epoch_finished = pyqtSignal(dict)
    model_download_failed = pyqtSignal(str, str)
    training_stopped = pyqtSignal(dict)
    metrics_updated = pyqtSignal(dict)
    tensorboard_updated = pyqtSignal(str, float, int)
    
    def __init__(self, config=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.stop_training = False
        self.training_thread = None
        self.writer = None
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.tensorboard_log_dir = None
        
    def _apply_activation_function(self, model, activation_name):
        """将指定的激活函数应用到模型的所有合适层中
        
        Args:
            model: PyTorch模型
            activation_name: 激活函数名称
            
        Returns:
            修改后的模型
        """
        self.status_updated.emit(f"正在应用激活函数: {activation_name}")
        
        # 如果选择无激活函数，则保持模型原样
        if activation_name == "None":
            self.status_updated.emit("保持模型原有的激活函数不变")
            return model
            
        # 创建激活函数实例
        if activation_name == "ReLU":
            activation = nn.ReLU(inplace=True)
        elif activation_name == "LeakyReLU":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_name == "PReLU":
            activation = nn.PReLU()
        elif activation_name == "ELU":
            activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_name == "SELU":
            activation = nn.SELU(inplace=True)
        elif activation_name == "GELU":
            activation = nn.GELU()
        elif activation_name == "Mish":
            try:
                activation = nn.Mish(inplace=True)
            except AttributeError:
                # 如果PyTorch版本不支持Mish，则手动实现
                class Mish(nn.Module):
                    def forward(self, x):
                        return x * torch.tanh(nn.functional.softplus(x))
                activation = Mish()
        elif activation_name == "Swish" or activation_name == "SiLU":
            try:
                activation = nn.SiLU(inplace=True)
            except AttributeError:
                # 如果PyTorch版本不支持SiLU，则手动实现
                class Swish(nn.Module):
                    def forward(self, x):
                        return x * torch.sigmoid(x)
                activation = Swish()
        else:
            self.status_updated.emit(f"未知的激活函数 {activation_name}，使用默认的ReLU")
            activation = nn.ReLU(inplace=True)
        
        # 递归替换模型中的激活函数
        def replace_activations(module):
            for name, child in module.named_children():
                if isinstance(child, (nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ELU, nn.SELU, nn.GELU)):
                    # 替换为新的激活函数
                    setattr(module, name, activation)
                else:
                    # 递归处理子模块
                    replace_activations(child)
        
        # 应用激活函数替换
        replace_activations(model)
        
        return model

    def start_training(self, config=None):
        """启动训练过程，使用传入的配置或初始化时的配置"""
        try:
            # 如果传入了新的配置，则使用新配置，否则使用初始化时的配置
            if config is not None:
                self.config = config
            
            # 确保有配置可用
            if self.config is None:
                self.training_error.emit("没有可用的训练配置")
                return
                
            # 直接调用已有的train_model方法
            self.train_model(self.config)
            
        except Exception as e:
            self.training_error.emit(f"启动训练时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def _update_metrics(self, metrics):
        """更新训练指标并发送信号"""
        try:
            # 验证数据格式
            required_fields = ['epoch', 'train_loss']
            if not all(field in metrics for field in required_fields):
                self.logger.warning(f"缺少必要的训练指标字段: {metrics}")
                return
                
            # 发送指标更新信号
            self.metrics_updated.emit(metrics)
            
            # 如果有TensorBoard，记录指标
            if self.writer:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and key != 'epoch' and key != 'batch':
                        self.writer.add_scalar(f'{key}', value, metrics['epoch'])
                
                # 立即刷新数据，确保实时写入
                self.writer.flush()
                        
                # 如果有tensorboard_updated信号的接收者
                if hasattr(self, 'tensorboard_log_dir') and self.tensorboard_log_dir:
                    # 发送val_map作为主要指标
                    val_map = metrics.get('val_map', 0.0)
                    self.tensorboard_updated.emit(self.tensorboard_log_dir, val_map, metrics['epoch'])
            
        except Exception as e:
            self.logger.error(f"更新训练指标时出错: {str(e)}")
            self.training_error.emit(f"更新训练指标失败: {str(e)}")
            
    def _train_model_common(self, model_name, train_loader, val_loader, model, optimizer, 
                          num_epochs, model_save_dir, device, lr_scheduler=None, config=None):
        """通用的训练循环实现"""
        try:
            best_map = 0.0
            model_save_path = None
            
            for epoch in range(num_epochs):
                if self.stop_training:
                    self.logger.info("训练已停止")
                    break
                    
                self.status_updated.emit(f'Epoch {epoch+1}/{num_epochs}')
                
                # 训练阶段
                model.train()
                epoch_loss = 0.0
                for i, (images, targets) in enumerate(train_loader):
                    if self.stop_training:
                        break
                        
                    images = [image.to(device) for image in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    optimizer.zero_grad()
                    losses.backward()
                    # 梯度裁剪（可选）
                    if config and config.get('clip_grad', False):
                        if max(p.grad.data.abs().max() for p in model.parameters() if p.grad is not None) > 1:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    
                    epoch_loss += losses.item()
                    
                    # 更新进度
                    progress = int(((epoch * len(train_loader) + i + 1) /
                                 (num_epochs * len(train_loader))) * 100)
                    self.progress_updated.emit(progress)
                    
                    # 实时更新训练指标
                    if i % 10 == 0:  # 每10个batch更新一次
                        current_loss = epoch_loss / (i + 1)
                        metrics = {
                            'epoch': epoch + 1,
                            'batch': i + 1,
                            'train_loss': current_loss,
                            'learning_rate': optimizer.param_groups[0]['lr']
                        }
                        self._update_metrics(metrics)
                
                # 更新学习率（如果有调度器）
                if lr_scheduler:
                    lr_scheduler.step()
                
                # 验证阶段
                eval_metrics = evaluate_model(model, val_loader, device)
                current_map = eval_metrics['map']
                avg_val_loss = eval_metrics['loss']
                
                # 更新完整的训练指标
                metrics = {
                    'epoch': epoch + 1,
                    'train_loss': epoch_loss / len(train_loader),
                    'val_loss': avg_val_loss,
                    'val_map': current_map,
                    'learning_rate': optimizer.param_groups[0]['lr']
                }
                self._update_metrics(metrics)
                
                # 发送epoch完成信号
                self.epoch_finished.emit(metrics)
                
                # 保存最佳模型
                if current_map > best_map:
                    best_map = current_map
                    # 使用统一的命名格式
                    model_note = config.get('model_note', '') if config else ''
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    model_file_suffix = f"_{model_note}" if model_note else ""
                    model_save_path = os.path.join(model_save_dir, f'{model_name.lower()}{model_file_suffix}_{timestamp}_best.pth')
                    torch.save(model.state_dict(), model_save_path)
                    self.status_updated.emit(f'保存最佳模型，mAP: {best_map:.4f}')
            
            # 保存最终模型
            model_note = config.get('model_note', '') if config else ''
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_file_suffix = f"_{model_note}" if model_note else ""
            final_model_path = os.path.join(model_save_dir, f'{model_name.lower()}{model_file_suffix}_{timestamp}_final.pth')
            torch.save(model.state_dict(), final_model_path)
            self.status_updated.emit(f'保存最终模型: {final_model_path}')
            
            # 确保最后一轮的tensorboard数据写入并更新
            if self.writer:
                # 显式记录最终的训练完成状态
                self.writer.add_scalar('Final/mAP', best_map, num_epochs)
                self.writer.add_scalar('Final/Epochs', num_epochs, 0)
                # 确保所有数据被写入
                self.writer.flush()
                
                # 发送tensorboard更新信号
                if hasattr(self, 'tensorboard_log_dir') and self.tensorboard_log_dir:
                    self.tensorboard_updated.emit(self.tensorboard_log_dir, best_map, num_epochs)
            
            return best_map, model_save_path
        
        except Exception as e:
            self.training_error.emit(f"训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0, None
            
    def train_model(self, config: Dict[str, Any]) -> None:
        """使用配置字典启动训练"""
        try:
            # 重置停止标志
            self.stop_training = False
            
            # 提取基本参数
            data_dir = config.get('data_dir', '')
            model_name = config.get('model_name', 'YOLOv8')
            num_epochs = config.get('num_epochs', 50)
            batch_size = config.get('batch_size', 16)
            learning_rate = config.get('learning_rate', 0.001)
            model_save_dir = config.get('model_save_dir', 'models/saved_models')
            use_tensorboard = config.get('use_tensorboard', True)
            
            # 创建模型保存目录
            os.makedirs(model_save_dir, exist_ok=True)
            
            # 创建TensorBoard写入器
            if use_tensorboard:
                # 获取TensorBoard日志目录
                tensorboard_dir = config.get('tensorboard_log_dir', os.path.join(model_save_dir, 'tensorboard_logs'))
                
                # 创建带有时间戳的唯一运行目录
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                model_run_name = f"{model_name}_{timestamp}"
                tensorboard_run_dir = os.path.join(tensorboard_dir, model_run_name)
                os.makedirs(tensorboard_run_dir, exist_ok=True)
                self.writer = SummaryWriter(tensorboard_run_dir)
                self.logger.info(f"已创建TensorBoard写入器，日志目录: {tensorboard_run_dir}")
                
                # 保存TensorBoard日志目录路径，便于后续访问
                self.tensorboard_log_dir = tensorboard_run_dir
            
            # 根据模型名称选择训练方法
            if model_name.startswith('YOLOv8'):
                self._train_yolov8(data_dir, model_name, num_epochs, batch_size, learning_rate,
                                 model_save_dir, use_tensorboard, config)
            elif model_name == 'Faster R-CNN':
                self._train_fasterrcnn(data_dir, num_epochs, batch_size, learning_rate,
                                     model_save_dir, use_tensorboard, config)
            elif model_name == 'SSD':
                self._train_ssd(data_dir, num_epochs, batch_size, learning_rate,
                              model_save_dir, use_tensorboard, config)
            elif model_name == 'RetinaNet':
                self._train_retinanet(data_dir, num_epochs, batch_size, learning_rate,
                                    model_save_dir, use_tensorboard, config)
            elif model_name == 'DETR':
                self._train_detr(data_dir, num_epochs, batch_size, learning_rate,
                               model_save_dir, use_tensorboard, config)
            else:
                error_msg = f"不支持的模型类型: {model_name}"
                self.logger.error(error_msg)
                self.training_error.emit(error_msg)
                return
            
            # 关闭TensorBoard写入器
            if self.writer:
                # 添加显式刷新操作，确保最后一轮数据被记录
                try:
                    # 添加训练完成标记
                    self.writer.add_text('Training', 'Training completed', 0)
                    
                    # 强制刷新最后的数据
                    self.writer.flush()
                    
                    # 发送最终的tensorboard更新信号
                    if hasattr(self, 'tensorboard_log_dir') and self.tensorboard_log_dir:
                        self.tensorboard_updated.emit(
                            self.tensorboard_log_dir, 
                            0.0,  # 这里传0是因为我们只是想触发UI刷新
                            -1    # 使用-1表示这是训练结束信号
                        )
                    
                    # 等待200ms确保数据被写入
                    time.sleep(0.2)
                    
                    # 关闭写入器
                    self.writer.close()
                    self.logger.info("已关闭TensorBoard写入器")
                except Exception as e:
                    self.logger.error(f"关闭TensorBoard写入器时出错: {str(e)}")
                
            self.training_finished.emit()
                
        except Exception as e:
            error_msg = f"训练过程中出错: {str(e)}"
            self.logger.error(error_msg)
            self.training_error.emit(error_msg)
            import traceback
            traceback.print_exc()
            
    def stop(self, is_intelligent_restart=False):
        """停止训练过程"""
        try:
            self.stop_training = True
            if is_intelligent_restart:
                self.status_updated.emit("智能训练重启中...")
            else:
                self.status_updated.emit("训练已停止")
            self.training_stopped.emit({'is_intelligent_restart': is_intelligent_restart})
            self.logger.info("已发送训练停止信号")
        except Exception as e:
            self.logger.error(f"停止训练时出错: {str(e)}")
            self.training_error.emit(f"停止训练失败: {str(e)}")

    def _train_yolov8(self, data_dir, model_name, num_epochs, batch_size, learning_rate, model_save_dir, use_tensorboard, config=None):
        """使用YOLOv8进行目标检测训练"""
        try:
            self.status_updated.emit("开始YOLOv8模型训练...")
            
            # 检查ultralytics包
            try:
                from ultralytics import YOLO
            except ImportError:
                self.model_download_failed.emit("YOLOv8", "https://github.com/ultralytics/ultralytics")
                self.training_error.emit("未安装ultralytics包，请使用pip install ultralytics安装")
                return
                
            # 提取YOLOv8的型号大小 (s, m, l, x)
            yolo_size = 's'  # 默认使用YOLOv8s
            if model_name in ['YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x']:
                yolo_size = model_name[-1].lower()
                
            # 配置训练参数
            model_path = f'yolov8{yolo_size}'
            model = YOLO(model_path)
            
            # 准备训练数据路径
            # YOLO格式需要data.yaml文件
            data_yaml_path = os.path.join(data_dir, 'data.yaml')
            
            if not os.path.exists(data_yaml_path):
                error_msg = f"训练数据配置文件不存在: {data_yaml_path}"
                self.status_updated.emit(error_msg)
                self.training_error.emit(error_msg)
                return
                
            # 处理空model_note的情况
            model_note = config.get('model_note', '') if config else ''
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # 获取参数保存目录，如果未指定则使用模型保存目录
            param_save_dir = config.get('default_param_save_dir', model_save_dir) if config else model_save_dir
            
            # 标准化路径格式
            model_save_dir = os.path.normpath(model_save_dir)
            param_save_dir = os.path.normpath(param_save_dir)
            
            os.makedirs(param_save_dir, exist_ok=True)
            os.makedirs(model_save_dir, exist_ok=True)
            
            # 生成使用统一格式的保存路径
            model_save_path = os.path.join(model_save_dir, f'{model_name}_{timestamp}_{model_note}_best.pt')
            
            # 设置训练参数
            train_args = {
                'data': data_yaml_path,
                'epochs': num_epochs,
                'batch': batch_size,
                'imgsz': config.get('resolution', 640) if config else 640,
                'device': '0' if torch.cuda.is_available() else 'cpu',
                'project': os.path.dirname(model_save_dir),
                'name': os.path.basename(model_save_dir),
                'exist_ok': True,
                'pretrained': True,
                'optimizer': config.get('optimizer', 'SGD') if config else 'SGD',
                'lr0': learning_rate,
                'save_json': False,
                'save': True,
                'save_period': -1,  # 每个epoch都保存
                'patience': config.get('patience', 50) if config else 50,
                'verbose': True,
                'seed': 42
            }
            
            # 添加激活函数配置（如果提供）
            if config and 'activation_function' in config:
                activation = config.get('activation_function')
                self.status_updated.emit(f"为YOLOv8配置激活函数: {activation}")
                
                # YOLOv8中常用的激活函数映射
                yolo_activation_map = {
                    'None': None,  # 无激活函数
                    'ReLU': 'ReLU',
                    'LeakyReLU': 'LeakyReLU',
                    'PReLU': 'PReLU', 
                    'SiLU': 'SiLU',
                    'Swish': 'SiLU',  # Swish和SiLU是相同的
                    'Mish': 'Mish',
                    'Hardswish': 'Hardswish',
                    'GELU': 'GELU'
                }
                
                # 将我们的激活函数名映射到YOLOv8支持的名称
                if activation in yolo_activation_map:
                    train_args['act'] = yolo_activation_map[activation]
                else:
                    self.status_updated.emit(f"YOLOv8不支持激活函数 {activation}，使用默认的SiLU")
            
            # 开始训练
            self.status_updated.emit(f"启动YOLOv8{yolo_size}训练，epochs={num_epochs}, batch={batch_size}, lr={learning_rate}")
            
            # 创建训练结果回调
            class YOLOCallback:
                def __init__(self, trainer):
                    self.trainer = trainer
                    self.best_map = 0.0
                    self.best_map_path = ''
                    
                def on_train_start(self):
                    self.trainer.status_updated.emit("YOLOv8训练开始")
                    
                def on_train_epoch_end(self, epoch, metrics):
                    # 更新进度
                    progress = int(((epoch + 1) / num_epochs) * 100)
                    self.trainer.progress_updated.emit(progress)
                    
                    # 记录指标
                    try:
                        val_map = metrics.get('metrics/mAP50-95', 0.0)
                        train_loss = metrics.get('train/box_loss', 0.0)
                        val_loss = metrics.get('val/box_loss', 0.0)
                        
                        # 发送epoch完成信号
                        epoch_info = {
                            'epoch': epoch + 1,
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'val_map': val_map
                        }
                        self.trainer.epoch_finished.emit(epoch_info)
                        
                        # 如果有TensorBoard，记录指标
                        if self.trainer.writer:
                            self.trainer.writer.add_scalar('Loss/train', train_loss, epoch)
                            self.trainer.writer.add_scalar('Loss/val', val_loss, epoch)
                            self.trainer.writer.add_scalar('mAP/val', val_map, epoch)
                            
                            # 确保每轮数据都被写入
                            self.trainer.writer.flush()
                                
                        # 记录最佳模型
                        if val_map > self.best_map:
                            self.best_map = val_map
                            self.trainer.status_updated.emit(f"新的最佳mAP: {val_map:.4f} (Epoch {epoch+1})")
                    except Exception as e:
                        print(f"处理YOLO训练指标时出错: {str(e)}")
                    
                def on_train_end(self, results):
                    # 训练结束，重命名最佳模型
                    try:
                        # 确保最后一轮的数据被写入TensorBoard
                        if self.trainer.writer:
                            # 记录最终轮次信息
                            final_val_map = results.get('metrics/mAP50-95', self.best_map)
                            final_train_loss = results.get('train/box_loss', 0.0)
                            final_val_loss = results.get('val/box_loss', 0.0)
                            
                            # 记录最终训练结果
                            self.trainer.writer.add_scalar('Final/mAP', final_val_map, num_epochs)
                            self.trainer.writer.add_scalar('Final/train_loss', final_train_loss, num_epochs)
                            self.trainer.writer.add_scalar('Final/val_loss', final_val_loss, num_epochs)
                            self.trainer.writer.add_scalar('Final/Epochs', num_epochs, 0)
                            
                            # 强制刷新确保数据写入
                            self.trainer.writer.flush()
                            
                            # 发送tensorboard更新信号，确保UI能够获取最新数据
                            if hasattr(self.trainer, 'tensorboard_log_dir') and self.trainer.tensorboard_log_dir:
                                self.trainer.tensorboard_updated.emit(
                                    self.trainer.tensorboard_log_dir, 
                                    final_val_map, 
                                    num_epochs
                                )
                            
                        # 获取YOLO保存的最佳模型路径
                        yolo_best_path = getattr(results, 'best', None)
                        
                        if yolo_best_path and os.path.exists(yolo_best_path):
                            # 将YOLO生成的模型复制到我们指定的统一命名路径
                            from shutil import copyfile
                            copyfile(yolo_best_path, model_save_path)
                            self.trainer.status_updated.emit(f"最佳模型已保存到: {model_save_path}")
                            
                            # 保存训练信息
                            training_info = {
                                'model_name': model_name,
                                'num_epochs': num_epochs,
                                'batch_size': batch_size,
                                'learning_rate': learning_rate,
                                'best_map': self.best_map,
                                'model_path': model_save_path,
                                'timestamp': timestamp  # 添加时间戳到训练信息
                            }
                            
                            with open(os.path.join(model_save_dir, 'training_info.json'), 'w') as f:
                                json.dump(training_info, f, indent=4)
                    except Exception as e:
                        self.trainer.status_updated.emit(f"保存最佳模型时出错: {str(e)}")
            
            # 创建回调并开始训练
            callback = YOLOCallback(self)
            results = model.train(**train_args, callbacks=[callback])
            
            self.status_updated.emit(f"YOLOv8训练完成，最佳mAP: {callback.best_map:.4f}")
            
        except Exception as e:
            self.training_error.emit(f"YOLOv8训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def _train_fasterrcnn(self, data_dir, num_epochs, batch_size, learning_rate, model_save_dir, use_tensorboard=True, config=None):
        """使用Faster R-CNN进行目标检测训练"""
        try:
            self.status_updated.emit("开始Faster R-CNN模型训练...")
            
            # 检查是否安装了torchvision
            try:
                from torchvision.models.detection import fasterrcnn_resnet50_fpn
                from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            except ImportError:
                self.training_error.emit("未安装torchvision包")
                return
            
            # 创建数据转换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
            
            # 创建数据集和数据加载器
            train_dataset = DetectionDataset(os.path.join(data_dir, 'train'), transform=transform)
            val_dataset = DetectionDataset(os.path.join(data_dir, 'val'), transform=transform, is_train=False)
            
            # 根据操作系统选择合适的num_workers
            import platform
            num_workers = 0 if platform.system() == 'Windows' else 4
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                collate_fn=lambda x: tuple(zip(*x)))
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                collate_fn=lambda x: tuple(zip(*x)))
            
            # 创建模型
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            num_classes = len(train_dataset.classes) + 1  # 背景 + 目标类别
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
            # 应用激活函数
            if config and 'activation_function' in config:
                activation_function = config.get('activation_function', 'ReLU')
                self.status_updated.emit(f"应用自定义激活函数: {activation_function}")
                model = self._apply_activation_function(model, activation_function)
            
            # 将模型移到设备
            model.to(self.device)
            
            # 定义优化器
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(params, lr=learning_rate)
            
            # 使用通用训练循环
            best_map, best_model_path = self._train_model_common(
                model_name='fasterrcnn',
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                optimizer=optimizer,
                num_epochs=num_epochs,
                model_save_dir=model_save_dir,
                device=self.device,
                config=config
            )
            
            self.status_updated.emit(f'Faster R-CNN训练完成，最佳mAP: {best_map:.4f}')
            
        except Exception as e:
            self.training_error.emit(f"Faster R-CNN训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def _train_ssd(self, data_dir, num_epochs, batch_size, learning_rate, model_save_dir, use_tensorboard=True, config=None):
        """使用SSD进行目标检测训练"""
        try:
            self.status_updated.emit("开始SSD模型训练...")
            
            # 检查是否安装了torchvision
            try:
                from torchvision.models.detection import ssd300_vgg16
                from torchvision.models.detection.ssd import SSDHead, SSDLoss
            except ImportError:
                self.training_error.emit("未安装torchvision包")
                return
            
            # 创建数据转换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
            
            # 创建数据集和数据加载器
            train_dataset = DetectionDataset(os.path.join(data_dir, 'train'), transform=transform)
            val_dataset = DetectionDataset(os.path.join(data_dir, 'val'), transform=transform, is_train=False)
            
            # 根据操作系统选择合适的num_workers
            import platform
            num_workers = 0 if platform.system() == 'Windows' else 4
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                collate_fn=lambda x: tuple(zip(*x)))
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                collate_fn=lambda x: tuple(zip(*x)))
            
            # 创建模型
            model = ssd300_vgg16(pretrained=True)
            num_classes = len(train_dataset.classes) + 1  # 背景 + 目标类别
            in_channels = model.backbone.out_channels
            num_anchors = model.anchor_generator.num_anchors_per_location()
            model.head = SSDHead(in_channels, num_anchors, num_classes)
            
            # 应用激活函数
            if config and 'activation_function' in config:
                activation_function = config.get('activation_function', 'ReLU')
                self.status_updated.emit(f"应用自定义激活函数: {activation_function}")
                model = self._apply_activation_function(model, activation_function)
            
            # 将模型移到设备
            model.to(self.device)
            
            # 定义优化器
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(params, lr=learning_rate)
            
            # 使用通用训练循环
            best_map, best_model_path = self._train_model_common(
                model_name='ssd',
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                optimizer=optimizer,
                num_epochs=num_epochs,
                model_save_dir=model_save_dir,
                device=self.device,
                config=config
            )
            
            self.status_updated.emit(f'SSD训练完成，最佳mAP: {best_map:.4f}')
            
        except Exception as e:
            self.training_error.emit(f"SSD训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def _train_retinanet(self, data_dir, num_epochs, batch_size, learning_rate, model_save_dir, use_tensorboard=True, config=None):
        """使用RetinaNet进行目标检测训练"""
        try:
            self.status_updated.emit("开始RetinaNet模型训练...")
            
            # 检查是否安装了torchvision
            try:
                from torchvision.models.detection import retinanet_resnet50_fpn
                from torchvision.models.detection.retinanet import RetinaNetHead
            except ImportError:
                self.training_error.emit("未安装torchvision包")
                return
            
            # 创建数据转换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
            
            # 创建数据集和数据加载器
            train_dataset = DetectionDataset(os.path.join(data_dir, 'train'), transform=transform)
            val_dataset = DetectionDataset(os.path.join(data_dir, 'val'), transform=transform, is_train=False)
            
            # 根据操作系统选择合适的num_workers
            import platform
            num_workers = 0 if platform.system() == 'Windows' else 4
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                collate_fn=lambda x: tuple(zip(*x)))
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                collate_fn=lambda x: tuple(zip(*x)))
            
            # 创建模型
            model = retinanet_resnet50_fpn(pretrained=True)
            num_classes = len(train_dataset.classes) + 1  # 背景 + 目标类别
            in_channels = model.backbone.out_channels
            num_anchors = model.anchor_generator.num_anchors_per_location()[0]
            model.head = RetinaNetHead(in_channels, num_anchors, num_classes)
            
            # 应用激活函数
            if config and 'activation_function' in config:
                activation_function = config.get('activation_function', 'ReLU')
                self.status_updated.emit(f"应用自定义激活函数: {activation_function}")
                model = self._apply_activation_function(model, activation_function)
            
            # 将模型移到设备
            model.to(self.device)
            
            # 定义优化器
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(params, lr=learning_rate)
            
            # 使用通用训练循环
            best_map, best_model_path = self._train_model_common(
                model_name='retinanet',
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                optimizer=optimizer,
                num_epochs=num_epochs,
                model_save_dir=model_save_dir,
                device=self.device,
                config=config
            )
            
            self.status_updated.emit(f'RetinaNet训练完成，最佳mAP: {best_map:.4f}')
            
        except Exception as e:
            self.training_error.emit(f"RetinaNet训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def _train_detr(self, data_dir, num_epochs, batch_size, learning_rate, model_save_dir, use_tensorboard=True, config=None):
        """使用DETR进行目标检测训练"""
        try:
            self.status_updated.emit("开始DETR模型训练...")
            
            # 检查是否安装了torchvision
            try:
                from torchvision.models.detection import detr_resnet50
            except ImportError:
                self.training_error.emit("未安装torchvision包或版本太低")
                return
            
            # 创建数据转换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
            
            # 创建数据集和数据加载器
            train_dataset = DetectionDataset(os.path.join(data_dir, 'train'), transform=transform)
            val_dataset = DetectionDataset(os.path.join(data_dir, 'val'), transform=transform, is_train=False)
            
            # 根据操作系统选择合适的num_workers
            import platform
            num_workers = 0 if platform.system() == 'Windows' else 4
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                collate_fn=lambda x: tuple(zip(*x)))
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                collate_fn=lambda x: tuple(zip(*x)))
            
            # 创建模型
            model = detr_resnet50(pretrained=True)
            num_classes = len(train_dataset.classes) + 1  # 背景 + 目标类别
            
            # 修改分类头以适应我们的类别数
            hidden_dim = model.transformer.d_model
            model.class_embed = nn.Linear(hidden_dim, num_classes)
            
            # 应用激活函数
            if config and 'activation_function' in config:
                activation_function = config.get('activation_function', 'ReLU')
                self.status_updated.emit(f"应用自定义激活函数: {activation_function}")
                model = self._apply_activation_function(model, activation_function)
            
            # 将模型移到设备
            model.to(self.device)
            
            # 定义优化器，使用不同的学习率
            param_dicts = [
                {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": learning_rate * 0.1,
                },
            ]
            optimizer = torch.optim.AdamW(param_dicts, lr=learning_rate, weight_decay=0.0001)
            
            # 创建学习率调度器
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)
            
            # 使用通用训练循环
            best_map, best_model_path = self._train_model_common(
                model_name='detr',
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                optimizer=optimizer,
                num_epochs=num_epochs,
                model_save_dir=model_save_dir,
                device=self.device,
                lr_scheduler=lr_scheduler,
                config=config
            )
            
            self.status_updated.emit(f'DETR训练完成，最佳mAP: {best_map:.4f}')
            
        except Exception as e:
            self.training_error.emit(f"DETR训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def _calculate_map(self, predictions, targets, iou_threshold=0.5):
        """计算mAP (mean Average Precision)
        
        Args:
            predictions: 模型预测结果列表，每个元素包含boxes, labels, scores
            targets: 真实标签列表，每个元素包含boxes, labels
            iou_threshold: IoU阈值，默认0.5
            
        Returns:
            float: mAP值
        """
        try:
            # 初始化每个类别的AP列表
            class_aps = []
            
            # 获取所有唯一的类别标签
            all_labels = set()
            for target in targets:
                all_labels.update(target['labels'].tolist())
            
            # 对每个类别计算AP
            for label in all_labels:
                # 收集当前类别的预测和真实标签
                class_predictions = []
                class_targets = []
                
                for pred, target in zip(predictions, targets):
                    # 获取当前类别的预测
                    mask = pred['labels'] == label
                    if mask.any():
                        class_predictions.append({
                            'boxes': pred['boxes'][mask],
                            'scores': pred['scores'][mask]
                        })
                    
                    # 获取当前类别的真实标签
                    mask = target['labels'] == label
                    if mask.any():
                        class_targets.append({
                            'boxes': target['boxes'][mask]
                        })
                
                if not class_predictions or not class_targets:
                    continue
                
                # 计算当前类别的AP
                ap = self._calculate_ap(class_predictions, class_targets, iou_threshold)
                class_aps.append(ap)
            
            # 计算mAP
            if not class_aps:
                return 0.0
                
            return sum(class_aps) / len(class_aps)
            
        except Exception as e:
            self.logger.error(f"计算mAP时出错: {str(e)}")
            return 0.0
            
    def _calculate_ap(self, predictions, targets, iou_threshold):
        """计算单个类别的Average Precision
        
        Args:
            predictions: 预测结果列表
            targets: 真实标签列表
            iou_threshold: IoU阈值
            
        Returns:
            float: AP值
        """
        try:
            # 收集所有预测框和对应的分数
            all_boxes = []
            all_scores = []
            for pred in predictions:
                all_boxes.extend(pred['boxes'].tolist())
                all_scores.extend(pred['scores'].tolist())
            
            # 按分数降序排序
            sorted_indices = np.argsort(all_scores)[::-1]
            all_boxes = np.array(all_boxes)[sorted_indices]
            all_scores = np.array(all_scores)[sorted_indices]
            
            # 收集所有真实标签框
            all_target_boxes = []
            for target in targets:
                all_target_boxes.extend(target['boxes'].tolist())
            all_target_boxes = np.array(all_target_boxes)
            
            # 计算每个预测框与所有真实标签框的IoU
            ious = self._calculate_iou_matrix(all_boxes, all_target_boxes)
            
            # 计算precision和recall
            tp = np.zeros(len(all_boxes))
            fp = np.zeros(len(all_boxes))
            
            for i in range(len(all_boxes)):
                if len(all_target_boxes) > 0:
                    max_iou = np.max(ious[i])
                    if max_iou >= iou_threshold:
                        tp[i] = 1
                    else:
                        fp[i] = 1
                else:
                    fp[i] = 1
            
            # 计算累积值
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            # 计算precision和recall
            recalls = tp_cumsum / float(len(all_target_boxes))
            precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)
            
            # 计算AP (使用11点插值)
            ap = 0.0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap = ap + p / 11.0
            
            return ap
            
        except Exception as e:
            self.logger.error(f"计算AP时出错: {str(e)}")
            return 0.0
            
    def _calculate_iou_matrix(self, boxes1, boxes2):
        """计算两组框之间的IoU矩阵
        
        Args:
            boxes1: 第一组框，shape为(N, 4)
            boxes2: 第二组框，shape为(M, 4)
            
        Returns:
            numpy.ndarray: IoU矩阵，shape为(N, M)
        """
        try:
            # 确保输入是numpy数组
            boxes1 = np.array(boxes1)
            boxes2 = np.array(boxes2)
            
            # 计算交集
            x1 = np.maximum(boxes1[:, 0][:, np.newaxis], boxes2[:, 0])
            y1 = np.maximum(boxes1[:, 1][:, np.newaxis], boxes2[:, 1])
            x2 = np.minimum(boxes1[:, 2][:, np.newaxis], boxes2[:, 2])
            y2 = np.minimum(boxes1[:, 3][:, np.newaxis], boxes2[:, 3])
            
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            
            # 计算并集
            area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
            area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
            union = area1[:, np.newaxis] + area2 - intersection
            
            # 计算IoU
            iou = intersection / np.maximum(union, np.finfo(np.float64).eps)
            
            return iou
            
        except Exception as e:
            self.logger.error(f"计算IoU矩阵时出错: {str(e)}")
            return np.zeros((len(boxes1), len(boxes2))) 