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

class DetectionTrainingThread(QThread):
    """负责在单独线程中执行目标检测训练过程的类"""
    
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
    
    def run(self):
        """线程运行入口，执行模型训练"""
        try:
            # 重置停止标志
            self.stop_training = False
            
            # 提取基本参数
            data_dir = self.config.get('data_dir', '')
            model_name = self.config.get('model_name', 'YOLOv5')
            num_epochs = self.config.get('num_epochs', 100)
            batch_size = self.config.get('batch_size', 16)
            learning_rate = self.config.get('learning_rate', 0.001)
            model_save_dir = self.config.get('model_save_dir', 'models/saved_models')
            use_tensorboard = self.config.get('use_tensorboard', True)
            
            # 提取目标检测特有参数
            iou_threshold = self.config.get('iou_threshold', 0.5)
            conf_threshold = self.config.get('conf_threshold', 0.25)
            resolution = self.config.get('resolution', '640x640')
            use_pretrained = self.config.get('use_pretrained', True)
            pretrained_path = self.config.get('pretrained_path', '')
            model_note = self.config.get('model_note', '')
            
            # 检查数据集目录
            if not os.path.exists(data_dir):
                self.training_error.emit(f"数据集目录不存在: {data_dir}")
                return
                
            # 创建模型保存目录
            os.makedirs(model_save_dir, exist_ok=True)
            
            # 根据模型名称选择训练方法
            if model_name.startswith('YOLOv8'):
                self._train_yolov8(data_dir, model_name, num_epochs, batch_size, learning_rate,
                                 model_save_dir, iou_threshold, conf_threshold, resolution,
                                 use_pretrained, pretrained_path, model_note)
            elif model_name == 'Faster R-CNN':
                self._train_fasterrcnn(data_dir, num_epochs, batch_size, learning_rate,
                                     model_save_dir, use_pretrained, pretrained_path, model_note)
            elif model_name == 'SSD':
                self._train_ssd(data_dir, num_epochs, batch_size, learning_rate,
                              model_save_dir, use_pretrained, pretrained_path, model_note)
            elif model_name == 'RetinaNet':
                self._train_retinanet(data_dir, num_epochs, batch_size, learning_rate,
                                    model_save_dir, use_pretrained, pretrained_path, model_note)
            elif model_name == 'DETR':
                self._train_detr(data_dir, num_epochs, batch_size, learning_rate,
                               model_save_dir, use_pretrained, pretrained_path, model_note)
            else:
                self.training_error.emit(f"不支持的模型类型: {model_name}")
                return
                
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
                   model_save_dir, use_tensorboard=True):
        """目标检测模型训练的主要实现"""
        try:
            # 标准化路径格式
            data_dir = os.path.normpath(data_dir).replace('\\', '/')
            model_save_dir = os.path.normpath(model_save_dir).replace('\\', '/')
            
            # 检查数据集目录
            train_dir = os.path.join(data_dir, 'train')
            val_dir = os.path.join(data_dir, 'val')
            
            if not os.path.exists(train_dir):
                self.training_error.emit(f"训练数据集目录不存在: {train_dir}")
                return
                
            if not os.path.exists(val_dir):
                self.training_error.emit(f"验证数据集目录不存在: {val_dir}")
                return
            
            # 创建模型保存目录
            os.makedirs(model_save_dir, exist_ok=True)
            
            # 根据模型名称选择不同的训练方法
            model_name = model_name.lower()
            if model_name in ['yolov5', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']:
                self._train_yolov5(data_dir, num_epochs, batch_size, learning_rate, model_save_dir)
            elif model_name in ['yolov8', 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']:
                self._train_yolov8(data_dir, num_epochs, batch_size, learning_rate, model_save_dir)
            elif model_name in ['fasterrcnn', 'faster-rcnn']:
                self._train_fasterrcnn(data_dir, num_epochs, batch_size, learning_rate, model_save_dir)
            elif model_name == 'ssd':
                self._train_ssd(data_dir, num_epochs, batch_size, learning_rate, model_save_dir)
            elif model_name == 'retinanet':
                self._train_retinanet(data_dir, num_epochs, batch_size, learning_rate, model_save_dir)
            elif model_name == 'detr':
                self._train_detr(data_dir, num_epochs, batch_size, learning_rate, model_save_dir)
            else:
                self.training_error.emit(f"不支持的目标检测模型: {model_name}")
                return
                
        except Exception as e:
            self.training_error.emit(f"训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _train_yolov5(self, data_dir, num_epochs, batch_size, learning_rate, model_save_dir):
        """使用YOLOv5进行目标检测训练"""
        try:
            self.status_updated.emit("开始YOLOv5模型训练...")
            
            # 检查是否安装了ultralytics包
            try:
                from ultralytics import YOLO
            except ImportError:
                self.status_updated.emit("正在安装ultralytics包...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
                from ultralytics import YOLO
            
            # 创建YOLO模型
            model = YOLO('yolov5s.pt')  # 使用预训练的YOLOv5s模型
            
            # 准备训练配置
            train_config = {
                'data': os.path.join(data_dir, 'data.yaml'),
                'epochs': num_epochs,
                'batch': batch_size,
                'imgsz': 640,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'workers': 4,
                'project': model_save_dir,
                'name': 'yolov5_train',
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'Adam',
                'lr0': learning_rate,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 0.05,
                'cls': 0.5,
                'cls_pw': 1.0,
                'obj': 1.0,
                'obj_pw': 1.0,
                'iou_t': 0.2,
                'anchor_t': 4.0,
                'fl_gamma': 0.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'val': True,
                'save': True,
                'save_json': False,
                'save_hybrid': False,
                'conf': 0.001,
                'iou': 0.6,
                'max_det': 300,
                'half': False,
                'dnn': False,
                'plots': True
            }
            
            # 开始训练
            results = model.train(**train_config)
            
            # 保存训练结果
            training_info = {
                'model_name': 'YOLOv5',
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'best_map': float(results.best_map),
                'model_path': os.path.join(model_save_dir, 'yolov5_train/weights/best.pt')
            }
            
            with open(os.path.join(model_save_dir, 'training_info.json'), 'w') as f:
                json.dump(training_info, f, indent=4)
            
            self.status_updated.emit(f'YOLOv5训练完成，最佳mAP: {results.best_map:.4f}')
            
        except Exception as e:
            self.training_error.emit(f"YOLOv5训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _train_yolov8(self, data_dir, num_epochs, batch_size, learning_rate, model_save_dir):
        """使用YOLOv8进行目标检测训练"""
        try:
            self.status_updated.emit("开始YOLOv8模型训练...")
            
            # 检查是否安装了ultralytics包
            try:
                from ultralytics import YOLO
            except ImportError:
                self.status_updated.emit("正在安装ultralytics包...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
                from ultralytics import YOLO
            
            # 创建YOLO模型
            model = YOLO('yolov8n.pt')  # 使用预训练的YOLOv8n模型
            
            # 准备训练配置
            train_config = {
                'data': os.path.join(data_dir, 'data.yaml'),
                'epochs': num_epochs,
                'batch': batch_size,
                'imgsz': 640,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'workers': 4,
                'project': model_save_dir,
                'name': 'yolov8_train',
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'Adam',
                'lr0': learning_rate,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 0.05,
                'cls': 0.5,
                'cls_pw': 1.0,
                'obj': 1.0,
                'obj_pw': 1.0,
                'iou_t': 0.2,
                'anchor_t': 4.0,
                'fl_gamma': 0.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'val': True,
                'save': True,
                'save_json': False,
                'save_hybrid': False,
                'conf': 0.001,
                'iou': 0.6,
                'max_det': 300,
                'half': False,
                'dnn': False,
                'plots': True
            }
            
            # 开始训练
            results = model.train(**train_config)
            
            # 保存训练结果
            training_info = {
                'model_name': 'YOLOv8',
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'best_map': float(results.best_map),
                'model_path': os.path.join(model_save_dir, 'yolov8_train/weights/best.pt')
            }
            
            with open(os.path.join(model_save_dir, 'training_info.json'), 'w') as f:
                json.dump(training_info, f, indent=4)
            
            self.status_updated.emit(f'YOLOv8训练完成，最佳mAP: {results.best_map:.4f}')
            
        except Exception as e:
            self.training_error.emit(f"YOLOv8训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _train_fasterrcnn(self, data_dir, num_epochs, batch_size, learning_rate, model_save_dir):
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
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                collate_fn=lambda x: tuple(zip(*x)))
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                collate_fn=lambda x: tuple(zip(*x)))
            
            # 创建模型
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            num_classes = 2  # 背景 + 目标类别
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
            # 将模型移到设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # 定义优化器
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(params, lr=learning_rate)
            
            # 创建TensorBoard写入器
            writer = SummaryWriter(os.path.join(model_save_dir, 'runs'))
            
            # 训练循环
            best_map = 0.0
            for epoch in range(num_epochs):
                if self.stop_training:
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
                    optimizer.step()
                    
                    epoch_loss += losses.item()
                    
                    # 更新进度
                    progress = int(((epoch * len(train_loader) + i + 1) /
                                 (num_epochs * len(train_loader))) * 100)
                    self.progress_updated.emit(progress)
                
                # 记录训练损失
                avg_loss = epoch_loss / len(train_loader)
                writer.add_scalar('Loss/train', avg_loss, epoch)
                
                # 验证阶段
                eval_metrics = evaluate_model(model, val_loader, device)
                current_map = eval_metrics['map']
                avg_val_loss = eval_metrics['loss']
                
                # 记录验证指标
                writer.add_scalar('Loss/val', avg_val_loss, epoch)
                writer.add_scalar('mAP/val', current_map, epoch)
                
                # 保存最佳模型
                if current_map > best_map:
                    best_map = current_map
                    model_save_path = os.path.join(model_save_dir, 'fasterrcnn_best.pth')
                    torch.save(model.state_dict(), model_save_path)
                
                # 发送epoch完成信号
                epoch_info = {
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    'val_loss': avg_val_loss,
                    'val_map': current_map
                }
                self.epoch_finished.emit(epoch_info)
            
            # 保存最终模型
            final_model_path = os.path.join(model_save_dir, 'fasterrcnn_final.pth')
            torch.save(model.state_dict(), final_model_path)
            
            # 保存训练信息
            training_info = {
                'model_name': 'Faster R-CNN',
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'best_map': best_map,
                'model_path': final_model_path,
                'best_model_path': os.path.join(model_save_dir, 'fasterrcnn_best.pth')
            }
            
            with open(os.path.join(model_save_dir, 'training_info.json'), 'w') as f:
                json.dump(training_info, f, indent=4)
            
            self.status_updated.emit(f'Faster R-CNN训练完成，最佳mAP: {best_map:.4f}')
            
        except Exception as e:
            self.training_error.emit(f"Faster R-CNN训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _train_ssd(self, data_dir, num_epochs, batch_size, learning_rate, model_save_dir):
        """使用SSD进行目标检测训练"""
        try:
            self.status_updated.emit("开始SSD模型训练...")
            
            # 检查是否安装了torchvision
            try:
                from torchvision.models.detection import ssd300_vgg16
                from torchvision.models.detection.ssd import SSDLoss
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
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                collate_fn=lambda x: tuple(zip(*x)))
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                collate_fn=lambda x: tuple(zip(*x)))
            
            # 创建模型
            model = ssd300_vgg16(pretrained=True)
            num_classes = 2  # 背景 + 目标类别
            in_channels = model.backbone.out_channels
            num_anchors = model.anchor_generator.num_anchors_per_location()
            model.head = torchvision.models.detection.ssd.SSDHead(
                in_channels, num_anchors, num_classes)
            
            # 将模型移到设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # 定义损失函数和优化器
            criterion = SSDLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # 创建TensorBoard写入器
            writer = SummaryWriter(os.path.join(model_save_dir, 'runs'))
            
            # 训练循环
            best_map = 0.0
            for epoch in range(num_epochs):
                if self.stop_training:
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
                    optimizer.step()
                    
                    epoch_loss += losses.item()
                    
                    # 更新进度
                    progress = int(((epoch * len(train_loader) + i + 1) /
                                 (num_epochs * len(train_loader))) * 100)
                    self.progress_updated.emit(progress)
                
                # 记录训练损失
                avg_loss = epoch_loss / len(train_loader)
                writer.add_scalar('Loss/train', avg_loss, epoch)
                
                # 验证阶段
                eval_metrics = evaluate_model(model, val_loader, device)
                current_map = eval_metrics['map']
                avg_val_loss = eval_metrics['loss']
                
                # 记录验证指标
                writer.add_scalar('Loss/val', avg_val_loss, epoch)
                writer.add_scalar('mAP/val', current_map, epoch)
                
                # 保存最佳模型
                if current_map > best_map:
                    best_map = current_map
                    model_save_path = os.path.join(model_save_dir, 'ssd_best.pth')
                    torch.save(model.state_dict(), model_save_path)
                
                # 发送epoch完成信号
                epoch_info = {
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    'val_loss': avg_val_loss,
                    'val_map': current_map
                }
                self.epoch_finished.emit(epoch_info)
            
            # 保存最终模型
            final_model_path = os.path.join(model_save_dir, 'ssd_final.pth')
            torch.save(model.state_dict(), final_model_path)
            
            # 保存训练信息
            training_info = {
                'model_name': 'SSD',
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'best_map': best_map,
                'model_path': final_model_path,
                'best_model_path': os.path.join(model_save_dir, 'ssd_best.pth')
            }
            
            with open(os.path.join(model_save_dir, 'training_info.json'), 'w') as f:
                json.dump(training_info, f, indent=4)
            
            self.status_updated.emit(f'SSD训练完成，最佳mAP: {best_map:.4f}')
            
        except Exception as e:
            self.training_error.emit(f"SSD训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def _train_retinanet(self, data_dir, num_epochs, batch_size, learning_rate, model_save_dir):
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
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                collate_fn=lambda x: tuple(zip(*x)))
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                collate_fn=lambda x: tuple(zip(*x)))
            
            # 创建模型
            model = retinanet_resnet50_fpn(pretrained=True)
            num_classes = 2  # 背景 + 目标类别
            in_channels = model.backbone.out_channels
            num_anchors = model.anchor_generator.num_anchors_per_location()
            model.head = RetinaNetHead(in_channels, num_anchors, num_classes)
            
            # 将模型移到设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # 定义优化器
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(params, lr=learning_rate)
            
            # 创建TensorBoard写入器
            writer = SummaryWriter(os.path.join(model_save_dir, 'runs'))
            
            # 训练循环
            best_map = 0.0
            for epoch in range(num_epochs):
                if self.stop_training:
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
                    optimizer.step()
                    
                    epoch_loss += losses.item()
                    
                    # 更新进度
                    progress = int(((epoch * len(train_loader) + i + 1) /
                                 (num_epochs * len(train_loader))) * 100)
                    self.progress_updated.emit(progress)
                
                # 记录训练损失
                avg_loss = epoch_loss / len(train_loader)
                writer.add_scalar('Loss/train', avg_loss, epoch)
                
                # 验证阶段
                eval_metrics = evaluate_model(model, val_loader, device)
                current_map = eval_metrics['map']
                avg_val_loss = eval_metrics['loss']
                
                # 记录验证指标
                writer.add_scalar('Loss/val', avg_val_loss, epoch)
                writer.add_scalar('mAP/val', current_map, epoch)
                
                # 保存最佳模型
                if current_map > best_map:
                    best_map = current_map
                    model_save_path = os.path.join(model_save_dir, 'retinanet_best.pth')
                    torch.save(model.state_dict(), model_save_path)
                
                # 发送epoch完成信号
                epoch_info = {
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    'val_loss': avg_val_loss,
                    'val_map': current_map
                }
                self.epoch_finished.emit(epoch_info)
            
            # 保存最终模型
            final_model_path = os.path.join(model_save_dir, 'retinanet_final.pth')
            torch.save(model.state_dict(), final_model_path)
            
            # 保存训练信息
            training_info = {
                'model_name': 'RetinaNet',
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'best_map': best_map,
                'model_path': final_model_path,
                'best_model_path': os.path.join(model_save_dir, 'retinanet_best.pth')
            }
            
            with open(os.path.join(model_save_dir, 'training_info.json'), 'w') as f:
                json.dump(training_info, f, indent=4)
            
            self.status_updated.emit(f'RetinaNet训练完成，最佳mAP: {best_map:.4f}')
            
        except Exception as e:
            self.training_error.emit(f"RetinaNet训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def _train_detr(self, data_dir, num_epochs, batch_size, learning_rate, model_save_dir):
        """使用DETR进行目标检测训练"""
        try:
            self.status_updated.emit("开始DETR模型训练...")
            
            # 检查是否安装了torchvision
            try:
                from torchvision.models.detection import detr_resnet50
                from torchvision.models.detection import DETRHead
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
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                collate_fn=lambda x: tuple(zip(*x)))
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                collate_fn=lambda x: tuple(zip(*x)))
            
            # 创建模型
            model = detr_resnet50(pretrained=True)
            num_classes = 2  # 背景 + 目标类别
            model.num_classes = num_classes
            hidden_dim = model.transformer.d_model
            model.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            
            # 将模型移到设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # 定义优化器
            param_dicts = [
                {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": learning_rate * 0.1,
                },
            ]
            optimizer = torch.optim.AdamW(param_dicts, lr=learning_rate, weight_decay=0.0001)
            
            # 创建学习率调度器
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3)
            
            # 创建TensorBoard写入器
            writer = SummaryWriter(os.path.join(model_save_dir, 'runs'))
            
            # 训练循环
            best_map = 0.0
            for epoch in range(num_epochs):
                if self.stop_training:
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
                    if max(p.grad.data.abs().max() for p in model.parameters()) > 1:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    
                    epoch_loss += losses.item()
                    
                    # 更新进度
                    progress = int(((epoch * len(train_loader) + i + 1) /
                                 (num_epochs * len(train_loader))) * 100)
                    self.progress_updated.emit(progress)
                
                # 更新学习率
                lr_scheduler.step()
                
                # 记录训练损失
                avg_loss = epoch_loss / len(train_loader)
                writer.add_scalar('Loss/train', avg_loss, epoch)
                
                # 验证阶段
                eval_metrics = evaluate_model(model, val_loader, device)
                current_map = eval_metrics['map']
                avg_val_loss = eval_metrics['loss']
                
                # 记录验证指标
                writer.add_scalar('Loss/val', avg_val_loss, epoch)
                writer.add_scalar('mAP/val', current_map, epoch)
                
                # 保存最佳模型
                if current_map > best_map:
                    best_map = current_map
                    model_save_path = os.path.join(model_save_dir, 'detr_best.pth')
                    torch.save(model.state_dict(), model_save_path)
                
                # 发送epoch完成信号
                epoch_info = {
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    'val_loss': avg_val_loss,
                    'val_map': current_map,
                    'learning_rate': optimizer.param_groups[0]['lr']
                }
                self.epoch_finished.emit(epoch_info)
            
            # 保存最终模型
            final_model_path = os.path.join(model_save_dir, 'detr_final.pth')
            torch.save(model.state_dict(), final_model_path)
            
            # 保存训练信息
            training_info = {
                'model_name': 'DETR',
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'best_map': best_map,
                'model_path': final_model_path,
                'best_model_path': os.path.join(model_save_dir, 'detr_best.pth')
            }
            
            with open(os.path.join(model_save_dir, 'training_info.json'), 'w') as f:
                json.dump(training_info, f, indent=4)
            
            self.status_updated.emit(f'DETR训练完成，最佳mAP: {best_map:.4f}')
            
        except Exception as e:
            self.training_error.emit(f"DETR训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()

class DetectionTrainer(QObject):
    """目标检测训练器主类"""
    
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
        self.writer = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def _update_metrics(self, metrics_dict):
        """更新训练指标并发送信号"""
        try:
            # 验证数据格式
            required_fields = ['epoch', 'train_loss', 'val_loss', 'val_map', 'learning_rate']
            if not all(field in metrics_dict for field in required_fields):
                self.logger.warning(f"缺少必要的训练指标字段: {metrics_dict}")
                return
                
            # 发送到训练曲线组件
            self.metrics_updated.emit(metrics_dict)
            
            # 发送到TensorBoard
            for metric_name, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_updated.emit(metric_name, value, metrics_dict.get('epoch', 0))
                    
            self.logger.debug(f"已更新训练指标: {metrics_dict}")
            
        except Exception as e:
            self.logger.error(f"更新训练指标时出错: {str(e)}")
            self.training_error.emit(f"更新训练指标失败: {str(e)}")
            
    def _train_model_common(self, model_name, train_loader, val_loader, model, optimizer, 
                          num_epochs, model_save_dir, device):
        """通用的训练循环实现"""
        try:
            best_map = 0.0
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
                    model_save_path = os.path.join(model_save_dir, f'{model_name}_best.pth')
                    torch.save(model.state_dict(), model_save_path)
                    self.status_updated.emit(f'保存最佳模型，mAP: {best_map:.4f}')
            
            return best_map
            
        except Exception as e:
            self.logger.error(f"训练过程中出错: {str(e)}")
            self.training_error.emit(f"训练失败: {str(e)}")
            raise
            
    def train_model(self, config: Dict[str, Any]) -> None:
        """使用配置字典启动训练"""
        try:
            # 提取基本参数
            data_dir = config.get('data_dir', '')
            model_name = config.get('model_name', 'YOLOv8')
            num_epochs = config.get('num_epochs', 50)
            batch_size = config.get('batch_size', 16)
            learning_rate = config.get('learning_rate', 0.001)
            model_save_dir = config.get('model_save_dir', 'models/saved_models')
            use_tensorboard = config.get('use_tensorboard', True)
            
            # 创建TensorBoard写入器
            if use_tensorboard:
                self.writer = SummaryWriter(os.path.join(model_save_dir, 'runs'))
                self.logger.info("已创建TensorBoard写入器")
            
            # 根据模型名称选择训练方法
            if model_name.startswith('YOLOv8'):
                self._train_yolov8(data_dir, model_name, num_epochs, batch_size, learning_rate,
                                 model_save_dir, use_tensorboard)
            elif model_name == 'Faster R-CNN':
                self._train_fasterrcnn(data_dir, num_epochs, batch_size, learning_rate,
                                     model_save_dir, use_tensorboard)
            elif model_name == 'SSD':
                self._train_ssd(data_dir, num_epochs, batch_size, learning_rate,
                              model_save_dir, use_tensorboard)
            elif model_name == 'RetinaNet':
                self._train_retinanet(data_dir, num_epochs, batch_size, learning_rate,
                                    model_save_dir, use_tensorboard)
            elif model_name == 'DETR':
                self._train_detr(data_dir, num_epochs, batch_size, learning_rate,
                               model_save_dir, use_tensorboard)
            else:
                error_msg = f"不支持的模型类型: {model_name}"
                self.logger.error(error_msg)
                self.training_error.emit(error_msg)
                return
            
            # 关闭TensorBoard写入器
            if self.writer:
                self.writer.close()
                self.logger.info("已关闭TensorBoard写入器")
                
            self.training_finished.emit()
                
        except Exception as e:
            error_msg = f"训练过程中出错: {str(e)}"
            self.logger.error(error_msg)
            self.training_error.emit(error_msg)
            import traceback
            traceback.print_exc()
            
    def stop(self):
        """停止训练过程"""
        try:
            self.stop_training = True
            self.status_updated.emit("训练已停止")
            self.training_stopped.emit()
            self.logger.info("已发送训练停止信号")
        except Exception as e:
            self.logger.error(f"停止训练时出错: {str(e)}")
            self.training_error.emit(f"停止训练失败: {str(e)}")

    def _train_yolov8(self, data_dir, model_name, num_epochs, batch_size, learning_rate, model_save_dir, use_tensorboard):
        """使用YOLOv8进行目标检测训练"""
        try:
            self.status_updated.emit("开始YOLOv8模型训练...")
            
            # 检查是否安装了ultralytics包
            try:
                from ultralytics import YOLO
            except ImportError:
                self.status_updated.emit("正在安装ultralytics包...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
                from ultralytics import YOLO
            
            # 创建YOLO模型
            model = YOLO('yolov8n.pt')  # 使用预训练的YOLOv8n模型
            
            # 准备训练配置
            train_config = {
                'data': os.path.join(data_dir, 'data.yaml'),
                'epochs': num_epochs,
                'batch': batch_size,
                'imgsz': 640,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'workers': 4,
                'project': model_save_dir,
                'name': 'yolov8_train',
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'Adam',
                'lr0': learning_rate,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 0.05,
                'cls': 0.5,
                'cls_pw': 1.0,
                'obj': 1.0,
                'obj_pw': 1.0,
                'iou_t': 0.2,
                'anchor_t': 4.0,
                'fl_gamma': 0.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'val': True,
                'save': True,
                'save_json': False,
                'save_hybrid': False,
                'conf': 0.001,
                'iou': 0.6,
                'max_det': 300,
                'half': False,
                'dnn': False,
                'plots': True
            }
            
            # 开始训练
            results = model.train(**train_config)
            
            # 保存训练结果
            training_info = {
                'model_name': 'YOLOv8',
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'best_map': float(results.best_map),
                'model_path': os.path.join(model_save_dir, 'yolov8_train/weights/best.pt')
            }
            
            with open(os.path.join(model_save_dir, 'training_info.json'), 'w') as f:
                json.dump(training_info, f, indent=4)
            
            self.status_updated.emit(f'YOLOv8训练完成，最佳mAP: {results.best_map:.4f}')
            
        except Exception as e:
            self.training_error.emit(f"YOLOv8训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def _train_fasterrcnn(self, data_dir, num_epochs, batch_size, learning_rate, model_save_dir):
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
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                collate_fn=lambda x: tuple(zip(*x)))
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                collate_fn=lambda x: tuple(zip(*x)))
            
            # 创建模型
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            num_classes = 2  # 背景 + 目标类别
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
            # 将模型移到设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # 定义优化器
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(params, lr=learning_rate)
            
            # 创建TensorBoard写入器
            writer = SummaryWriter(os.path.join(model_save_dir, 'runs'))
            
            # 训练循环
            best_map = 0.0
            for epoch in range(num_epochs):
                if self.stop_training:
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
                    optimizer.step()
                    
                    epoch_loss += losses.item()
                    
                    # 更新进度
                    progress = int(((epoch * len(train_loader) + i + 1) /
                                 (num_epochs * len(train_loader))) * 100)
                    self.progress_updated.emit(progress)
                
                # 记录训练损失
                avg_loss = epoch_loss / len(train_loader)
                writer.add_scalar('Loss/train', avg_loss, epoch)
                
                # 验证阶段
                eval_metrics = evaluate_model(model, val_loader, device)
                current_map = eval_metrics['map']
                avg_val_loss = eval_metrics['loss']
                
                # 记录验证指标
                writer.add_scalar('Loss/val', avg_val_loss, epoch)
                writer.add_scalar('mAP/val', current_map, epoch)
                
                # 保存最佳模型
                if current_map > best_map:
                    best_map = current_map
                    model_save_path = os.path.join(model_save_dir, 'fasterrcnn_best.pth')
                    torch.save(model.state_dict(), model_save_path)
                
                # 发送epoch完成信号
                epoch_info = {
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    'val_loss': avg_val_loss,
                    'val_map': current_map
                }
                self.epoch_finished.emit(epoch_info)
            
            # 保存最终模型
            final_model_path = os.path.join(model_save_dir, 'fasterrcnn_final.pth')
            torch.save(model.state_dict(), final_model_path)
            
            # 保存训练信息
            training_info = {
                'model_name': 'Faster R-CNN',
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'best_map': best_map,
                'model_path': final_model_path,
                'best_model_path': os.path.join(model_save_dir, 'fasterrcnn_best.pth')
            }
            
            with open(os.path.join(model_save_dir, 'training_info.json'), 'w') as f:
                json.dump(training_info, f, indent=4)
            
            self.status_updated.emit(f'Faster R-CNN训练完成，最佳mAP: {best_map:.4f}')
            
        except Exception as e:
            self.training_error.emit(f"Faster R-CNN训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def _train_ssd(self, data_dir, num_epochs, batch_size, learning_rate, model_save_dir):
        """使用SSD进行目标检测训练"""
        try:
            self.status_updated.emit("开始SSD模型训练...")
            
            # 检查是否安装了torchvision
            try:
                from torchvision.models.detection import ssd300_vgg16
                from torchvision.models.detection.ssd import SSDLoss
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
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                collate_fn=lambda x: tuple(zip(*x)))
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                collate_fn=lambda x: tuple(zip(*x)))
            
            # 创建模型
            model = ssd300_vgg16(pretrained=True)
            num_classes = 2  # 背景 + 目标类别
            in_channels = model.backbone.out_channels
            num_anchors = model.anchor_generator.num_anchors_per_location()
            model.head = torchvision.models.detection.ssd.SSDHead(
                in_channels, num_anchors, num_classes)
            
            # 将模型移到设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # 定义损失函数和优化器
            criterion = SSDLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # 创建TensorBoard写入器
            writer = SummaryWriter(os.path.join(model_save_dir, 'runs'))
            
            # 训练循环
            best_map = 0.0
            for epoch in range(num_epochs):
                if self.stop_training:
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
                    optimizer.step()
                    
                    epoch_loss += losses.item()
                    
                    # 更新进度
                    progress = int(((epoch * len(train_loader) + i + 1) /
                                 (num_epochs * len(train_loader))) * 100)
                    self.progress_updated.emit(progress)
                
                # 记录训练损失
                avg_loss = epoch_loss / len(train_loader)
                writer.add_scalar('Loss/train', avg_loss, epoch)
                
                # 验证阶段
                eval_metrics = evaluate_model(model, val_loader, device)
                current_map = eval_metrics['map']
                avg_val_loss = eval_metrics['loss']
                
                # 记录验证指标
                writer.add_scalar('Loss/val', avg_val_loss, epoch)
                writer.add_scalar('mAP/val', current_map, epoch)
                
                # 保存最佳模型
                if current_map > best_map:
                    best_map = current_map
                    model_save_path = os.path.join(model_save_dir, 'ssd_best.pth')
                    torch.save(model.state_dict(), model_save_path)
                
                # 发送epoch完成信号
                epoch_info = {
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    'val_loss': avg_val_loss,
                    'val_map': current_map
                }
                self.epoch_finished.emit(epoch_info)
            
            # 保存最终模型
            final_model_path = os.path.join(model_save_dir, 'ssd_final.pth')
            torch.save(model.state_dict(), final_model_path)
            
            # 保存训练信息
            training_info = {
                'model_name': 'SSD',
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'best_map': best_map,
                'model_path': final_model_path,
                'best_model_path': os.path.join(model_save_dir, 'ssd_best.pth')
            }
            
            with open(os.path.join(model_save_dir, 'training_info.json'), 'w') as f:
                json.dump(training_info, f, indent=4)
            
            self.status_updated.emit(f'SSD训练完成，最佳mAP: {best_map:.4f}')
            
        except Exception as e:
            self.training_error.emit(f"SSD训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def _train_retinanet(self, data_dir, num_epochs, batch_size, learning_rate, model_save_dir):
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
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                collate_fn=lambda x: tuple(zip(*x)))
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                collate_fn=lambda x: tuple(zip(*x)))
            
            # 创建模型
            model = retinanet_resnet50_fpn(pretrained=True)
            num_classes = 2  # 背景 + 目标类别
            in_channels = model.backbone.out_channels
            num_anchors = model.anchor_generator.num_anchors_per_location()
            model.head = RetinaNetHead(in_channels, num_anchors, num_classes)
            
            # 将模型移到设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # 定义优化器
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(params, lr=learning_rate)
            
            # 创建TensorBoard写入器
            writer = SummaryWriter(os.path.join(model_save_dir, 'runs'))
            
            # 训练循环
            best_map = 0.0
            for epoch in range(num_epochs):
                if self.stop_training:
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
                    optimizer.step()
                    
                    epoch_loss += losses.item()
                    
                    # 更新进度
                    progress = int(((epoch * len(train_loader) + i + 1) /
                                 (num_epochs * len(train_loader))) * 100)
                    self.progress_updated.emit(progress)
                
                # 记录训练损失
                avg_loss = epoch_loss / len(train_loader)
                writer.add_scalar('Loss/train', avg_loss, epoch)
                
                # 验证阶段
                eval_metrics = evaluate_model(model, val_loader, device)
                current_map = eval_metrics['map']
                avg_val_loss = eval_metrics['loss']
                
                # 记录验证指标
                writer.add_scalar('Loss/val', avg_val_loss, epoch)
                writer.add_scalar('mAP/val', current_map, epoch)
                
                # 保存最佳模型
                if current_map > best_map:
                    best_map = current_map
                    model_save_path = os.path.join(model_save_dir, 'retinanet_best.pth')
                    torch.save(model.state_dict(), model_save_path)
                
                # 发送epoch完成信号
                epoch_info = {
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    'val_loss': avg_val_loss,
                    'val_map': current_map
                }
                self.epoch_finished.emit(epoch_info)
            
            # 保存最终模型
            final_model_path = os.path.join(model_save_dir, 'retinanet_final.pth')
            torch.save(model.state_dict(), final_model_path)
            
            # 保存训练信息
            training_info = {
                'model_name': 'RetinaNet',
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'best_map': best_map,
                'model_path': final_model_path,
                'best_model_path': os.path.join(model_save_dir, 'retinanet_best.pth')
            }
            
            with open(os.path.join(model_save_dir, 'training_info.json'), 'w') as f:
                json.dump(training_info, f, indent=4)
            
            self.status_updated.emit(f'RetinaNet训练完成，最佳mAP: {best_map:.4f}')
            
        except Exception as e:
            self.training_error.emit(f"RetinaNet训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def _train_detr(self, data_dir, num_epochs, batch_size, learning_rate, model_save_dir):
        """使用DETR进行目标检测训练"""
        try:
            self.status_updated.emit("开始DETR模型训练...")
            
            # 检查是否安装了torchvision
            try:
                from torchvision.models.detection import detr_resnet50
                from torchvision.models.detection import DETRHead
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
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                collate_fn=lambda x: tuple(zip(*x)))
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                collate_fn=lambda x: tuple(zip(*x)))
            
            # 创建模型
            model = detr_resnet50(pretrained=True)
            num_classes = 2  # 背景 + 目标类别
            model.num_classes = num_classes
            hidden_dim = model.transformer.d_model
            model.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            
            # 将模型移到设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # 定义优化器
            param_dicts = [
                {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": learning_rate * 0.1,
                },
            ]
            optimizer = torch.optim.AdamW(param_dicts, lr=learning_rate, weight_decay=0.0001)
            
            # 创建学习率调度器
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3)
            
            # 创建TensorBoard写入器
            writer = SummaryWriter(os.path.join(model_save_dir, 'runs'))
            
            # 训练循环
            best_map = 0.0
            for epoch in range(num_epochs):
                if self.stop_training:
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
                    if max(p.grad.data.abs().max() for p in model.parameters()) > 1:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    
                    epoch_loss += losses.item()
                    
                    # 更新进度
                    progress = int(((epoch * len(train_loader) + i + 1) /
                                 (num_epochs * len(train_loader))) * 100)
                    self.progress_updated.emit(progress)
                
                # 更新学习率
                lr_scheduler.step()
                
                # 记录训练损失
                avg_loss = epoch_loss / len(train_loader)
                writer.add_scalar('Loss/train', avg_loss, epoch)
                
                # 验证阶段
                eval_metrics = evaluate_model(model, val_loader, device)
                current_map = eval_metrics['map']
                avg_val_loss = eval_metrics['loss']
                
                # 记录验证指标
                writer.add_scalar('Loss/val', avg_val_loss, epoch)
                writer.add_scalar('mAP/val', current_map, epoch)
                
                # 保存最佳模型
                if current_map > best_map:
                    best_map = current_map
                    model_save_path = os.path.join(model_save_dir, 'detr_best.pth')
                    torch.save(model.state_dict(), model_save_path)
                
                # 发送epoch完成信号
                epoch_info = {
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    'val_loss': avg_val_loss,
                    'val_map': current_map,
                    'learning_rate': optimizer.param_groups[0]['lr']
                }
                self.epoch_finished.emit(epoch_info)
            
            # 保存最终模型
            final_model_path = os.path.join(model_save_dir, 'detr_final.pth')
            torch.save(model.state_dict(), final_model_path)
            
            # 保存训练信息
            training_info = {
                'model_name': 'DETR',
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'best_map': best_map,
                'model_path': final_model_path,
                'best_model_path': os.path.join(model_save_dir, 'detr_best.pth')
            }
            
            with open(os.path.join(model_save_dir, 'training_info.json'), 'w') as f:
                json.dump(training_info, f, indent=4)
            
            self.status_updated.emit(f'DETR训练完成，最佳mAP: {best_map:.4f}')
            
        except Exception as e:
            self.training_error.emit(f"DETR训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc() 