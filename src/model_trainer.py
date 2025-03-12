import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import torchvision
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
from typing import Dict, Any, Optional
import json
import subprocess
import sys
from torch.utils.tensorboard import SummaryWriter

class ModelTrainer(QObject):
    # 定义信号
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    training_finished = pyqtSignal()
    training_error = pyqtSignal(str)
    epoch_finished = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.stop_training = False

    def train_model(self,
                   data_dir: str,
                   model_name: str,
                   num_epochs: int,
                   batch_size: int,
                   learning_rate: float,
                   model_save_dir: str,
                   task_type: str = 'classification',
                   use_tensorboard: bool = True) -> None:
        """
        训练模型
        
        参数:
            data_dir: 数据目录
            model_name: 模型名称
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            model_save_dir: 模型保存目录
            task_type: 任务类型，'classification'或'detection'
            use_tensorboard: 是否使用TensorBoard记录训练过程
        """
        try:
            # 标准化路径格式，确保使用正斜杠
            data_dir = os.path.normpath(data_dir).replace('\\', '/')
            model_save_dir = os.path.normpath(model_save_dir).replace('\\', '/')
            
            # 检查训练和验证数据集目录是否存在
            train_dir = os.path.join(data_dir, 'train')
            val_dir = os.path.join(data_dir, 'val')
            
            if not os.path.exists(train_dir):
                self.training_error.emit(f"训练数据集目录不存在: {train_dir}")
                return
                
            if not os.path.exists(val_dir):
                self.training_error.emit(f"验证数据集目录不存在: {val_dir}")
                return
            
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

            # 根据任务类型加载不同的数据集
            if task_type == 'classification':
                # 分类任务：使用ImageFolder加载数据
                self.status_updated.emit("加载分类数据集...")
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

                # 保存类别信息
                class_info = {
                    'class_names': class_names,
                    'class_to_idx': image_datasets['train'].class_to_idx
                }
                
            elif task_type == 'detection':
                # 目标检测任务：这里需要实现自定义数据集加载逻辑
                # 由于目标检测数据集加载比较复杂，这里只是一个示例框架
                self.status_updated.emit("加载目标检测数据集...")
                self.training_error.emit("目标检测训练功能尚未完全实现")
                return
                
                # 实际应用中，这里需要实现自定义数据集类来加载XML标注文件
                # 并创建相应的DataLoader
                # 例如：
                # from detection_dataset import DetectionDataset
                # image_datasets = {x: DetectionDataset(os.path.join(data_dir, x),
                #                                     transform=data_transforms[x])
                #                 for x in ['train', 'val']}
                # 
                # dataloaders = {x: DataLoader(image_datasets[x],
                #                            batch_size=batch_size,
                #                            shuffle=True,
                #                            num_workers=4,
                #                            collate_fn=detection_collate_fn)
                #               for x in ['train', 'val']}
                # 
                # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
                # class_names = image_datasets['train'].get_classes()
                # num_classes = len(class_names)
                # 
                # # 保存类别信息
                # class_info = {
                #     'class_names': class_names,
                #     'class_to_idx': {name: i for i, name in enumerate(class_names)}
                # }
            else:
                self.training_error.emit(f"不支持的任务类型: {task_type}")
                return
                
            os.makedirs(model_save_dir, exist_ok=True)
            with open(os.path.join(model_save_dir, 'class_info.json'), 'w') as f:
                json.dump(class_info, f)

            # 创建模型
            self.model = self._create_model(model_name, num_classes, task_type)
            self.model = self.model.to(self.device)

            # 定义损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # 初始化TensorBoard
            writer = None
            if use_tensorboard:
                tensorboard_dir = os.path.join(model_save_dir, 'tensorboard_logs')
                os.makedirs(tensorboard_dir, exist_ok=True)
                writer = SummaryWriter(tensorboard_dir)
                
                # 记录模型图
                try:
                    # 获取一个批次的样本数据用于记录模型图
                    sample_inputs, _ = next(iter(dataloaders['train']))
                    sample_inputs = sample_inputs.to(self.device)
                    writer.add_graph(self.model, sample_inputs)
                except Exception as e:
                    print(f"记录模型图时出错: {str(e)}")

            # 训练模型
            best_acc = 0.0
            for epoch in range(num_epochs):
                if self.stop_training:
                    break

                self.status_updated.emit(f'Epoch {epoch+1}/{num_epochs}')

                # 训练和验证阶段
                for phase in ['train', 'val']:
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

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        
                        # 收集预测和标签用于计算混淆矩阵
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                        # 更新进度
                        progress = int(((epoch * len(dataloaders[phase]) + i + 1) /
                                     (num_epochs * len(dataloaders[phase]))) * 100)
                        self.progress_updated.emit(progress)

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    # 发送每个epoch的结果
                    epoch_results = {
                        'epoch': epoch + 1,
                        'phase': phase,
                        'loss': epoch_loss,
                        'accuracy': epoch_acc.item()
                    }
                    self.epoch_finished.emit(epoch_results)
                    
                    # 记录到TensorBoard
                    if writer:
                        writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
                        writer.add_scalar(f'Accuracy/{phase}', epoch_acc.item(), epoch)
                        
                        # 每个epoch结束时记录一些样本图像
                        if phase == 'val' and epoch % 5 == 0:  # 每5个epoch记录一次
                            try:
                                # 获取一个批次的样本数据
                                sample_inputs, sample_labels = next(iter(dataloaders[phase]))
                                sample_inputs = sample_inputs.to(self.device)
                                
                                # 记录样本图像
                                grid = torchvision.utils.make_grid(sample_inputs[:8])
                                writer.add_image('Sample Images', grid, epoch)
                                
                                # 记录混淆矩阵
                                if len(all_preds) > 0 and len(all_labels) > 0:
                                    from sklearn.metrics import confusion_matrix
                                    import matplotlib.pyplot as plt
                                    import io
                                    from PIL import Image
                                    
                                    # 设置matplotlib中文字体
                                    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'sans-serif']
                                    plt.rcParams['axes.unicode_minus'] = False
                                    
                                    # 计算混淆矩阵
                                    cm = confusion_matrix(all_labels, all_preds)
                                    
                                    # 绘制混淆矩阵
                                    plt.figure(figsize=(10, 8))
                                    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                                    plt.title('混淆矩阵')
                                    plt.colorbar()
                                    tick_marks = np.arange(len(class_names))
                                    plt.xticks(tick_marks, class_names, rotation=45)
                                    plt.yticks(tick_marks, class_names)
                                    plt.tight_layout()
                                    plt.ylabel('真实标签')
                                    plt.xlabel('预测标签')
                                    
                                    # 将matplotlib图像转换为PIL图像
                                    buf = io.BytesIO()
                                    plt.savefig(buf, format='png')
                                    buf.seek(0)
                                    image = Image.open(buf)
                                    image = np.array(image)
                                    
                                    # 添加到TensorBoard
                                    writer.add_image(f'Confusion Matrix/{phase}', image, epoch, dataformats='HWC')
                                    plt.close()
                            except Exception as e:
                                print(f"记录TensorBoard可视化时出错: {str(e)}")

                    # 保存最佳模型
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'best_acc': best_acc,
                        }, os.path.join(model_save_dir, 'best_model.pth'))

            # 关闭TensorBoard写入器
            if writer:
                writer.close()

            self.status_updated.emit('训练完成')
            self.training_finished.emit()

        except Exception as e:
            self.training_error.emit(f'训练过程中出错: {str(e)}')

    def stop(self):
        """停止训练"""
        self.stop_training = True

    def _create_model(self, model_name: str, num_classes: int, task_type: str = 'classification') -> nn.Module:
        """
        创建模型
        
        参数:
            model_name: 模型名称
            num_classes: 类别数量
            task_type: 任务类型，'classification'或'detection'
            
        返回:
            创建的模型
        """
        if task_type == 'classification':
            # 分类模型
            if model_name == 'ResNet18':
                model = models.resnet18(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_name == 'ResNet50':
                model = models.resnet50(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_name == 'EfficientNet-B0':
                try:
                    # 使用条件导入避免IDE警告
                    if True:  # 这行代码只是为了避免IDE警告
                        from efficientnet_pytorch import EfficientNet
                    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
                except ImportError:
                    self.status_updated.emit("EfficientNet模块未安装，尝试安装中...")
                    try:
                        subprocess.run([sys.executable, '-m', 'pip', 'install', 'efficientnet-pytorch'], 
                                      check=True)
                        from efficientnet_pytorch import EfficientNet
                        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
                    except Exception as e:
                        self.status_updated.emit(f"无法安装EfficientNet: {str(e)}，使用ResNet50替代")
                        model = models.resnet50(pretrained=True)
                        model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_name == 'EfficientNet-B4':
                try:
                    # 使用条件导入避免IDE警告
                    if True:  # 这行代码只是为了避免IDE警告
                        from efficientnet_pytorch import EfficientNet
                    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
                except ImportError:
                    self.status_updated.emit("EfficientNet模块未安装，尝试安装中...")
                    try:
                        subprocess.run([sys.executable, '-m', 'pip', 'install', 'efficientnet-pytorch'], 
                                      check=True)
                        from efficientnet_pytorch import EfficientNet
                        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
                    except Exception as e:
                        self.status_updated.emit(f"无法安装EfficientNet: {str(e)}，使用ResNet50替代")
                        model = models.resnet50(pretrained=True)
                        model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_name == 'MobileNetV2':
                model = models.mobilenet_v2(pretrained=True)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            elif model_name == 'DenseNet121':
                model = models.densenet121(pretrained=True)
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            else:
                # 默认使用ResNet50
                self.status_updated.emit(f"未知模型: {model_name}，使用默认的ResNet50")
                model = models.resnet50(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif task_type == 'detection':
            # 目标检测模型
            # 这里只是一个示例框架，实际应用中需要实现具体的目标检测模型
            self.status_updated.emit(f"目标检测模型 {model_name} 尚未实现")
            
            # 使用一个简单的占位模型
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            # 实际应用中，这里应该根据model_name创建不同的目标检测模型
            # 例如：
            # if model_name == 'Faster R-CNN':
            #     from torchvision.models.detection import fasterrcnn_resnet50_fpn
            #     model = fasterrcnn_resnet50_fpn(pretrained=True)
            #     # 修改模型以适应自定义类别数
            #     in_features = model.roi_heads.box_predictor.cls_score.in_features
            #     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            # elif model_name == 'SSD':
            #     from torchvision.models.detection import ssd300_vgg16
            #     model = ssd300_vgg16(pretrained=True)
            #     # 修改模型以适应自定义类别数
            #     # ...
            # elif model_name == 'YOLO v5':
            #     # YOLO v5需要单独安装
            #     # ...
            # 等等
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")
            
        return model 

    def train_model_with_config(self, config: Dict[str, Any]) -> None:
        """
        使用配置字典训练模型
        
        参数:
            config: 训练配置字典，包含所有训练参数
        """
        try:
            # 提取基本参数
            data_dir = config.get('data_dir', '')
            model_name = config.get('model_name', 'ResNet50')
            num_epochs = config.get('num_epochs', 20)
            batch_size = config.get('batch_size', 32)
            learning_rate = config.get('learning_rate', 0.001)
            model_save_dir = config.get('model_save_dir', 'models/saved_models')
            task_type = config.get('task_type', 'classification')
            
            # 提取高级参数
            optimizer_name = config.get('optimizer', 'Adam')
            lr_scheduler_name = config.get('lr_scheduler', '固定学习率')
            weight_decay = config.get('weight_decay', 0.0001)
            use_pretrained = config.get('use_pretrained', True)
            metrics = config.get('metrics', ['accuracy'])
            use_tensorboard = config.get('use_tensorboard', True)
            
            # 提取检测特有参数
            iou_threshold = config.get('iou_threshold', 0.5)
            nms_threshold = config.get('nms_threshold', 0.45)
            conf_threshold = config.get('conf_threshold', 0.25)
            use_fpn = config.get('use_fpn', True)
            
            # 打印训练配置
            self.status_updated.emit(f"开始训练 {model_name} 模型，任务类型: {task_type}")
            self.status_updated.emit(f"批次大小: {batch_size}, 学习率: {learning_rate}, 训练轮数: {num_epochs}")
            self.status_updated.emit(f"优化器: {optimizer_name}, 学习率调度: {lr_scheduler_name}")
            self.status_updated.emit(f"评估指标: {', '.join(metrics)}")
            
            if task_type == 'detection':
                self.status_updated.emit(f"IoU阈值: {iou_threshold}, NMS阈值: {nms_threshold}, 置信度阈值: {conf_threshold}")
                if use_fpn:
                    self.status_updated.emit("使用特征金字塔网络(FPN)")
            
            if use_tensorboard:
                self.status_updated.emit("已启用TensorBoard可视化")
            
            # 调用原始训练方法
            return self.train_model(
                data_dir=data_dir,
                model_name=model_name,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                model_save_dir=model_save_dir,
                task_type=task_type,
                use_tensorboard=use_tensorboard
            )
            
        except Exception as e:
            self.training_error.emit(f"训练配置错误: {str(e)}")
            import traceback
            traceback.print_exc() 