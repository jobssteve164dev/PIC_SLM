import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
from typing import Dict, Any, Optional
import json

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
                   model_save_dir: str) -> None:
        """
        训练模型
        """
        try:
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

            # 保存类别信息
            class_info = {
                'class_names': class_names,
                'class_to_idx': image_datasets['train'].class_to_idx
            }
            os.makedirs(model_save_dir, exist_ok=True)
            with open(os.path.join(model_save_dir, 'class_info.json'), 'w') as f:
                json.dump(class_info, f)

            # 创建模型
            self.model = self._create_model(model_name, num_classes)
            self.model = self.model.to(self.device)

            # 定义损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

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

                    # 保存最佳模型
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'best_acc': best_acc,
                        }, os.path.join(model_save_dir, 'best_model.pth'))

            self.status_updated.emit('训练完成')
            self.training_finished.emit()

        except Exception as e:
            self.training_error.emit(f'训练过程中出错: {str(e)}')

    def stop(self):
        """停止训练"""
        self.stop_training = True

    def _create_model(self, model_name: str, num_classes: int) -> nn.Module:
        """创建模型"""
        if model_name == 'ResNet18':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'ResNet50':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'EfficientNet-B0':
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'EfficientNet-B4':
            model = models.efficientnet_b4(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f'不支持的模型: {model_name}')
        
        return model 