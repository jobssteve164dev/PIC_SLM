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

class TrainingThread(QThread):
    """负责在单独线程中执行训练过程的类"""
    
    # 定义与ModelTrainer相同的信号，用于在线程中发送训练状态
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    training_finished = pyqtSignal()
    training_error = pyqtSignal(str)
    epoch_finished = pyqtSignal(dict)
    # 添加预训练模型下载失败信号
    model_download_failed = pyqtSignal(str, str) # 模型名称，下载链接
    # 添加训练停止信号
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
        # 仅设置停止标志，不发送任何信号
        self.stop_training = True
        self.status_updated.emit("训练线程正在停止...")
        # 不发送训练停止信号，由ModelTrainer类统一管理
    
    def train_model(self, data_dir, model_name, num_epochs, batch_size, learning_rate, 
                   model_save_dir, task_type='classification', use_tensorboard=True):
        """与ModelTrainer.train_model相同的实现，但发送信号到主线程"""
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
                                'accuracy': float(current_acc.item()),
                                'batch': i + 1,
                                'total_batches': len(dataloaders[phase])
                            }
                            print(f"发送训练状态更新: {epoch_data}")  # 添加调试信息
                            self.epoch_finished.emit(epoch_data)

                    if self.stop_training:
                        break

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    # 发送每个epoch的结果
                    epoch_data = {
                        'epoch': epoch + 1,
                        'phase': phase,
                        'loss': float(epoch_loss),
                        'accuracy': float(epoch_acc.item()),
                        'batch': len(dataloaders[phase]),
                        'total_batches': len(dataloaders[phase])
                    }
                    print(f"发送epoch结果: {epoch_data}")  # 添加调试信息
                    self.epoch_finished.emit(epoch_data)
                    
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
                            except Exception as e:
                                print(f"记录样本图像时出错: {str(e)}")
                        
                        # 如果是验证阶段，记录混淆矩阵
                        if phase == 'val':
                            try:
                                from sklearn.metrics import confusion_matrix
                                import matplotlib.pyplot as plt
                                import io
                                from PIL import Image
                                
                                # 计算混淆矩阵
                                cm = confusion_matrix(all_labels, all_preds)
                                
                                # 绘制混淆矩阵
                                plt.figure(figsize=(10, 10))
                                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                                plt.title('Confusion matrix')
                                plt.colorbar()
                                tick_marks = np.arange(len(class_names))
                                plt.xticks(tick_marks, class_names, rotation=45)
                                plt.yticks(tick_marks, class_names)
                                
                                # 在格子中添加数值
                                thresh = cm.max() / 2.
                                for i in range(cm.shape[0]):
                                    for j in range(cm.shape[1]):
                                        plt.text(j, i, format(cm[i, j], 'd'),
                                                horizontalalignment="center",
                                                color="white" if cm[i, j] > thresh else "black")
                                
                                plt.tight_layout()
                                plt.ylabel('True label')
                                plt.xlabel('Predicted label')
                                
                                # 将图像转换为tensor
                                buf = io.BytesIO()
                                plt.savefig(buf, format='png')
                                buf.seek(0)
                                img = Image.open(buf)
                                img_tensor = transforms.ToTensor()(img)
                                
                                # 添加到TensorBoard
                                writer.add_image('Confusion Matrix', img_tensor, epoch)
                                
                                plt.close()
                            except Exception as e:
                                print(f"记录混淆矩阵时出错: {str(e)}")

                if self.stop_training:
                    break

                # 如果是验证阶段，检查是否为最佳模型
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    # 保存最佳模型
                    model_note = self.config.get('model_note', '')
                    model_file_suffix = f"_{model_note}" if model_note else ""
                    model_save_path = os.path.join(model_save_dir, f'{model_name}{model_file_suffix}_best.pth')
                    torch.save(self.model.state_dict(), model_save_path)
                    self.status_updated.emit(f'保存最佳模型，Epoch {epoch+1}, Acc: {best_acc:.4f}')
                    
                    # 导出ONNX模型
                    try:
                        onnx_save_path = os.path.join(model_save_dir, f'{model_name}{model_file_suffix}_best.onnx')
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
            model_file_suffix = f"_{model_note}" if model_note else ""
            final_model_path = os.path.join(model_save_dir, f'{model_name}{model_file_suffix}_final.pth')
            torch.save(self.model.state_dict(), final_model_path)
            self.status_updated.emit(f'保存最终模型: {final_model_path}')
            
            # 记录训练完成信息
            training_info = {
                'model_name': model_name,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'best_accuracy': best_acc.item() if isinstance(best_acc, torch.Tensor) else best_acc,
                'class_names': class_names,
                'model_path': final_model_path
            }
            
            with open(os.path.join(model_save_dir, 'training_info.json'), 'w') as f:
                json.dump(training_info, f, indent=4)
                
            self.status_updated.emit(f'训练完成，最佳准确率: {best_acc:.4f}')
            
            # 关闭TensorBoard写入器
            if writer:
                writer.close()
                
        except Exception as e:
            self.training_error.emit(f"训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _create_model(self, model_name, num_classes, task_type='classification'):
        """与ModelTrainer._create_model相同的实现"""
        try:
            if task_type == 'classification':
                try:
                    if model_name == 'ResNet18':
                        model = models.resnet18(pretrained=True)
                        model.fc = nn.Linear(model.fc.in_features, num_classes)
                        return model
                    elif model_name == 'ResNet34':
                        model = models.resnet34(pretrained=True)
                        model.fc = nn.Linear(model.fc.in_features, num_classes)
                        return model
                    elif model_name == 'ResNet50':
                        model = models.resnet50(pretrained=True)
                        model.fc = nn.Linear(model.fc.in_features, num_classes)
                        return model
                    elif model_name == 'ResNet101':
                        # 使用新的权重参数方式加载模型，避免警告和下载问题
                        try:
                            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        except Exception as e:
                            # 如果新API不可用，尝试旧的方式
                            self.status_updated.emit(f"使用新API加载模型失败: {str(e)}")
                            model = models.resnet101(pretrained=True)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                    elif model_name == 'ResNet152':
                        model = models.resnet152(pretrained=True)
                        model.fc = nn.Linear(model.fc.in_features, num_classes)
                        return model
                    elif model_name == 'MobileNetV2':
                        model = models.mobilenet_v2(pretrained=True)
                        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                        return model
                    elif model_name == 'MobileNetV3':
                        model = models.mobilenet_v3_large(pretrained=True)
                        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
                        return model
                    elif model_name == 'VGG16':
                        model = models.vgg16(pretrained=True)
                        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                        return model
                    elif model_name == 'VGG19':
                        model = models.vgg19(pretrained=True)
                        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                        return model
                    elif model_name == 'DenseNet121':
                        model = models.densenet121(pretrained=True)
                        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                        return model
                    elif model_name == 'DenseNet169':
                        model = models.densenet169(pretrained=True)
                        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                        return model
                    elif model_name == 'DenseNet201':
                        model = models.densenet201(pretrained=True)
                        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                        return model
                    elif model_name == 'InceptionV3':
                        model = models.inception_v3(pretrained=True, aux_logits=False)
                        model.fc = nn.Linear(model.fc.in_features, num_classes)
                        return model
                    elif model_name in ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4']:
                        # 检查是否安装了efficientnet_pytorch
                        try:
                            from efficientnet_pytorch import EfficientNet
                            
                            # 根据模型名称选择对应版本
                            if model_name == 'EfficientNetB0':
                                model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
                            elif model_name == 'EfficientNetB1':
                                model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
                            elif model_name == 'EfficientNetB2':
                                model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
                            elif model_name == 'EfficientNetB3':
                                model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
                            elif model_name == 'EfficientNetB4':
                                model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
                            
                            return model
                        except ImportError:
                            # 如果没有安装efficientnet_pytorch，提示用户安装
                            self.status_updated.emit(f"未安装EfficientNet库，尝试自动安装...")
                            try:
                                subprocess.check_call([sys.executable, "-m", "pip", "install", "efficientnet_pytorch"])
                                from efficientnet_pytorch import EfficientNet
                                
                                # 根据模型名称选择对应版本
                                if model_name == 'EfficientNetB0':
                                    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
                                elif model_name == 'EfficientNetB1':
                                    model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
                                elif model_name == 'EfficientNetB2':
                                    model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
                                elif model_name == 'EfficientNetB3':
                                    model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
                                elif model_name == 'EfficientNetB4':
                                    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
                                
                                return model
                            except Exception as install_err:
                                raise Exception(f"无法安装EfficientNet库: {str(install_err)}")
                    elif model_name == 'Xception':
                        try:
                            # 使用torchvision替代timm
                            model = models.resnet50(pretrained=True)  # 临时使用ResNet50作为替代
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            self.status_updated.emit("注意：Xception模型暂未实现，使用ResNet50替代")
                            return model
                        except Exception as e:
                            raise Exception(f"创建Xception模型失败: {str(e)}")
                    else:
                        raise ValueError(f"不支持的模型名称: {model_name}")
                except Exception as e:
                    # 处理模型下载失败的情况
                    import urllib.error
                    download_failed = False
                    
                    # 检查是否为下载失败的错误
                    if isinstance(e, urllib.error.URLError) or isinstance(e, ConnectionError) or isinstance(e, TimeoutError):
                        download_failed = True
                    elif any(keyword in str(e).lower() for keyword in ["download", "connection", "timeout", "url", "connect", "network", "internet"]):
                        download_failed = True
                    
                    if download_failed:
                        # 根据模型名称构建下载链接和说明
                        model_links = {
                            'ResNet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
                            'ResNet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
                            'ResNet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
                            'ResNet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
                            'ResNet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
                            'MobileNetV2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
                            'MobileNetV3Small': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
                            'MobileNetV3Large': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth',
                            'VGG16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
                            'VGG19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
                            'DenseNet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
                            'DenseNet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
                            'DenseNet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
                            'InceptionV3': 'https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth',
                            'EfficientNetB0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
                            'EfficientNetB1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
                            'EfficientNetB2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
                            'EfficientNetB3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
                            'EfficientNetB4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth'
                        }
                        
                        # 尝试查找更精确的模型名称匹配
                        exact_model_name = model_name
                        if model_name not in model_links:
                            for key in model_links.keys():
                                if model_name.lower() in key.lower() or key.lower() in model_name.lower():
                                    exact_model_name = key
                                    break
                        
                        model_link = model_links.get(exact_model_name, "")
                        if model_link:
                            self.model_download_failed.emit(exact_model_name, model_link)
                            self.status_updated.emit(f"预训练模型 {model_name} 下载失败，已提供手动下载链接")
                        else:
                            self.training_error.emit(f"预训练模型 {model_name} 下载失败，无法找到下载链接: {str(e)}")
                        
                        # 尝试使用不带预训练权重的模型来继续
                        self.status_updated.emit("尝试使用未预训练的模型继续训练...")
                        
                        # 根据模型名称创建不带预训练权重的模型
                        if model_name == 'ResNet101':
                            self.status_updated.emit("使用不带预训练权重的ResNet101模型")
                            try:
                                # 尝试新API
                                model = models.resnet101(weights=None)
                            except:
                                # 如果新API不可用，使用旧API
                                model = models.resnet101(pretrained=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        elif model_name == 'ResNet50':
                            model = models.resnet50(pretrained=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        elif model_name == 'ResNet18':
                            model = models.resnet18(pretrained=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        elif model_name == 'ResNet34':
                            model = models.resnet34(pretrained=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        elif model_name == 'ResNet152':
                            model = models.resnet152(pretrained=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        elif model_name == 'MobileNetV2':
                            model = models.mobilenet_v2(pretrained=False)
                            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                            return model
                        elif model_name == 'MobileNetV3':
                            model = models.mobilenet_v3_large(pretrained=False)
                            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
                            return model
                        elif model_name == 'VGG16':
                            model = models.vgg16(pretrained=False)
                            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                            return model
                        elif model_name == 'VGG19':
                            model = models.vgg19(pretrained=False)
                            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                            return model
                        elif model_name == 'DenseNet121':
                            model = models.densenet121(pretrained=False)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                        elif model_name == 'DenseNet169':
                            model = models.densenet169(pretrained=False)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                        elif model_name == 'DenseNet201':
                            model = models.densenet201(pretrained=False)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                        elif model_name == 'InceptionV3':
                            model = models.inception_v3(pretrained=False, aux_logits=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        elif model_name in ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4']:
                            try:
                                from efficientnet_pytorch import EfficientNet
                                # 创建一个不带预训练权重的模型
                                if model_name == 'EfficientNetB0':
                                    model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
                                elif model_name == 'EfficientNetB1':
                                    model = EfficientNet.from_name('efficientnet-b1', num_classes=num_classes)
                                elif model_name == 'EfficientNetB2':
                                    model = EfficientNet.from_name('efficientnet-b2', num_classes=num_classes)
                                elif model_name == 'EfficientNetB3':
                                    model = EfficientNet.from_name('efficientnet-b3', num_classes=num_classes)
                                elif model_name == 'EfficientNetB4':
                                    model = EfficientNet.from_name('efficientnet-b4', num_classes=num_classes)
                                return model
                            except ImportError:
                                # 如果没有安装，尝试安装并再次创建
                                try:
                                    subprocess.check_call([sys.executable, "-m", "pip", "install", "efficientnet_pytorch"])
                                    from efficientnet_pytorch import EfficientNet
                                    if model_name == 'EfficientNetB0':
                                        model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
                                    elif model_name == 'EfficientNetB1':
                                        model = EfficientNet.from_name('efficientnet-b1', num_classes=num_classes)
                                    elif model_name == 'EfficientNetB2':
                                        model = EfficientNet.from_name('efficientnet-b2', num_classes=num_classes)
                                    elif model_name == 'EfficientNetB3':
                                        model = EfficientNet.from_name('efficientnet-b3', num_classes=num_classes)
                                    elif model_name == 'EfficientNetB4':
                                        model = EfficientNet.from_name('efficientnet-b4', num_classes=num_classes)
                                    return model
                                except Exception:
                                    # 如果安装失败，回退到ResNet50作为替代
                                    self.status_updated.emit("无法安装或创建EfficientNet模型，使用ResNet50替代")
                                    model = models.resnet50(pretrained=False)
                                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                                    return model
                        else:
                            # 对于所有其他情况，使用ResNet50作为后备方案
                            self.status_updated.emit(f"未知模型 {model_name}，使用ResNet50替代")
                            model = models.resnet50(pretrained=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                    
                    # 重新抛出其他类型的错误
                    raise
                    
            elif task_type == 'detection':
                # 目标检测模型
                # 这里只是一个示例框架，实际应用中需要实现具体的目标检测模型
                self.status_updated.emit(f"目标检测模型 {model_name} 尚未实现")
                
                # 使用一个简单的占位模型
                model = models.resnet50(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                return model
                
            else:
                raise ValueError(f"不支持的任务类型: {task_type}")
        except Exception as e:
            # 确保即使发生错误也返回一个有效的模型
            self.status_updated.emit(f"创建模型时出错: {str(e)}，使用默认的ResNet50模型")
            try:
                model = models.resnet50(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                return model
            except:
                # 如果连ResNet50都创建失败，尝试创建最简单的模型
                self.status_updated.emit("创建ResNet50模型失败，使用简单自定义模型")
                model = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, num_classes)
                )
                return model

class ModelTrainer(QObject):
    # 定义信号
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    training_finished = pyqtSignal()
    training_error = pyqtSignal(str)
    epoch_finished = pyqtSignal(dict)
    # 添加预训练模型下载失败信号
    model_download_failed = pyqtSignal(str, str)  # 模型名称，下载链接
    # 添加训练停止信号
    training_stopped = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.stop_training = False
        self.training_thread = None

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
                            self.epoch_finished.emit({
                                'epoch': epoch + 1,
                                'phase': phase,
                                'loss': current_loss,
                                'accuracy': current_acc.item(),
                                'batch': i + 1,
                                'total_batches': len(dataloaders[phase])
                            })

                    if self.stop_training:
                        break

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    # 发送每个epoch的结果
                    self.epoch_finished.emit({
                        'epoch': epoch + 1,
                        'phase': phase,
                        'loss': epoch_loss,
                        'accuracy': epoch_acc.item(),
                        'batch': len(dataloaders[phase]),
                        'total_batches': len(dataloaders[phase])
                    })
                    
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
                            except Exception as e:
                                print(f"记录样本图像时出错: {str(e)}")
                        
                        # 如果是验证阶段，记录混淆矩阵
                        if phase == 'val':
                            try:
                                from sklearn.metrics import confusion_matrix
                                import matplotlib.pyplot as plt
                                import io
                                from PIL import Image
                                
                                # 计算混淆矩阵
                                cm = confusion_matrix(all_labels, all_preds)
                                
                                # 绘制混淆矩阵
                                plt.figure(figsize=(10, 10))
                                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                                plt.title('Confusion matrix')
                                plt.colorbar()
                                tick_marks = np.arange(len(class_names))
                                plt.xticks(tick_marks, class_names, rotation=45)
                                plt.yticks(tick_marks, class_names)
                                
                                # 在格子中添加数值
                                thresh = cm.max() / 2.
                                for i in range(cm.shape[0]):
                                    for j in range(cm.shape[1]):
                                        plt.text(j, i, format(cm[i, j], 'd'),
                                                horizontalalignment="center",
                                                color="white" if cm[i, j] > thresh else "black")
                                
                                plt.tight_layout()
                                plt.ylabel('True label')
                                plt.xlabel('Predicted label')
                                
                                # 将图像转换为tensor
                                buf = io.BytesIO()
                                plt.savefig(buf, format='png')
                                buf.seek(0)
                                img = Image.open(buf)
                                img_tensor = transforms.ToTensor()(img)
                                
                                # 添加到TensorBoard
                                writer.add_image('Confusion Matrix', img_tensor, epoch)
                                
                                plt.close()
                            except Exception as e:
                                print(f"记录混淆矩阵时出错: {str(e)}")

                if self.stop_training:
                    break

                # 如果是验证阶段，检查是否为最佳模型
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    # 保存最佳模型
                    model_note = self.config.get('model_note', '')
                    model_file_suffix = f"_{model_note}" if model_note else ""
                    model_save_path = os.path.join(model_save_dir, f'{model_name}{model_file_suffix}_best.pth')
                    torch.save(self.model.state_dict(), model_save_path)
                    self.status_updated.emit(f'保存最佳模型，Epoch {epoch+1}, Acc: {best_acc:.4f}')
                    
                    # 导出ONNX模型
                    try:
                        onnx_save_path = os.path.join(model_save_dir, f'{model_name}{model_file_suffix}_best.onnx')
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
            model_file_suffix = f"_{model_note}" if model_note else ""
            final_model_path = os.path.join(model_save_dir, f'{model_name}{model_file_suffix}_final.pth')
            torch.save(self.model.state_dict(), final_model_path)
            self.status_updated.emit(f'保存最终模型: {final_model_path}')
            
            # 记录训练完成信息
            training_info = {
                'model_name': model_name,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'best_accuracy': best_acc.item() if isinstance(best_acc, torch.Tensor) else best_acc,
                'class_names': class_names,
                'model_path': final_model_path
            }
            
            with open(os.path.join(model_save_dir, 'training_info.json'), 'w') as f:
                json.dump(training_info, f, indent=4)
                
            self.status_updated.emit(f'训练完成，最佳准确率: {best_acc:.4f}')
            
            # 关闭TensorBoard写入器
            if writer:
                writer.close()
                
        except Exception as e:
            self.training_error.emit(f"训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def stop(self):
        """停止训练过程"""
        # 设置停止标志
        self.stop_training = True
        self.status_updated.emit("正在停止训练...")
        
        # 如果有训练线程在运行
        if self.training_thread and self.training_thread.isRunning():
            # 设置训练线程的停止标志
            if hasattr(self.training_thread, 'stop_training'):
                self.training_thread.stop_training = True
            
            # 调用训练线程的stop方法
            if hasattr(self.training_thread, 'stop'):
                # 断开所有训练线程的信号连接，避免产生竞争条件
                try:
                    self.training_thread.progress_updated.disconnect()
                    self.training_thread.status_updated.disconnect()
                    self.training_thread.training_finished.disconnect()
                    self.training_thread.training_error.disconnect()
                    self.training_thread.epoch_finished.disconnect()
                    self.training_thread.model_download_failed.disconnect()
                    if hasattr(self.training_thread, 'training_stopped'):
                        self.training_thread.training_stopped.disconnect()
                except (TypeError, RuntimeError):
                    # 忽略信号断开可能产生的错误
                    pass
                
                # 优雅地停止线程
                self.training_thread.stop()
            
            # 等待线程结束，最多等待3秒
            if not self.training_thread.wait(3000):
                # 如果3秒后线程还在运行，记录警告但不强制终止
                # 避免使用terminate()方法，因为它可能导致资源泄漏和程序崩溃
                self.status_updated.emit("警告：训练线程未能及时停止，但训练已被标记为停止状态")
            else:
                self.status_updated.emit("训练线程已停止")
        
        # 无论线程是否正常结束，都发射一次训练停止信号
        self.training_stopped.emit()

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
            elif model_name == 'ResNet34':
                model = models.resnet34(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_name == 'ResNet50':
                model = models.resnet50(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                return model
            elif model_name == 'ResNet101':
                # 使用新的权重参数方式加载模型，避免警告和下载问题
                try:
                    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                    return model
                except Exception as e:
                    # 如果新API不可用，尝试旧的方式
                    self.status_updated.emit(f"使用新API加载模型失败: {str(e)}")
                    model = models.resnet101(pretrained=True)
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                    return model
            elif model_name == 'ResNet152':
                model = models.resnet152(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                return model
            elif model_name == 'MobileNetV2':
                model = models.mobilenet_v2(pretrained=True)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                return model
            elif model_name == 'MobileNetV3':
                model = models.mobilenet_v3_large(pretrained=True)
                model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
                return model
            elif model_name == 'VGG16':
                model = models.vgg16(pretrained=True)
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                return model
            elif model_name == 'VGG19':
                model = models.vgg19(pretrained=True)
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                return model
            elif model_name == 'DenseNet121':
                model = models.densenet121(pretrained=True)
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                return model
            elif model_name == 'DenseNet169':
                model = models.densenet169(pretrained=True)
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                return model
            elif model_name == 'DenseNet201':
                model = models.densenet201(pretrained=True)
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                return model
            elif model_name == 'InceptionV3':
                model = models.inception_v3(pretrained=True, aux_logits=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                return model
            elif model_name in ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4']:
                # 检查是否安装了efficientnet_pytorch
                try:
                    from efficientnet_pytorch import EfficientNet
                    
                    # 根据模型名称选择对应版本
                    if model_name == 'EfficientNetB0':
                        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
                    elif model_name == 'EfficientNetB1':
                        model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
                    elif model_name == 'EfficientNetB2':
                        model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
                    elif model_name == 'EfficientNetB3':
                        model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
                    elif model_name == 'EfficientNetB4':
                        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
                    
                    return model
                except ImportError:
                    # 如果没有安装efficientnet_pytorch，提示用户安装
                    self.status_updated.emit(f"未安装EfficientNet库，尝试自动安装...")
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "efficientnet_pytorch"])
                        from efficientnet_pytorch import EfficientNet
                        
                        # 根据模型名称选择对应版本
                        if model_name == 'EfficientNetB0':
                            model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
                        elif model_name == 'EfficientNetB1':
                            model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
                        elif model_name == 'EfficientNetB2':
                            model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
                        elif model_name == 'EfficientNetB3':
                            model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
                        elif model_name == 'EfficientNetB4':
                            model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
                        
                        return model
                    except Exception as install_err:
                        raise Exception(f"无法安装EfficientNet库: {str(install_err)}")
            elif model_name == 'Xception':
                try:
                    # 使用torchvision替代timm
                    model = models.resnet50(pretrained=True)  # 临时使用ResNet50作为替代
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                    self.status_updated.emit("注意：Xception模型暂未实现，使用ResNet50替代")
                    return model
                except Exception as e:
                    raise Exception(f"创建Xception模型失败: {str(e)}")
            else:
                raise ValueError(f"不支持的模型名称: {model_name}")
        elif task_type == 'detection':
            # 目标检测模型
            # 这里只是一个示例框架，实际应用中需要实现具体的目标检测模型
            self.status_updated.emit(f"目标检测模型 {model_name} 尚未实现")
            
            # 使用一个简单的占位模型
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
            
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")

    def train_model_with_config(self, config: Dict[str, Any]) -> None:
        """
        使用配置字典启动训练线程
        
        参数:
            config: 训练配置字典，包含所有训练参数
        """
        try:
            # 如果有正在运行的训练线程，先停止它
            if self.training_thread and self.training_thread.isRunning():
                self.stop()
            
            # 创建新的训练线程
            self.training_thread = TrainingThread(config)
            
            # 连接信号
            self.training_thread.progress_updated.connect(self.progress_updated)
            self.training_thread.status_updated.connect(self.status_updated)
            self.training_thread.training_finished.connect(self.training_finished)
            self.training_thread.training_error.connect(self.training_error)
            self.training_thread.epoch_finished.connect(self.epoch_finished)
            self.training_thread.model_download_failed.connect(self.model_download_failed)
            
            # 打印训练配置
            self.status_updated.emit(f"开始训练 {config.get('model_name', 'unknown')} 模型，任务类型: {config.get('task_type', 'classification')}")
            self.status_updated.emit(f"批次大小: {config.get('batch_size', 32)}, 学习率: {config.get('learning_rate', 0.001)}, 训练轮数: {config.get('num_epochs', 20)}")
            
            # 启动训练线程
            self.training_thread.start()
            
        except Exception as e:
            self.training_error.emit(f"启动训练线程失败: {str(e)}")
            import traceback
            traceback.print_exc() 