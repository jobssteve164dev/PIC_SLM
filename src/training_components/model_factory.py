"""
模型工厂类 - 负责创建各种深度学习模型

支持的模型：
- ResNet系列（18, 34, 50, 101, 152）
- MobileNet系列（V2, V3）
- VGG系列（16, 19）
- DenseNet系列（121, 169, 201）
- InceptionV3
- EfficientNet系列（B0-B4）
"""

import torch
import torch.nn as nn
from torchvision import models
import subprocess
import sys
from PyQt5.QtCore import QObject, pyqtSignal


class ModelFactory(QObject):
    """模型创建工厂类"""
    
    # 信号定义
    status_updated = pyqtSignal(str)
    model_download_failed = pyqtSignal(str, str)  # 模型名称，下载链接
    
    def __init__(self):
        super().__init__()
        self.model_links = {
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
    
    def create_model(self, model_name, num_classes, task_type='classification'):
        """
        创建指定的模型
        
        Args:
            model_name: 模型名称
            num_classes: 类别数量
            task_type: 任务类型，默认为分类
            
        Returns:
            创建的PyTorch模型
        """
        if task_type != 'classification':
            raise ValueError(f"当前仅支持分类任务，不支持: {task_type}")
            
        try:
            return self._create_classification_model(model_name, num_classes)
        except Exception as e:
            return self._handle_model_creation_error(e, model_name, num_classes)
    
    def _create_classification_model(self, model_name, num_classes):
        """创建分类模型"""
        # ResNet系列
        if model_name == 'ResNet18':
            return self._create_resnet_model('resnet18', num_classes, models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == 'ResNet34':
            return self._create_resnet_model('resnet34', num_classes, models.ResNet34_Weights.IMAGENET1K_V1)
        elif model_name == 'ResNet50':
            return self._create_resnet_model('resnet50', num_classes, models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == 'ResNet101':
            return self._create_resnet_model('resnet101', num_classes, models.ResNet101_Weights.IMAGENET1K_V1)
        elif model_name == 'ResNet152':
            return self._create_resnet_model('resnet152', num_classes, models.ResNet152_Weights.IMAGENET1K_V1)
        
        # MobileNet系列
        elif model_name == 'MobileNetV2':
            return self._create_mobilenet_v2_model(num_classes)
        elif model_name == 'MobileNetV3':
            return self._create_mobilenet_v3_model(num_classes)
        
        # VGG系列
        elif model_name == 'VGG16':
            return self._create_vgg_model('vgg16', num_classes, models.VGG16_Weights.IMAGENET1K_V1)
        elif model_name == 'VGG19':
            return self._create_vgg_model('vgg19', num_classes, models.VGG19_Weights.IMAGENET1K_V1)
        
        # DenseNet系列
        elif model_name == 'DenseNet121':
            return self._create_densenet_model('densenet121', num_classes, models.DenseNet121_Weights.IMAGENET1K_V1)
        elif model_name == 'DenseNet169':
            return self._create_densenet_model('densenet169', num_classes, models.DenseNet169_Weights.IMAGENET1K_V1)
        elif model_name == 'DenseNet201':
            return self._create_densenet_model('densenet201', num_classes, models.DenseNet201_Weights.IMAGENET1K_V1)
        
        # InceptionV3
        elif model_name == 'InceptionV3':
            return self._create_inception_v3_model(num_classes)
        
        # EfficientNet系列
        elif model_name in ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4']:
            return self._create_efficientnet_model(model_name, num_classes)
        
        # Xception（暂时使用ResNet50替代）
        elif model_name == 'Xception':
            self.status_updated.emit("注意：Xception模型暂未实现，使用ResNet50替代")
            return self._create_resnet_model('resnet50', num_classes, models.ResNet50_Weights.IMAGENET1K_V1)
        
        else:
            raise ValueError(f"不支持的模型名称: {model_name}")
    
    def _create_resnet_model(self, model_type, num_classes, weights):
        """创建ResNet系列模型"""
        try:
            model_func = getattr(models, model_type)
            model = model_func(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        except Exception as e:
            self.status_updated.emit(f"使用新API加载{model_type}模型失败: {str(e)}")
            # 降级到旧API
            model = model_func(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
    
    def _create_mobilenet_v2_model(self, num_classes):
        """创建MobileNetV2模型"""
        try:
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            return model
        except Exception as e:
            self.status_updated.emit(f"使用新API加载MobileNetV2模型失败: {str(e)}")
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            return model
    
    def _create_mobilenet_v3_model(self, num_classes):
        """创建MobileNetV3模型"""
        try:
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
            return model
        except Exception as e:
            self.status_updated.emit(f"使用新API加载MobileNetV3模型失败: {str(e)}")
            model = models.mobilenet_v3_large(pretrained=True)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
            return model
    
    def _create_vgg_model(self, model_type, num_classes, weights):
        """创建VGG系列模型"""
        try:
            model_func = getattr(models, model_type)
            model = model_func(weights=weights)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            return model
        except Exception as e:
            self.status_updated.emit(f"使用新API加载{model_type}模型失败: {str(e)}")
            model = model_func(pretrained=True)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            return model
    
    def _create_densenet_model(self, model_type, num_classes, weights):
        """创建DenseNet系列模型"""
        try:
            model_func = getattr(models, model_type)
            model = model_func(weights=weights)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            return model
        except Exception as e:
            self.status_updated.emit(f"使用新API加载{model_type}模型失败: {str(e)}")
            model = model_func(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            return model
    
    def _create_inception_v3_model(self, num_classes):
        """创建InceptionV3模型"""
        try:
            model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        except Exception as e:
            self.status_updated.emit(f"使用新API加载InceptionV3模型失败: {str(e)}")
            model = models.inception_v3(pretrained=True, aux_logits=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
    
    def _create_efficientnet_model(self, model_name, num_classes):
        """创建EfficientNet系列模型"""
        try:
            from efficientnet_pytorch import EfficientNet
            
            model_mapping = {
                'EfficientNetB0': 'efficientnet-b0',
                'EfficientNetB1': 'efficientnet-b1',
                'EfficientNetB2': 'efficientnet-b2',
                'EfficientNetB3': 'efficientnet-b3',
                'EfficientNetB4': 'efficientnet-b4'
            }
            
            efficientnet_name = model_mapping[model_name]
            model = EfficientNet.from_pretrained(efficientnet_name, num_classes=num_classes)
            return model
            
        except ImportError:
            self.status_updated.emit(f"未安装EfficientNet库，尝试自动安装...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "efficientnet_pytorch"])
                from efficientnet_pytorch import EfficientNet
                
                model_mapping = {
                    'EfficientNetB0': 'efficientnet-b0',
                    'EfficientNetB1': 'efficientnet-b1',
                    'EfficientNetB2': 'efficientnet-b2',
                    'EfficientNetB3': 'efficientnet-b3',
                    'EfficientNetB4': 'efficientnet-b4'
                }
                
                efficientnet_name = model_mapping[model_name]
                model = EfficientNet.from_pretrained(efficientnet_name, num_classes=num_classes)
                return model
                
            except Exception as install_err:
                raise Exception(f"无法安装EfficientNet库: {str(install_err)}")
    
    def _handle_model_creation_error(self, error, model_name, num_classes):
        """处理模型创建错误"""
        import urllib.error
        download_failed = False
        
        # 检查是否为下载失败的错误
        if isinstance(error, urllib.error.URLError) or isinstance(error, ConnectionError) or isinstance(error, TimeoutError):
            download_failed = True
        elif any(keyword in str(error).lower() for keyword in ["download", "connection", "timeout", "url", "connect", "network", "internet"]):
            download_failed = True
        
        if download_failed:
            # 发送下载失败信号
            exact_model_name = self._find_exact_model_name(model_name)
            model_link = self.model_links.get(exact_model_name, "")
            
            if model_link:
                self.model_download_failed.emit(exact_model_name, model_link)
                self.status_updated.emit(f"预训练模型 {model_name} 下载失败，已提供手动下载链接")
            else:
                self.status_updated.emit(f"预训练模型 {model_name} 下载失败，无法找到下载链接: {str(error)}")
            
            # 尝试使用不带预训练权重的模型
            self.status_updated.emit("尝试使用未预训练的模型继续训练...")
            return self._create_model_without_pretrained_weights(model_name, num_classes)
        
        # 对于其他错误，返回默认模型
        return self._create_fallback_model(num_classes)
    
    def _find_exact_model_name(self, model_name):
        """查找精确的模型名称"""
        if model_name in self.model_links:
            return model_name
        
        for key in self.model_links.keys():
            if model_name.lower() in key.lower() or key.lower() in model_name.lower():
                return key
        
        return model_name
    
    def _create_model_without_pretrained_weights(self, model_name, num_classes):
        """创建不带预训练权重的模型"""
        self.status_updated.emit(f"使用不带预训练权重的{model_name}模型")
        
        try:
            # ResNet系列
            if model_name in ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']:
                model_type = model_name.lower()
                model_func = getattr(models, model_type)
                try:
                    model = model_func(weights=None)
                except:
                    model = model_func(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                return model
            
            # MobileNet系列
            elif model_name == 'MobileNetV2':
                model = models.mobilenet_v2(pretrained=False)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                return model
            elif model_name == 'MobileNetV3':
                model = models.mobilenet_v3_large(pretrained=False)
                model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
                return model
            
            # VGG系列
            elif model_name in ['VGG16', 'VGG19']:
                model_type = model_name.lower()
                model_func = getattr(models, model_type)
                model = model_func(pretrained=False)
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                return model
            
            # DenseNet系列
            elif model_name in ['DenseNet121', 'DenseNet169', 'DenseNet201']:
                model_type = model_name.lower()
                model_func = getattr(models, model_type)
                model = model_func(pretrained=False)
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                return model
            
            # InceptionV3
            elif model_name == 'InceptionV3':
                model = models.inception_v3(pretrained=False, aux_logits=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                return model
            
            # EfficientNet系列
            elif model_name in ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4']:
                try:
                    from efficientnet_pytorch import EfficientNet
                    model_mapping = {
                        'EfficientNetB0': 'efficientnet-b0',
                        'EfficientNetB1': 'efficientnet-b1',
                        'EfficientNetB2': 'efficientnet-b2',
                        'EfficientNetB3': 'efficientnet-b3',
                        'EfficientNetB4': 'efficientnet-b4'
                    }
                    efficientnet_name = model_mapping[model_name]
                    model = EfficientNet.from_name(efficientnet_name, num_classes=num_classes)
                    return model
                except:
                    # 如果失败，使用ResNet50替代
                    self.status_updated.emit("无法创建EfficientNet模型，使用ResNet50替代")
                    return self._create_fallback_model(num_classes)
            
            else:
                return self._create_fallback_model(num_classes)
                
        except Exception:
            return self._create_fallback_model(num_classes)
    
    def _create_fallback_model(self, num_classes):
        """创建后备模型"""
        self.status_updated.emit("使用默认的ResNet50模型")
        try:
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        except:
            # 如果连ResNet50都创建失败，创建简单的自定义模型
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