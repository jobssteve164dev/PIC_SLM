import torch
import torch.nn as nn
import torchvision.models as models
import os
import json
from collections import OrderedDict
from PyQt5.QtWidgets import QMessageBox
import traceback


class ModelLoader:
    """模型加载器，专门处理PyTorch模型的加载和创建"""
    
    def __init__(self):
        self.model = None
        self.model_path = None
        self.class_names = []
    
    def set_model_path(self, model_path):
        """设置模型路径"""
        self.model_path = model_path
        
        # 尝试寻找同目录下的类别信息文件
        model_dir = os.path.dirname(model_path)
        class_info_file = os.path.join(model_dir, "class_info.json")
        
        if os.path.exists(class_info_file):
            try:
                with open(class_info_file, 'r', encoding='utf-8') as f:
                    class_info = json.load(f)
                    self.class_names = class_info.get('class_names', [])
            except Exception as e:
                print(f"加载类别信息出错: {str(e)}")
                self.class_names = []
    
    def _create_densenet_model(self, model_type, num_classes):
        """创建DenseNet模型，兼容新旧PyTorch API"""
        try:
            # 尝试使用新API
            print(f"尝试使用新API加载{model_type}模型...")
            if model_type == "densenet121":
                model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            elif model_type == "densenet169":
                model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
            elif model_type == "densenet201":
                model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
            else:
                raise ValueError(f"不支持的DenseNet模型类型: {model_type}")
                
            # 修改分类器头
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            print(f"使用新API成功加载{model_type}模型")
            return model
        except Exception as e:
            # 新API失败，尝试使用旧API
            print(f"使用新API加载{model_type}模型失败: {str(e)}")
            try:
                print(f"尝试使用旧API加载{model_type}模型...")
                if model_type == "densenet121":
                    model = models.densenet121(pretrained=False)
                elif model_type == "densenet169":
                    model = models.densenet169(pretrained=False)
                elif model_type == "densenet201":
                    model = models.densenet201(pretrained=False)
                else:
                    raise ValueError(f"不支持的DenseNet模型类型: {model_type}")
                    
                # 修改分类器头
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                print(f"使用旧API成功加载{model_type}模型")
                return model
            except Exception as e2:
                # 再次失败，尝试直接从torchvision导入
                print(f"使用旧API加载{model_type}模型失败: {str(e2)}")
                try:
                    print(f"尝试直接从torchvision.models导入{model_type}...")
                    if model_type == "densenet121":
                        from torchvision.models import densenet121
                        model = densenet121(pretrained=False)
                    elif model_type == "densenet169":
                        from torchvision.models import densenet169
                        model = densenet169(pretrained=False)
                    elif model_type == "densenet201":
                        from torchvision.models import densenet201
                        model = densenet201(pretrained=False)
                    
                    # 使用安全的方式设置分类器
                    if hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
                        in_features = model.classifier.in_features
                        model.classifier = nn.Linear(in_features, num_classes)
                    else:
                        print(f"警告: 无法确定{model_type}的分类器结构，使用默认分类器")
                        
                    print(f"通过直接导入成功加载{model_type}模型")
                    return model
                except Exception as e3:
                    # 所有方法均失败
                    error_msg = f"所有方法加载{model_type}模型均失败:\n"
                    error_msg += f"新API错误: {str(e)}\n"
                    error_msg += f"旧API错误: {str(e2)}\n"
                    error_msg += f"直接导入错误: {str(e3)}"
                    print(error_msg)
                    raise ValueError(error_msg)
    
    def _create_model_by_name(self, model_name, num_classes):
        """根据模型名称创建模型实例"""
        model_name = model_name.lower()
        
        if "resnet18" in model_name:
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif "resnet34" in model_name:
            model = models.resnet34(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif "resnet50" in model_name:
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif "resnet101" in model_name:
            model = models.resnet101(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif "resnet152" in model_name:
            model = models.resnet152(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif "mobilenetv2" in model_name or "mobilenet_v2" in model_name:
            model = models.mobilenet_v2(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif "mobilenetv3" in model_name or "mobilenet_v3" in model_name:
            model = models.mobilenet_v3_large(pretrained=False)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        elif "vgg16" in model_name:
            model = models.vgg16(pretrained=False)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        elif "vgg19" in model_name:
            model = models.vgg19(pretrained=False)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        elif "densenet121" in model_name:
            model = self._create_densenet_model("densenet121", num_classes)
        elif "densenet169" in model_name:
            model = self._create_densenet_model("densenet169", num_classes)
        elif "densenet201" in model_name:
            model = self._create_densenet_model("densenet201", num_classes)
        elif "efficientnet" in model_name:
            # 尝试使用EfficientNet
            try:
                from efficientnet_pytorch import EfficientNet
                if "b0" in model_name:
                    model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
                elif "b1" in model_name:
                    model = EfficientNet.from_name('efficientnet-b1', num_classes=num_classes)
                elif "b2" in model_name:
                    model = EfficientNet.from_name('efficientnet-b2', num_classes=num_classes)
                elif "b3" in model_name:
                    model = EfficientNet.from_name('efficientnet-b3', num_classes=num_classes)
                elif "b4" in model_name:
                    model = EfficientNet.from_name('efficientnet-b4', num_classes=num_classes)
                else:
                    model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
            except ImportError:
                # 如果没有安装EfficientNet库，使用ResNet50替代
                model = models.resnet50(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            # 如果无法从文件名判断，默认使用ResNet50
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        return model
    
    def _load_state_dict(self, model, state_dict):
        """加载模型权重，处理各种兼容性问题"""
        try:
            # 尝试直接加载
            model.load_state_dict(state_dict)
            return True
        except Exception as e:
            print(f"直接加载模型权重失败：{str(e)}")
            # 可能是DataParallel保存的模型，尝试移除 'module.' 前缀
            try:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '')
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                return True
            except Exception as e2:
                print(f"尝试调整模型权重键名后仍然失败：{str(e2)}")
                return False
    
    def load_model(self):
        """加载模型"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError("模型文件不存在")
            
            # 获取模型类型
            model_name = os.path.basename(self.model_path)
            
            # 创建模型实例
            num_classes = len(self.class_names) if self.class_names else 1000
            model = self._create_model_by_name(model_name, num_classes)
            
            # 加载模型权重
            if torch.cuda.is_available():
                state_dict = torch.load(self.model_path)
            else:
                state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
            
            # 尝试加载权重
            if not self._load_state_dict(model, state_dict):
                print("警告: 无法加载模型权重，只返回模型结构")
            
            self.model = model
            return model
            
        except Exception as e:
            error_msg = f"加载模型失败: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            raise Exception(error_msg)
    
    def set_external_model(self, model, class_names=None):
        """从外部设置模型"""
        self.model = model
        if class_names:
            self.class_names = class_names
    
    def is_densenet_model(self, model):
        """判断是否为DenseNet模型"""
        try:
            return any(isinstance(model, getattr(models.densenet, name)) 
                      for name in dir(models.densenet) 
                      if 'DenseNet' in name and isinstance(getattr(models.densenet, name), type))
        except:
            return False
    
    def get_model_info(self):
        """获取模型基本信息"""
        if self.model is None:
            return None
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'name': self.model.__class__.__name__,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'class_names': self.class_names,
            'is_densenet': self.is_densenet_model(self.model)
        } 