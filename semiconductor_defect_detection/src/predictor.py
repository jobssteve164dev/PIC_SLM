import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
from PyQt5.QtCore import QObject, pyqtSignal
from typing import Dict, Tuple, List

class Predictor(QObject):
    # 定义信号
    prediction_finished = pyqtSignal(dict)
    prediction_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path: str, class_info_path: str) -> None:
        """
        加载模型和类别信息
        """
        try:
            # 加载类别信息
            with open(class_info_path, 'r') as f:
                class_info = json.load(f)
            self.class_names = class_info['class_names']

            # 加载模型
            checkpoint = torch.load(model_path, map_location=self.device)
            num_classes = len(self.class_names)
            
            # 创建模型
            self.model = self._create_model('ResNet50', num_classes)  # 默认使用ResNet50
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            self.prediction_error.emit(f'加载模型时出错: {str(e)}')

    def predict(self, image_path: str) -> None:
        """
        预测图片类别
        """
        try:
            # 加载和预处理图片
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # 预测
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_prob, top_class = torch.topk(probabilities, 3)

            # 获取预测结果
            predictions = []
            for i in range(3):
                class_idx = top_class[0][i].item()
                prob = top_prob[0][i].item()
                predictions.append({
                    'class_name': self.class_names[class_idx],
                    'probability': prob * 100  # 转换为百分比
                })

            # 发送预测结果
            result = {
                'predictions': predictions,
                'image_path': image_path
            }
            self.prediction_finished.emit(result)

        except Exception as e:
            self.prediction_error.emit(f'预测过程中出错: {str(e)}')

    def _create_model(self, model_name: str, num_classes: int) -> nn.Module:
        """创建模型"""
        if model_name == 'ResNet50':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            raise ValueError(f'不支持的模型: {model_name}')
        
        return model 