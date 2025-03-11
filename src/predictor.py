import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import shutil
from PyQt5.QtCore import QObject, pyqtSignal
from typing import Dict, Tuple, List, Optional

class Predictor(QObject):
    # 定义信号
    prediction_finished = pyqtSignal(dict)
    prediction_error = pyqtSignal(str)
    batch_prediction_progress = pyqtSignal(int)
    batch_prediction_status = pyqtSignal(str)
    batch_prediction_finished = pyqtSignal(dict)

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
        self._stop_batch_processing = False

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
        预测单张图片类别
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

    def predict_image(self, image_path: str) -> Optional[Dict]:
        """
        预测单张图片类别并返回结果（不发送信号）
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

            return {
                'predictions': predictions,
                'image_path': image_path
            }

        except Exception as e:
            print(f'预测图片 {image_path} 时出错: {str(e)}')
            return None

    def batch_predict(self, params: Dict) -> None:
        """
        批量预测图片并根据预测结果分类
        
        参数:
            params: 包含批量预测参数的字典
                - source_folder: 源图片文件夹
                - target_folder: 目标文件夹（分类后的图片将保存在这里）
                - confidence_threshold: 置信度阈值，只有高于此阈值的预测才会被接受
                - copy_mode: 'copy'（复制）或 'move'（移动）
                - create_subfolders: 是否为每个类别创建子文件夹
        """
        try:
            if self.model is None:
                self.prediction_error.emit('请先加载模型')
                return
                
            source_folder = params.get('source_folder')
            target_folder = params.get('target_folder')
            confidence_threshold = params.get('confidence_threshold', 50.0)  # 默认50%
            copy_mode = params.get('copy_mode', 'copy')
            create_subfolders = params.get('create_subfolders', True)
            
            # 重置停止标志
            self._stop_batch_processing = False
            
            # 获取所有图片文件
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            image_files = [f for f in os.listdir(source_folder) 
                          if os.path.isfile(os.path.join(source_folder, f)) and 
                          os.path.splitext(f.lower())[1] in valid_extensions]
            
            if not image_files:
                self.batch_prediction_status.emit('未找到图片文件')
                return
                
            # 创建目标文件夹
            os.makedirs(target_folder, exist_ok=True)
            
            # 如果需要创建子文件夹，为每个类别创建一个子文件夹
            if create_subfolders:
                for class_name in self.class_names:
                    os.makedirs(os.path.join(target_folder, class_name), exist_ok=True)
            
            # 统计结果
            results = {
                'total': len(image_files),
                'processed': 0,
                'classified': 0,
                'unclassified': 0,
                'class_counts': {class_name: 0 for class_name in self.class_names}
            }
            
            # 批量处理图片
            for i, image_file in enumerate(image_files):
                if self._stop_batch_processing:
                    self.batch_prediction_status.emit('批量处理已停止')
                    break
                    
                image_path = os.path.join(source_folder, image_file)
                result = self.predict_image(image_path)
                
                if result:
                    # 获取最高置信度的预测
                    top_prediction = result['predictions'][0]
                    class_name = top_prediction['class_name']
                    probability = top_prediction['probability']
                    
                    # 更新进度
                    progress = int((i + 1) / len(image_files) * 100)
                    self.batch_prediction_progress.emit(progress)
                    self.batch_prediction_status.emit(f'处理图片 {i+1}/{len(image_files)}: {image_file}')
                    
                    results['processed'] += 1
                    
                    # 如果置信度高于阈值，则分类图片
                    if probability >= confidence_threshold:
                        # 确定目标路径
                        if create_subfolders:
                            target_path = os.path.join(target_folder, class_name, image_file)
                        else:
                            # 如果不创建子文件夹，则在文件名前添加类别名称
                            base_name, ext = os.path.splitext(image_file)
                            target_path = os.path.join(target_folder, f"{class_name}_{base_name}{ext}")
                        
                        # 复制或移动文件
                        try:
                            if copy_mode == 'copy':
                                shutil.copy2(image_path, target_path)
                            else:  # move
                                shutil.move(image_path, target_path)
                                
                            results['classified'] += 1
                            results['class_counts'][class_name] += 1
                        except Exception as e:
                            self.batch_prediction_status.emit(f'处理文件 {image_file} 时出错: {str(e)}')
                    else:
                        results['unclassified'] += 1
            
            # 发送完成信号
            self.batch_prediction_finished.emit(results)
            self.batch_prediction_status.emit('批量处理完成')
            
        except Exception as e:
            self.prediction_error.emit(f'批量预测过程中出错: {str(e)}')

    def stop_batch_processing(self) -> None:
        """停止批量处理"""
        self._stop_batch_processing = True

    def _create_model(self, model_name: str, num_classes: int) -> nn.Module:
        """创建模型"""
        if model_name == 'ResNet50':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            raise ValueError(f'不支持的模型: {model_name}')
        
        return model 