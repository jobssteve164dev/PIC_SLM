from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                           QFileDialog, QMessageBox, QHBoxLayout)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import logging

class FeatureVisualizationWidget(QWidget):
    """特征可视化组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.image = None
        self.image_tensor = None
        self.features = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 说明标签
        info_label = QLabel(
            "特征可视化可以帮助您理解模型在不同层提取的特征。\n"
            "选择一张图片后，将显示模型中间层的特征图。"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 图片选择按钮
        self.select_image_btn = QPushButton("选择图片")
        self.select_image_btn.clicked.connect(self.select_image)
        layout.addWidget(self.select_image_btn)
        
        # 显示原始图片
        self.original_image_label = QLabel("原始图片")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(300, 300)
        self.original_image_label.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.original_image_label)
        
        # 特征图显示区域
        self.feature_label = QLabel("特征图")
        self.feature_label.setAlignment(Qt.AlignCenter)
        self.feature_label.setMinimumSize(600, 400)
        self.feature_label.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.feature_label)
        
        # 添加弹性空间
        layout.addStretch()
        
    def select_image(self):
        """选择图片"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择图片",
                "",
                "图片文件 (*.png *.jpg *.jpeg *.bmp)"
            )
            
            if file_path:
                # 加载图片
                self.image = Image.open(file_path)
                # 转换为RGB模式
                if self.image.mode != 'RGB':
                    self.image = self.image.convert('RGB')
                # 调整大小
                self.image = self.image.resize((224, 224))
                # 转换为tensor
                self.image_tensor = torch.from_numpy(np.array(self.image)).float()
                self.image_tensor = self.image_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
                
                # 显示原始图片
                self.display_image(self.image, self.original_image_label)
                
                # 如果模型已加载，则提取特征
                if self.model is not None:
                    self.extract_features()
                    
        except Exception as e:
            self.logger.error(f"选择图片时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"选择图片时出错: {str(e)}")
            
    def display_image(self, image, label):
        """显示图片"""
        try:
            # 将PIL图片转换为QPixmap
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            qimage = QImage.fromData(buffer.getvalue())
            pixmap = QPixmap.fromImage(qimage)
            
            # 调整大小以适应标签
            scaled_pixmap = pixmap.scaled(
                label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.logger.error(f"显示图片时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"显示图片时出错: {str(e)}")
            
    def extract_features(self):
        """提取特征"""
        try:
            if self.model is None or self.image_tensor is None:
                return
                
            # 将图片移到正确的设备上
            device = next(self.model.parameters()).device
            image_tensor = self.image_tensor.to(device)
            
            # 提取特征
            self.features = self.model.get_intermediate_features(image_tensor)
            
            # 可视化特征
            self.visualize_features()
            
        except Exception as e:
            self.logger.error(f"提取特征时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"提取特征时出错: {str(e)}")
            
    def visualize_features(self):
        """可视化特征图"""
        try:
            if self.features is None:
                return
                
            # 创建matplotlib图形
            plt.figure(figsize=(15, 5))
            
            # 显示前5个特征图
            for i, feature in enumerate(self.features[:5]):
                plt.subplot(1, 5, i+1)
                # 获取第一个通道的特征图
                feature_map = feature[0, 0].detach().cpu().numpy()
                # 归一化
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
                plt.imshow(feature_map, cmap='viridis')
                plt.title(f'Feature Map {i+1}')
                plt.axis('off')
            
            # 将matplotlib图形转换为QPixmap
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # 显示特征图
            qimage = QImage.fromData(buffer.getvalue())
            pixmap = QPixmap.fromImage(qimage)
            
            # 调整大小以适应标签
            scaled_pixmap = pixmap.scaled(
                self.feature_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.feature_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.logger.error(f"可视化特征时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"可视化特征时出错: {str(e)}")
            
    def set_model(self, model):
        """设置模型"""
        self.model = model
        if self.image_tensor is not None:
            self.extract_features() 