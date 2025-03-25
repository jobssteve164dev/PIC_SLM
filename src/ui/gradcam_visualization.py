from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                           QFileDialog, QMessageBox, QHBoxLayout, QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import logging

class GradCAMVisualizationWidget(QWidget):
    """Grad-CAM可视化组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.image = None
        self.class_names = []
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 说明标签
        info_label = QLabel(
            "Grad-CAM可视化可以帮助您理解模型在做出预测时关注的图像区域。\n"
            "选择一张图片和目标类别后，将显示热力图。"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 图片选择按钮
        self.select_image_btn = QPushButton("选择图片")
        self.select_image_btn.clicked.connect(self.select_image)
        layout.addWidget(self.select_image_btn)
        
        # 类别选择下拉框
        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self.update_gradcam)
        layout.addWidget(self.class_combo)
        
        # 创建水平布局用于显示原始图片和热力图
        image_layout = QHBoxLayout()
        
        # 显示原始图片
        self.original_image_label = QLabel("原始图片")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(300, 300)
        self.original_image_label.setStyleSheet("border: 1px solid gray;")
        image_layout.addWidget(self.original_image_label)
        
        # 显示热力图
        self.heatmap_label = QLabel("热力图")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setMinimumSize(300, 300)
        self.heatmap_label.setStyleSheet("border: 1px solid gray;")
        image_layout.addWidget(self.heatmap_label)
        
        layout.addLayout(image_layout)
        
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
                
                # 如果模型已加载，则更新Grad-CAM
                if self.model is not None:
                    self.update_gradcam()
                    
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
            
    def generate_gradcam(self, target_class):
        """生成Grad-CAM热力图"""
        try:
            if self.model is None or self.image_tensor is None:
                return None
                
            # 将图片移到正确的设备上
            device = next(self.model.parameters()).device
            image_tensor = self.image_tensor.to(device)
            
            # 获取模型输出和梯度
            output = self.model(image_tensor)
            
            # 清除之前的梯度
            self.model.zero_grad()
            
            # 计算目标类别的梯度
            output[:, target_class].backward()
            
            # 获取最后一个卷积层的梯度
            gradients = self.model.last_conv_layer.weight.grad
            pooled_gradients = torch.mean(gradients, dim=[2, 3])
            
            # 生成热力图
            heatmap = torch.zeros_like(self.image_tensor[0])
            for i in range(len(pooled_gradients)):
                heatmap += pooled_gradients[i] * self.model.last_conv_layer.output[0, i]
            
            # 归一化热力图
            heatmap = heatmap.detach().cpu().numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
            return heatmap
            
        except Exception as e:
            self.logger.error(f"生成Grad-CAM时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"生成Grad-CAM时出错: {str(e)}")
            return None
            
    def update_gradcam(self):
        """更新Grad-CAM显示"""
        try:
            if self.class_combo.currentIndex() < 0:
                return
                
            target_class = self.class_combo.currentIndex()
            heatmap = self.generate_gradcam(target_class)
            
            if heatmap is not None:
                # 创建matplotlib图形
                plt.figure(figsize=(10, 10))
                plt.imshow(heatmap, cmap='jet')
                plt.axis('off')
                
                # 将matplotlib图形转换为QPixmap
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
                plt.close()
                
                # 显示热力图
                qimage = QImage.fromData(buffer.getvalue())
                pixmap = QPixmap.fromImage(qimage)
                
                # 调整大小以适应标签
                scaled_pixmap = pixmap.scaled(
                    self.heatmap_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                
                self.heatmap_label.setPixmap(scaled_pixmap)
                
        except Exception as e:
            self.logger.error(f"更新Grad-CAM显示时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"更新Grad-CAM显示时出错: {str(e)}")
            
    def set_model(self, model, class_names=None):
        """设置模型和类别名称"""
        self.model = model
        if class_names is not None:
            self.class_names = class_names
            self.class_combo.clear()
            self.class_combo.addItems(class_names)
        if self.image_tensor is not None:
            self.update_gradcam() 