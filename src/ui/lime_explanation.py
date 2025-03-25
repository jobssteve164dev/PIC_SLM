from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                           QFileDialog, QMessageBox, QHBoxLayout, QComboBox,
                           QSpinBox, QDoubleSpinBox, QFormLayout)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import logging
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

class LIMEExplanationWidget(QWidget):
    """LIME解释组件"""
    
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
            "LIME (Local Interpretable Model-agnostic Explanations) 可以帮助您理解\n"
            "模型在特定预测中使用的特征。选择一张图片后，将显示解释结果。"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 图片选择按钮
        self.select_image_btn = QPushButton("选择图片")
        self.select_image_btn.clicked.connect(self.select_image)
        layout.addWidget(self.select_image_btn)
        
        # 创建参数设置表单
        param_layout = QFormLayout()
        
        # 类别选择下拉框
        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self.update_explanation)
        param_layout.addRow("目标类别:", self.class_combo)
        
        # 超像素数量设置
        self.num_superpixels = QSpinBox()
        self.num_superpixels.setRange(10, 1000)
        self.num_superpixels.setValue(100)
        self.num_superpixels.valueChanged.connect(self.update_explanation)
        param_layout.addRow("超像素数量:", self.num_superpixels)
        
        # 样本数量设置
        self.num_samples = QSpinBox()
        self.num_samples.setRange(100, 10000)
        self.num_samples.setValue(1000)
        self.num_samples.valueChanged.connect(self.update_explanation)
        param_layout.addRow("样本数量:", self.num_samples)
        
        layout.addLayout(param_layout)
        
        # 创建水平布局用于显示原始图片和解释结果
        image_layout = QHBoxLayout()
        
        # 显示原始图片
        self.original_image_label = QLabel("原始图片")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(300, 300)
        self.original_image_label.setStyleSheet("border: 1px solid gray;")
        image_layout.addWidget(self.original_image_label)
        
        # 显示解释结果
        self.explanation_label = QLabel("解释结果")
        self.explanation_label.setAlignment(Qt.AlignCenter)
        self.explanation_label.setMinimumSize(300, 300)
        self.explanation_label.setStyleSheet("border: 1px solid gray;")
        image_layout.addWidget(self.explanation_label)
        
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
                
                # 如果模型已加载，则更新解释
                if self.model is not None:
                    self.update_explanation()
                    
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
            
    def generate_lime_explanation(self, target_class):
        """生成LIME解释"""
        try:
            if self.model is None or self.image_tensor is None:
                return None
                
            # 将图片移到正确的设备上
            device = next(self.model.parameters()).device
            image_tensor = self.image_tensor.to(device)
            
            # 定义预测函数
            def predict_fn(images):
                # 将numpy数组转换为tensor
                images_tensor = torch.from_numpy(images).float()
                images_tensor = images_tensor.permute(0, 3, 1, 2) / 255.0
                images_tensor = images_tensor.to(device)
                
                # 获取预测
                with torch.no_grad():
                    outputs = self.model(images_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    return probs.cpu().numpy()
            
            # 创建LIME解释器
            explainer = lime_image.LimeImageExplainer()
            
            # 设置超像素分割参数
            segmentation_fn = SegmentationAlgorithm(
                'quickshift',
                kernel_size=1,
                max_dist=200,
                ratio=0.2
            )
            
            # 生成解释
            explanation = explainer.explain_instance(
                np.array(self.image),
                predict_fn,
                segmentation_fn=segmentation_fn,
                num_features=self.num_superpixels.value(),
                num_samples=self.num_samples.value()
            )
            
            # 获取解释图像
            temp, mask = explanation.get_image_and_mask(
                target_class,
                positive_only=True,
                num_features=5,
                hide_rest=True
            )
            
            return temp, mask
            
        except Exception as e:
            self.logger.error(f"生成LIME解释时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"生成LIME解释时出错: {str(e)}")
            return None, None
            
    def update_explanation(self):
        """更新LIME解释显示"""
        try:
            if self.class_combo.currentIndex() < 0:
                return
                
            target_class = self.class_combo.currentIndex()
            temp, mask = self.generate_lime_explanation(target_class)
            
            if temp is not None and mask is not None:
                # 创建matplotlib图形
                plt.figure(figsize=(10, 10))
                plt.imshow(temp)
                plt.axis('off')
                plt.title(f'类别 {self.class_names[target_class]} 的LIME解释')
                
                # 将matplotlib图形转换为QPixmap
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
                plt.close()
                
                # 显示解释结果
                qimage = QImage.fromData(buffer.getvalue())
                pixmap = QPixmap.fromImage(qimage)
                
                # 调整大小以适应标签
                scaled_pixmap = pixmap.scaled(
                    self.explanation_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                
                self.explanation_label.setPixmap(scaled_pixmap)
                
        except Exception as e:
            self.logger.error(f"更新LIME解释显示时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"更新LIME解释显示时出错: {str(e)}")
            
    def set_model(self, model, class_names=None):
        """设置模型和类别名称"""
        self.model = model
        if class_names is not None:
            self.class_names = class_names
            self.class_combo.clear()
            self.class_combo.addItems(class_names)
        if self.image_tensor is not None:
            self.update_explanation() 