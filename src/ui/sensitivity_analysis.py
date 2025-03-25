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

class SensitivityAnalysisWidget(QWidget):
    """敏感性分析组件"""
    
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
            "敏感性分析可以帮助您理解模型对输入变化的敏感程度。\n"
            "选择一张图片和目标类别后，将显示不同扰动下的预测变化。"
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
        self.class_combo.currentIndexChanged.connect(self.update_analysis)
        param_layout.addRow("目标类别:", self.class_combo)
        
        # 扰动范围设置
        self.perturbation_range = QDoubleSpinBox()
        self.perturbation_range.setRange(0.01, 1.0)
        self.perturbation_range.setValue(0.1)
        self.perturbation_range.setSingleStep(0.01)
        self.perturbation_range.valueChanged.connect(self.update_analysis)
        param_layout.addRow("扰动范围:", self.perturbation_range)
        
        # 扰动步数设置
        self.num_steps = QSpinBox()
        self.num_steps.setRange(10, 100)
        self.num_steps.setValue(20)
        self.num_steps.valueChanged.connect(self.update_analysis)
        param_layout.addRow("扰动步数:", self.num_steps)
        
        layout.addLayout(param_layout)
        
        # 创建水平布局用于显示原始图片和敏感性曲线
        image_layout = QHBoxLayout()
        
        # 显示原始图片
        self.original_image_label = QLabel("原始图片")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(300, 300)
        self.original_image_label.setStyleSheet("border: 1px solid gray;")
        image_layout.addWidget(self.original_image_label)
        
        # 显示敏感性曲线
        self.sensitivity_label = QLabel("敏感性曲线")
        self.sensitivity_label.setAlignment(Qt.AlignCenter)
        self.sensitivity_label.setMinimumSize(300, 300)
        self.sensitivity_label.setStyleSheet("border: 1px solid gray;")
        image_layout.addWidget(self.sensitivity_label)
        
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
                
                # 如果模型已加载，则更新分析
                if self.model is not None:
                    self.update_analysis()
                    
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
            
    def perform_sensitivity_analysis(self, target_class):
        """执行敏感性分析"""
        try:
            if self.model is None or self.image_tensor is None:
                return None, None
                
            # 将图片移到正确的设备上
            device = next(self.model.parameters()).device
            image_tensor = self.image_tensor.to(device)
            
            # 获取扰动参数
            perturbation_range = self.perturbation_range.value()
            num_steps = self.num_steps.value()
            
            # 创建扰动序列
            perturbations = torch.linspace(-perturbation_range, perturbation_range, num_steps)
            predictions = []
            
            # 对每个扰动值进行预测
            for delta in perturbations:
                # 创建扰动后的图片
                perturbed_image = image_tensor + delta
                # 确保像素值在[0,1]范围内
                perturbed_image = torch.clamp(perturbed_image, 0, 1)
                
                # 获取预测
                with torch.no_grad():
                    output = self.model(perturbed_image)
                    prob = torch.softmax(output, dim=1)[0, target_class].item()
                    predictions.append(prob)
            
            return perturbations.numpy(), predictions
            
        except Exception as e:
            self.logger.error(f"执行敏感性分析时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"执行敏感性分析时出错: {str(e)}")
            return None, None
            
    def update_analysis(self):
        """更新敏感性分析显示"""
        try:
            if self.class_combo.currentIndex() < 0:
                return
                
            target_class = self.class_combo.currentIndex()
            perturbations, predictions = self.perform_sensitivity_analysis(target_class)
            
            if perturbations is not None and predictions is not None:
                # 创建matplotlib图形
                plt.figure(figsize=(10, 6))
                plt.plot(perturbations, predictions, 'b-', label='预测概率')
                plt.title(f'类别 {self.class_names[target_class]} 的敏感性分析')
                plt.xlabel('输入扰动')
                plt.ylabel('预测概率')
                plt.grid(True)
                plt.legend()
                
                # 将matplotlib图形转换为QPixmap
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
                plt.close()
                
                # 显示敏感性曲线
                qimage = QImage.fromData(buffer.getvalue())
                pixmap = QPixmap.fromImage(qimage)
                
                # 调整大小以适应标签
                scaled_pixmap = pixmap.scaled(
                    self.sensitivity_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                
                self.sensitivity_label.setPixmap(scaled_pixmap)
                
        except Exception as e:
            self.logger.error(f"更新敏感性分析显示时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"更新敏感性分析显示时出错: {str(e)}")
            
    def set_model(self, model, class_names=None):
        """设置模型和类别名称"""
        self.model = model
        if class_names is not None:
            self.class_names = class_names
            self.class_combo.clear()
            self.class_combo.addItems(class_names)
        if self.image_tensor is not None:
            self.update_analysis() 