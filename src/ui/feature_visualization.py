from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                           QFileDialog, QMessageBox, QHBoxLayout, QComboBox, 
                           QLineEdit, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import logging
import json

class FeatureVisualizationWidget(QWidget):
    """特征可视化组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.image = None
        self.image_tensor = None
        self.features = None
        self.model_file = None
        self.class_info_file = None
        self.class_names = []
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
        
        # 创建模型文件选择组
        model_group = QGroupBox("模型文件")
        model_layout = QGridLayout()
        
        # 添加模型类型选择
        model_layout.addWidget(QLabel("模型类型:"), 0, 0)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["分类模型", "检测模型"])
        self.model_type_combo.currentIndexChanged.connect(self.switch_model_type)
        model_layout.addWidget(self.model_type_combo, 0, 1, 1, 1)
        
        # 添加模型架构选择
        model_layout.addWidget(QLabel("模型架构:"), 0, 2)
        self.model_arch_combo = QComboBox()
        self.model_arch_combo.addItems([
            "MobileNetV2", "MobileNetV3", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
            "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4",
            "VGG16", "VGG19", "DenseNet121", "DenseNet169", "DenseNet201", "InceptionV3", "Xception"
        ])
        model_layout.addWidget(self.model_arch_combo, 0, 3, 1, 1)
        
        # 模型文件选择
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText("请选择训练好的模型文件")
        
        model_btn = QPushButton("浏览...")
        model_btn.clicked.connect(self.select_model_file)
        
        model_layout.addWidget(QLabel("模型文件:"), 1, 0)
        model_layout.addWidget(self.model_path_edit, 1, 1, 1, 2)
        model_layout.addWidget(model_btn, 1, 3)
        
        # 类别信息文件
        self.class_info_path_edit = QLineEdit()
        self.class_info_path_edit.setReadOnly(True)
        self.class_info_path_edit.setPlaceholderText("请选择类别信息文件")
        
        class_info_btn = QPushButton("浏览...")
        class_info_btn.clicked.connect(self.select_class_info_file)
        
        model_layout.addWidget(QLabel("类别信息:"), 2, 0)
        model_layout.addWidget(self.class_info_path_edit, 2, 1, 1, 2)
        model_layout.addWidget(class_info_btn, 2, 3)
        
        # 加载模型按钮
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.load_model)
        self.load_model_btn.setEnabled(False)
        model_layout.addWidget(self.load_model_btn, 3, 0, 1, 4)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
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
        
    def switch_model_type(self, index):
        """切换模型类型"""
        if index == 0:  # 分类模型
            self.model_arch_combo.clear()
            self.model_arch_combo.addItems([
                "MobileNetV2", "MobileNetV3", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
                "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4",
                "VGG16", "VGG19", "DenseNet121", "DenseNet169", "DenseNet201", "InceptionV3", "Xception"
            ])
        else:  # 检测模型
            self.model_arch_combo.clear()
            self.model_arch_combo.addItems([
                "YOLOv5", "YOLOv8", "YOLOv7", "YOLOv6", "YOLOv4", "YOLOv3",
                "SSD", "SSD512", "SSD300", "Faster R-CNN", "Mask R-CNN",
                "RetinaNet", "DETR"
            ])
    
    def select_model_file(self):
        """选择模型文件"""
        file, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "模型文件 (*.h5 *.pb *.tflite *.pt *.pth);;所有文件 (*)")
        if file:
            self.model_file = file
            self.model_path_edit.setText(file)
            self.check_model_ready()
    
    def select_class_info_file(self):
        """选择类别信息文件"""
        file, _ = QFileDialog.getOpenFileName(self, "选择类别信息文件", "", "JSON文件 (*.json);;所有文件 (*)")
        if file:
            self.class_info_file = file
            self.class_info_path_edit.setText(file)
            self.check_model_ready()
    
    def check_model_ready(self):
        """检查是否可以加载模型"""
        is_ready = bool(self.model_file and self.class_info_file)
        self.load_model_btn.setEnabled(is_ready)
        return is_ready
        
    def load_model(self):
        """加载模型"""
        if not self.model_file or not self.class_info_file:
            QMessageBox.warning(self, "警告", "请先选择模型文件和类别信息文件!")
            return
            
        try:
            # 获取模型类型和架构
            model_type = self.model_type_combo.currentText()
            model_arch = self.model_arch_combo.currentText()
            
            # 创建模型信息字典
            model_info = {
                "model_path": self.model_file,
                "class_info_path": self.class_info_file,
                "model_type": model_type,
                "model_arch": model_arch
            }
            
            # 寻找主窗口引用
            main_window = None
            parent = self.parent()
            
            # 逐级向上查找主窗口
            while parent:
                if hasattr(parent, 'worker') and hasattr(parent.worker, 'predictor'):
                    main_window = parent
                    break
                elif hasattr(parent, 'main_window'):
                    main_window = parent.main_window
                    break
                parent = parent.parent()
                
            if main_window and hasattr(main_window, 'worker') and hasattr(main_window.worker, 'predictor'):
                # 使用找到的主窗口加载模型
                main_window.worker.predictor.load_model_with_info(model_info)
                
                # 获取加载后的模型
                self.model = main_window.worker.predictor.model
                
                # 加载类别名称
                try:
                    with open(self.class_info_file, 'r', encoding='utf-8') as f:
                        self.class_names = json.load(f)
                except Exception as e:
                    self.logger.error(f"读取类别信息文件失败: {str(e)}")
                    QMessageBox.critical(self, "错误", f"读取类别信息文件失败: {str(e)}")
                
                # 模型加载成功
                QMessageBox.information(self, "成功", f"模型 {model_arch} 加载成功！\n现在可以进行特征可视化。")
                
                # 禁用加载模型按钮，表示模型已加载
                self.load_model_btn.setEnabled(False)
                
                # 如果已经选择了图像，提取特征并可视化
                if self.image_tensor is not None:
                    self.extract_features()
            else:
                QMessageBox.critical(self, "错误", "无法访问模型加载器，请检查应用配置")
                
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            return
        
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
            
            # 检查模型是否有get_intermediate_features方法
            if hasattr(self.model, 'get_intermediate_features'):
                # 如果有，直接调用
                self.features = self.model.get_intermediate_features(image_tensor)
            else:
                # 如果没有，使用钩子机制捕获中间层特征
                self.features = []
                hooks = []
                
                # 定义钩子函数
                def hook_fn(module, input, output):
                    self.features.append(output.detach())
                
                # 添加钩子到卷积层
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        hooks.append(module.register_forward_hook(hook_fn))
                
                # 前向传播
                with torch.no_grad():
                    self.model(image_tensor)
                
                # 移除钩子
                for hook in hooks:
                    hook.remove()
            
            # 可视化特征
            self.visualize_features()
            
        except Exception as e:
            self.logger.error(f"提取特征时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"提取特征时出错: {str(e)}")
            
    def visualize_features(self):
        """可视化特征图"""
        try:
            if not self.features or len(self.features) == 0:
                QMessageBox.warning(self, "警告", "未能提取到任何特征图")
                return
                
            # 创建matplotlib图形
            plt.figure(figsize=(15, 5))
            
            # 显示前5个特征图或者全部（如果少于5个）
            num_features = min(5, len(self.features))
            for i in range(num_features):
                plt.subplot(1, num_features, i+1)
                # 获取第一个通道的特征图
                feature = self.features[i]
                if isinstance(feature, torch.Tensor):
                    # 确保特征图是4D张量 [batch, channel, height, width]
                    if len(feature.shape) == 4:
                        # 获取第一个样本的第一个通道
                        feature_map = feature[0, 0].cpu().numpy()
                    else:
                        # 如果形状不是4D，则尝试适配
                        feature = feature.view(1, -1, feature.shape[-2], feature.shape[-1])
                        feature_map = feature[0, 0].cpu().numpy()
                else:
                    # 如果不是张量，尝试转换
                    feature_map = np.array(feature)
                    
                # 归一化
                if feature_map.max() > feature_map.min():  # 避免除零错误
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
            import traceback
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"可视化特征时出错: {str(e)}")
            
    def set_model(self, model):
        """设置模型"""
        self.model = model
        # 只在有图片时才更新特征
        if self.image_tensor is not None and self.isVisible():
            self.extract_features() 