from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                            QHBoxLayout, QGroupBox, QGridLayout, QTextEdit, QLineEdit,
                            QComboBox, QSplitter, QMessageBox, QSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import torch
import torch.nn as nn
import torchvision.models as models
import os
import json
from torchsummary import summary
import io
from contextlib import redirect_stdout
import sys

class ModelStructureViewer(QWidget):
    """模型结构可视化组件，使用torchsummary显示PyTorch模型结构"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.model_path = None
        self.class_names = []
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        
        # 模型选择组
        model_group = QGroupBox("模型选择")
        model_layout = QGridLayout()
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText("请选择模型文件")
        
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_model)
        
        self.input_size_label = QLabel("输入尺寸:")
        self.input_height = QSpinBox()
        self.input_height.setRange(1, 1024)
        self.input_height.setValue(224)
        self.input_width = QSpinBox()
        self.input_width.setRange(1, 1024)
        self.input_width.setValue(224)
        self.input_channels = QSpinBox()
        self.input_channels.setRange(1, 16)
        self.input_channels.setValue(3)
        
        size_layout = QHBoxLayout()
        size_layout.addWidget(self.input_height)
        size_layout.addWidget(QLabel("×"))
        size_layout.addWidget(self.input_width)
        size_layout.addWidget(QLabel("×"))
        size_layout.addWidget(self.input_channels)
        
        model_layout.addWidget(QLabel("模型文件:"), 0, 0)
        model_layout.addWidget(self.model_path_edit, 0, 1)
        model_layout.addWidget(browse_btn, 0, 2)
        model_layout.addWidget(self.input_size_label, 1, 0)
        model_layout.addLayout(size_layout, 1, 1)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # 控制按钮
        control_layout = QHBoxLayout()
        
        self.visualize_btn = QPushButton("可视化模型结构")
        self.visualize_btn.clicked.connect(self.visualize_model_structure)
        self.visualize_btn.setEnabled(False)
        
        control_layout.addStretch()
        control_layout.addWidget(self.visualize_btn)
        control_layout.addStretch()
        
        main_layout.addLayout(control_layout)
        
        # 输出文本区域
        output_group = QGroupBox("模型结构")
        output_layout = QVBoxLayout()
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setLineWrapMode(QTextEdit.NoWrap)
        self.output_text.setFont(QFont("Courier New", 10))
        
        output_layout.addWidget(self.output_text)
        output_group.setLayout(output_layout)
        
        main_layout.addWidget(output_group)
    
    def browse_model(self):
        """浏览并选择模型文件"""
        file_dialog = QFileDialog()
        model_path, _ = file_dialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch模型 (*.pth *.pt);;所有文件 (*.*)"
        )
        
        if model_path:
            self.model_path = model_path
            self.model_path_edit.setText(model_path)
            
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
            
            self.visualize_btn.setEnabled(True)
    
    def load_model(self):
        """加载模型"""
        try:
            if not os.path.exists(self.model_path):
                QMessageBox.warning(self, "错误", "模型文件不存在")
                return None
            
            # 获取模型类型
            model_name = os.path.basename(self.model_path).lower()
            
            # 创建模型实例 - 首先尝试从文件名猜测模型类型
            num_classes = len(self.class_names) if self.class_names else 1000
            
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
                model = models.densenet121(pretrained=False)
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            elif "densenet169" in model_name:
                model = models.densenet169(pretrained=False)
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            elif "densenet201" in model_name:
                model = models.densenet201(pretrained=False)
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
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
            
            # 加载模型权重
            if torch.cuda.is_available():
                state_dict = torch.load(self.model_path)
            else:
                state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
            
            # 尝试直接加载，可能会因为键名不匹配而失败
            try:
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"直接加载模型权重失败：{str(e)}")
                # 可能是DataParallel保存的模型，尝试移除 'module.' 前缀
                try:
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k.replace('module.', '')
                        new_state_dict[name] = v
                    model.load_state_dict(new_state_dict)
                except Exception as e2:
                    print(f"尝试调整模型权重键名后仍然失败：{str(e2)}")
                    # 加载失败可能是因为模型结构不匹配，直接返回创建的模型而不加载权重
                    QMessageBox.warning(self, "警告", f"无法加载模型权重，只显示模型结构: {str(e2)}")
            
            return model
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
            return None
    
    def visualize_model_structure(self):
        """可视化模型结构"""
        if not self.model_path:
            QMessageBox.warning(self, "警告", "请先选择模型文件")
            return
        
        # 加载模型
        model = self.load_model()
        if model is None:
            return
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 获取输入尺寸
        height = self.input_height.value()
        width = self.input_width.value()
        channels = self.input_channels.value()
        
        # 使用torchsummary显示模型结构
        try:
            # 重定向stdout以捕获summary输出
            string_io = io.StringIO()
            with redirect_stdout(string_io):
                summary(model, input_size=(channels, height, width), device=str(device))
            
            # 获取捕获的输出并显示在文本区域
            output = string_io.getvalue()
            self.output_text.setText(output)
            
            # 添加总参数数量和可训练参数数量信息
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            additional_info = f"\n\n总参数数量：{total_params:,}\n可训练参数数量：{trainable_params:,}"
            self.output_text.append(additional_info)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"可视化模型结构失败: {str(e)}")
    
    def set_model(self, model, class_names=None):
        """从外部设置模型"""
        if model is not None:
            self.model = model
            if class_names:
                self.class_names = class_names
            self.visualize_btn.setEnabled(True)
            
            # 自动触发可视化
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            # 获取输入尺寸
            height = self.input_height.value()
            width = self.input_width.value()
            channels = self.input_channels.value()
            
            # 使用torchsummary显示模型结构
            try:
                # 重定向stdout以捕获summary输出
                string_io = io.StringIO()
                with redirect_stdout(string_io):
                    summary(model, input_size=(channels, height, width), device=str(device))
                
                # 获取捕获的输出并显示在文本区域
                output = string_io.getvalue()
                self.output_text.setText(output)
                
                # 添加总参数数量和可训练参数数量信息
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                additional_info = f"\n\n总参数数量：{total_params:,}\n可训练参数数量：{trainable_params:,}"
                self.output_text.append(additional_info)
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"可视化模型结构失败: {str(e)}") 