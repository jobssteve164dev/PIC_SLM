from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QFileDialog, QMessageBox, QComboBox, QLineEdit, QGroupBox, 
                           QGridLayout, QTabWidget, QSpinBox, QDoubleSpinBox, QFormLayout,
                           QCheckBox, QScrollArea, QSplitter, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import logging
import json
import cv2
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import traceback


class ModelAnalysisWorker(QThread):
    """模型分析工作线程"""
    
    analysis_finished = pyqtSignal(str, object)  # 分析类型, 结果
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.image = None
        self.image_tensor = None
        self.class_names = []
        self.analysis_type = ""
        self.target_class = 0
        self.analysis_params = {}
        
    def set_analysis_task(self, analysis_type, model, image, image_tensor, class_names, target_class, params=None):
        """设置分析任务"""
        self.analysis_type = analysis_type
        self.model = model
        self.image = image
        self.image_tensor = image_tensor
        self.class_names = class_names
        self.target_class = target_class
        self.analysis_params = params or {}
        
    def run(self):
        """执行分析任务"""
        try:
            if self.analysis_type == "特征可视化":
                result = self._feature_visualization()
            elif self.analysis_type == "GradCAM":
                result = self._gradcam_analysis()
            elif self.analysis_type == "LIME解释":
                result = self._lime_analysis()
            elif self.analysis_type == "敏感性分析":
                result = self._sensitivity_analysis()
            else:
                raise ValueError(f"未知的分析类型: {self.analysis_type}")
                
            self.analysis_finished.emit(self.analysis_type, result)
            
        except Exception as e:
            self.error_occurred.emit(f"{self.analysis_type}分析失败: {str(e)}")
            
    def _feature_visualization(self):
        """特征可视化"""
        self.status_updated.emit("正在提取特征...")
        features = {}
        
        def hook_fn(module, input, output):
            features[module] = output.detach()
            
        # 注册钩子
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                hooks.append(module.register_forward_hook(hook_fn))
                
        # 前向传播
        with torch.no_grad():
            _ = self.model(self.image_tensor.unsqueeze(0))
            
        # 移除钩子
        for hook in hooks:
            hook.remove()
            
        self.progress_updated.emit(100)
        return features
        
    def _gradcam_analysis(self):
        """GradCAM分析"""
        self.status_updated.emit("正在生成GradCAM...")
        
        gradients = {}
        activations = {}
        
        def forward_hook(module, input, output):
            activations['value'] = output
            
        def backward_hook(module, grad_input, grad_output):
            gradients['value'] = grad_output[0]
            
        # 找到最后一个卷积层
        last_conv_layer = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv_layer = module
                
        if last_conv_layer is None:
            raise ValueError("未找到卷积层")
            
        # 注册钩子
        forward_handle = last_conv_layer.register_forward_hook(forward_hook)
        backward_handle = last_conv_layer.register_backward_hook(backward_hook)
        
        try:
            # 前向传播
            self.model.eval()
            input_tensor = self.image_tensor.unsqueeze(0)
            input_tensor.requires_grad_()
            
            output = self.model(input_tensor)
            
            # 反向传播
            self.model.zero_grad()
            output[0, self.target_class].backward(retain_graph=True)
            
            # 计算GradCAM
            grads = gradients['value']
            acts = activations['value']
            
            weights = torch.mean(grads, dim=(2, 3))
            gradcam = torch.zeros(acts.shape[2:])
            
            for i in range(acts.shape[1]):
                gradcam += weights[0, i] * acts[0, i]
                
            gradcam = torch.relu(gradcam)
            gradcam = gradcam / torch.max(gradcam)
            
            self.progress_updated.emit(100)
            return gradcam.cpu().numpy()
            
        finally:
            forward_handle.remove()
            backward_handle.remove()
            
    def _lime_analysis(self):
        """LIME分析"""
        self.status_updated.emit("正在进行LIME分析...")
        
        # 获取参数
        num_superpixels = self.analysis_params.get('num_superpixels', 100)
        num_samples = self.analysis_params.get('num_samples', 1000)
        
        def predict_fn(images):
            """预测函数"""
            batch_predictions = []
            for img in images:
                # 转换为tensor
                img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)
                
                with torch.no_grad():
                    pred = self.model(img_tensor)
                    pred = torch.softmax(pred, dim=1)
                    
                batch_predictions.append(pred.cpu().numpy()[0])
                
            return np.array(batch_predictions)
            
        # 创建LIME解释器
        explainer = lime_image.LimeImageExplainer()
        
        # 转换图像
        image_array = np.array(self.image)
        
        # 生成解释
        explanation = explainer.explain_instance(
            image_array,
            predict_fn,
            top_labels=len(self.class_names),
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=SegmentationAlgorithm('quickshift', kernel_size=4,
                                                max_dist=200, ratio=0.2)
        )
        
        self.progress_updated.emit(100)
        return explanation
        
    def _sensitivity_analysis(self):
        """敏感性分析"""
        self.status_updated.emit("正在进行敏感性分析...")
        
        # 获取参数
        perturbation_range = self.analysis_params.get('perturbation_range', 0.1)
        num_steps = self.analysis_params.get('num_steps', 20)
        
        # 生成扰动
        epsilons = np.linspace(0, perturbation_range, num_steps)
        predictions = []
        
        original_tensor = self.image_tensor.unsqueeze(0)
        
        with torch.no_grad():
            original_pred = self.model(original_tensor)
            original_confidence = torch.softmax(original_pred, dim=1)[0, self.target_class].item()
            
        for i, epsilon in enumerate(epsilons):
            # 添加随机噪声
            noise = torch.randn_like(original_tensor) * epsilon
            perturbed_tensor = original_tensor + noise
            
            with torch.no_grad():
                pred = self.model(perturbed_tensor)
                confidence = torch.softmax(pred, dim=1)[0, self.target_class].item()
                predictions.append(confidence)
                
            self.progress_updated.emit(int((i + 1) / num_steps * 100))
            
        return {
            'epsilons': epsilons,
            'predictions': predictions,
            'original_confidence': original_confidence
        }


class ModelAnalysisWidget(QWidget):
    """整合的模型分析组件"""
    
    status_updated = pyqtSignal(str)
    
    def __init__(self, parent=None):
        try:
            print("ModelAnalysisWidget: 开始初始化...")
            super().__init__(parent)
            self.model = None
            self.image = None
            self.image_tensor = None
            self.class_names = []
            self.model_file = None
            self.class_info_file = None
            
            print("ModelAnalysisWidget: 创建工作线程...")
            # 创建工作线程
            self.worker = ModelAnalysisWorker()
            self.worker.analysis_finished.connect(self.on_analysis_finished)
            self.worker.progress_updated.connect(self.on_progress_updated)
            self.worker.status_updated.connect(self.status_updated)
            self.worker.error_occurred.connect(self.on_error_occurred)
            
            # 存储当前结果用于resize时重新显示
            self.current_results = {}
            
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            
            print("ModelAnalysisWidget: 初始化UI...")
            self.init_ui()
            
            # 注册事件过滤器处理resize事件
            self.installEventFilter(self)
            
            print("ModelAnalysisWidget: 初始化完成")
            
        except Exception as e:
            print(f"ModelAnalysisWidget: 初始化失败 - {str(e)}")
            import traceback
            traceback.print_exc()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 说明标签
        info_label = QLabel(
            "模型分析工具集：一次加载模型和图片，选择需要的分析方法。\n"
            "支持特征可视化、GradCAM、LIME解释和敏感性分析。"
        )
        info_label.setWordWrap(True)
        info_label.setFont(QFont("Arial", 10))
        layout.addWidget(info_label)
        
        # 创建模型加载区域
        self.create_model_section(layout)
        
        # 创建图片选择区域
        self.create_image_section(layout)
        
        # 创建分析选择区域
        self.create_analysis_section(layout)
        
        # 创建结果显示区域
        self.create_results_section(layout)
        
    def create_model_section(self, parent_layout):
        """创建模型加载区域"""
        model_group = QGroupBox("模型配置")
        model_layout = QGridLayout()
        
        # 模型类型选择
        model_layout.addWidget(QLabel("模型类型:"), 0, 0)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["分类模型", "检测模型"])
        self.model_type_combo.currentIndexChanged.connect(self.switch_model_type)
        model_layout.addWidget(self.model_type_combo, 0, 1)
        
        # 模型架构选择
        model_layout.addWidget(QLabel("模型架构:"), 0, 2)
        self.model_arch_combo = QComboBox()
        self.model_arch_combo.addItems([
            "MobileNetV2", "MobileNetV3", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
            "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4",
            "VGG16", "VGG19", "DenseNet121", "DenseNet169", "DenseNet201", "InceptionV3", "Xception"
        ])
        model_layout.addWidget(self.model_arch_combo, 0, 3)
        
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
        parent_layout.addWidget(model_group)
        
    def create_image_section(self, parent_layout):
        """创建图片选择区域"""
        image_group = QGroupBox("图片配置")
        image_layout = QHBoxLayout()
        
        # 图片选择按钮
        self.select_image_btn = QPushButton("选择图片")
        self.select_image_btn.clicked.connect(self.select_image)
        image_layout.addWidget(self.select_image_btn)
        
        # 显示选中的图片路径
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setReadOnly(True)
        self.image_path_edit.setPlaceholderText("请选择要分析的图片")
        image_layout.addWidget(self.image_path_edit)
        
        # 原始图片显示
        self.original_image_label = QLabel("原始图片")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(200, 200)
        self.original_image_label.setStyleSheet("border: 1px solid gray;")
        image_layout.addWidget(self.original_image_label)
        
        image_group.setLayout(image_layout)
        parent_layout.addWidget(image_group)
        
    def create_analysis_section(self, parent_layout):
        """创建分析选择区域"""
        analysis_group = QGroupBox("分析配置")
        analysis_layout = QVBoxLayout()
        
        # 分析方法选择
        methods_layout = QHBoxLayout()
        methods_layout.addWidget(QLabel("选择分析方法:"))
        
        self.feature_checkbox = QCheckBox("特征可视化")
        self.gradcam_checkbox = QCheckBox("GradCAM")
        self.lime_checkbox = QCheckBox("LIME解释")
        self.sensitivity_checkbox = QCheckBox("敏感性分析")
        
        methods_layout.addWidget(self.feature_checkbox)
        methods_layout.addWidget(self.gradcam_checkbox)
        methods_layout.addWidget(self.lime_checkbox)
        methods_layout.addWidget(self.sensitivity_checkbox)
        methods_layout.addStretch()
        
        analysis_layout.addLayout(methods_layout)
        
        # 公共参数
        common_layout = QHBoxLayout()
        
        # 目标类别选择
        common_layout.addWidget(QLabel("目标类别:"))
        self.class_combo = QComboBox()
        common_layout.addWidget(self.class_combo)
        
        common_layout.addStretch()
        analysis_layout.addLayout(common_layout)
        
        # 分析参数设置
        params_layout = QFormLayout()
        
        # LIME参数
        self.num_superpixels = QSpinBox()
        self.num_superpixels.setRange(10, 1000)
        self.num_superpixels.setValue(100)
        params_layout.addRow("LIME超像素数量:", self.num_superpixels)
        
        self.num_samples = QSpinBox()
        self.num_samples.setRange(100, 10000)
        self.num_samples.setValue(1000)
        params_layout.addRow("LIME样本数量:", self.num_samples)
        
        # 敏感性分析参数
        self.perturbation_range = QDoubleSpinBox()
        self.perturbation_range.setRange(0.01, 1.0)
        self.perturbation_range.setValue(0.1)
        self.perturbation_range.setSingleStep(0.01)
        params_layout.addRow("敏感性扰动范围:", self.perturbation_range)
        
        self.num_steps = QSpinBox()
        self.num_steps.setRange(10, 100)
        self.num_steps.setValue(20)
        params_layout.addRow("敏感性扰动步数:", self.num_steps)
        
        analysis_layout.addLayout(params_layout)
        
        # 开始分析按钮
        button_layout = QHBoxLayout()
        self.start_analysis_btn = QPushButton("开始分析")
        self.start_analysis_btn.clicked.connect(self.start_analysis)
        self.start_analysis_btn.setEnabled(False)
        button_layout.addWidget(self.start_analysis_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        button_layout.addWidget(self.progress_bar)
        
        analysis_layout.addLayout(button_layout)
        
        analysis_group.setLayout(analysis_layout)
        parent_layout.addWidget(analysis_group)
        
    def create_results_section(self, parent_layout):
        """创建结果显示区域"""
        results_group = QGroupBox("分析结果")
        results_layout = QVBoxLayout()
        
        # 创建标签页widget
        self.results_tabs = QTabWidget()
        
        # 特征可视化结果页
        self.feature_scroll = QScrollArea()
        self.feature_label = QLabel("特征可视化结果将在这里显示")
        self.feature_label.setAlignment(Qt.AlignCenter)
        self.feature_label.setMinimumSize(400, 300)
        self.feature_scroll.setWidget(self.feature_label)
        self.feature_scroll.setWidgetResizable(True)
        self.results_tabs.addTab(self.feature_scroll, "特征可视化")
        
        # GradCAM结果页
        self.gradcam_label = QLabel("GradCAM结果将在这里显示")
        self.gradcam_label.setAlignment(Qt.AlignCenter)
        self.gradcam_label.setMinimumSize(400, 300)
        self.results_tabs.addTab(self.gradcam_label, "GradCAM")
        
        # LIME结果页
        self.lime_label = QLabel("LIME解释结果将在这里显示")
        self.lime_label.setAlignment(Qt.AlignCenter)
        self.lime_label.setMinimumSize(400, 300)
        self.results_tabs.addTab(self.lime_label, "LIME解释")
        
        # 敏感性分析结果页
        self.sensitivity_label = QLabel("敏感性分析结果将在这里显示")
        self.sensitivity_label.setAlignment(Qt.AlignCenter)
        self.sensitivity_label.setMinimumSize(400, 300)
        self.results_tabs.addTab(self.sensitivity_label, "敏感性分析")
        
        results_layout.addWidget(self.results_tabs)
        results_group.setLayout(results_layout)
        parent_layout.addWidget(results_group)
        
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
            self.check_ready_state()
    
    def select_class_info_file(self):
        """选择类别信息文件"""
        file, _ = QFileDialog.getOpenFileName(self, "选择类别信息文件", "", "JSON文件 (*.json);;所有文件 (*)")
        if file:
            self.class_info_file = file
            self.class_info_path_edit.setText(file)
            self.check_ready_state()
    
    def select_image(self):
        """选择图片"""
        file, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)")
        if file:
            try:
                # 加载图片
                self.image = Image.open(file).convert('RGB')
                self.image_path_edit.setText(file)
                
                # 显示图片
                self.display_image(self.image, self.original_image_label)
                
                # 转换为tensor
                image_array = np.array(self.image)
                image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float() / 255.0
                # 标准化 (假设使用ImageNet预训练模型)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                self.image_tensor = (image_tensor - mean) / std
                
                self.check_ready_state()
                
            except Exception as e:
                QMessageBox.warning(self, "错误", f"无法加载图片: {str(e)}")
    
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
                        class_info = json.load(f)
                        self.class_names = class_info['class_names']
                        
                    # 更新类别下拉框
                    self.class_combo.clear()
                    self.class_combo.addItems(self.class_names)
                    
                    QMessageBox.information(self, "成功", "模型加载成功!")
                    self.check_ready_state()
                    
                except Exception as e:
                    QMessageBox.warning(self, "错误", f"无法加载类别信息: {str(e)}")
            else:
                QMessageBox.warning(self, "错误", "无法找到主窗口预测器，请重启应用程序")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
            
    def check_ready_state(self):
        """检查是否准备就绪"""
        model_ready = bool(self.model_file and self.class_info_file)
        self.load_model_btn.setEnabled(model_ready)
        
        analysis_ready = bool(self.model and self.image)
        self.start_analysis_btn.setEnabled(analysis_ready)
    
    def start_analysis(self):
        """开始分析"""
        if not self.model or not self.image:
            QMessageBox.warning(self, "警告", "请先加载模型和选择图片!")
            return
            
        # 检查是否选择了至少一种分析方法
        selected_methods = []
        if self.feature_checkbox.isChecked():
            selected_methods.append("特征可视化")
        if self.gradcam_checkbox.isChecked():
            selected_methods.append("GradCAM")
        if self.lime_checkbox.isChecked():
            selected_methods.append("LIME解释")
        if self.sensitivity_checkbox.isChecked():
            selected_methods.append("敏感性分析")
            
        if not selected_methods:
            QMessageBox.warning(self, "警告", "请至少选择一种分析方法!")
            return
            
        # 获取目标类别
        target_class = self.class_combo.currentIndex()
        
        # 准备分析参数
        analysis_params = {
            'num_superpixels': self.num_superpixels.value(),
            'num_samples': self.num_samples.value(),
            'perturbation_range': self.perturbation_range.value(),
            'num_steps': self.num_steps.value()
        }
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.start_analysis_btn.setEnabled(False)
        
        # 逐个执行分析
        self.current_methods = selected_methods.copy()
        self.execute_next_analysis(target_class, analysis_params)
    
    def execute_next_analysis(self, target_class, analysis_params):
        """执行下一个分析"""
        if not self.current_methods:
            # 所有分析完成
            self.progress_bar.setVisible(False)
            self.start_analysis_btn.setEnabled(True)
            QMessageBox.information(self, "完成", "所有分析已完成!")
            return
            
        # 获取下一个分析方法
        analysis_type = self.current_methods.pop(0)
        
        # 设置工作线程任务
        self.worker.set_analysis_task(
            analysis_type, self.model, self.image, self.image_tensor,
            self.class_names, target_class, analysis_params
        )
        
        # 启动分析
        self.worker.start()
    
    @pyqtSlot(str, object)
    def on_analysis_finished(self, analysis_type, result):
        """处理分析完成事件"""
        try:
            # 保存结果用于重新显示
            self.current_results[analysis_type] = result
            
            if analysis_type == "特征可视化":
                self.display_feature_visualization(result)
            elif analysis_type == "GradCAM":
                self.display_gradcam(result)
            elif analysis_type == "LIME解释":
                self.display_lime_explanation(result)
            elif analysis_type == "敏感性分析":
                self.display_sensitivity_analysis(result)
                
            # 继续下一个分析
            target_class = self.class_combo.currentIndex()
            analysis_params = {
                'num_superpixels': self.num_superpixels.value(),
                'num_samples': self.num_samples.value(),
                'perturbation_range': self.perturbation_range.value(),
                'num_steps': self.num_steps.value()
            }
            self.execute_next_analysis(target_class, analysis_params)
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"显示{analysis_type}结果失败: {str(e)}")
    
    @pyqtSlot(int)
    def on_progress_updated(self, progress):
        """更新进度"""
        self.progress_bar.setValue(progress)
    
    @pyqtSlot(str)
    def on_error_occurred(self, error_msg):
        """处理错误"""
        QMessageBox.critical(self, "分析错误", error_msg)
        self.progress_bar.setVisible(False)
        self.start_analysis_btn.setEnabled(True)
    
    def display_image(self, image, label):
        """显示图片"""
        try:
            # 获取label的大小，用于自适应缩放
            label_size = label.size()
            
            # 确保有合理的最小尺寸，避免label未初始化时的问题
            if label_size.width() > 50 and label_size.height() > 50:
                max_width = max(200, label_size.width() - 20)
                max_height = max(200, label_size.height() - 20)
            else:
                # 使用默认尺寸
                max_width = 300
                max_height = 300
            
            # 保持比例调整图片大小
            original_width, original_height = image.size
            scale = min(max_width / original_width, max_height / original_height)
            
            # 确保缩放后的尺寸至少为1像素
            new_width = max(1, int(original_width * scale))
            new_height = max(1, int(original_height * scale))
            
            image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 转换为QPixmap
            image_array = np.array(image_resized)
            height, width, channel = image_array.shape
            bytes_per_line = 3 * width
            
            q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            label.setPixmap(pixmap)
            
        except Exception as e:
            self.logger.error(f"显示图片失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def display_feature_visualization(self, features):
        """显示特征可视化结果"""
        try:
            # 创建特征图网格
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            fig.suptitle('特征可视化')
            
            feature_items = list(features.items())
            for i, (module, feature) in enumerate(feature_items[:16]):
                row, col = i // 4, i % 4
                
                # 取第一个特征图
                feature_map = feature[0, 0].cpu().numpy()
                
                axes[row, col].imshow(feature_map, cmap='viridis')
                axes[row, col].set_title(f'Layer {i+1}')
                axes[row, col].axis('off')
            
            # 隐藏多余的子图
            for i in range(len(feature_items), 16):
                row, col = i // 4, i % 4
                axes[row, col].axis('off')
            
            plt.tight_layout()
            
            # 转换为QPixmap并显示
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            
            # 直接从buffer创建QPixmap，避免数据转换问题
            pixmap = QPixmap()
            if not pixmap.loadFromData(buffer.getvalue()):
                self.logger.error("无法加载特征可视化图像数据")
                return
            
            # 获取可用空间进行自适应缩放
            # 优先使用标签页的尺寸，如果不可用则使用默认值
            tab_size = self.results_tabs.size()
            if tab_size.width() > 200 and tab_size.height() > 200:
                max_width = tab_size.width() - 40  # 减去边距和滚动条
                max_height = tab_size.height() - 80  # 减去标签页头部和边距
            else:
                max_width, max_height = 800, 600
            
            # 确保最小尺寸
            max_width = max(600, max_width)
            max_height = max(450, max_height)
            
            scaled_pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.feature_label.setPixmap(scaled_pixmap)
            
            # 调整label大小以适应图片
            self.feature_label.resize(scaled_pixmap.size())
            
            plt.close()
            buffer.close()
            
        except Exception as e:
            self.logger.error(f"显示特征可视化失败: {str(e)}")
    
    def display_gradcam(self, gradcam):
        """显示GradCAM结果"""
        try:
            # 调整GradCAM到原图大小
            gradcam_resized = cv2.resize(gradcam, (self.image.width, self.image.height))
            
            # 创建热力图
            plt.figure(figsize=(12, 6))
            
            # 原图
            plt.subplot(1, 2, 1)
            plt.imshow(self.image)
            plt.title('原始图片')
            plt.axis('off')
            
            # GradCAM叠加
            plt.subplot(1, 2, 2)
            plt.imshow(self.image)
            plt.imshow(gradcam_resized, alpha=0.4, cmap='jet')
            plt.title(f'GradCAM - {self.class_names[self.class_combo.currentIndex()]}')
            plt.axis('off')
            
            plt.tight_layout()
            
            # 转换为QPixmap并显示
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            
            # 直接从buffer创建QPixmap，避免数据转换问题
            pixmap = QPixmap()
            if not pixmap.loadFromData(buffer.getvalue()):
                self.logger.error("无法加载GradCAM图像数据")
                return
            
            # 获取可用空间进行自适应缩放
            tab_size = self.results_tabs.size()
            if tab_size.width() > 200 and tab_size.height() > 200:
                max_width = tab_size.width() - 40
                max_height = tab_size.height() - 80
            else:
                max_width, max_height = 800, 400
            
            # 确保最小尺寸
            max_width = max(600, max_width)
            max_height = max(300, max_height)
            
            self.gradcam_label.setPixmap(pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            plt.close()
            buffer.close()
            
        except Exception as e:
            self.logger.error(f"显示GradCAM失败: {str(e)}")
    
    def display_lime_explanation(self, explanation):
        """显示LIME解释结果"""
        try:
            # 获取解释图像
            temp, mask = explanation.get_image_and_mask(
                self.class_combo.currentIndex(), 
                positive_only=True, 
                num_features=5, 
                hide_rest=False
            )
            
            # 创建图像显示
            plt.figure(figsize=(12, 6))
            
            # 原图
            plt.subplot(1, 2, 1)
            plt.imshow(self.image)
            plt.title('原始图片')
            plt.axis('off')
            
            # LIME解释
            plt.subplot(1, 2, 2)
            plt.imshow(temp)
            plt.title(f'LIME解释 - {self.class_names[self.class_combo.currentIndex()]}')
            plt.axis('off')
            
            plt.tight_layout()
            
            # 转换为QPixmap并显示
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            
            # 直接从buffer创建QPixmap，避免数据转换问题
            pixmap = QPixmap()
            if not pixmap.loadFromData(buffer.getvalue()):
                self.logger.error("无法加载LIME图像数据")
                return
            
            # 获取可用空间进行自适应缩放
            tab_size = self.results_tabs.size()
            if tab_size.width() > 200 and tab_size.height() > 200:
                max_width = tab_size.width() - 40
                max_height = tab_size.height() - 80
            else:
                max_width, max_height = 800, 400
            
            # 确保最小尺寸
            max_width = max(600, max_width)
            max_height = max(300, max_height)
            
            self.lime_label.setPixmap(pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            plt.close()
            buffer.close()
            
        except Exception as e:
            self.logger.error(f"显示LIME解释失败: {str(e)}")
    
    def display_sensitivity_analysis(self, result):
        """显示敏感性分析结果"""
        try:
            epsilons = result['epsilons']
            predictions = result['predictions']
            original_confidence = result['original_confidence']
            
            # 创建敏感性曲线
            plt.figure(figsize=(10, 6))
            plt.plot(epsilons, predictions, 'b-', linewidth=2, label='预测置信度')
            plt.axhline(y=original_confidence, color='r', linestyle='--', label='原始置信度')
            plt.xlabel('扰动强度')
            plt.ylabel('预测置信度')
            plt.title(f'敏感性分析 - {self.class_names[self.class_combo.currentIndex()]}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 转换为QPixmap并显示
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            
            # 直接从buffer创建QPixmap，避免数据转换问题
            pixmap = QPixmap()
            if not pixmap.loadFromData(buffer.getvalue()):
                self.logger.error("无法加载敏感性分析图像数据")
                return
            
            # 获取可用空间进行自适应缩放
            tab_size = self.results_tabs.size()
            if tab_size.width() > 200 and tab_size.height() > 200:
                max_width = tab_size.width() - 40
                max_height = tab_size.height() - 80
            else:
                max_width, max_height = 800, 480
            
            # 确保最小尺寸
            max_width = max(600, max_width)
            max_height = max(360, max_height)
            
            self.sensitivity_label.setPixmap(pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            plt.close()
            buffer.close()
            
        except Exception as e:
            self.logger.error(f"显示敏感性分析失败: {str(e)}")
    
    def eventFilter(self, obj, event):
        """事件过滤器，处理resize事件"""
        if event.type() == event.Resize and obj == self:
            # 延迟重新显示图片，避免频繁更新
            QTimer.singleShot(200, self.refresh_image_displays)
        return super().eventFilter(obj, event)
    
    def refresh_image_displays(self):
        """刷新所有图片显示"""
        try:
            # 重新显示所有已有的分析结果
            for analysis_type, result in self.current_results.items():
                if analysis_type == "特征可视化":
                    self.display_feature_visualization(result)
                elif analysis_type == "GradCAM":
                    self.display_gradcam(result)
                elif analysis_type == "LIME解释":
                    self.display_lime_explanation(result)
                elif analysis_type == "敏感性分析":
                    self.display_sensitivity_analysis(result)
                    
            # 重新显示原始图片
            if self.image:
                self.display_image(self.image, self.original_image_label)
                
        except Exception as e:
            self.logger.error(f"刷新图片显示失败: {str(e)}")
    
    def set_model(self, model, class_names=None):
        """从外部设置模型"""
        self.model = model
        if class_names:
            self.class_names = class_names
            self.class_combo.clear()
            self.class_combo.addItems(self.class_names)
        self.check_ready_state() 