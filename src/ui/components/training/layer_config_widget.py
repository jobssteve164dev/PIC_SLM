import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                           QLabel, QSpinBox, QDoubleSpinBox, QComboBox, 
                           QCheckBox, QPushButton, QListWidget, QScrollArea,
                           QMessageBox, QLineEdit, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal
from ...model_structure_editor import ModelStructureEditor

class LayerConfigWidget(QWidget):
    """模型层配置组件"""
    
    # 定义信号
    config_changed = pyqtSignal(dict)  # 当配置发生变化时发出信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_structure = None  # 存储模型结构
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        
        # 创建启用开关
        self.enable_checkbox = QCheckBox("启用自定义层配置")
        self.enable_checkbox.setChecked(False)
        self.enable_checkbox.stateChanged.connect(self.on_enable_changed)
        main_layout.addWidget(self.enable_checkbox)
        
        # 创建配置区域
        self.config_group = QGroupBox("层配置")
        self.config_layout = QVBoxLayout()
        
        # 基础网络配置
        basic_group = QGroupBox("基础网络配置")
        basic_layout = QGridLayout()
        
        # 网络深度
        basic_layout.addWidget(QLabel("网络深度:"), 0, 0)
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 200)
        self.depth_spin.setValue(50)
        basic_layout.addWidget(self.depth_spin, 0, 1)
        
        # 特征提取层数
        basic_layout.addWidget(QLabel("特征提取层数:"), 1, 0)
        self.feature_layers_spin = QSpinBox()
        self.feature_layers_spin.setRange(1, 100)
        self.feature_layers_spin.setValue(20)
        basic_layout.addWidget(self.feature_layers_spin, 1, 1)
        
        basic_group.setLayout(basic_layout)
        self.config_layout.addWidget(basic_group)
        
        # 卷积层配置
        conv_group = QGroupBox("卷积层配置")
        conv_layout = QGridLayout()
        
        # 卷积核大小
        conv_layout.addWidget(QLabel("卷积核大小:"), 0, 0)
        self.kernel_size_combo = QComboBox()
        self.kernel_size_combo.addItems(['3x3', '5x5', '7x7', '9x9'])
        conv_layout.addWidget(self.kernel_size_combo, 0, 1)
        
        # 卷积核数量
        conv_layout.addWidget(QLabel("卷积核数量:"), 1, 0)
        self.kernel_num_spin = QSpinBox()
        self.kernel_num_spin.setRange(16, 512)
        self.kernel_num_spin.setValue(64)
        conv_layout.addWidget(self.kernel_num_spin, 1, 1)
        
        conv_group.setLayout(conv_layout)
        self.config_layout.addWidget(conv_group)
        
        # 分类模型特有配置
        self.classification_group = QGroupBox("分类模型配置")
        class_layout = QGridLayout()
        
        # Backbone网络
        class_layout.addWidget(QLabel("Backbone网络:"), 0, 0)
        self.backbone_combo = QComboBox()
        self.backbone_combo.addItems([
            "MobileNetV2", "MobileNetV3", "ResNet18", "ResNet34", "ResNet50", 
            "ResNet101", "ResNet152", "EfficientNetB0", "EfficientNetB1", 
            "EfficientNetB2", "EfficientNetB3", "EfficientNetB4", "VGG16", 
            "VGG19", "DenseNet121", "DenseNet169", "DenseNet201", 
            "InceptionV3", "Xception"
        ])
        class_layout.addWidget(self.backbone_combo, 0, 1)
        
        # 全连接层数量
        class_layout.addWidget(QLabel("全连接层数量:"), 1, 0)
        self.fc_layers_spin = QSpinBox()
        self.fc_layers_spin.setRange(1, 10)
        self.fc_layers_spin.setValue(3)
        class_layout.addWidget(self.fc_layers_spin, 1, 1)
        
        self.classification_group.setLayout(class_layout)
        self.config_layout.addWidget(self.classification_group)
        
        # 检测模型特有配置
        self.detection_group = QGroupBox("检测模型配置")
        detect_layout = QGridLayout()
        
        # Anchor大小
        detect_layout.addWidget(QLabel("Anchor大小:"), 0, 0)
        self.anchor_combo = QComboBox()
        self.anchor_combo.addItems(['小', '中', '大', '自动'])
        detect_layout.addWidget(self.anchor_combo, 0, 1)
        
        # 特征金字塔层级
        detect_layout.addWidget(QLabel("特征金字塔层级:"), 1, 0)
        self.fpn_levels_spin = QSpinBox()
        self.fpn_levels_spin.setRange(3, 7)
        self.fpn_levels_spin.setValue(5)
        detect_layout.addWidget(self.fpn_levels_spin, 1, 1)
        
        # 检测头结构
        detect_layout.addWidget(QLabel("检测头结构:"), 2, 0)
        self.head_combo = QComboBox()
        self.head_combo.addItems(['单阶段', '两阶段', '密集头'])
        detect_layout.addWidget(self.head_combo, 2, 1)
        
        self.detection_group.setLayout(detect_layout)
        self.config_layout.addWidget(self.detection_group)
        
        # 高级配置
        advanced_group = QGroupBox("高级配置")
        advanced_layout = QGridLayout()
        
        # 跳跃连接
        self.skip_connection_cb = QCheckBox("启用跳跃连接")
        advanced_layout.addWidget(self.skip_connection_cb, 0, 0)
        
        # 自定义层结构
        self.custom_layers_cb = QCheckBox("启用自定义层结构")
        advanced_layout.addWidget(self.custom_layers_cb, 1, 0)
        
        # 添加编辑按钮
        self.edit_structure_btn = QPushButton("编辑模型结构")
        self.edit_structure_btn.clicked.connect(self.open_structure_editor)
        self.edit_structure_btn.setEnabled(False)  # 初始禁用
        advanced_layout.addWidget(self.edit_structure_btn, 1, 1)
        
        advanced_group.setLayout(advanced_layout)
        self.config_layout.addWidget(advanced_group)
        
        self.config_group.setLayout(self.config_layout)
        main_layout.addWidget(self.config_group)
        
        # 初始禁用配置组
        self.config_group.setEnabled(False)
        
        # 为所有控件添加值变更信号连接
        self.depth_spin.valueChanged.connect(self.emit_config)
        self.feature_layers_spin.valueChanged.connect(self.emit_config)
        self.kernel_size_combo.currentTextChanged.connect(self.emit_config)
        self.kernel_num_spin.valueChanged.connect(self.emit_config)
        self.backbone_combo.currentTextChanged.connect(self.emit_config)
        self.fc_layers_spin.valueChanged.connect(self.emit_config)
        self.anchor_combo.currentTextChanged.connect(self.emit_config)
        self.fpn_levels_spin.valueChanged.connect(self.emit_config)
        self.head_combo.currentTextChanged.connect(self.emit_config)
        self.skip_connection_cb.stateChanged.connect(self.emit_config)
        self.custom_layers_cb.stateChanged.connect(self.on_custom_layers_changed)
    
    def on_enable_changed(self, state):
        """启用状态改变时的处理"""
        enabled = state == Qt.Checked
        
        # 启用/禁用整个配置组
        self.config_group.setEnabled(enabled)
        
        # 启用/禁用所有子控件
        self.depth_spin.setEnabled(enabled)
        self.feature_layers_spin.setEnabled(enabled)
        self.kernel_size_combo.setEnabled(enabled)
        self.kernel_num_spin.setEnabled(enabled)
        self.skip_connection_cb.setEnabled(enabled)
        self.custom_layers_cb.setEnabled(enabled)
        
        # 启用/禁用分类模型特有控件
        if self.classification_group.isVisible():
            self.backbone_combo.setEnabled(enabled)
            self.fc_layers_spin.setEnabled(enabled)
        
        # 启用/禁用检测模型特有控件
        if self.detection_group.isVisible():
            self.anchor_combo.setEnabled(enabled)
            self.fpn_levels_spin.setEnabled(enabled)
            self.head_combo.setEnabled(enabled)
        
        # 发送配置变更信号
        self.emit_config()
    
    def on_custom_layers_changed(self, state):
        """自定义层结构启用状态改变时的处理"""
        enabled = state == Qt.Checked
        self.edit_structure_btn.setEnabled(enabled)
        
        # 如果禁用自定义层结构，清除已有的结构
        if not enabled:
            self.model_structure = None
            
        self.emit_config()
    
    def open_structure_editor(self):
        """打开模型结构编辑器"""
        editor = ModelStructureEditor(self)
        
        # 如果已有模型结构，加载到编辑器中
        if self.model_structure:
            editor.layers = self.model_structure['layers']
            editor.connections = self.model_structure['connections']
            editor.clear_all()  # 清除现有显示
            
            # 重新创建层部件
            for layer_info in editor.layers:
                layer_widget = editor.LayerWidget(layer_info)
                layer_widget.layer_selected.connect(editor.on_layer_selected)
                layer_widget.layer_modified.connect(editor.on_layer_modified)
                layer_widget.layer_deleted.connect(editor.on_layer_deleted)
                editor.layer_layout.addWidget(layer_widget)
                
            editor.update_connection_display()
        
        if editor.exec_() == editor.Accepted:
            self.model_structure = editor.get_model_structure()
            self.emit_config()
    
    def set_task_type(self, task_type):
        """设置任务类型，显示/隐藏相应的配置选项"""
        self.classification_group.setVisible(task_type == "classification")
        self.detection_group.setVisible(task_type == "detection")
        
    def get_config(self):
        """获取当前配置"""
        if not self.enable_checkbox.isChecked():
            return {
                "enabled": False,
                "network_depth": self.depth_spin.value(),
                "feature_layers": self.feature_layers_spin.value(),
                "kernel_size": self.kernel_size_combo.currentText(),
                "kernel_num": self.kernel_num_spin.value(),
                "skip_connection": self.skip_connection_cb.isChecked(),
                "custom_layers": self.custom_layers_cb.isChecked()
            }
            
        config = {
            "enabled": True,
            "network_depth": self.depth_spin.value(),
            "feature_layers": self.feature_layers_spin.value(),
            "kernel_size": self.kernel_size_combo.currentText(),
            "kernel_num": self.kernel_num_spin.value(),
            "skip_connection": self.skip_connection_cb.isChecked(),
            "custom_layers": self.custom_layers_cb.isChecked()
        }
        
        # 如果启用了自定义层结构，添加结构信息
        if self.custom_layers_cb.isChecked() and self.model_structure:
            config['model_structure'] = self.model_structure
        
        # 添加分类模型特有配置
        if self.classification_group.isVisible():
            config.update({
                "backbone": self.backbone_combo.currentText(),
                "fc_layers": self.fc_layers_spin.value()
            })
            
        # 添加检测模型特有配置
        if self.detection_group.isVisible():
            config.update({
                "anchor_size": self.anchor_combo.currentText(),
                "fpn_levels": self.fpn_levels_spin.value(),
                "head_structure": self.head_combo.currentText()
            })
            
        return config
        
    def emit_config(self):
        """发出配置变更信号"""
        config = self.get_config()
        if config is not None:  # 添加空值检查
            self.config_changed.emit(config) 