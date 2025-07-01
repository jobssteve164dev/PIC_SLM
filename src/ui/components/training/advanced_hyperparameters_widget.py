"""高级超参数配置组件"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QLabel, QDoubleSpinBox, QSpinBox, QComboBox, 
                             QCheckBox, QGridLayout, QFrame)
from PyQt5.QtCore import pyqtSignal, Qt


class AdvancedHyperparametersWidget(QWidget):
    """高级超参数配置组件"""
    
    params_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.connect_signals()
    
    def setup_ui(self):
        """设置用户界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        
        # 创建各个参数组
        self.create_optimizer_advanced_group(main_layout)
        self.create_learning_rate_warmup_group(main_layout)
        self.create_advanced_scheduler_group(main_layout)
        self.create_label_smoothing_group(main_layout)
        
        # 添加弹性空间
        main_layout.addStretch()
    
    def create_optimizer_advanced_group(self, main_layout):
        """创建优化器高级参数组"""
        group = QGroupBox("优化器高级参数")
        layout = QGridLayout(group)
        
        # Beta1
        beta1_label = QLabel("Beta1:")
        beta1_label.setToolTip("Adam优化器的一阶矩估计衰减率，典型值: 0.9")
        layout.addWidget(beta1_label, 0, 0)
        self.beta1_spin = QDoubleSpinBox()
        self.beta1_spin.setRange(0.01, 0.999)
        self.beta1_spin.setSingleStep(0.01)
        self.beta1_spin.setDecimals(3)
        self.beta1_spin.setValue(0.9)
        layout.addWidget(self.beta1_spin, 0, 1)
        
        # Beta2
        beta2_label = QLabel("Beta2:")
        beta2_label.setToolTip("Adam优化器的二阶矩估计衰减率，典型值: 0.999")
        layout.addWidget(beta2_label, 1, 0)
        self.beta2_spin = QDoubleSpinBox()
        self.beta2_spin.setRange(0.9, 0.9999)
        self.beta2_spin.setSingleStep(0.001)
        self.beta2_spin.setDecimals(4)
        self.beta2_spin.setValue(0.999)
        layout.addWidget(self.beta2_spin, 1, 1)
        
        # Momentum
        momentum_label = QLabel("Momentum:")
        momentum_label.setToolTip("SGD动量参数，典型值: 0.9")
        layout.addWidget(momentum_label, 2, 0)
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 0.99)
        self.momentum_spin.setSingleStep(0.01)
        self.momentum_spin.setDecimals(2)
        self.momentum_spin.setValue(0.9)
        layout.addWidget(self.momentum_spin, 2, 1)
        
        # Nesterov
        self.nesterov_checkbox = QCheckBox("启用Nesterov动量")
        self.nesterov_checkbox.setToolTip("启用Nesterov动量，SGD推荐启用")
        layout.addWidget(self.nesterov_checkbox, 3, 0, 1, 2)
        
        main_layout.addWidget(group)
    
    def create_learning_rate_warmup_group(self, main_layout):
        """创建学习率预热参数组"""
        group = QGroupBox("学习率预热")
        layout = QGridLayout(group)
        
        # 预热步数
        warmup_steps_label = QLabel("预热步数:")
        warmup_steps_label.setToolTip("预热阶段的训练步数，0表示不使用预热")
        layout.addWidget(warmup_steps_label, 0, 0)
        self.warmup_steps_spin = QSpinBox()
        self.warmup_steps_spin.setRange(0, 50000)
        self.warmup_steps_spin.setValue(0)
        layout.addWidget(self.warmup_steps_spin, 0, 1)
        
        # 预热比例
        warmup_ratio_label = QLabel("预热比例:")
        warmup_ratio_label.setToolTip("预热步数占总训练步数的比例")
        layout.addWidget(warmup_ratio_label, 1, 0)
        self.warmup_ratio_spin = QDoubleSpinBox()
        self.warmup_ratio_spin.setRange(0.0, 0.5)
        self.warmup_ratio_spin.setSingleStep(0.01)
        self.warmup_ratio_spin.setDecimals(2)
        self.warmup_ratio_spin.setValue(0.0)
        layout.addWidget(self.warmup_ratio_spin, 1, 1)
        
        # 预热方法
        warmup_method_label = QLabel("预热方法:")
        layout.addWidget(warmup_method_label, 2, 0)
        self.warmup_method_combo = QComboBox()
        self.warmup_method_combo.addItems(['linear', 'cosine'])
        layout.addWidget(self.warmup_method_combo, 2, 1)
        
        main_layout.addWidget(group)
    
    def create_advanced_scheduler_group(self, main_layout):
        """创建高级学习率调度参数组"""
        group = QGroupBox("高级学习率调度")
        layout = QGridLayout(group)
        
        # 最小学习率
        min_lr_label = QLabel("最小学习率:")
        layout.addWidget(min_lr_label, 0, 0)
        self.min_lr_spin = QDoubleSpinBox()
        self.min_lr_spin.setRange(1e-10, 1e-3)
        self.min_lr_spin.setSingleStep(1e-7)
        self.min_lr_spin.setDecimals(10)
        self.min_lr_spin.setValue(1e-6)
        layout.addWidget(self.min_lr_spin, 0, 1)
        
        main_layout.addWidget(group)
    
    def create_label_smoothing_group(self, main_layout):
        """创建标签平滑参数组"""
        group = QGroupBox("标签平滑")
        layout = QGridLayout(group)
        
        # 标签平滑系数
        label_smoothing_label = QLabel("平滑系数:")
        label_smoothing_label.setToolTip("标签平滑的强度，0.0表示不使用标签平滑")
        layout.addWidget(label_smoothing_label, 0, 0)
        self.label_smoothing_spin = QDoubleSpinBox()
        self.label_smoothing_spin.setRange(0.0, 0.4)
        self.label_smoothing_spin.setSingleStep(0.01)
        self.label_smoothing_spin.setDecimals(2)
        self.label_smoothing_spin.setValue(0.0)
        layout.addWidget(self.label_smoothing_spin, 0, 1)
        
        main_layout.addWidget(group)
    
    def connect_signals(self):
        """连接信号"""
        self.beta1_spin.valueChanged.connect(self.params_changed)
        self.beta2_spin.valueChanged.connect(self.params_changed)
        self.momentum_spin.valueChanged.connect(self.params_changed)
        self.nesterov_checkbox.toggled.connect(self.params_changed)
        self.warmup_steps_spin.valueChanged.connect(self.params_changed)
        self.warmup_ratio_spin.valueChanged.connect(self.params_changed)
        self.warmup_method_combo.currentTextChanged.connect(self.params_changed)
        self.min_lr_spin.valueChanged.connect(self.params_changed)
        self.label_smoothing_spin.valueChanged.connect(self.params_changed)
    
    def get_config(self):
        """获取高级超参数配置"""
        return {
            'beta1': self.beta1_spin.value(),
            'beta2': self.beta2_spin.value(),
            'momentum': self.momentum_spin.value(),
            'nesterov': self.nesterov_checkbox.isChecked(),
            'warmup_steps': self.warmup_steps_spin.value(),
            'warmup_ratio': self.warmup_ratio_spin.value(),
            'warmup_method': self.warmup_method_combo.currentText(),
            'min_lr': self.min_lr_spin.value(),
            'label_smoothing': self.label_smoothing_spin.value(),
        }
    
    def set_config(self, config):
        """设置高级超参数配置"""
        if 'beta1' in config:
            self.beta1_spin.setValue(config['beta1'])
        if 'beta2' in config:
            self.beta2_spin.setValue(config['beta2'])
        if 'momentum' in config:
            self.momentum_spin.setValue(config['momentum'])
        if 'nesterov' in config:
            self.nesterov_checkbox.setChecked(config['nesterov'])
        if 'warmup_steps' in config:
            self.warmup_steps_spin.setValue(config['warmup_steps'])
        if 'warmup_ratio' in config:
            self.warmup_ratio_spin.setValue(config['warmup_ratio'])
        if 'warmup_method' in config:
            index = self.warmup_method_combo.findText(config['warmup_method'])
            if index >= 0:
                self.warmup_method_combo.setCurrentIndex(index)
        if 'min_lr' in config:
            self.min_lr_spin.setValue(config['min_lr'])
        if 'label_smoothing' in config:
            self.label_smoothing_spin.setValue(config['label_smoothing']) 