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
        
        # 阶段一功能组
        self.create_optimizer_advanced_group(main_layout)
        self.create_learning_rate_warmup_group(main_layout)
        self.create_advanced_scheduler_group(main_layout)
        self.create_label_smoothing_group(main_layout)
        
        # 阶段二新增功能组
        self.create_model_ema_group(main_layout)
        self.create_gradient_accumulation_group(main_layout)
        self.create_advanced_augmentation_group(main_layout)
        self.create_loss_scaling_group(main_layout)
        
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
    
    def create_model_ema_group(self, main_layout):
        """创建模型EMA参数组（阶段二新增）"""
        group = QGroupBox("模型EMA - 指数移动平均")
        group.setToolTip("模型权重的指数移动平均，提升模型稳定性和泛化能力")
        layout = QGridLayout(group)
        
        # 启用EMA
        self.model_ema_checkbox = QCheckBox("启用模型EMA")
        self.model_ema_checkbox.setToolTip("启用模型权重的指数移动平均，推荐用于提升模型稳定性")
        self.model_ema_checkbox.setChecked(False)
        layout.addWidget(self.model_ema_checkbox, 0, 0, 1, 2)
        
        # EMA衰减率
        ema_decay_label = QLabel("EMA衰减率:")
        ema_decay_label.setToolTip("EMA权重更新的衰减率，越接近1.0更新越平滑")
        layout.addWidget(ema_decay_label, 1, 0)
        self.model_ema_decay_spin = QDoubleSpinBox()
        self.model_ema_decay_spin.setRange(0.9, 0.9999)
        self.model_ema_decay_spin.setSingleStep(0.0001)
        self.model_ema_decay_spin.setDecimals(4)
        self.model_ema_decay_spin.setValue(0.9999)
        layout.addWidget(self.model_ema_decay_spin, 1, 1)
        
        main_layout.addWidget(group)
    
    def create_gradient_accumulation_group(self, main_layout):
        """创建梯度累积参数组（阶段二新增）"""
        group = QGroupBox("梯度累积")
        group.setToolTip("梯度累积可以模拟更大的批次大小，在GPU内存有限时很有用")
        layout = QGridLayout(group)
        
        # 梯度累积步数
        grad_accum_label = QLabel("累积步数:")
        grad_accum_label.setToolTip("每次优化器更新前累积的梯度步数，1表示不使用梯度累积")
        layout.addWidget(grad_accum_label, 0, 0)
        self.gradient_accumulation_steps_spin = QSpinBox()
        self.gradient_accumulation_steps_spin.setRange(1, 32)
        self.gradient_accumulation_steps_spin.setValue(1)
        layout.addWidget(self.gradient_accumulation_steps_spin, 0, 1)
        
        main_layout.addWidget(group)
    
    def create_advanced_augmentation_group(self, main_layout):
        """创建高级数据增强参数组（阶段二新增）"""
        group = QGroupBox("高级数据增强")
        group.setToolTip("高级数据增强技术：CutMix和MixUp，可提升模型泛化能力")
        layout = QGridLayout(group)
        
        # CutMix概率
        cutmix_prob_label = QLabel("CutMix概率:")
        cutmix_prob_label.setToolTip("CutMix数据增强的使用概率，0.0表示不使用")
        layout.addWidget(cutmix_prob_label, 0, 0)
        self.cutmix_prob_spin = QDoubleSpinBox()
        self.cutmix_prob_spin.setRange(0.0, 1.0)
        self.cutmix_prob_spin.setSingleStep(0.1)
        self.cutmix_prob_spin.setDecimals(1)
        self.cutmix_prob_spin.setValue(0.0)
        layout.addWidget(self.cutmix_prob_spin, 0, 1)
        
        # MixUp Alpha参数
        mixup_alpha_label = QLabel("MixUp Alpha:")
        mixup_alpha_label.setToolTip("MixUp数据增强的Alpha参数，0.0表示不使用MixUp")
        layout.addWidget(mixup_alpha_label, 1, 0)
        self.mixup_alpha_spin = QDoubleSpinBox()
        self.mixup_alpha_spin.setRange(0.0, 1.0)
        self.mixup_alpha_spin.setSingleStep(0.1)
        self.mixup_alpha_spin.setDecimals(1)
        self.mixup_alpha_spin.setValue(0.0)
        layout.addWidget(self.mixup_alpha_spin, 1, 1)
        
        main_layout.addWidget(group)
    
    def create_loss_scaling_group(self, main_layout):
        """创建损失缩放参数组（阶段二新增）"""
        group = QGroupBox("损失缩放")
        group.setToolTip("混合精度训练的损失缩放策略，防止梯度下溢")
        layout = QGridLayout(group)
        
        # 损失缩放策略
        loss_scale_label = QLabel("缩放策略:")
        loss_scale_label.setToolTip("损失缩放的策略：dynamic为动态缩放，static为固定缩放")
        layout.addWidget(loss_scale_label, 0, 0)
        self.loss_scale_combo = QComboBox()
        self.loss_scale_combo.addItems(['dynamic', 'static'])
        layout.addWidget(self.loss_scale_combo, 0, 1)
        
        # 静态缩放值
        static_scale_label = QLabel("静态缩放值:")
        static_scale_label.setToolTip("静态损失缩放的缩放因子，仅在静态缩放时生效")
        layout.addWidget(static_scale_label, 1, 0)
        self.static_loss_scale_spin = QDoubleSpinBox()
        self.static_loss_scale_spin.setRange(1.0, 65536.0)
        self.static_loss_scale_spin.setSingleStep(1.0)
        self.static_loss_scale_spin.setDecimals(0)
        self.static_loss_scale_spin.setValue(128.0)
        layout.addWidget(self.static_loss_scale_spin, 1, 1)
        
        main_layout.addWidget(group)
    
    def connect_signals(self):
        """连接信号"""
        # 阶段一信号
        self.beta1_spin.valueChanged.connect(self.params_changed)
        self.beta2_spin.valueChanged.connect(self.params_changed)
        self.momentum_spin.valueChanged.connect(self.params_changed)
        self.nesterov_checkbox.toggled.connect(self.params_changed)
        self.warmup_steps_spin.valueChanged.connect(self.params_changed)
        self.warmup_ratio_spin.valueChanged.connect(self.params_changed)
        self.warmup_method_combo.currentTextChanged.connect(self.params_changed)
        self.min_lr_spin.valueChanged.connect(self.params_changed)
        self.label_smoothing_spin.valueChanged.connect(self.params_changed)
        
        # 阶段二新增信号
        self.model_ema_checkbox.toggled.connect(self.params_changed)
        self.model_ema_decay_spin.valueChanged.connect(self.params_changed)
        self.gradient_accumulation_steps_spin.valueChanged.connect(self.params_changed)
        self.cutmix_prob_spin.valueChanged.connect(self.params_changed)
        self.mixup_alpha_spin.valueChanged.connect(self.params_changed)
        self.loss_scale_combo.currentTextChanged.connect(self.params_changed)
        self.static_loss_scale_spin.valueChanged.connect(self.params_changed)
    
    def get_config(self):
        """获取高级超参数配置"""
        config = {
            # 阶段一配置
            'beta1': self.beta1_spin.value(),
            'beta2': self.beta2_spin.value(),
            'momentum': self.momentum_spin.value(),
            'nesterov': self.nesterov_checkbox.isChecked(),
            'warmup_steps': self.warmup_steps_spin.value(),
            'warmup_ratio': self.warmup_ratio_spin.value(),
            'warmup_method': self.warmup_method_combo.currentText(),
            'min_lr': self.min_lr_spin.value(),
            'label_smoothing': self.label_smoothing_spin.value(),
            
            # 阶段二新增配置
            'model_ema': self.model_ema_checkbox.isChecked(),
            'model_ema_decay': self.model_ema_decay_spin.value(),
            'gradient_accumulation_steps': self.gradient_accumulation_steps_spin.value(),
            'cutmix_prob': self.cutmix_prob_spin.value(),
            'mixup_alpha': self.mixup_alpha_spin.value(),
            'loss_scale': self.loss_scale_combo.currentText(),
            'static_loss_scale': self.static_loss_scale_spin.value(),
        }
        return config
    
    def set_config(self, config):
        """设置高级超参数配置"""
        # 阶段一配置
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
        
        # 阶段二新增配置
        if 'model_ema' in config:
            self.model_ema_checkbox.setChecked(config['model_ema'])
        if 'model_ema_decay' in config:
            self.model_ema_decay_spin.setValue(config['model_ema_decay'])
        if 'gradient_accumulation_steps' in config:
            self.gradient_accumulation_steps_spin.setValue(config['gradient_accumulation_steps'])
        if 'cutmix_prob' in config:
            self.cutmix_prob_spin.setValue(config['cutmix_prob'])
        if 'mixup_alpha' in config:
            self.mixup_alpha_spin.setValue(config['mixup_alpha'])
        if 'loss_scale' in config:
            index = self.loss_scale_combo.findText(config['loss_scale'])
            if index >= 0:
                self.loss_scale_combo.setCurrentIndex(index)
        if 'static_loss_scale' in config:
            self.static_loss_scale_spin.setValue(config['static_loss_scale']) 