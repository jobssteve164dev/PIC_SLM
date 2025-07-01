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
        beta1_label.setToolTip("Adam优化器的一阶矩估计衰减率\n• 控制梯度的动量项，影响训练稳定性\n• 推荐值：0.9（默认）\n• 训练不稳定时可降至0.8-0.85\n• 较大值使训练更平滑但可能收敛慢")
        layout.addWidget(beta1_label, 0, 0)
        self.beta1_spin = QDoubleSpinBox()
        self.beta1_spin.setRange(0.01, 0.999)
        self.beta1_spin.setSingleStep(0.01)
        self.beta1_spin.setDecimals(3)
        self.beta1_spin.setValue(0.9)
        self.beta1_spin.setToolTip("Adam优化器的一阶矩估计衰减率\n• 控制梯度的动量项，影响训练稳定性\n• 推荐值：0.9（默认）\n• 训练不稳定时可降至0.8-0.85\n• 较大值使训练更平滑但可能收敛慢")
        layout.addWidget(self.beta1_spin, 0, 1)
        
        # Beta2
        beta2_label = QLabel("Beta2:")
        beta2_label.setToolTip("Adam优化器的二阶矩估计衰减率\n• 控制梯度平方的累积，影响自适应学习率\n• 推荐值：0.999（默认）\n• 稀疏梯度场景可降至0.99\n• 较小值对梯度变化更敏感，收敛可能更快")
        layout.addWidget(beta2_label, 1, 0)
        self.beta2_spin = QDoubleSpinBox()
        self.beta2_spin.setRange(0.9, 0.9999)
        self.beta2_spin.setSingleStep(0.001)
        self.beta2_spin.setDecimals(4)
        self.beta2_spin.setValue(0.999)
        self.beta2_spin.setToolTip("Adam优化器的二阶矩估计衰减率\n• 控制梯度平方的累积，影响自适应学习率\n• 推荐值：0.999（默认）\n• 稀疏梯度场景可降至0.99\n• 较小值对梯度变化更敏感，收敛可能更快")
        layout.addWidget(self.beta2_spin, 1, 1)
        
        # Momentum
        momentum_label = QLabel("Momentum:")
        momentum_label.setToolTip("SGD动量参数，加速收敛并减少振荡\n• 推荐值：0.9（经典设置）\n• 目标检测通常用0.937\n• 较大值（0.95-0.99）适合大数据集\n• 较小值（0.5-0.8）适合小数据集或不稳定训练")
        layout.addWidget(momentum_label, 2, 0)
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 0.99)
        self.momentum_spin.setSingleStep(0.01)
        self.momentum_spin.setDecimals(2)
        self.momentum_spin.setValue(0.9)
        self.momentum_spin.setToolTip("SGD动量参数，加速收敛并减少振荡\n• 推荐值：0.9（经典设置）\n• 目标检测通常用0.937\n• 较大值（0.95-0.99）适合大数据集\n• 较小值（0.5-0.8）适合小数据集或不稳定训练")
        layout.addWidget(self.momentum_spin, 2, 1)
        
        # Nesterov
        self.nesterov_checkbox = QCheckBox("启用Nesterov动量")
        self.nesterov_checkbox.setToolTip("Nesterov加速梯度（NAG）\n• 在动量方向上预先计算梯度，收敛更快\n• 推荐：SGD优化器默认启用\n• 特别适合凸优化问题\n• 可能在某些非凸问题上不稳定")
        layout.addWidget(self.nesterov_checkbox, 3, 0, 1, 2)
        
        main_layout.addWidget(group)
    
    def create_learning_rate_warmup_group(self, main_layout):
        """创建学习率预热参数组"""
        group = QGroupBox("学习率预热")
        layout = QGridLayout(group)
        
        # 预热步数
        warmup_steps_label = QLabel("预热步数:")
        warmup_steps_label.setToolTip("学习率预热的训练步数\n• 推荐：总步数的5-10%\n• 大模型或大批次训练建议使用\n• 典型值：1000-5000步\n• 0表示不使用预热\n• 有助于训练初期的稳定性")
        layout.addWidget(warmup_steps_label, 0, 0)
        self.warmup_steps_spin = QSpinBox()
        self.warmup_steps_spin.setRange(0, 50000)
        self.warmup_steps_spin.setValue(0)
        self.warmup_steps_spin.setToolTip("学习率预热的训练步数\n• 推荐：总步数的5-10%\n• 大模型或大批次训练建议使用\n• 典型值：1000-5000步\n• 0表示不使用预热\n• 有助于训练初期的稳定性")
        layout.addWidget(self.warmup_steps_spin, 0, 1)
        
        # 预热比例
        warmup_ratio_label = QLabel("预热比例:")
        warmup_ratio_label.setToolTip("预热步数占总训练步数的比例\n• 推荐值：0.05-0.1（5%-10%）\n• 与预热步数二选一设置\n• 较大数据集可用更小比例\n• 微调预训练模型时可用较小比例（0.01-0.03）")
        layout.addWidget(warmup_ratio_label, 1, 0)
        self.warmup_ratio_spin = QDoubleSpinBox()
        self.warmup_ratio_spin.setRange(0.0, 0.5)
        self.warmup_ratio_spin.setSingleStep(0.01)
        self.warmup_ratio_spin.setDecimals(2)
        self.warmup_ratio_spin.setValue(0.0)
        self.warmup_ratio_spin.setToolTip("预热步数占总训练步数的比例\n• 推荐值：0.05-0.1（5%-10%）\n• 与预热步数二选一设置\n• 较大数据集可用更小比例\n• 微调预训练模型时可用较小比例（0.01-0.03）")
        layout.addWidget(self.warmup_ratio_spin, 1, 1)
        
        # 预热方法
        warmup_method_label = QLabel("预热方法:")
        warmup_method_label.setToolTip("学习率预热的策略\n• linear：线性增长（推荐）\n• cosine：余弦增长，更平滑\n• linear适合大多数场景\n• cosine适合对训练稳定性要求高的场景")
        layout.addWidget(warmup_method_label, 2, 0)
        self.warmup_method_combo = QComboBox()
        self.warmup_method_combo.addItems(['linear', 'cosine'])
        self.warmup_method_combo.setToolTip("学习率预热的策略\n• linear：线性增长（推荐）\n• cosine：余弦增长，更平滑\n• linear适合大多数场景\n• cosine适合对训练稳定性要求高的场景")
        layout.addWidget(self.warmup_method_combo, 2, 1)
        
        main_layout.addWidget(group)
    
    def create_advanced_scheduler_group(self, main_layout):
        """创建高级学习率调度参数组"""
        group = QGroupBox("高级学习率调度")
        layout = QGridLayout(group)
        
        # 最小学习率
        min_lr_label = QLabel("最小学习率:")
        min_lr_label.setToolTip("学习率调度的最小值\n• 防止学习率过小导致训练停滞\n• 推荐值：初始学习率的1/100到1/1000\n• 典型值：1e-6到1e-5\n• CosineAnnealing调度器特别重要\n• 过小可能导致欠拟合")
        layout.addWidget(min_lr_label, 0, 0)
        self.min_lr_spin = QDoubleSpinBox()
        self.min_lr_spin.setRange(1e-10, 1e-3)
        self.min_lr_spin.setSingleStep(1e-7)
        self.min_lr_spin.setDecimals(10)
        self.min_lr_spin.setValue(1e-6)
        self.min_lr_spin.setToolTip("学习率调度的最小值\n• 防止学习率过小导致训练停滞\n• 推荐值：初始学习率的1/100到1/1000\n• 典型值：1e-6到1e-5\n• CosineAnnealing调度器特别重要\n• 过小可能导致欠拟合")
        layout.addWidget(self.min_lr_spin, 0, 1)
        
        main_layout.addWidget(group)
    
    def create_label_smoothing_group(self, main_layout):
        """创建标签平滑参数组"""
        group = QGroupBox("标签平滑")
        layout = QGridLayout(group)
        
        # 标签平滑系数
        label_smoothing_label = QLabel("平滑系数:")
        label_smoothing_label.setToolTip("标签平滑的强度，减少过拟合\n• 推荐值：0.1（分类）或0.0（不使用）\n• 目标检测通常不使用（0.0）\n• 较大数据集可用0.1-0.2\n• 较小数据集建议0.05-0.1\n• 过大会导致欠拟合")
        layout.addWidget(label_smoothing_label, 0, 0)
        self.label_smoothing_spin = QDoubleSpinBox()
        self.label_smoothing_spin.setRange(0.0, 0.4)
        self.label_smoothing_spin.setSingleStep(0.01)
        self.label_smoothing_spin.setDecimals(2)
        self.label_smoothing_spin.setValue(0.0)
        self.label_smoothing_spin.setToolTip("标签平滑的强度，减少过拟合\n• 推荐值：0.1（分类）或0.0（不使用）\n• 目标检测通常不使用（0.0）\n• 较大数据集可用0.1-0.2\n• 较小数据集建议0.05-0.1\n• 过大会导致欠拟合")
        layout.addWidget(self.label_smoothing_spin, 0, 1)
        
        main_layout.addWidget(group)
    
    def create_model_ema_group(self, main_layout):
        """创建模型EMA参数组（阶段二新增）"""
        group = QGroupBox("模型EMA - 指数移动平均")
        group.setToolTip("模型权重的指数移动平均，显著提升模型稳定性和泛化能力")
        layout = QGridLayout(group)
        
        # 启用EMA
        self.model_ema_checkbox = QCheckBox("启用模型EMA")
        self.model_ema_checkbox.setToolTip("启用模型权重的指数移动平均\n• 强烈推荐：通常提升1-3%精度\n• 几乎无额外计算成本\n• 特别适合目标检测和大模型\n• 产生更稳定的推理结果\n• YOLO、EfficientNet等都默认使用")
        self.model_ema_checkbox.setChecked(False)
        layout.addWidget(self.model_ema_checkbox, 0, 0, 1, 2)
        
        # EMA衰减率
        ema_decay_label = QLabel("EMA衰减率:")
        ema_decay_label.setToolTip("EMA权重更新的衰减率\n• 推荐值：0.9999（默认）\n• 大数据集：0.9999-0.99999\n• 小数据集：0.999-0.9999\n• 越接近1.0更新越平滑但响应越慢\n• 训练轮数少时可用较小值（0.999）")
        layout.addWidget(ema_decay_label, 1, 0)
        self.model_ema_decay_spin = QDoubleSpinBox()
        self.model_ema_decay_spin.setRange(0.9, 0.9999)
        self.model_ema_decay_spin.setSingleStep(0.0001)
        self.model_ema_decay_spin.setDecimals(4)
        self.model_ema_decay_spin.setValue(0.9999)
        self.model_ema_decay_spin.setToolTip("EMA权重更新的衰减率\n• 推荐值：0.9999（默认）\n• 大数据集：0.9999-0.99999\n• 小数据集：0.999-0.9999\n• 越接近1.0更新越平滑但响应越慢\n• 训练轮数少时可用较小值（0.999）")
        layout.addWidget(self.model_ema_decay_spin, 1, 1)
        
        main_layout.addWidget(group)
    
    def create_gradient_accumulation_group(self, main_layout):
        """创建梯度累积参数组（阶段二新增）"""
        group = QGroupBox("梯度累积")
        group.setToolTip("梯度累积：模拟更大批次训练，在GPU内存受限时非常有用")
        layout = QGridLayout(group)
        
        # 梯度累积步数
        grad_accum_label = QLabel("累积步数:")
        grad_accum_label.setToolTip("每次优化器更新前累积的梯度步数\n• 有效批次 = 原批次 × 累积步数\n• 推荐：2-8步（常用4步）\n• GPU内存不足时增大此值\n• 1表示不使用梯度累积\n• 过大可能影响批归一化效果")
        layout.addWidget(grad_accum_label, 0, 0)
        self.gradient_accumulation_steps_spin = QSpinBox()
        self.gradient_accumulation_steps_spin.setRange(1, 32)
        self.gradient_accumulation_steps_spin.setValue(1)
        self.gradient_accumulation_steps_spin.setToolTip("每次优化器更新前累积的梯度步数\n• 有效批次 = 原批次 × 累积步数\n• 推荐：2-8步（常用4步）\n• GPU内存不足时增大此值\n• 1表示不使用梯度累积\n• 过大可能影响批归一化效果")
        layout.addWidget(self.gradient_accumulation_steps_spin, 0, 1)
        
        main_layout.addWidget(group)
    
    def create_advanced_augmentation_group(self, main_layout):
        """创建高级数据增强参数组（阶段二新增）"""
        group = QGroupBox("高级数据增强")
        group.setToolTip("高级数据增强：CutMix和MixUp，显著提升模型泛化能力")
        layout = QGridLayout(group)
        
        # CutMix概率
        cutmix_prob_label = QLabel("CutMix概率:")
        cutmix_prob_label.setToolTip("CutMix数据增强的使用概率\n• 推荐值：0.5-1.0（分类）\n• 目标检测：0.0（不推荐使用）\n• 数据量少时建议使用\n• 与MixUp可以同时使用\n• 0.0表示不使用CutMix")
        layout.addWidget(cutmix_prob_label, 0, 0)
        self.cutmix_prob_spin = QDoubleSpinBox()
        self.cutmix_prob_spin.setRange(0.0, 1.0)
        self.cutmix_prob_spin.setSingleStep(0.1)
        self.cutmix_prob_spin.setDecimals(1)
        self.cutmix_prob_spin.setValue(0.0)
        self.cutmix_prob_spin.setToolTip("CutMix数据增强的使用概率\n• 推荐值：0.5-1.0（分类）\n• 目标检测：0.0（不推荐使用）\n• 数据量少时建议使用\n• 与MixUp可以同时使用\n• 0.0表示不使用CutMix")
        layout.addWidget(self.cutmix_prob_spin, 0, 1)
        
        # MixUp Alpha参数
        mixup_alpha_label = QLabel("MixUp Alpha:")
        mixup_alpha_label.setToolTip("MixUp数据增强的Alpha参数\n• 推荐值：0.2-0.4（分类）\n• 目标检测：0.0（不推荐使用）\n• 控制混合强度，越大混合越均匀\n• 数据不平衡时特别有效\n• 0.0表示不使用MixUp")
        layout.addWidget(mixup_alpha_label, 1, 0)
        self.mixup_alpha_spin = QDoubleSpinBox()
        self.mixup_alpha_spin.setRange(0.0, 1.0)
        self.mixup_alpha_spin.setSingleStep(0.1)
        self.mixup_alpha_spin.setDecimals(1)
        self.mixup_alpha_spin.setValue(0.0)
        self.mixup_alpha_spin.setToolTip("MixUp数据增强的Alpha参数\n• 推荐值：0.2-0.4（分类）\n• 目标检测：0.0（不推荐使用）\n• 控制混合强度，越大混合越均匀\n• 数据不平衡时特别有效\n• 0.0表示不使用MixUp")
        layout.addWidget(self.mixup_alpha_spin, 1, 1)
        
        main_layout.addWidget(group)
    
    def create_loss_scaling_group(self, main_layout):
        """创建损失缩放参数组（阶段二新增）"""
        group = QGroupBox("损失缩放")
        group.setToolTip("混合精度训练的损失缩放策略，防止梯度下溢问题")
        layout = QGridLayout(group)
        
        # 损失缩放策略
        loss_scale_label = QLabel("缩放策略:")
        loss_scale_label.setToolTip("损失缩放的策略选择\n• dynamic：动态调整（推荐）\n• static：固定缩放值\n• 动态缩放自动适应训练过程\n• 静态缩放需要手动调整\n• 大多数情况推荐使用动态缩放")
        layout.addWidget(loss_scale_label, 0, 0)
        self.loss_scale_combo = QComboBox()
        self.loss_scale_combo.addItems(['dynamic', 'static'])
        self.loss_scale_combo.setToolTip("损失缩放的策略选择\n• dynamic：动态调整（推荐）\n• static：固定缩放值\n• 动态缩放自动适应训练过程\n• 静态缩放需要手动调整\n• 大多数情况推荐使用动态缩放")
        layout.addWidget(self.loss_scale_combo, 0, 1)
        
        # 静态缩放值
        static_scale_label = QLabel("静态缩放值:")
        static_scale_label.setToolTip("静态损失缩放的缩放因子\n• 推荐值：128-512\n• 仅在选择静态缩放时生效\n• 过大可能导致梯度爆炸\n• 过小可能无法防止梯度下溢\n• 需要根据模型和数据调整")
        layout.addWidget(static_scale_label, 1, 0)
        self.static_loss_scale_spin = QDoubleSpinBox()
        self.static_loss_scale_spin.setRange(1.0, 65536.0)
        self.static_loss_scale_spin.setSingleStep(1.0)
        self.static_loss_scale_spin.setDecimals(0)
        self.static_loss_scale_spin.setValue(128.0)
        self.static_loss_scale_spin.setToolTip("静态损失缩放的缩放因子\n• 推荐值：128-512\n• 仅在选择静态缩放时生效\n• 过大可能导致梯度爆炸\n• 过小可能无法防止梯度下溢\n• 需要根据模型和数据调整")
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