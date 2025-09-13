"""
训练配置验证器 - 负责验证训练配置的有效性

主要功能：
- 验证数据集路径是否存在
- 验证训练参数的合理性
- 验证模型配置的正确性
- 验证保存路径的可写性
- 检测超参数冲突并提供用户交互
"""

import os
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QCheckBox


class HyperparameterConflictDialog(QDialog):
    """超参数冲突对话框"""
    
    def __init__(self, conflicts, suggestions, parent=None):
        super().__init__(parent)
        self.conflicts = conflicts
        self.suggestions = suggestions
        self.user_choice = None
        self.setup_ui()
    
    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle("超参数冲突检测")
        self.setMinimumSize(600, 500)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel("⚠️ 检测到超参数配置冲突")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #d32f2f; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # 冲突描述
        conflicts_label = QLabel("发现以下参数冲突：")
        conflicts_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(conflicts_label)
        
        # 冲突列表
        conflicts_text = QTextEdit()
        conflicts_text.setMaximumHeight(150)
        conflicts_text.setReadOnly(True)
        conflicts_content = ""
        for i, conflict in enumerate(self.conflicts, 1):
            conflicts_content += f"{i}. {conflict['type']}: {conflict['description']}\n"
            conflicts_content += f"   影响: {conflict['impact']}\n\n"
        conflicts_text.setPlainText(conflicts_content)
        layout.addWidget(conflicts_text)
        
        # 建议修改
        suggestions_label = QLabel("建议的修改方案：")
        suggestions_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(suggestions_label)
        
        # 建议列表
        suggestions_text = QTextEdit()
        suggestions_text.setMaximumHeight(150)
        suggestions_text.setReadOnly(True)
        suggestions_content = ""
        for i, suggestion in enumerate(self.suggestions, 1):
            suggestions_content += f"{i}. {suggestion['parameter']}: {suggestion['action']}\n"
            suggestions_content += f"   原因: {suggestion['reason']}\n\n"
        suggestions_text.setPlainText(suggestions_content)
        layout.addWidget(suggestions_text)
        
        # 自动修复选项
        self.auto_fix_checkbox = QCheckBox("自动应用建议的修改（推荐）")
        self.auto_fix_checkbox.setChecked(True)
        layout.addWidget(self.auto_fix_checkbox)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.apply_button = QPushButton("应用修改并继续")
        self.apply_button.setStyleSheet("QPushButton { background-color: #4caf50; color: white; padding: 8px 16px; }")
        self.apply_button.clicked.connect(self.apply_changes)
        
        self.ignore_button = QPushButton("忽略冲突并继续")
        self.ignore_button.setStyleSheet("QPushButton { background-color: #ff9800; color: white; padding: 8px 16px; }")
        self.ignore_button.clicked.connect(self.ignore_conflicts)
        
        self.cancel_button = QPushButton("取消训练")
        self.cancel_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; padding: 8px 16px; }")
        self.cancel_button.clicked.connect(self.cancel_training)
        
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.ignore_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        # 警告信息
        warning_label = QLabel("⚠️ 忽略冲突可能导致训练不稳定或失败")
        warning_label.setStyleSheet("color: #ff5722; font-style: italic; margin-top: 10px;")
        layout.addWidget(warning_label)
    
    def apply_changes(self):
        """应用修改"""
        self.user_choice = 'apply'
        self.accept()
    
    def ignore_conflicts(self):
        """忽略冲突"""
        self.user_choice = 'ignore'
        self.accept()
    
    def cancel_training(self):
        """取消训练"""
        self.user_choice = 'cancel'
        self.reject()
    
    def should_auto_fix(self):
        """是否自动修复"""
        return self.auto_fix_checkbox.isChecked()


class TrainingValidator(QObject):
    """训练配置验证器"""
    
    # 信号定义
    status_updated = pyqtSignal(str)
    validation_error = pyqtSignal(str)
    conflict_detected = pyqtSignal(list, list)  # 冲突列表，建议列表
    training_cancelled = pyqtSignal()  # 训练被取消
    
    def __init__(self):
        super().__init__()
    
    def validate_config(self, config, parent_widget=None):
        """
        验证训练配置
        
        Args:
            config: 训练配置字典
            parent_widget: 父窗口，用于显示对话框
            
        Returns:
            tuple: (验证是否通过, 修改后的配置)
        """
        self.status_updated.emit("开始验证训练配置...")
        
        try:
            # 验证数据集路径
            if not self.validate_dataset_paths(config):
                return False, config
            
            # 验证训练参数
            if not self.validate_training_parameters(config):
                return False, config
            
            # 验证模型配置
            if not self.validate_model_config(config):
                return False, config
            
            # 验证保存路径
            if not self.validate_save_paths(config):
                return False, config
            
            # 检测超参数冲突
            conflicts, suggestions = self.detect_hyperparameter_conflicts(config)
            
            if conflicts:
                self.status_updated.emit(f"检测到 {len(conflicts)} 个超参数冲突")
                
                # 显示冲突对话框
                dialog = HyperparameterConflictDialog(conflicts, suggestions, parent_widget)
                result = dialog.exec_()
                
                if result == QDialog.Accepted:
                    if dialog.user_choice == 'apply':
                        # 应用建议的修改
                        if dialog.should_auto_fix():
                            modified_config = self.apply_conflict_fixes(config, suggestions)
                            self.status_updated.emit("已自动修复参数冲突")
                            return True, modified_config
                        else:
                            self.status_updated.emit("用户选择手动修改参数")
                            return False, config
                    elif dialog.user_choice == 'ignore':
                        self.status_updated.emit("用户选择忽略冲突，继续训练")
                        return True, config
                else:
                    # 用户取消训练
                    self.training_cancelled.emit()
                    self.status_updated.emit("用户取消训练")
                    return False, config
            
            self.status_updated.emit("配置验证通过")
            return True, config
            
        except Exception as e:
            self.validation_error.emit(f"配置验证时发生错误: {str(e)}")
            return False, config
    
    def detect_hyperparameter_conflicts(self, config):
        """
        检测超参数冲突
        
        Returns:
            tuple: (冲突列表, 建议列表)
        """
        conflicts = []
        suggestions = []
        
        # 检测优化器参数冲突
        optimizer = config.get('optimizer', 'Adam')
        beta1 = config.get('beta1', 0.9)
        beta2 = config.get('beta2', 0.999)
        momentum = config.get('momentum', 0.9)
        nesterov = config.get('nesterov', False)
        
        if optimizer == 'SGD' and (beta1 != 0.9 or beta2 != 0.999):
            conflicts.append({
                'type': '优化器参数不匹配',
                'description': f'SGD优化器不使用beta1/beta2参数，但检测到非默认值',
                'impact': '这些参数将被忽略，可能造成配置混淆'
            })
            suggestions.append({
                'parameter': 'beta1/beta2',
                'action': '重置为默认值 (beta1=0.9, beta2=0.999)',
                'reason': 'SGD优化器不使用这些参数'
            })
        
        if optimizer in ['Adam', 'AdamW'] and (momentum != 0.9 or nesterov):
            conflicts.append({
                'type': '优化器参数不匹配',
                'description': f'{optimizer}优化器不使用momentum/nesterov参数',
                'impact': '这些参数将被忽略，可能造成配置混淆'
            })
            suggestions.append({
                'parameter': 'momentum/nesterov',
                'action': '重置为默认值 (momentum=0.9, nesterov=False)',
                'reason': f'{optimizer}优化器不使用这些参数'
            })
        
        # 检测学习率预热冲突
        warmup_enabled = config.get('warmup_enabled', False)
        if warmup_enabled:
            warmup_steps = config.get('warmup_steps', 0)
            warmup_ratio = config.get('warmup_ratio', 0.0)
            
            if warmup_steps > 0 and warmup_ratio > 0:
                conflicts.append({
                    'type': '学习率预热参数冲突',
                    'description': '同时设置了warmup_steps和warmup_ratio',
                    'impact': 'warmup_steps将优先使用，warmup_ratio被忽略'
                })
                suggestions.append({
                    'parameter': 'warmup_ratio',
                    'action': '设置为0.0',
                    'reason': '避免参数歧义，优先使用warmup_steps'
                })
            
            # 检查预热参数是否合理
            if warmup_steps == 0 and warmup_ratio == 0.0:
                conflicts.append({
                    'type': '预热参数缺失',
                    'description': '启用了预热但未设置预热参数',
                    'impact': '预热功能不会生效'
                })
                suggestions.append({
                    'parameter': 'warmup_ratio',
                    'action': '设置为0.05',
                    'reason': '提供合理的预热比例'
                })
        
        # 检测数据增强冲突
        advanced_augmentation_enabled = config.get('advanced_augmentation_enabled', False)
        if advanced_augmentation_enabled:
            cutmix_prob = config.get('cutmix_prob', 0.0)
            mixup_alpha = config.get('mixup_alpha', 0.0)
            task_type = config.get('task_type', 'classification')
            
            if task_type == 'detection':
                conflicts.append({
                    'type': '任务类型与数据增强冲突',
                    'description': '目标检测任务不应使用CutMix/MixUp数据增强',
                    'impact': '会破坏边界框标注，导致训练失败'
                })
                suggestions.append({
                    'parameter': 'advanced_augmentation_enabled',
                    'action': '设置为False',
                    'reason': '目标检测任务不兼容这些数据增强方法'
                })
            
            if cutmix_prob > 0 and mixup_alpha > 0:
                conflicts.append({
                    'type': '数据增强过度',
                    'description': '同时启用CutMix和MixUp可能导致过度增强',
                    'impact': '可能影响训练稳定性和收敛速度'
                })
                suggestions.append({
                    'parameter': 'cutmix_prob或mixup_alpha',
                    'action': '建议只启用其中一种',
                    'reason': '避免过度数据增强'
                })
        
        # 检测梯度累积与批次大小冲突
        gradient_accumulation_enabled = config.get('gradient_accumulation_enabled', False)
        if gradient_accumulation_enabled:
            batch_size = config.get('batch_size', 32)
            gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
            
            effective_batch_size = batch_size * gradient_accumulation_steps
            if effective_batch_size > 512:
                conflicts.append({
                    'type': '有效批次过大',
                    'description': f'有效批次大小 = {batch_size} × {gradient_accumulation_steps} = {effective_batch_size}',
                    'impact': '可能导致内存不足或BatchNorm统计不准确'
                })
                suggestions.append({
                    'parameter': 'gradient_accumulation_steps',
                    'action': f'减少到 {512 // batch_size}',
                    'reason': '保持有效批次大小在合理范围内'
                })
        
        # 检测EMA与优化器动量冲突
        model_ema = config.get('model_ema', False)
        if model_ema and optimizer in ['Adam', 'AdamW'] and beta1 > 0.95:
            conflicts.append({
                'type': 'EMA与优化器动量冲突',
                'description': f'EMA与{optimizer}的高beta1值可能产生双重平滑效果',
                'impact': '可能导致收敛过慢或陷入局部最优'
            })
            suggestions.append({
                'parameter': 'beta1',
                'action': '降低到0.9',
                'reason': '避免与EMA的双重平滑效果'
            })
        
        # 检测标签平滑与任务类型冲突
        label_smoothing_enabled = config.get('label_smoothing_enabled', False)
        if label_smoothing_enabled:
            task_type = config.get('task_type', 'classification')
            if task_type == 'detection':
                conflicts.append({
                    'type': '标签平滑与任务类型冲突',
                    'description': '目标检测任务通常不使用标签平滑',
                    'impact': '可能影响检测精度'
                })
                suggestions.append({
                    'parameter': 'label_smoothing_enabled',
                    'action': '设置为False',
                    'reason': '目标检测任务不推荐使用标签平滑'
                })
        
        # 检测学习率调度器冲突
        lr_scheduler = config.get('lr_scheduler', 'StepLR')
        warmup_enabled = config.get('warmup_enabled', False)
        if lr_scheduler == 'OneCycleLR' and warmup_enabled:
            conflicts.append({
                'type': '学习率调度器冲突',
                'description': 'OneCycleLR已包含预热机制，不需要额外的warmup设置',
                'impact': '可能导致学习率调度混乱'
            })
            suggestions.append({
                'parameter': 'warmup_enabled',
                'action': '设置为False',
                'reason': 'OneCycleLR内置预热机制'
            })
        
        # 检测混合精度与损失缩放冲突
        mixed_precision = config.get('mixed_precision', True)
        loss_scaling_enabled = config.get('loss_scaling_enabled', False)
        if loss_scaling_enabled and not mixed_precision:
            conflicts.append({
                'type': '混合精度与损失缩放冲突',
                'description': '未启用混合精度时，损失缩放设置无效',
                'impact': '损失缩放参数被忽略'
            })
            suggestions.append({
                'parameter': 'loss_scaling_enabled',
                'action': '设置为False或启用mixed_precision',
                'reason': '损失缩放需要混合精度支持'
            })
        
        return conflicts, suggestions
    
    def apply_conflict_fixes(self, config, suggestions):
        """
        应用冲突修复建议
        
        Args:
            config: 原始配置
            suggestions: 修复建议列表
            
        Returns:
            dict: 修复后的配置
        """
        modified_config = config.copy()
        
        for suggestion in suggestions:
            parameter = suggestion['parameter']
            action = suggestion['action']
            
            if parameter == 'beta1/beta2':
                modified_config['beta1'] = 0.9
                modified_config['beta2'] = 0.999
            elif parameter == 'momentum/nesterov':
                modified_config['momentum'] = 0.9
                modified_config['nesterov'] = False
            elif parameter == 'warmup_ratio':
                if '设置为0.05' in action:
                    modified_config['warmup_ratio'] = 0.05
                else:
                    modified_config['warmup_ratio'] = 0.0
            elif parameter == 'warmup_enabled':
                modified_config['warmup_enabled'] = False
            elif parameter == 'advanced_augmentation_enabled':
                modified_config['advanced_augmentation_enabled'] = False
            elif parameter == 'cutmix_prob/mixup_alpha':
                modified_config['cutmix_prob'] = 0.0
                modified_config['mixup_alpha'] = 0.0
            elif parameter == 'cutmix_prob或mixup_alpha':
                # 保留其中一个，优先保留CutMix
                if modified_config.get('cutmix_prob', 0) > 0:
                    modified_config['mixup_alpha'] = 0.0
                else:
                    modified_config['cutmix_prob'] = 0.0
            elif parameter == 'gradient_accumulation_steps':
                batch_size = modified_config.get('batch_size', 32)
                modified_config['gradient_accumulation_steps'] = max(1, 512 // batch_size)
            elif parameter == 'beta1':
                modified_config['beta1'] = 0.9
            elif parameter == 'label_smoothing':
                modified_config['label_smoothing'] = 0.0
            elif parameter == 'label_smoothing_enabled':
                modified_config['label_smoothing_enabled'] = False
            elif parameter == 'warmup_steps/warmup_ratio':
                modified_config['warmup_steps'] = 0
                modified_config['warmup_ratio'] = 0.0
            elif parameter == 'loss_scale':
                modified_config['loss_scale'] = 'dynamic'
            elif parameter == 'loss_scaling_enabled':
                modified_config['loss_scaling_enabled'] = False
        
        return modified_config
    
    def validate_dataset_paths(self, config):
        """验证数据集路径"""
        data_dir = config.get('data_dir', '')
        
        if not data_dir:
            self.validation_error.emit("数据集路径不能为空")
            return False
        
        if not os.path.exists(data_dir):
            self.validation_error.emit(f"数据集路径不存在: {data_dir}")
            return False
        
        # 检查训练和验证数据集目录
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        
        if not os.path.exists(train_dir):
            self.validation_error.emit(f"训练数据集目录不存在: {train_dir}")
            return False
        
        if not os.path.exists(val_dir):
            self.validation_error.emit(f"验证数据集目录不存在: {val_dir}")
            return False
        
        # 检查数据集是否为空
        if not self._check_directory_has_data(train_dir):
            self.validation_error.emit("训练数据集目录为空")
            return False
        
        if not self._check_directory_has_data(val_dir):
            self.validation_error.emit("验证数据集目录为空")
            return False
        
        self.status_updated.emit("数据集路径验证通过")
        return True
    
    def validate_training_parameters(self, config):
        """验证训练参数"""
        # 验证num_epochs
        num_epochs = config.get('num_epochs', 20)
        if not isinstance(num_epochs, int) or num_epochs <= 0:
            self.validation_error.emit("训练轮数必须为正整数")
            return False
        
        if num_epochs > 1000:
            self.status_updated.emit("警告: 训练轮数过大，可能需要很长时间")
        
        # 验证batch_size
        batch_size = config.get('batch_size', 32)
        if not isinstance(batch_size, int) or batch_size <= 0:
            self.validation_error.emit("批次大小必须为正整数")
            return False
        
        if batch_size > 256:
            self.status_updated.emit("警告: 批次大小较大，可能会消耗大量内存")
        
        # 验证learning_rate
        learning_rate = config.get('learning_rate', 0.001)
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            self.validation_error.emit("学习率必须为正数")
            return False
        
        if learning_rate > 1.0:
            self.status_updated.emit("警告: 学习率较大，可能导致训练不稳定")
        elif learning_rate < 1e-6:
            self.status_updated.emit("警告: 学习率较小，可能导致训练缓慢")
        
        # 验证dropout_rate
        dropout_rate = config.get('dropout_rate', 0.0)
        if not isinstance(dropout_rate, (int, float)) or dropout_rate < 0 or dropout_rate >= 1:
            self.validation_error.emit("Dropout率必须在[0, 1)范围内")
            return False
        
        # 验证新增的高级超参数
        if not self.validate_advanced_hyperparameters(config):
            return False
        
        # 验证第二阶段新增超参数
        if not self.validate_stage_two_hyperparameters(config):
            return False
        
        self.status_updated.emit("训练参数验证通过")
        return True
    
    def validate_advanced_hyperparameters(self, config):
        """验证高级超参数（阶段一新增）"""
        # 验证优化器高级参数
        beta1 = config.get('beta1', 0.9)
        if not isinstance(beta1, (int, float)) or beta1 <= 0 or beta1 >= 1:
            self.validation_error.emit("Beta1参数必须在(0, 1)范围内")
            return False
        
        beta2 = config.get('beta2', 0.999)
        if not isinstance(beta2, (int, float)) or beta2 <= 0 or beta2 >= 1:
            self.validation_error.emit("Beta2参数必须在(0, 1)范围内")
            return False
        
        eps = config.get('eps', 1e-8)
        if not isinstance(eps, (int, float)) or eps <= 0:
            self.validation_error.emit("Eps参数必须为正数")
            return False
        
        momentum = config.get('momentum', 0.9)
        if not isinstance(momentum, (int, float)) or momentum < 0 or momentum >= 1:
            self.validation_error.emit("Momentum参数必须在[0, 1)范围内")
            return False
        
        # 验证学习率预热参数 - 只在启用时验证
        warmup_enabled = config.get('warmup_enabled', False)
        if warmup_enabled:
            warmup_steps = config.get('warmup_steps', 0)
            if not isinstance(warmup_steps, int) or warmup_steps < 0:
                self.validation_error.emit("预热步数必须为非负整数")
                return False
            
            warmup_ratio = config.get('warmup_ratio', 0.0)
            if not isinstance(warmup_ratio, (int, float)) or warmup_ratio < 0 or warmup_ratio > 0.5:
                self.validation_error.emit("预热比例必须在[0, 0.5]范围内")
                return False
            
            warmup_method = config.get('warmup_method', 'linear')
            if warmup_method not in ['linear', 'cosine']:
                self.validation_error.emit("预热方法必须是'linear'或'cosine'")
                return False
        
        # 验证最小学习率参数 - 只在启用时验证
        min_lr_enabled = config.get('min_lr_enabled', False)
        if min_lr_enabled:
            min_lr = config.get('min_lr', 1e-6)
            if not isinstance(min_lr, (int, float)) or min_lr <= 0:
                self.validation_error.emit("最小学习率必须为正数")
                return False
            
            learning_rate = config.get('learning_rate', 0.001)
            if min_lr >= learning_rate:
                self.validation_error.emit("最小学习率必须小于初始学习率")
                return False
        
        # 验证标签平滑参数 - 只在启用时验证
        label_smoothing_enabled = config.get('label_smoothing_enabled', False)
        if label_smoothing_enabled:
            label_smoothing = config.get('label_smoothing', 0.0)
            if not isinstance(label_smoothing, (int, float)) or label_smoothing < 0 or label_smoothing >= 0.5:
                self.validation_error.emit("标签平滑系数必须在[0, 0.5)范围内")
                return False
        
        # 验证优化器名称
        optimizer = config.get('optimizer', 'Adam')
        from .optimizer_factory import OptimizerFactory
        supported_optimizers = OptimizerFactory.get_supported_optimizers()
        if optimizer not in supported_optimizers:
            self.validation_error.emit(f"不支持的优化器: {optimizer}")
            return False
        
        # 验证学习率调度器名称
        lr_scheduler = config.get('lr_scheduler', 'StepLR')
        supported_schedulers = OptimizerFactory.get_supported_schedulers()
        if lr_scheduler not in supported_schedulers:
            self.validation_error.emit(f"不支持的学习率调度器: {lr_scheduler}")
            return False
        
        self.status_updated.emit("高级超参数验证通过")
        return True
    
    def validate_stage_two_hyperparameters(self, config):
        """验证第二阶段新增超参数"""
        # 验证模型EMA参数
        model_ema = config.get('model_ema', False)
        if isinstance(model_ema, bool):
            if model_ema:
                model_ema_decay = config.get('model_ema_decay', 0.9999)
                if not isinstance(model_ema_decay, (int, float)) or model_ema_decay < 0.9 or model_ema_decay >= 1.0:
                    self.validation_error.emit("EMA衰减率必须在[0.9, 1.0)范围内")
                    return False
        else:
            self.validation_error.emit("模型EMA设置必须为布尔值")
            return False
        
        # 验证梯度累积参数 - 只在启用时验证
        gradient_accumulation_enabled = config.get('gradient_accumulation_enabled', False)
        if gradient_accumulation_enabled:
            gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
            if not isinstance(gradient_accumulation_steps, int) or gradient_accumulation_steps < 1:
                self.validation_error.emit("梯度累积步数必须为正整数")
                return False
            
            if gradient_accumulation_steps > 32:
                self.status_updated.emit("警告: 梯度累积步数较大，可能会影响训练动态")
        
        # 验证高级数据增强参数 - 只在启用时验证
        advanced_augmentation_enabled = config.get('advanced_augmentation_enabled', False)
        if advanced_augmentation_enabled:
            cutmix_prob = config.get('cutmix_prob', 0.0)
            if not isinstance(cutmix_prob, (int, float)) or cutmix_prob < 0 or cutmix_prob > 1.0:
                self.validation_error.emit("CutMix概率必须在[0, 1]范围内")
                return False
            
            mixup_alpha = config.get('mixup_alpha', 0.0)
            if not isinstance(mixup_alpha, (int, float)) or mixup_alpha < 0 or mixup_alpha > 2.0:
                self.validation_error.emit("MixUp Alpha参数必须在[0, 2.0]范围内")
                return False
            
            # 兼容性检查
            if cutmix_prob > 0 and mixup_alpha > 0:
                self.status_updated.emit("警告: 同时启用CutMix和MixUp，将随机选择使用")
        
        # 验证损失缩放参数 - 只在启用时验证
        loss_scaling_enabled = config.get('loss_scaling_enabled', False)
        if loss_scaling_enabled:
            loss_scale = config.get('loss_scale', 'dynamic')
            if loss_scale == 'none':
                # 如果loss_scale为'none'但启用状态为True，这是矛盾的，重置为禁用
                self.status_updated.emit("警告: 损失缩放参数矛盾，已自动禁用")
            else:
                if loss_scale not in ['dynamic', 'static']:
                    self.validation_error.emit("损失缩放策略必须是'dynamic'或'static'")
                    return False
                
                if loss_scale == 'static':
                    static_loss_scale = config.get('static_loss_scale', 128.0)
                    if not isinstance(static_loss_scale, (int, float)) or static_loss_scale < 1.0:
                        self.validation_error.emit("静态损失缩放值必须大于等于1.0")
                        return False
                
                # 检查与混合精度的兼容性
                mixed_precision = config.get('mixed_precision', True)
                if not mixed_precision:
                    self.status_updated.emit("警告: 未启用混合精度时损失缩放可能无效")
        
        self.status_updated.emit("第二阶段超参数验证通过")
        return True
    
    def validate_model_config(self, config):
        """验证模型配置"""
        # 验证model_name
        model_name = config.get('model_name', '')
        if not model_name:
            self.validation_error.emit("模型名称不能为空")
            return False
        
        # 支持的模型列表
        supported_models = [
            'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
            'MobileNetV2', 'MobileNetV3',
            'VGG16', 'VGG19',
            'DenseNet121', 'DenseNet169', 'DenseNet201',
            'InceptionV3',
            'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4',
            'Xception'
        ]
        
        if model_name not in supported_models:
            self.validation_error.emit(f"不支持的模型: {model_name}")
            return False
        
        # 验证task_type
        task_type = config.get('task_type', 'classification')
        if task_type not in ['classification', 'detection']:
            self.validation_error.emit(f"不支持的任务类型: {task_type}")
            return False
        
        # 验证activation_function
        activation_function = config.get('activation_function')
        if activation_function:
            supported_activations = [
                'None', 'ReLU', 'LeakyReLU', 'PReLU', 'ELU', 'SELU', 'GELU', 'Mish', 'Swish', 'SiLU'
            ]
            if activation_function not in supported_activations:
                self.validation_error.emit(f"不支持的激活函数: {activation_function}")
                return False
        
        # 验证weight_strategy
        weight_strategy = config.get('weight_strategy', 'balanced')
        if weight_strategy not in ['balanced', 'inverse', 'log_inverse', 'custom']:
            self.validation_error.emit(f"不支持的权重策略: {weight_strategy}")
            return False
        
        self.status_updated.emit("模型配置验证通过")
        return True
    
    def validate_save_paths(self, config):
        """验证保存路径"""
        # 验证model_save_dir
        model_save_dir = config.get('model_save_dir', 'models/saved_models')
        
        # 标准化路径，处理相对路径引用
        try:
            model_save_dir = os.path.normpath(model_save_dir)
            # 如果是相对路径，转换为绝对路径
            if not os.path.isabs(model_save_dir):
                # 获取当前工作目录
                current_dir = os.getcwd()
                model_save_dir = os.path.join(current_dir, model_save_dir)
        except Exception as e:
            self.validation_error.emit(f"路径标准化失败: {model_save_dir}, 错误: {str(e)}")
            return False
        
        try:
            # 尝试创建目录
            os.makedirs(model_save_dir, exist_ok=True)
            
            # 检查目录是否可写（带清理机制）
            if not self._check_directory_writable_with_cleanup(model_save_dir):
                return False
                
        except Exception as e:
            self.validation_error.emit(f"无法创建模型保存目录: {model_save_dir}, 错误: {str(e)}")
            return False
        
        # 验证参数保存目录（如果指定）
        param_save_dir = config.get('default_param_save_dir')
        if param_save_dir:
            # 标准化路径，处理相对路径引用
            try:
                param_save_dir = os.path.normpath(param_save_dir)
                # 如果是相对路径，转换为绝对路径
                if not os.path.isabs(param_save_dir):
                    # 获取当前工作目录
                    current_dir = os.getcwd()
                    param_save_dir = os.path.join(current_dir, param_save_dir)
            except Exception as e:
                self.validation_error.emit(f"参数保存路径标准化失败: {param_save_dir}, 错误: {str(e)}")
                return False
            
            try:
                os.makedirs(param_save_dir, exist_ok=True)
                
                # 检查目录是否可写（带清理机制）
                if not self._check_directory_writable_with_cleanup(param_save_dir):
                    return False
                    
            except Exception as e:
                self.validation_error.emit(f"无法创建参数保存目录: {param_save_dir}, 错误: {str(e)}")
                return False
        
        self.status_updated.emit("保存路径验证通过")
        return True
    
    def _check_directory_has_data(self, directory):
        """检查目录是否包含数据"""
        try:
            # 检查是否有子目录（类别目录）
            subdirs = [d for d in os.listdir(directory) 
                      if os.path.isdir(os.path.join(directory, d))]
            
            if not subdirs:
                return False
            
            # 检查子目录中是否有文件
            for subdir in subdirs:
                subdir_path = os.path.join(directory, subdir)
                files = [f for f in os.listdir(subdir_path) 
                        if os.path.isfile(os.path.join(subdir_path, f))]
                
                # 过滤图像文件
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
                image_files = [f for f in files 
                              if any(f.lower().endswith(ext) for ext in image_extensions)]
                
                if image_files:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def validate_runtime_config(self, config, class_names):
        """
        验证运行时配置
        
        Args:
            config: 训练配置
            class_names: 类别名称列表
            
        Returns:
            bool: 验证是否通过
        """
        # 验证类别权重配置（如果使用自定义权重）
        if config.get('weight_strategy') == 'custom':
            return self._validate_custom_weights(config, class_names)
        
        return True
    
    def _validate_custom_weights(self, config, class_names):
        """验证自定义权重配置"""
        custom_weights = {}
        
        # 从配置中获取权重
        if 'class_weights' in config:
            custom_weights = config.get('class_weights', {})
        elif 'custom_class_weights' in config:
            custom_weights = config.get('custom_class_weights', {})
        elif 'weight_config_file' in config:
            weight_config_file = config.get('weight_config_file')
            if not weight_config_file or not os.path.exists(weight_config_file):
                self.validation_error.emit("权重配置文件不存在")
                return False
        
        # 检查权重是否覆盖所有类别
        if custom_weights:
            missing_classes = set(class_names) - set(custom_weights.keys())
            if missing_classes:
                self.status_updated.emit(f"警告: 以下类别缺少权重配置: {list(missing_classes)[:5]}...")
        
        return True 
    
    def _check_directory_writable_with_cleanup(self, directory: str) -> bool:
        """检查目录是否可写，带清理机制处理云同步冲突"""
        import time
        import random
        
        # 清理可能存在的临时文件
        self._cleanup_temp_files(directory)
        
        # 尝试多次写入测试，处理云同步冲突
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # 生成唯一的测试文件名，避免冲突
                timestamp = int(time.time() * 1000)
                random_suffix = random.randint(1000, 9999)
                test_file = os.path.join(directory, f'test_write_{timestamp}_{random_suffix}.tmp')
                
                # 尝试写入测试文件
                with open(test_file, 'w') as f:
                    f.write('test')
                
                # 立即删除测试文件
                os.remove(test_file)
                
                # 如果成功，返回True
                return True
                
            except PermissionError as e:
                if "另一个程序正在使用此文件" in str(e) or "WinError 32" in str(e):
                    # 云同步冲突，等待后重试
                    if attempt < max_attempts - 1:
                        wait_time = (attempt + 1) * 2  # 递增等待时间：2秒、4秒
                        self.status_updated.emit(f"检测到云同步冲突，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # 最后一次尝试失败，提供详细错误信息
                        self.validation_error.emit(
                            f"模型保存目录被云同步服务占用: {directory}\n"
                            f"建议解决方案:\n"
                            f"1. 暂停 iCloudDrive 或 Qsync 同步\n"
                            f"2. 等待同步完成后重试\n"
                            f"3. 将项目移动到非云同步目录\n"
                            f"错误详情: {str(e)}"
                        )
                        return False
                else:
                    # 其他权限错误
                    self.validation_error.emit(f"模型保存目录权限不足: {directory}, 错误: {str(e)}")
                    return False
                    
            except Exception as e:
                # 其他错误
                self.validation_error.emit(f"模型保存目录不可写: {directory}, 错误: {str(e)}")
                return False
        
        return False
    
    def _cleanup_temp_files(self, directory: str):
        """清理目录中的临时文件"""
        try:
            import glob
            import time
            
            # 查找所有临时文件
            temp_patterns = [
                os.path.join(directory, 'test_write*.tmp'),
                os.path.join(directory, '*.tmp'),
                os.path.join(directory, '*.lock')
            ]
            
            for pattern in temp_patterns:
                temp_files = glob.glob(pattern)
                for temp_file in temp_files:
                    try:
                        # 检查文件是否超过5分钟
                        file_age = time.time() - os.path.getmtime(temp_file)
                        if file_age > 300:  # 5分钟
                            os.remove(temp_file)
                            self.status_updated.emit(f"已清理过期临时文件: {os.path.basename(temp_file)}")
                    except Exception:
                        # 忽略清理失败的文件
                        pass
                        
        except Exception:
            # 忽略清理过程中的错误
            pass