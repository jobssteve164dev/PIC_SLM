"""
智能训练设置组件

此组件提供智能训练控制器的配置界面，包括：
- 干预阈值设置
- 参数调优策略
- 配置保存/加载/重置功能
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                           QLabel, QDoubleSpinBox, QSpinBox, QComboBox, 
                           QPushButton, QMessageBox, QFileDialog, QFormLayout, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
import json
import os
from typing import Dict, Any


class IntelligentTrainingSettingsWidget(QWidget):
    """智能训练设置组件"""
    
    # 定义信号
    config_changed = pyqtSignal(dict)  # 配置变更信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 默认配置
        self.default_config = {
            'overfitting_threshold': 0.80,
            'underfitting_threshold': 0.70,
            'stagnation_epochs': 5,
            'divergence_threshold': 2.00,
            'min_training_epochs': 3,
            'tuning_strategy': 'conservative',
            'enable_auto_intervention': True,
            'intervention_cooldown': 2,
            'max_interventions_per_session': 10,
            'llm_analysis_enabled': True,
            'confidence_threshold': 0.7
        }
        
        # 当前配置
        self.current_config = self.default_config.copy()
        
        # 初始化UI
        self.init_ui()
        
        # 连接信号
        self.connect_signals()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # 添加标题
        title_label = QLabel("智能训练设置")
        title_label.setFont(QFont('微软雅黑', 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 干预阈值设置组
        self.create_intervention_thresholds_group(layout)
        
        # 参数调优策略组
        self.create_tuning_strategy_group(layout)
        
        # 高级设置组
        self.create_advanced_settings_group(layout)
        
        # 配置管理按钮
        self.create_config_management_buttons(layout)
        
        # 添加弹性空间
        layout.addStretch()
    
    def create_intervention_thresholds_group(self, parent_layout):
        """创建干预阈值设置组"""
        group = QGroupBox("干预阈值设置")
        group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        layout = QFormLayout(group)
        layout.setSpacing(10)
        
        # 过拟合风险阈值
        self.overfitting_spinbox = QDoubleSpinBox()
        self.overfitting_spinbox.setRange(0.1, 1.0)
        self.overfitting_spinbox.setSingleStep(0.05)
        self.overfitting_spinbox.setDecimals(2)
        self.overfitting_spinbox.setValue(self.current_config['overfitting_threshold'])
        self.overfitting_spinbox.setToolTip("当验证损失与训练损失的比值超过此阈值时，触发过拟合干预")
        layout.addRow("过拟合风险阈值:", self.overfitting_spinbox)
        
        # 欠拟合风险阈值
        self.underfitting_spinbox = QDoubleSpinBox()
        self.underfitting_spinbox.setRange(0.1, 1.0)
        self.underfitting_spinbox.setSingleStep(0.05)
        self.underfitting_spinbox.setDecimals(2)
        self.underfitting_spinbox.setValue(self.current_config['underfitting_threshold'])
        self.underfitting_spinbox.setToolTip("当训练准确率低于此阈值时，触发欠拟合干预")
        layout.addRow("欠拟合风险阈值:", self.underfitting_spinbox)
        
        # 停滞轮数阈值
        self.stagnation_spinbox = QSpinBox()
        self.stagnation_spinbox.setRange(1, 50)
        self.stagnation_spinbox.setValue(self.current_config['stagnation_epochs'])
        self.stagnation_spinbox.setToolTip("当验证指标连续N轮无改善时，触发停滞干预")
        layout.addRow("停滞轮数阈值:", self.stagnation_spinbox)
        
        # 发散阈值
        self.divergence_spinbox = QDoubleSpinBox()
        self.divergence_spinbox.setRange(0.1, 10.0)
        self.divergence_spinbox.setSingleStep(0.1)
        self.divergence_spinbox.setDecimals(2)
        self.divergence_spinbox.setValue(self.current_config['divergence_threshold'])
        self.divergence_spinbox.setToolTip("当损失值增长超过此倍数时，触发发散干预")
        layout.addRow("发散阈值:", self.divergence_spinbox)
        
        # 最小训练轮数
        self.min_epochs_spinbox = QSpinBox()
        self.min_epochs_spinbox.setRange(1, 100)
        self.min_epochs_spinbox.setValue(self.current_config['min_training_epochs'])
        self.min_epochs_spinbox.setToolTip("训练至少进行N轮后才允许干预")
        layout.addRow("最小训练轮数:", self.min_epochs_spinbox)
        
        parent_layout.addWidget(group)
    
    def create_tuning_strategy_group(self, parent_layout):
        """创建参数调优策略组"""
        group = QGroupBox("参数调优策略")
        group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        layout = QFormLayout(group)
        layout.setSpacing(10)
        
        # 调优策略
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(['conservative', 'moderate', 'aggressive'])
        self.strategy_combo.setCurrentText(self.current_config['tuning_strategy'])
        self.strategy_combo.setToolTip("保守：小幅调整参数\n适中：中等幅度调整\n激进：大幅调整参数")
        layout.addRow("调优策略:", self.strategy_combo)
        
        # 干预冷却时间
        self.cooldown_spinbox = QSpinBox()
        self.cooldown_spinbox.setRange(1, 60)
        self.cooldown_spinbox.setValue(self.current_config['intervention_cooldown'])
        self.cooldown_spinbox.setSuffix(" 轮")
        self.cooldown_spinbox.setToolTip("两次干预之间的最小间隔轮数")
        layout.addRow("干预冷却时间:", self.cooldown_spinbox)
        
        # 最大干预次数
        self.max_interventions_spinbox = QSpinBox()
        self.max_interventions_spinbox.setRange(1, 100)
        self.max_interventions_spinbox.setValue(self.current_config['max_interventions_per_session'])
        self.max_interventions_spinbox.setToolTip("单次训练会话中允许的最大干预次数")
        layout.addRow("最大干预次数:", self.max_interventions_spinbox)
        
        parent_layout.addWidget(group)
    
    def create_advanced_settings_group(self, parent_layout):
        """创建高级设置组"""
        group = QGroupBox("高级设置")
        group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        layout = QFormLayout(group)
        layout.setSpacing(10)
        
        # 启用自动干预
        self.auto_intervention_checkbox = QCheckBox("启用自动干预")
        self.auto_intervention_checkbox.setChecked(self.current_config['enable_auto_intervention'])
        self.auto_intervention_checkbox.setToolTip("是否允许系统自动执行参数调优")
        layout.addRow("", self.auto_intervention_checkbox)
        
        # 启用LLM分析
        self.llm_analysis_checkbox = QCheckBox("启用LLM分析")
        self.llm_analysis_checkbox.setChecked(self.current_config['llm_analysis_enabled'])
        self.llm_analysis_checkbox.setToolTip("是否使用大语言模型进行训练分析")
        layout.addRow("", self.llm_analysis_checkbox)
        
        # 置信度阈值
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.1, 1.0)
        self.confidence_spinbox.setSingleStep(0.05)
        self.confidence_spinbox.setDecimals(2)
        self.confidence_spinbox.setValue(self.current_config['confidence_threshold'])
        self.confidence_spinbox.setToolTip("LLM分析结果的置信度阈值")
        layout.addRow("置信度阈值:", self.confidence_spinbox)
        
        parent_layout.addWidget(group)
    
    def create_config_management_buttons(self, parent_layout):
        """创建配置管理按钮"""
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        # 保存配置按钮
        self.save_btn = QPushButton("保存配置")
        self.save_btn.setIcon(QIcon(":/icons/save.png"))
        self.save_btn.setToolTip("将当前配置保存到应用设置中")
        button_layout.addWidget(self.save_btn)
        
        # 重置默认按钮
        self.reset_btn = QPushButton("重置默认")
        self.reset_btn.setIcon(QIcon(":/icons/reset.png"))
        self.reset_btn.setToolTip("重置为默认配置")
        button_layout.addWidget(self.reset_btn)
        
        # 添加弹性空间
        button_layout.addStretch()
        
        parent_layout.addLayout(button_layout)
    
    def connect_signals(self):
        """连接信号"""
        # 配置变更信号
        self.overfitting_spinbox.valueChanged.connect(self.on_config_changed)
        self.underfitting_spinbox.valueChanged.connect(self.on_config_changed)
        self.stagnation_spinbox.valueChanged.connect(self.on_config_changed)
        self.divergence_spinbox.valueChanged.connect(self.on_config_changed)
        self.min_epochs_spinbox.valueChanged.connect(self.on_config_changed)
        self.strategy_combo.currentTextChanged.connect(self.on_config_changed)
        self.cooldown_spinbox.valueChanged.connect(self.on_config_changed)
        self.max_interventions_spinbox.valueChanged.connect(self.on_config_changed)
        self.auto_intervention_checkbox.toggled.connect(self.on_config_changed)
        self.llm_analysis_checkbox.toggled.connect(self.on_config_changed)
        self.confidence_spinbox.valueChanged.connect(self.on_config_changed)
        
        # 按钮信号
        self.save_btn.clicked.connect(self.save_config)
        self.reset_btn.clicked.connect(self.reset_to_default)
    
    def on_config_changed(self):
        """配置变更处理"""
        self.update_current_config()
        self.config_changed.emit(self.current_config)
    
    def update_current_config(self):
        """更新当前配置"""
        self.current_config = {
            'overfitting_threshold': self.overfitting_spinbox.value(),
            'underfitting_threshold': self.underfitting_spinbox.value(),
            'stagnation_epochs': self.stagnation_spinbox.value(),
            'divergence_threshold': self.divergence_spinbox.value(),
            'min_training_epochs': self.min_epochs_spinbox.value(),
            'tuning_strategy': self.strategy_combo.currentText(),
            'enable_auto_intervention': self.auto_intervention_checkbox.isChecked(),
            'intervention_cooldown': self.cooldown_spinbox.value(),
            'max_interventions_per_session': self.max_interventions_spinbox.value(),
            'llm_analysis_enabled': self.llm_analysis_checkbox.isChecked(),
            'confidence_threshold': self.confidence_spinbox.value()
        }
    
    def get_current_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        self.update_current_config()
        return self.current_config.copy()
    
    def set_config(self, config: Dict[str, Any]):
        """设置配置"""
        # 更新UI控件
        self.overfitting_spinbox.setValue(config.get('overfitting_threshold', self.default_config['overfitting_threshold']))
        self.underfitting_spinbox.setValue(config.get('underfitting_threshold', self.default_config['underfitting_threshold']))
        self.stagnation_spinbox.setValue(config.get('stagnation_epochs', self.default_config['stagnation_epochs']))
        self.divergence_spinbox.setValue(config.get('divergence_threshold', self.default_config['divergence_threshold']))
        self.min_epochs_spinbox.setValue(config.get('min_training_epochs', self.default_config['min_training_epochs']))
        self.strategy_combo.setCurrentText(config.get('tuning_strategy', self.default_config['tuning_strategy']))
        self.cooldown_spinbox.setValue(config.get('intervention_cooldown', self.default_config['intervention_cooldown']))
        self.max_interventions_spinbox.setValue(config.get('max_interventions_per_session', self.default_config['max_interventions_per_session']))
        self.auto_intervention_checkbox.setChecked(config.get('enable_auto_intervention', self.default_config['enable_auto_intervention']))
        self.llm_analysis_checkbox.setChecked(config.get('llm_analysis_enabled', self.default_config['llm_analysis_enabled']))
        self.confidence_spinbox.setValue(config.get('confidence_threshold', self.default_config['confidence_threshold']))
        
        # 更新内部配置
        self.current_config = config.copy()
    
    def save_config(self):
        """保存配置"""
        try:
            self.update_current_config()
            self.config_changed.emit(self.current_config)
            QMessageBox.information(self, "成功", "智能训练配置已保存到应用设置中")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存配置失败: {str(e)}")
    
    def reset_to_default(self):
        """重置为默认配置"""
        reply = QMessageBox.question(
            self, "确认重置", 
            "确定要重置为默认配置吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.set_config(self.default_config)
            self.config_changed.emit(self.current_config)
            QMessageBox.information(self, "成功", "已重置为默认配置")
    
    def load_from_file(self, file_path: str):
        """从文件加载配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.set_config(config)
            QMessageBox.information(self, "成功", f"配置已从 {file_path} 加载")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载配置失败: {str(e)}")
    
    def save_to_file(self, file_path: str):
        """保存配置到文件"""
        try:
            self.update_current_config()
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_config, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "成功", f"配置已保存到 {file_path}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存配置失败: {str(e)}")
