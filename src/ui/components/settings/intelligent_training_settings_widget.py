"""
智能训练设置组件

此组件提供智能训练系统的配置界面，包括：
- 智能训练编排器配置
- 配置生成器参数设置
- 干预阈值和策略设置
- LLM分析配置
- 监控和报告设置
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                           QLabel, QDoubleSpinBox, QSpinBox, QComboBox, 
                           QPushButton, QMessageBox, QFileDialog, QFormLayout, QCheckBox,
                           QTabWidget, QTextEdit, QScrollArea)
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
        
        # 默认配置 - 基于新的智能训练系统
        self.default_config = {
            # 智能训练编排器配置
            'enabled': True,
            'max_iterations': 5,
            'min_iteration_epochs': 2,
            'analysis_interval': 2,
            'convergence_threshold': 0.01,
            'improvement_threshold': 0.02,
            'auto_restart': True,
            'preserve_best_model': True,
            
            # 干预阈值设置
            'overfitting_threshold': 0.80,
            'underfitting_threshold': 0.70,
            'stagnation_epochs': 5,
            'divergence_threshold': 2.00,
            'min_training_epochs': 3,
            
            # 参数调优策略
            'tuning_strategy': 'conservative',
            'enable_auto_intervention': True,
            'intervention_cooldown': 2,
            'max_interventions_per_session': 10,
            
            # LLM分析配置
            'llm_analysis_enabled': True,
            'confidence_threshold': 0.7,
            'adapter_type': 'openai',  # 生产环境使用真实LLM适配器
            'analysis_frequency': 'epoch_based',
            'min_data_points': 5,
            
            # 监控配置
            'check_interval': 5,
            'metrics_buffer_size': 100,
            'trend_analysis_window': 10,
            'alert_on_intervention': True,
            
            # 报告配置
            'auto_generate_reports': True,
            'report_format': 'json',
            'include_visualizations': True,
            'save_intervention_details': True
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
        title_label = QLabel("智能训练系统设置")
        title_label.setFont(QFont('微软雅黑', 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 创建选项卡
        self.settings_tabs = QTabWidget()
        layout.addWidget(self.settings_tabs)
        
        # 创建各个设置选项卡
        self.create_orchestrator_tab()
        self.create_intervention_tab()
        self.create_llm_analysis_tab()
        self.create_monitoring_tab()
        self.create_reporting_tab()
        
        # 配置管理按钮
        self.create_config_management_buttons(layout)
        
        # 添加弹性空间
        layout.addStretch()
    
    def create_orchestrator_tab(self):
        """创建智能训练编排器设置选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 基本设置组
        basic_group = QGroupBox("基本设置")
        basic_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        basic_layout = QFormLayout(basic_group)
        
        # 启用智能训练
        self.enabled_checkbox = QCheckBox("启用智能训练")
        self.enabled_checkbox.setChecked(self.current_config['enabled'])
        self.enabled_checkbox.setToolTip("是否启用智能训练系统")
        basic_layout.addRow("", self.enabled_checkbox)
        
        # 最大迭代次数
        self.max_iterations_spinbox = QSpinBox()
        self.max_iterations_spinbox.setRange(1, 20)
        self.max_iterations_spinbox.setValue(self.current_config['max_iterations'])
        self.max_iterations_spinbox.setToolTip("智能训练的最大迭代次数")
        basic_layout.addRow("最大迭代次数:", self.max_iterations_spinbox)
        
        # 最小迭代轮数
        self.min_iteration_epochs_spinbox = QSpinBox()
        self.min_iteration_epochs_spinbox.setRange(1, 50)
        self.min_iteration_epochs_spinbox.setValue(self.current_config['min_iteration_epochs'])
        self.min_iteration_epochs_spinbox.setToolTip("每次迭代的最小训练轮数")
        basic_layout.addRow("最小迭代轮数:", self.min_iteration_epochs_spinbox)
        
        # 分析间隔
        self.analysis_interval_spinbox = QSpinBox()
        self.analysis_interval_spinbox.setRange(1, 50)
        self.analysis_interval_spinbox.setValue(self.current_config['analysis_interval'])
        self.analysis_interval_spinbox.setToolTip("分析间隔（轮数）")
        basic_layout.addRow("分析间隔:", self.analysis_interval_spinbox)
        
        layout.addWidget(basic_group)
        
        # 收敛设置组
        convergence_group = QGroupBox("收敛设置")
        convergence_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        convergence_layout = QFormLayout(convergence_group)
        
        # 收敛阈值
        self.convergence_threshold_spinbox = QDoubleSpinBox()
        self.convergence_threshold_spinbox.setRange(0.001, 0.1)
        self.convergence_threshold_spinbox.setSingleStep(0.001)
        self.convergence_threshold_spinbox.setDecimals(3)
        self.convergence_threshold_spinbox.setValue(self.current_config['convergence_threshold'])
        self.convergence_threshold_spinbox.setToolTip("收敛判断阈值")
        convergence_layout.addRow("收敛阈值:", self.convergence_threshold_spinbox)
        
        # 改进阈值
        self.improvement_threshold_spinbox = QDoubleSpinBox()
        self.improvement_threshold_spinbox.setRange(0.001, 0.1)
        self.improvement_threshold_spinbox.setSingleStep(0.001)
        self.improvement_threshold_spinbox.setDecimals(3)
        self.improvement_threshold_spinbox.setValue(self.current_config['improvement_threshold'])
        self.improvement_threshold_spinbox.setToolTip("改进判断阈值")
        convergence_layout.addRow("改进阈值:", self.improvement_threshold_spinbox)
        
        layout.addWidget(convergence_group)
        
        # 重启设置组
        restart_group = QGroupBox("重启设置")
        restart_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        restart_layout = QFormLayout(restart_group)
        
        # 自动重启
        self.auto_restart_checkbox = QCheckBox("自动重启训练")
        self.auto_restart_checkbox.setChecked(self.current_config['auto_restart'])
        self.auto_restart_checkbox.setToolTip("是否自动重启训练")
        restart_layout.addRow("", self.auto_restart_checkbox)
        
        # 保留最佳模型
        self.preserve_best_model_checkbox = QCheckBox("保留最佳模型")
        self.preserve_best_model_checkbox.setChecked(self.current_config['preserve_best_model'])
        self.preserve_best_model_checkbox.setToolTip("是否保留最佳模型")
        restart_layout.addRow("", self.preserve_best_model_checkbox)
        
        layout.addWidget(restart_group)
        layout.addStretch()
        
        self.settings_tabs.addTab(tab, "🎯 编排器")
    
    def create_intervention_tab(self):
        """创建干预设置选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 干预阈值设置组
        threshold_group = QGroupBox("干预阈值设置")
        threshold_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        threshold_layout = QFormLayout(threshold_group)
        
        # 过拟合风险阈值
        self.overfitting_spinbox = QDoubleSpinBox()
        self.overfitting_spinbox.setRange(0.1, 1.0)
        self.overfitting_spinbox.setSingleStep(0.05)
        self.overfitting_spinbox.setDecimals(2)
        self.overfitting_spinbox.setValue(self.current_config['overfitting_threshold'])
        self.overfitting_spinbox.setToolTip("当验证损失与训练损失的比值超过此阈值时，触发过拟合干预")
        threshold_layout.addRow("过拟合风险阈值:", self.overfitting_spinbox)
        
        # 欠拟合风险阈值
        self.underfitting_spinbox = QDoubleSpinBox()
        self.underfitting_spinbox.setRange(0.1, 1.0)
        self.underfitting_spinbox.setSingleStep(0.05)
        self.underfitting_spinbox.setDecimals(2)
        self.underfitting_spinbox.setValue(self.current_config['underfitting_threshold'])
        self.underfitting_spinbox.setToolTip("当训练准确率低于此阈值时，触发欠拟合干预")
        threshold_layout.addRow("欠拟合风险阈值:", self.underfitting_spinbox)
        
        # 停滞轮数阈值
        self.stagnation_spinbox = QSpinBox()
        self.stagnation_spinbox.setRange(1, 50)
        self.stagnation_spinbox.setValue(self.current_config['stagnation_epochs'])
        self.stagnation_spinbox.setToolTip("当验证指标连续N轮无改善时，触发停滞干预")
        threshold_layout.addRow("停滞轮数阈值:", self.stagnation_spinbox)
        
        # 发散阈值
        self.divergence_spinbox = QDoubleSpinBox()
        self.divergence_spinbox.setRange(0.1, 10.0)
        self.divergence_spinbox.setSingleStep(0.1)
        self.divergence_spinbox.setDecimals(2)
        self.divergence_spinbox.setValue(self.current_config['divergence_threshold'])
        self.divergence_spinbox.setToolTip("当损失值增长超过此倍数时，触发发散干预")
        threshold_layout.addRow("发散阈值:", self.divergence_spinbox)
        
        # 最小训练轮数
        self.min_epochs_spinbox = QSpinBox()
        self.min_epochs_spinbox.setRange(1, 100)
        self.min_epochs_spinbox.setValue(self.current_config['min_training_epochs'])
        self.min_epochs_spinbox.setToolTip("训练至少进行N轮后才允许干预")
        threshold_layout.addRow("最小训练轮数:", self.min_epochs_spinbox)
        
        layout.addWidget(threshold_group)
        
        # 参数调优策略组
        strategy_group = QGroupBox("参数调优策略")
        strategy_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        strategy_layout = QFormLayout(strategy_group)
        
        # 调优策略
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(['conservative', 'balanced', 'aggressive'])
        self.strategy_combo.setCurrentText(self.current_config['tuning_strategy'])
        self.strategy_combo.setToolTip("保守：小幅调整参数\n平衡：中等幅度调整\n激进：大幅调整参数")
        strategy_layout.addRow("调优策略:", self.strategy_combo)
        
        # 干预冷却时间
        self.cooldown_spinbox = QSpinBox()
        self.cooldown_spinbox.setRange(1, 60)
        self.cooldown_spinbox.setValue(self.current_config['intervention_cooldown'])
        self.cooldown_spinbox.setSuffix(" 轮")
        self.cooldown_spinbox.setToolTip("两次干预之间的最小间隔轮数")
        strategy_layout.addRow("干预冷却时间:", self.cooldown_spinbox)
        
        # 最大干预次数
        self.max_interventions_spinbox = QSpinBox()
        self.max_interventions_spinbox.setRange(1, 100)
        self.max_interventions_spinbox.setValue(self.current_config['max_interventions_per_session'])
        self.max_interventions_spinbox.setToolTip("单次训练会话中允许的最大干预次数")
        strategy_layout.addRow("最大干预次数:", self.max_interventions_spinbox)
        
        # 启用自动干预
        self.auto_intervention_checkbox = QCheckBox("启用自动干预")
        self.auto_intervention_checkbox.setChecked(self.current_config['enable_auto_intervention'])
        self.auto_intervention_checkbox.setToolTip("是否允许系统自动执行参数调优")
        strategy_layout.addRow("", self.auto_intervention_checkbox)
        
        layout.addWidget(strategy_group)
        layout.addStretch()
        
        self.settings_tabs.addTab(tab, "⚡ 干预")
    
    def create_llm_analysis_tab(self):
        """创建LLM分析设置选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # LLM分析设置组
        llm_group = QGroupBox("LLM分析设置")
        llm_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        llm_layout = QFormLayout(llm_group)
        
        # 启用LLM分析
        self.llm_analysis_checkbox = QCheckBox("启用LLM分析")
        self.llm_analysis_checkbox.setChecked(self.current_config['llm_analysis_enabled'])
        self.llm_analysis_checkbox.setToolTip("是否使用大语言模型进行训练分析")
        llm_layout.addRow("", self.llm_analysis_checkbox)
        
        # 适配器类型 - 引用AI设置中的适配器
        self.adapter_type_combo = QComboBox()
        self.adapter_type_combo.addItems(['openai', 'deepseek', 'ollama', 'custom', 'mock'])
        self.adapter_type_combo.setCurrentText(self.current_config['adapter_type'])
        self.adapter_type_combo.setToolTip("LLM适配器类型 - 生产环境请使用真实LLM服务，mock仅用于测试")
        llm_layout.addRow("适配器类型:", self.adapter_type_combo)
        
        # 添加警告标签
        warning_label = QLabel("⚠️ 生产环境请配置真实的LLM服务，避免使用mock适配器")
        warning_label.setStyleSheet("color: #ff6b35; font-weight: bold;")
        warning_label.setWordWrap(True)
        llm_layout.addRow("", warning_label)
        
        # 分析频率
        self.analysis_frequency_combo = QComboBox()
        self.analysis_frequency_combo.addItems(['epoch_based', 'time_based', 'metric_based'])
        self.analysis_frequency_combo.setCurrentText(self.current_config['analysis_frequency'])
        self.analysis_frequency_combo.setToolTip("分析触发频率")
        llm_layout.addRow("分析频率:", self.analysis_frequency_combo)
        
        # 最小数据点数
        self.min_data_points_spinbox = QSpinBox()
        self.min_data_points_spinbox.setRange(1, 100)
        self.min_data_points_spinbox.setValue(self.current_config['min_data_points'])
        self.min_data_points_spinbox.setToolTip("触发分析所需的最小数据点数")
        llm_layout.addRow("最小数据点数:", self.min_data_points_spinbox)
        
        # 置信度阈值
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.1, 1.0)
        self.confidence_spinbox.setSingleStep(0.05)
        self.confidence_spinbox.setDecimals(2)
        self.confidence_spinbox.setValue(self.current_config['confidence_threshold'])
        self.confidence_spinbox.setToolTip("LLM分析结果的置信度阈值")
        llm_layout.addRow("置信度阈值:", self.confidence_spinbox)
        
        layout.addWidget(llm_group)
        layout.addStretch()
        
        self.settings_tabs.addTab(tab, "🤖 LLM分析")
    
    def create_monitoring_tab(self):
        """创建监控设置选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 监控设置组
        monitoring_group = QGroupBox("监控设置")
        monitoring_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        monitoring_layout = QFormLayout(monitoring_group)
        
        # 检查间隔
        self.check_interval_spinbox = QSpinBox()
        self.check_interval_spinbox.setRange(1, 60)
        self.check_interval_spinbox.setValue(self.current_config['check_interval'])
        self.check_interval_spinbox.setSuffix(" 秒")
        self.check_interval_spinbox.setToolTip("监控检查间隔时间")
        monitoring_layout.addRow("检查间隔:", self.check_interval_spinbox)
        
        # 指标缓冲区大小
        self.metrics_buffer_size_spinbox = QSpinBox()
        self.metrics_buffer_size_spinbox.setRange(10, 1000)
        self.metrics_buffer_size_spinbox.setValue(self.current_config['metrics_buffer_size'])
        self.metrics_buffer_size_spinbox.setToolTip("指标数据缓冲区大小")
        monitoring_layout.addRow("指标缓冲区大小:", self.metrics_buffer_size_spinbox)
        
        # 趋势分析窗口
        self.trend_analysis_window_spinbox = QSpinBox()
        self.trend_analysis_window_spinbox.setRange(5, 100)
        self.trend_analysis_window_spinbox.setValue(self.current_config['trend_analysis_window'])
        self.trend_analysis_window_spinbox.setToolTip("趋势分析窗口大小")
        monitoring_layout.addRow("趋势分析窗口:", self.trend_analysis_window_spinbox)
        
        # 干预时发出警报
        self.alert_on_intervention_checkbox = QCheckBox("干预时发出警报")
        self.alert_on_intervention_checkbox.setChecked(self.current_config['alert_on_intervention'])
        self.alert_on_intervention_checkbox.setToolTip("是否在干预时发出警报")
        monitoring_layout.addRow("", self.alert_on_intervention_checkbox)
        
        layout.addWidget(monitoring_group)
        layout.addStretch()
        
        self.settings_tabs.addTab(tab, "📊 监控")
    
    def create_reporting_tab(self):
        """创建报告设置选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 报告设置组
        reporting_group = QGroupBox("报告设置")
        reporting_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        reporting_layout = QFormLayout(reporting_group)
        
        # 自动生成报告
        self.auto_generate_reports_checkbox = QCheckBox("自动生成报告")
        self.auto_generate_reports_checkbox.setChecked(self.current_config['auto_generate_reports'])
        self.auto_generate_reports_checkbox.setToolTip("是否自动生成训练报告")
        reporting_layout.addRow("", self.auto_generate_reports_checkbox)
        
        # 报告格式
        self.report_format_combo = QComboBox()
        self.report_format_combo.addItems(['json', 'html', 'markdown', 'pdf'])
        self.report_format_combo.setCurrentText(self.current_config['report_format'])
        self.report_format_combo.setToolTip("报告输出格式")
        reporting_layout.addRow("报告格式:", self.report_format_combo)
        
        # 包含可视化
        self.include_visualizations_checkbox = QCheckBox("包含可视化")
        self.include_visualizations_checkbox.setChecked(self.current_config['include_visualizations'])
        self.include_visualizations_checkbox.setToolTip("是否在报告中包含可视化图表")
        reporting_layout.addRow("", self.include_visualizations_checkbox)
        
        # 保存干预详情
        self.save_intervention_details_checkbox = QCheckBox("保存干预详情")
        self.save_intervention_details_checkbox.setChecked(self.current_config['save_intervention_details'])
        self.save_intervention_details_checkbox.setToolTip("是否保存详细的干预信息")
        reporting_layout.addRow("", self.save_intervention_details_checkbox)
        
        layout.addWidget(reporting_group)
        layout.addStretch()
        
        self.settings_tabs.addTab(tab, "📋 报告")
    
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
        # 编排器配置信号
        self.enabled_checkbox.toggled.connect(self.on_config_changed)
        self.max_iterations_spinbox.valueChanged.connect(self.on_config_changed)
        self.min_iteration_epochs_spinbox.valueChanged.connect(self.on_config_changed)
        self.analysis_interval_spinbox.valueChanged.connect(self.on_config_changed)
        self.convergence_threshold_spinbox.valueChanged.connect(self.on_config_changed)
        self.improvement_threshold_spinbox.valueChanged.connect(self.on_config_changed)
        self.auto_restart_checkbox.toggled.connect(self.on_config_changed)
        self.preserve_best_model_checkbox.toggled.connect(self.on_config_changed)
        
        # 干预配置信号
        self.overfitting_spinbox.valueChanged.connect(self.on_config_changed)
        self.underfitting_spinbox.valueChanged.connect(self.on_config_changed)
        self.stagnation_spinbox.valueChanged.connect(self.on_config_changed)
        self.divergence_spinbox.valueChanged.connect(self.on_config_changed)
        self.min_epochs_spinbox.valueChanged.connect(self.on_config_changed)
        self.strategy_combo.currentTextChanged.connect(self.on_config_changed)
        self.cooldown_spinbox.valueChanged.connect(self.on_config_changed)
        self.max_interventions_spinbox.valueChanged.connect(self.on_config_changed)
        self.auto_intervention_checkbox.toggled.connect(self.on_config_changed)
        
        # LLM分析配置信号
        self.llm_analysis_checkbox.toggled.connect(self.on_config_changed)
        self.adapter_type_combo.currentTextChanged.connect(self.on_config_changed)
        self.analysis_frequency_combo.currentTextChanged.connect(self.on_config_changed)
        self.min_data_points_spinbox.valueChanged.connect(self.on_config_changed)
        self.confidence_spinbox.valueChanged.connect(self.on_config_changed)
        
        # 监控配置信号
        self.check_interval_spinbox.valueChanged.connect(self.on_config_changed)
        self.metrics_buffer_size_spinbox.valueChanged.connect(self.on_config_changed)
        self.trend_analysis_window_spinbox.valueChanged.connect(self.on_config_changed)
        self.alert_on_intervention_checkbox.toggled.connect(self.on_config_changed)
        
        # 报告配置信号
        self.auto_generate_reports_checkbox.toggled.connect(self.on_config_changed)
        self.report_format_combo.currentTextChanged.connect(self.on_config_changed)
        self.include_visualizations_checkbox.toggled.connect(self.on_config_changed)
        self.save_intervention_details_checkbox.toggled.connect(self.on_config_changed)
        
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
            # 编排器配置
            'enabled': self.enabled_checkbox.isChecked(),
            'max_iterations': self.max_iterations_spinbox.value(),
            'min_iteration_epochs': self.min_iteration_epochs_spinbox.value(),
            'analysis_interval': self.analysis_interval_spinbox.value(),
            'convergence_threshold': self.convergence_threshold_spinbox.value(),
            'improvement_threshold': self.improvement_threshold_spinbox.value(),
            'auto_restart': self.auto_restart_checkbox.isChecked(),
            'preserve_best_model': self.preserve_best_model_checkbox.isChecked(),
            
            # 干预阈值设置
            'overfitting_threshold': self.overfitting_spinbox.value(),
            'underfitting_threshold': self.underfitting_spinbox.value(),
            'stagnation_epochs': self.stagnation_spinbox.value(),
            'divergence_threshold': self.divergence_spinbox.value(),
            'min_training_epochs': self.min_epochs_spinbox.value(),
            
            # 参数调优策略
            'tuning_strategy': self.strategy_combo.currentText(),
            'enable_auto_intervention': self.auto_intervention_checkbox.isChecked(),
            'intervention_cooldown': self.cooldown_spinbox.value(),
            'max_interventions_per_session': self.max_interventions_spinbox.value(),
            
            # LLM分析配置
            'llm_analysis_enabled': self.llm_analysis_checkbox.isChecked(),
            'adapter_type': self.adapter_type_combo.currentText(),
            'analysis_frequency': self.analysis_frequency_combo.currentText(),
            'min_data_points': self.min_data_points_spinbox.value(),
            'confidence_threshold': self.confidence_spinbox.value(),
            
            # 监控配置
            'check_interval': self.check_interval_spinbox.value(),
            'metrics_buffer_size': self.metrics_buffer_size_spinbox.value(),
            'trend_analysis_window': self.trend_analysis_window_spinbox.value(),
            'alert_on_intervention': self.alert_on_intervention_checkbox.isChecked(),
            
            # 报告配置
            'auto_generate_reports': self.auto_generate_reports_checkbox.isChecked(),
            'report_format': self.report_format_combo.currentText(),
            'include_visualizations': self.include_visualizations_checkbox.isChecked(),
            'save_intervention_details': self.save_intervention_details_checkbox.isChecked()
        }
    
    def get_current_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        self.update_current_config()
        return self.current_config.copy()
    
    def set_config(self, config: Dict[str, Any]):
        """设置配置"""
        # 更新编排器UI控件
        self.enabled_checkbox.setChecked(config.get('enabled', self.default_config['enabled']))
        self.max_iterations_spinbox.setValue(config.get('max_iterations', self.default_config['max_iterations']))
        self.min_iteration_epochs_spinbox.setValue(config.get('min_iteration_epochs', self.default_config['min_iteration_epochs']))
        self.analysis_interval_spinbox.setValue(config.get('analysis_interval', self.default_config['analysis_interval']))
        self.convergence_threshold_spinbox.setValue(config.get('convergence_threshold', self.default_config['convergence_threshold']))
        self.improvement_threshold_spinbox.setValue(config.get('improvement_threshold', self.default_config['improvement_threshold']))
        self.auto_restart_checkbox.setChecked(config.get('auto_restart', self.default_config['auto_restart']))
        self.preserve_best_model_checkbox.setChecked(config.get('preserve_best_model', self.default_config['preserve_best_model']))
        
        # 更新干预UI控件
        self.overfitting_spinbox.setValue(config.get('overfitting_threshold', self.default_config['overfitting_threshold']))
        self.underfitting_spinbox.setValue(config.get('underfitting_threshold', self.default_config['underfitting_threshold']))
        self.stagnation_spinbox.setValue(config.get('stagnation_epochs', self.default_config['stagnation_epochs']))
        self.divergence_spinbox.setValue(config.get('divergence_threshold', self.default_config['divergence_threshold']))
        self.min_epochs_spinbox.setValue(config.get('min_training_epochs', self.default_config['min_training_epochs']))
        self.strategy_combo.setCurrentText(config.get('tuning_strategy', self.default_config['tuning_strategy']))
        self.cooldown_spinbox.setValue(config.get('intervention_cooldown', self.default_config['intervention_cooldown']))
        self.max_interventions_spinbox.setValue(config.get('max_interventions_per_session', self.default_config['max_interventions_per_session']))
        self.auto_intervention_checkbox.setChecked(config.get('enable_auto_intervention', self.default_config['enable_auto_intervention']))
        
        # 更新LLM分析UI控件
        self.llm_analysis_checkbox.setChecked(config.get('llm_analysis_enabled', self.default_config['llm_analysis_enabled']))
        self.adapter_type_combo.setCurrentText(config.get('adapter_type', self.default_config['adapter_type']))
        self.analysis_frequency_combo.setCurrentText(config.get('analysis_frequency', self.default_config['analysis_frequency']))
        self.min_data_points_spinbox.setValue(config.get('min_data_points', self.default_config['min_data_points']))
        self.confidence_spinbox.setValue(config.get('confidence_threshold', self.default_config['confidence_threshold']))
        
        # 更新监控UI控件
        self.check_interval_spinbox.setValue(config.get('check_interval', self.default_config['check_interval']))
        self.metrics_buffer_size_spinbox.setValue(config.get('metrics_buffer_size', self.default_config['metrics_buffer_size']))
        self.trend_analysis_window_spinbox.setValue(config.get('trend_analysis_window', self.default_config['trend_analysis_window']))
        self.alert_on_intervention_checkbox.setChecked(config.get('alert_on_intervention', self.default_config['alert_on_intervention']))
        
        # 更新报告UI控件
        self.auto_generate_reports_checkbox.setChecked(config.get('auto_generate_reports', self.default_config['auto_generate_reports']))
        self.report_format_combo.setCurrentText(config.get('report_format', self.default_config['report_format']))
        self.include_visualizations_checkbox.setChecked(config.get('include_visualizations', self.default_config['include_visualizations']))
        self.save_intervention_details_checkbox.setChecked(config.get('save_intervention_details', self.default_config['save_intervention_details']))
        
        # 更新内部配置
        self.current_config = config.copy()
    
    def save_config(self):
        """保存配置"""
        try:
            self.update_current_config()
            
            # 保存到智能训练配置文件
            self._save_to_intelligent_training_config()
            
            self.config_changed.emit(self.current_config)
            QMessageBox.information(self, "成功", "智能训练配置已保存到应用设置中")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存配置失败: {str(e)}")
    
    def _save_to_intelligent_training_config(self):
        """保存到智能训练配置文件"""
        try:
            config_file = "setting/intelligent_training_config.json"
            
            # 创建配置目录
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            
            # 构建完整的配置
            full_config = {
                # 智能训练编排器配置
                'enabled': self.current_config['enabled'],
                'max_iterations': self.current_config['max_iterations'],
                'min_iteration_epochs': self.current_config['min_iteration_epochs'],
                'analysis_interval': self.current_config['analysis_interval'],
                'convergence_threshold': self.current_config['convergence_threshold'],
                'improvement_threshold': self.current_config['improvement_threshold'],
                'auto_restart': self.current_config['auto_restart'],
                'preserve_best_model': self.current_config['preserve_best_model'],
                
                # 干预阈值设置
                'overfitting_threshold': self.current_config['overfitting_threshold'],
                'underfitting_threshold': self.current_config['underfitting_threshold'],
                'stagnation_epochs': self.current_config['stagnation_epochs'],
                'divergence_threshold': self.current_config['divergence_threshold'],
                'min_training_epochs': self.current_config['min_training_epochs'],
                
                # 参数调优策略
                'tuning_strategy': self.current_config['tuning_strategy'],
                'enable_auto_intervention': self.current_config['enable_auto_intervention'],
                'intervention_cooldown': self.current_config['intervention_cooldown'],
                'max_interventions_per_session': self.current_config['max_interventions_per_session'],
                
                # LLM配置
                'llm_config': {
                    'adapter_type': self.current_config['adapter_type'],
                    'analysis_frequency': self.current_config['analysis_frequency'],
                    'min_data_points': self.current_config['min_data_points'],
                    'confidence_threshold': self.current_config['confidence_threshold']
                },
                
                # 监控配置
                'check_interval': self.current_config['check_interval'],
                'metrics_buffer_size': self.current_config['metrics_buffer_size'],
                'trend_analysis_window': self.current_config['trend_analysis_window'],
                'alert_on_intervention': self.current_config['alert_on_intervention'],
                
                # 报告配置
                'auto_generate_reports': self.current_config['auto_generate_reports'],
                'report_format': self.current_config['report_format'],
                'include_visualizations': self.current_config['include_visualizations'],
                'save_intervention_details': self.current_config['save_intervention_details']
            }
            
            # 保存到文件
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(full_config, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"保存智能训练配置失败: {str(e)}")
            raise
    
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
