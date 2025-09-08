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
                           QTabWidget, QTextEdit, QScrollArea, QLineEdit)
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
            'save_intervention_details': True,
            
            # 参数微调报告配置
            'parameter_tuning_reports': {
                'enabled': True,
                'save_path': 'reports/parameter_tuning',
                'format': 'markdown',
                'include_llm_analysis': True,
                'include_metrics_comparison': True,
                'include_config_changes': True
            }
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
        
        # 添加配置说明
        info_label = QLabel("💡 智能训练系统会自动分析训练过程，并根据效果调整参数。\n建议新手使用默认设置，有经验用户可根据需要调整参数。")
        info_label.setStyleSheet("color: #666; font-size: 10px; padding: 5px; background-color: #f0f8ff; border-radius: 5px;")
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
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
        self.max_iterations_spinbox.setToolTip("智能训练的最大迭代次数\n\n说明：系统会自动进行多次训练迭代，每次根据训练效果调整参数。\n建议值：3-8次（太少可能优化不充分，太多可能浪费时间）")
        basic_layout.addRow("最大迭代次数:", self.max_iterations_spinbox)
        
        # 最小迭代轮数
        self.min_iteration_epochs_spinbox = QSpinBox()
        self.min_iteration_epochs_spinbox.setRange(1, 50)
        self.min_iteration_epochs_spinbox.setValue(self.current_config['min_iteration_epochs'])
        self.min_iteration_epochs_spinbox.setToolTip("每次迭代的最小训练轮数\n\n说明：每次智能训练迭代至少训练这么多轮才开始分析效果。\n建议值：3-10轮（太少无法判断效果，太多浪费时间）")
        basic_layout.addRow("最小迭代轮数:", self.min_iteration_epochs_spinbox)
        
        # 分析间隔
        self.analysis_interval_spinbox = QSpinBox()
        self.analysis_interval_spinbox.setRange(1, 50)
        self.analysis_interval_spinbox.setValue(self.current_config['analysis_interval'])
        self.analysis_interval_spinbox.setToolTip("分析间隔（轮数）\n\n说明：每隔多少轮训练进行一次智能分析和参数优化。\n建议值：2-5轮（太频繁可能干扰训练，太少可能错过优化时机）")
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
        self.convergence_threshold_spinbox.setToolTip("收敛判断阈值\n\n说明：当连续几次迭代的准确率提升小于此阈值时，认为训练已收敛。\n建议值：0.01-0.05（越小越严格，越大越宽松）")
        convergence_layout.addRow("收敛阈值:", self.convergence_threshold_spinbox)
        
        # 改进阈值
        self.improvement_threshold_spinbox = QDoubleSpinBox()
        self.improvement_threshold_spinbox.setRange(0.001, 0.1)
        self.improvement_threshold_spinbox.setSingleStep(0.001)
        self.improvement_threshold_spinbox.setDecimals(3)
        self.improvement_threshold_spinbox.setValue(self.current_config['improvement_threshold'])
        self.improvement_threshold_spinbox.setToolTip("改进判断阈值\n\n说明：当新配置的准确率提升超过此阈值时，才认为是有意义的改进。\n建议值：0.01-0.03（避免微小的随机波动被误认为改进）")
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
        self.overfitting_spinbox.setToolTip("过拟合风险阈值\n\n说明：当验证损失与训练损失的比值超过此阈值时，系统会调整参数防止过拟合。\n建议值：0.7-0.9（越小越敏感，越大越宽松）")
        threshold_layout.addRow("过拟合风险阈值:", self.overfitting_spinbox)
        
        # 欠拟合风险阈值
        self.underfitting_spinbox = QDoubleSpinBox()
        self.underfitting_spinbox.setRange(0.1, 1.0)
        self.underfitting_spinbox.setSingleStep(0.05)
        self.underfitting_spinbox.setDecimals(2)
        self.underfitting_spinbox.setValue(self.current_config['underfitting_threshold'])
        self.underfitting_spinbox.setToolTip("欠拟合风险阈值\n\n说明：当训练准确率低于此阈值时，系统会调整参数提高模型学习能力。\n建议值：0.6-0.8（越小越宽松，越大越严格）")
        threshold_layout.addRow("欠拟合风险阈值:", self.underfitting_spinbox)
        
        # 停滞轮数阈值
        self.stagnation_spinbox = QSpinBox()
        self.stagnation_spinbox.setRange(1, 50)
        self.stagnation_spinbox.setValue(self.current_config['stagnation_epochs'])
        self.stagnation_spinbox.setToolTip("停滞轮数阈值\n\n说明：当验证指标连续N轮无改善时，系统会调整参数打破停滞。\n建议值：5-15轮（太小可能误判，太大可能浪费时间）")
        threshold_layout.addRow("停滞轮数阈值:", self.stagnation_spinbox)
        
        # 发散阈值
        self.divergence_spinbox = QDoubleSpinBox()
        self.divergence_spinbox.setRange(0.1, 10.0)
        self.divergence_spinbox.setSingleStep(0.1)
        self.divergence_spinbox.setDecimals(2)
        self.divergence_spinbox.setValue(self.current_config['divergence_threshold'])
        self.divergence_spinbox.setToolTip("发散阈值\n\n说明：当损失值增长超过此倍数时，系统会紧急调整参数防止训练发散。\n建议值：1.5-3.0（太小可能误判，太大可能错过挽救时机）")
        threshold_layout.addRow("发散阈值:", self.divergence_spinbox)
        
        # 最小训练轮数
        self.min_epochs_spinbox = QSpinBox()
        self.min_epochs_spinbox.setRange(1, 100)
        self.min_epochs_spinbox.setValue(self.current_config['min_training_epochs'])
        self.min_epochs_spinbox.setToolTip("最小训练轮数\n\n说明：训练至少进行N轮后才允许智能干预，避免过早干扰训练。\n建议值：3-10轮（给模型足够时间建立基础）")
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
        self.strategy_combo.setToolTip("参数调优策略\n\n• 保守：小幅调整参数，稳定但可能较慢\n• 平衡：中等幅度调整，兼顾稳定性和效率\n• 激进：大幅调整参数，快速但可能不稳定\n\n建议：新手选择保守，有经验用户可选择平衡或激进")
        strategy_layout.addRow("调优策略:", self.strategy_combo)
        
        # 干预冷却时间
        self.cooldown_spinbox = QSpinBox()
        self.cooldown_spinbox.setRange(1, 60)
        self.cooldown_spinbox.setValue(self.current_config['intervention_cooldown'])
        self.cooldown_spinbox.setSuffix(" 轮")
        self.cooldown_spinbox.setToolTip("干预冷却时间\n\n说明：两次干预之间的最小间隔轮数，避免频繁干预影响训练稳定性。\n建议值：2-5轮（太小可能过度干预，太大可能错过优化时机）")
        strategy_layout.addRow("干预冷却时间:", self.cooldown_spinbox)
        
        # 最大干预次数
        self.max_interventions_spinbox = QSpinBox()
        self.max_interventions_spinbox.setRange(1, 100)
        self.max_interventions_spinbox.setValue(self.current_config['max_interventions_per_session'])
        self.max_interventions_spinbox.setToolTip("最大干预次数\n\n说明：单次训练会话中允许的最大干预次数，防止过度干预。\n建议值：5-20次（太少可能优化不充分，太多可能影响训练稳定性）")
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
        self.adapter_type_combo.setToolTip("LLM适配器类型\n\n• openai: OpenAI GPT模型（需要API密钥）\n• deepseek: DeepSeek模型（需要API密钥）\n• ollama: 本地Ollama服务\n• custom: 自定义LLM服务\n• mock: 模拟模式（仅用于测试）\n\n⚠️ 生产环境请使用真实LLM服务，避免使用mock")
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
        self.analysis_frequency_combo.setToolTip("分析触发频率\n\n• epoch_based: 基于训练轮数触发分析（推荐）\n• time_based: 基于时间间隔触发分析\n• metric_based: 基于指标变化触发分析\n\n建议：新手选择epoch_based，更稳定可预测")
        llm_layout.addRow("分析频率:", self.analysis_frequency_combo)
        
        # 最小数据点数
        self.min_data_points_spinbox = QSpinBox()
        self.min_data_points_spinbox.setRange(1, 100)
        self.min_data_points_spinbox.setValue(self.current_config['min_data_points'])
        self.min_data_points_spinbox.setToolTip("最小数据点数\n\n说明：触发LLM分析所需的最小训练数据点数，确保有足够数据进行分析。\n建议值：3-10个（太少可能分析不准确，太多可能延迟分析）")
        llm_layout.addRow("最小数据点数:", self.min_data_points_spinbox)
        
        # 置信度阈值
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.1, 1.0)
        self.confidence_spinbox.setSingleStep(0.05)
        self.confidence_spinbox.setDecimals(2)
        self.confidence_spinbox.setValue(self.current_config['confidence_threshold'])
        self.confidence_spinbox.setToolTip("置信度阈值\n\n说明：LLM分析结果的置信度阈值，只有超过此阈值的分析结果才会被采用。\n建议值：0.6-0.8（太低可能采用不准确的分析，太高可能错过有效建议）")
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
        
        # 参数微调报告设置组
        tuning_report_group = QGroupBox("参数微调报告设置")
        tuning_report_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        tuning_report_layout = QFormLayout(tuning_report_group)
        
        # 启用参数微调报告
        self.enable_tuning_reports_checkbox = QCheckBox("启用参数微调报告")
        tuning_reports_config = self.current_config.get('parameter_tuning_reports', {})
        self.enable_tuning_reports_checkbox.setChecked(tuning_reports_config.get('enabled', True))
        self.enable_tuning_reports_checkbox.setToolTip("是否在每次参数微调时生成详细报告")
        tuning_report_layout.addRow("", self.enable_tuning_reports_checkbox)
        
        # 报告保存路径
        path_layout = QHBoxLayout()
        self.tuning_report_path_edit = QLineEdit()
        self.tuning_report_path_edit.setText(tuning_reports_config.get('save_path', 'reports/parameter_tuning'))
        self.tuning_report_path_edit.setToolTip("参数微调报告的保存路径")
        path_layout.addWidget(self.tuning_report_path_edit)
        
        self.browse_path_btn = QPushButton("浏览...")
        self.browse_path_btn.setToolTip("选择报告保存路径")
        self.browse_path_btn.clicked.connect(self.browse_report_path)
        path_layout.addWidget(self.browse_path_btn)
        
        tuning_report_layout.addRow("保存路径:", path_layout)
        
        # 报告格式
        self.tuning_report_format_combo = QComboBox()
        self.tuning_report_format_combo.addItems(['markdown', 'json', 'html'])
        self.tuning_report_format_combo.setCurrentText(tuning_reports_config.get('format', 'markdown'))
        self.tuning_report_format_combo.setToolTip("参数微调报告的格式")
        tuning_report_layout.addRow("报告格式:", self.tuning_report_format_combo)
        
        # 包含LLM分析
        self.include_llm_analysis_checkbox = QCheckBox("包含LLM分析")
        self.include_llm_analysis_checkbox.setChecked(tuning_reports_config.get('include_llm_analysis', True))
        self.include_llm_analysis_checkbox.setToolTip("是否在报告中包含LLM的详细分析")
        tuning_report_layout.addRow("", self.include_llm_analysis_checkbox)
        
        # 包含指标对比
        self.include_metrics_comparison_checkbox = QCheckBox("包含指标对比")
        self.include_metrics_comparison_checkbox.setChecked(tuning_reports_config.get('include_metrics_comparison', True))
        self.include_metrics_comparison_checkbox.setToolTip("是否在报告中包含训练指标对比")
        tuning_report_layout.addRow("", self.include_metrics_comparison_checkbox)
        
        # 包含配置变更
        self.include_config_changes_checkbox = QCheckBox("包含配置变更")
        self.include_config_changes_checkbox.setChecked(tuning_reports_config.get('include_config_changes', True))
        self.include_config_changes_checkbox.setToolTip("是否在报告中包含详细的配置变更信息")
        tuning_report_layout.addRow("", self.include_config_changes_checkbox)
        
        layout.addWidget(tuning_report_group)
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
        
        # 验证配置按钮
        self.validate_btn = QPushButton("验证配置")
        self.validate_btn.setIcon(QIcon(":/icons/check.png"))
        self.validate_btn.setToolTip("检查当前配置的合理性")
        button_layout.addWidget(self.validate_btn)
        
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
        
        # 参数微调报告配置信号
        self.enable_tuning_reports_checkbox.toggled.connect(self.on_config_changed)
        self.tuning_report_path_edit.textChanged.connect(self.on_config_changed)
        self.tuning_report_format_combo.currentTextChanged.connect(self.on_config_changed)
        self.include_llm_analysis_checkbox.toggled.connect(self.on_config_changed)
        self.include_metrics_comparison_checkbox.toggled.connect(self.on_config_changed)
        self.include_config_changes_checkbox.toggled.connect(self.on_config_changed)
        
        # 按钮信号
        self.save_btn.clicked.connect(self.save_config)
        self.reset_btn.clicked.connect(self.reset_to_default)
        self.validate_btn.clicked.connect(self.validate_config)
    
    def browse_report_path(self):
        """浏览报告保存路径"""
        try:
            current_path = self.tuning_report_path_edit.text()
            if not current_path:
                current_path = "reports/parameter_tuning"
            
            # 确保路径存在
            if not os.path.exists(current_path):
                os.makedirs(current_path, exist_ok=True)
            
            # 打开文件夹选择对话框
            folder_path = QFileDialog.getExistingDirectory(
                self, 
                "选择参数微调报告保存路径", 
                current_path,
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )
            
            if folder_path:
                self.tuning_report_path_edit.setText(folder_path)
                self.on_config_changed()
                
        except Exception as e:
            QMessageBox.warning(self, "错误", f"选择路径时发生错误: {str(e)}")
    
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
            'save_intervention_details': self.save_intervention_details_checkbox.isChecked(),
            
            # 参数微调报告配置
            'parameter_tuning_reports': {
                'enabled': self.enable_tuning_reports_checkbox.isChecked(),
                'save_path': self.tuning_report_path_edit.text(),
                'format': self.tuning_report_format_combo.currentText(),
                'include_llm_analysis': self.include_llm_analysis_checkbox.isChecked(),
                'include_metrics_comparison': self.include_metrics_comparison_checkbox.isChecked(),
                'include_config_changes': self.include_config_changes_checkbox.isChecked()
            }
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
        
        # 更新参数微调报告UI控件
        tuning_reports_config = config.get('parameter_tuning_reports', self.default_config['parameter_tuning_reports'])
        self.enable_tuning_reports_checkbox.setChecked(tuning_reports_config.get('enabled', True))
        self.tuning_report_path_edit.setText(tuning_reports_config.get('save_path', 'reports/parameter_tuning'))
        self.tuning_report_format_combo.setCurrentText(tuning_reports_config.get('format', 'markdown'))
        self.include_llm_analysis_checkbox.setChecked(tuning_reports_config.get('include_llm_analysis', True))
        self.include_metrics_comparison_checkbox.setChecked(tuning_reports_config.get('include_metrics_comparison', True))
        self.include_config_changes_checkbox.setChecked(tuning_reports_config.get('include_config_changes', True))
        
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
                'save_intervention_details': self.current_config['save_intervention_details'],
                
                # 参数微调报告配置
                'parameter_tuning_reports': self.current_config['parameter_tuning_reports']
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
    
    def validate_config(self):
        """验证配置的合理性"""
        try:
            self.update_current_config()
            config = self.current_config
            
            warnings = []
            errors = []
            
            # 检查基本参数合理性
            if config['max_iterations'] < 2:
                warnings.append("最大迭代次数建议至少为2，当前值可能过小")
            elif config['max_iterations'] > 15:
                warnings.append("最大迭代次数建议不超过15，当前值可能过大")
            
            if config['min_iteration_epochs'] < 2:
                warnings.append("最小迭代轮数建议至少为2，当前值可能过小")
            elif config['min_iteration_epochs'] > 20:
                warnings.append("最小迭代轮数建议不超过20，当前值可能过大")
            
            if config['analysis_interval'] < 1:
                errors.append("分析间隔必须至少为1")
            elif config['analysis_interval'] > 10:
                warnings.append("分析间隔建议不超过10，当前值可能过大")
            
            # 检查阈值合理性
            if config['convergence_threshold'] < 0.001:
                warnings.append("收敛阈值过小，可能导致过早停止训练")
            elif config['convergence_threshold'] > 0.1:
                warnings.append("收敛阈值过大，可能导致训练时间过长")
            
            if config['improvement_threshold'] < 0.001:
                warnings.append("改进阈值过小，可能对微小改进过于敏感")
            elif config['improvement_threshold'] > 0.1:
                warnings.append("改进阈值过大，可能错过有效改进")
            
            # 检查干预参数
            if config['overfitting_threshold'] < 0.5:
                warnings.append("过拟合阈值过小，可能过于敏感")
            elif config['overfitting_threshold'] > 0.95:
                warnings.append("过拟合阈值过大，可能错过过拟合问题")
            
            if config['underfitting_threshold'] < 0.3:
                warnings.append("欠拟合阈值过小，可能过于严格")
            elif config['underfitting_threshold'] > 0.9:
                warnings.append("欠拟合阈值过大，可能错过欠拟合问题")
            
            if config['stagnation_epochs'] < 3:
                warnings.append("停滞轮数阈值过小，可能误判正常波动")
            elif config['stagnation_epochs'] > 30:
                warnings.append("停滞轮数阈值过大，可能错过优化时机")
            
            # 检查LLM配置
            if config['adapter_type'] == 'mock' and config['llm_analysis_enabled']:
                warnings.append("⚠️ 当前使用mock适配器，LLM分析功能将无法正常工作")
            
            if config['min_data_points'] < 2:
                warnings.append("最小数据点数建议至少为2")
            elif config['min_data_points'] > 20:
                warnings.append("最小数据点数建议不超过20")
            
            # 显示验证结果
            if errors:
                error_msg = "发现配置错误：\n" + "\n".join(f"❌ {error}" for error in errors)
                QMessageBox.critical(self, "配置错误", error_msg)
            elif warnings:
                warning_msg = "配置验证完成，发现以下建议：\n\n" + "\n".join(f"⚠️ {warning}" for warning in warnings)
                QMessageBox.warning(self, "配置建议", warning_msg)
            else:
                QMessageBox.information(self, "配置验证", "✅ 配置验证通过，所有参数设置合理！")
                
        except Exception as e:
            QMessageBox.critical(self, "验证错误", f"配置验证失败: {str(e)}")
