"""
智能训练控制器UI组件

提供智能训练控制器的用户界面，包括：
- 监控状态显示
- 干预历史查看
- 配置参数调整
- 会话报告管理
"""

import os
import json
import time
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QLabel, QGroupBox, QTextEdit, QTableWidget, 
                           QTableWidgetItem, QHeaderView, QSpinBox, QDoubleSpinBox,
                           QCheckBox, QComboBox, QProgressBar, QSplitter, QFrame,
                           QMessageBox, QFileDialog, QTabWidget, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, pyqtSlot
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon

from src.training_components.intelligent_training_controller import IntelligentTrainingController


class IntelligentTrainingWidget(QWidget):
    """智能训练控制器UI组件"""
    
    # 信号定义
    start_monitoring_requested = pyqtSignal(dict)  # 请求开始监控
    stop_monitoring_requested = pyqtSignal()       # 请求停止监控
    restart_training_requested = pyqtSignal(dict)  # 请求重启训练
    
    def __init__(self, training_system=None, parent=None, use_external_controller=False, external_manager=None):
        super().__init__(parent)
        self.training_system = training_system
        self.intelligent_controller = None
        self._use_external_controller = use_external_controller
        self._external_manager = external_manager
        
        # UI组件
        self.status_label = None
        self.monitoring_btn = None
        self.progress_bar = None
        self.intervention_table = None
        self.session_info_display = None
        self.config_widgets = {}
        
        # 状态管理
        self.is_monitoring = False
        self.update_timer = None
        
        # 初始化UI
        self.init_ui()
        self.init_controller()
        self.setup_timers()
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 标题
        title_label = QLabel("🤖 智能训练控制器")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 主控制区域
        control_group = QGroupBox("🎮 控制面板")
        control_layout = QVBoxLayout()
        
        # 状态显示
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("状态:"))
        self.status_label = QLabel("未启动")
        self.status_label.setStyleSheet("color: gray; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        control_layout.addLayout(status_layout)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        
        self.monitoring_btn = QPushButton("启动智能监控")
        self.monitoring_btn.clicked.connect(self.on_monitoring_clicked)
        self.monitoring_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        button_layout.addWidget(self.monitoring_btn)
        
        self.stop_btn = QPushButton("停止监控")
        self.stop_btn.clicked.connect(self.on_stop_clicked)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.stop_btn)
        
        button_layout.addStretch()
        control_layout.addLayout(button_layout)
        
        # 进度条
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("监控进度:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        control_layout.addLayout(progress_layout)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # 创建标签页
        tab_widget = QTabWidget()
        
        # 监控状态标签页
        monitoring_tab = self.create_monitoring_tab()
        tab_widget.addTab(monitoring_tab, "📊 监控状态")
        
        # 干预历史标签页
        intervention_tab = self.create_intervention_tab()
        tab_widget.addTab(intervention_tab, "📝 干预历史")
        
        # 配置设置标签页
        config_tab = self.create_config_tab()
        tab_widget.addTab(config_tab, "⚙️ 配置设置")
        
        # 会话报告标签页
        report_tab = self.create_report_tab()
        tab_widget.addTab(report_tab, "📋 会话报告")
        
        layout.addWidget(tab_widget)
        
    def create_monitoring_tab(self):
        """创建监控状态标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 会话信息组
        session_group = QGroupBox("📋 当前训练会话")
        session_layout = QVBoxLayout()
        
        self.session_info_display = QTextEdit()
        self.session_info_display.setMaximumHeight(150)
        self.session_info_display.setReadOnly(True)
        self.session_info_display.setPlaceholderText("等待训练会话启动...")
        session_layout.addWidget(self.session_info_display)
        
        session_group.setLayout(session_layout)
        layout.addWidget(session_group)
        
        # 实时指标组
        metrics_group = QGroupBox("📈 实时训练指标")
        metrics_layout = QVBoxLayout()
        
        self.metrics_display = QTextEdit()
        self.metrics_display.setMaximumHeight(120)
        self.metrics_display.setReadOnly(True)
        self.metrics_display.setPlaceholderText("等待训练数据...")
        metrics_layout.addWidget(self.metrics_display)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # 分析结果组
        analysis_group = QGroupBox("🧠 AI分析结果")
        analysis_layout = QVBoxLayout()
        
        self.analysis_display = QTextEdit()
        self.analysis_display.setMaximumHeight(120)
        self.analysis_display.setReadOnly(True)
        self.analysis_display.setPlaceholderText("等待AI分析...")
        analysis_layout.addWidget(self.analysis_display)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        layout.addStretch()
        return widget
    
    def create_intervention_tab(self):
        """创建干预历史标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 干预历史表格
        self.intervention_table = QTableWidget()
        self.intervention_table.setColumnCount(7)
        self.intervention_table.setHorizontalHeaderLabels([
            "干预ID", "时间", "触发原因", "干预类型", "状态", "参数建议", "操作"
        ])
        
        # 设置表格属性
        header = self.intervention_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        
        layout.addWidget(self.intervention_table)
        
        # 操作按钮
        button_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.clicked.connect(self.refresh_intervention_table)
        button_layout.addWidget(refresh_btn)
        
        export_btn = QPushButton("📤 导出")
        export_btn.clicked.connect(self.export_intervention_history)
        button_layout.addWidget(export_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        return widget
    
    def create_config_tab(self):
        """创建配置设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 基本设置组
        basic_group = QGroupBox("🔧 基本设置")
        basic_layout = QVBoxLayout()
        
        # 自动干预开关
        self.config_widgets['auto_intervention_enabled'] = QCheckBox("启用自动干预")
        self.config_widgets['auto_intervention_enabled'].setChecked(True)
        basic_layout.addWidget(self.config_widgets['auto_intervention_enabled'])
        
        # 分析间隔
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("分析间隔:"))
        self.config_widgets['analysis_interval'] = QSpinBox()
        self.config_widgets['analysis_interval'].setRange(1, 100)
        self.config_widgets['analysis_interval'].setValue(10)
        self.config_widgets['analysis_interval'].setSuffix(" 轮")
        interval_layout.addWidget(self.config_widgets['analysis_interval'])
        interval_layout.addStretch()
        basic_layout.addLayout(interval_layout)
        
        # 最大干预次数
        max_interventions_layout = QHBoxLayout()
        max_interventions_layout.addWidget(QLabel("最大干预次数:"))
        self.config_widgets['max_interventions_per_session'] = QSpinBox()
        self.config_widgets['max_interventions_per_session'].setRange(1, 10)
        self.config_widgets['max_interventions_per_session'].setValue(3)
        max_interventions_layout.addWidget(self.config_widgets['max_interventions_per_session'])
        max_interventions_layout.addStretch()
        basic_layout.addLayout(max_interventions_layout)
        
        basic_group.setLayout(basic_layout)
        scroll_layout.addWidget(basic_group)
        
        # 阈值设置组
        thresholds_group = QGroupBox("📊 干预阈值设置")
        thresholds_layout = QVBoxLayout()
        
        # 过拟合风险阈值
        overfitting_layout = QHBoxLayout()
        overfitting_layout.addWidget(QLabel("过拟合风险阈值:"))
        self.config_widgets['overfitting_risk'] = QDoubleSpinBox()
        self.config_widgets['overfitting_risk'].setRange(0.1, 2.0)
        self.config_widgets['overfitting_risk'].setSingleStep(0.1)
        self.config_widgets['overfitting_risk'].setValue(0.8)
        overfitting_layout.addWidget(self.config_widgets['overfitting_risk'])
        overfitting_layout.addStretch()
        thresholds_layout.addLayout(overfitting_layout)
        
        # 欠拟合风险阈值
        underfitting_layout = QHBoxLayout()
        underfitting_layout.addWidget(QLabel("欠拟合风险阈值:"))
        self.config_widgets['underfitting_risk'] = QDoubleSpinBox()
        self.config_widgets['underfitting_risk'].setRange(0.1, 2.0)
        self.config_widgets['underfitting_risk'].setSingleStep(0.1)
        self.config_widgets['underfitting_risk'].setValue(0.7)
        underfitting_layout.addWidget(self.config_widgets['underfitting_risk'])
        underfitting_layout.addStretch()
        thresholds_layout.addLayout(underfitting_layout)
        
        # 停滞轮数阈值
        stagnation_layout = QHBoxLayout()
        stagnation_layout.addWidget(QLabel("停滞轮数阈值:"))
        self.config_widgets['stagnation_epochs'] = QSpinBox()
        self.config_widgets['stagnation_epochs'].setRange(1, 20)
        self.config_widgets['stagnation_epochs'].setValue(5)
        stagnation_layout.addWidget(self.config_widgets['stagnation_epochs'])
        stagnation_layout.addStretch()
        thresholds_layout.addLayout(stagnation_layout)
        
        # 发散阈值
        divergence_layout = QHBoxLayout()
        divergence_layout.addWidget(QLabel("发散阈值:"))
        self.config_widgets['divergence_threshold'] = QDoubleSpinBox()
        self.config_widgets['divergence_threshold'].setRange(0.5, 5.0)
        self.config_widgets['divergence_threshold'].setSingleStep(0.1)
        self.config_widgets['divergence_threshold'].setValue(2.0)
        divergence_layout.addWidget(self.config_widgets['divergence_threshold'])
        divergence_layout.addStretch()
        thresholds_layout.addLayout(divergence_layout)
        
        # 最小训练轮数
        min_epochs_layout = QHBoxLayout()
        min_epochs_layout.addWidget(QLabel("最小训练轮数:"))
        self.config_widgets['min_training_epochs'] = QSpinBox()
        self.config_widgets['min_training_epochs'].setRange(1, 10)
        self.config_widgets['min_training_epochs'].setValue(3)
        min_epochs_layout.addWidget(self.config_widgets['min_training_epochs'])
        min_epochs_layout.addStretch()
        thresholds_layout.addLayout(min_epochs_layout)
        
        thresholds_group.setLayout(thresholds_layout)
        scroll_layout.addWidget(thresholds_group)
        
        # 参数调优策略组
        strategy_group = QGroupBox("🎯 参数调优策略")
        strategy_layout = QVBoxLayout()
        
        strategy_layout.addWidget(QLabel("调优策略:"))
        self.config_widgets['parameter_tuning_strategy'] = QComboBox()
        self.config_widgets['parameter_tuning_strategy'].addItems([
            "保守", "平衡", "激进"
        ])
        strategy_layout.addWidget(self.config_widgets['parameter_tuning_strategy'])
        
        strategy_group.setLayout(strategy_layout)
        scroll_layout.addWidget(strategy_group)
        
        # 配置操作按钮
        config_buttons_layout = QHBoxLayout()
        
        save_config_btn = QPushButton("💾 保存配置")
        save_config_btn.clicked.connect(self.save_config)
        config_buttons_layout.addWidget(save_config_btn)
        
        load_config_btn = QPushButton("📂 加载配置")
        load_config_btn.clicked.connect(self.load_config)
        config_buttons_layout.addWidget(load_config_btn)
        
        reset_config_btn = QPushButton("🔄 重置默认")
        reset_config_btn.clicked.connect(self.reset_config)
        config_buttons_layout.addWidget(reset_config_btn)
        
        config_buttons_layout.addStretch()
        scroll_layout.addLayout(config_buttons_layout)
        
        scroll_layout.addStretch()
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        return widget
    
    def create_report_tab(self):
        """创建会话报告标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 会话信息显示
        report_group = QGroupBox("📋 会话报告")
        report_layout = QVBoxLayout()
        
        self.report_display = QTextEdit()
        self.report_display.setReadOnly(True)
        self.report_display.setPlaceholderText("等待会话数据...")
        report_layout.addWidget(self.report_display)
        
        report_group.setLayout(report_layout)
        layout.addWidget(report_group)
        
        # 操作按钮
        report_buttons_layout = QHBoxLayout()
        
        generate_report_btn = QPushButton("📊 生成报告")
        generate_report_btn.clicked.connect(self.generate_report)
        report_buttons_layout.addWidget(generate_report_btn)
        
        save_report_btn = QPushButton("💾 保存报告")
        save_report_btn.clicked.connect(self.save_report)
        report_buttons_layout.addWidget(save_report_btn)
        
        export_report_btn = QPushButton("📤 导出报告")
        export_report_btn.clicked.connect(self.export_report)
        report_buttons_layout.addWidget(export_report_btn)
        
        report_buttons_layout.addStretch()
        layout.addLayout(report_buttons_layout)
        
        return widget
    
    def init_controller(self):
        """初始化智能训练控制器"""
        try:
            if self._use_external_controller:
                self.intelligent_controller = None
                if hasattr(self, 'status_label') and self.status_label:
                    self.status_label.setText("已连接外部智能训练管理器")
                return
            self.intelligent_controller = IntelligentTrainingController(self.training_system)
            
            # 连接信号
            self.intelligent_controller.intervention_triggered.connect(self.on_intervention_triggered)
            self.intelligent_controller.training_restarted.connect(self.on_training_restarted)
            self.intelligent_controller.analysis_completed.connect(self.on_analysis_completed)
            self.intelligent_controller.status_updated.connect(self.on_status_updated)
            self.intelligent_controller.error_occurred.connect(self.on_error_occurred)
            
            self.status_updated.emit("智能训练控制器初始化完成")
            
        except Exception as e:
            self.on_error_occurred(f"初始化控制器失败: {str(e)}")
    
    def setup_timers(self):
        """设置定时器"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(2000)  # 每2秒更新一次
    
    def on_monitoring_clicked(self):
        """监控按钮点击事件"""
        if not self.is_monitoring:
            # 启动监控（外部/内部模式均发射请求，由外部或内部处理）
            config = self.get_current_config()
            self.start_monitoring_requested.emit(config)
        else:
            # 停止监控
            self.stop_monitoring_requested.emit()
    
    def on_stop_clicked(self):
        """停止按钮点击事件"""
        self.stop_monitoring_requested.emit()
    
    def get_current_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        config = {}
        
        # 基本设置
        config['auto_intervention_enabled'] = self.config_widgets['auto_intervention_enabled'].isChecked()
        config['analysis_interval'] = self.config_widgets['analysis_interval'].value()
        config['max_interventions_per_session'] = self.config_widgets['max_interventions_per_session'].value()
        
        # 阈值设置
        config['intervention_thresholds'] = {
            'overfitting_risk': self.config_widgets['overfitting_risk'].value(),
            'underfitting_risk': self.config_widgets['underfitting_risk'].value(),
            'stagnation_epochs': self.config_widgets['stagnation_epochs'].value(),
            'divergence_threshold': self.config_widgets['divergence_threshold'].value(),
            'min_training_epochs': self.config_widgets['min_training_epochs'].value(),
        }
        
        # 策略设置
        strategy_map = {"保守": "conservative", "平衡": "balanced", "激进": "aggressive"}
        config['parameter_tuning_strategy'] = strategy_map.get(
            self.config_widgets['parameter_tuning_strategy'].currentText(), "conservative"
        )
        
        return config
    
    def start_monitoring(self, training_config: Dict[str, Any]):
        """开始监控"""
        if not self._use_external_controller and self.intelligent_controller:
            self.intelligent_controller.start_monitoring(training_config)
        self.is_monitoring = True
        self.monitoring_btn.setText("停止智能监控")
        self.monitoring_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("监控中")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self._use_external_controller and self.intelligent_controller:
            self.intelligent_controller.stop_monitoring()
        self.is_monitoring = False
        self.monitoring_btn.setText("启动智能监控")
        self.monitoring_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("已停止")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def update_display(self):
        """更新显示"""
        if self.is_monitoring:
            # 更新会话信息
            session_info = None
            if self._use_external_controller and self._external_manager:
                session_info = self._external_manager.get_current_session_info()
            elif self.intelligent_controller:
                session_info = self.intelligent_controller.get_current_session_info()
            if session_info:
                self.update_session_display(session_info)
            
            # 更新干预历史表格
            self.refresh_intervention_table()
            
            # 更新进度条
            if session_info:
                progress = (session_info.get('completed_epochs', 0) / 
                           session_info.get('total_epochs', 1)) * 100
                self.progress_bar.setValue(int(progress))
    
    def update_session_display(self, session_info: Dict[str, Any]):
        """更新会话信息显示"""
        if not session_info:
            return
        
        info_text = f"""
📋 训练会话信息
================
会话ID: {session_info.get('session_id', 'N/A')}
状态: {session_info.get('status', 'N/A')}
开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_info.get('start_time', 0)))}
总轮数: {session_info.get('total_epochs', 0)}
已完成轮数: {session_info.get('completed_epochs', 0)}
干预次数: {len(session_info.get('interventions', []))}
        """
        
        if session_info.get('end_time'):
            info_text += f"\n结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_info['end_time']))}"
        
        self.session_info_display.setText(info_text)
    
    def refresh_intervention_table(self):
        """刷新干预历史表格"""
        if self._use_external_controller and self._external_manager:
            interventions = self._external_manager.get_intervention_history()
        elif self.intelligent_controller:
            interventions = self.intelligent_controller.get_intervention_history()
        else:
            return
        self.intervention_table.setRowCount(len(interventions))
        
        for row, intervention in enumerate(interventions):
            # 干预ID
            self.intervention_table.setItem(row, 0, QTableWidgetItem(intervention.get('intervention_id', '')))
            
            # 时间
            timestamp = intervention.get('timestamp', 0)
            time_str = time.strftime('%H:%M:%S', time.localtime(timestamp))
            self.intervention_table.setItem(row, 1, QTableWidgetItem(time_str))
            
            # 触发原因
            self.intervention_table.setItem(row, 2, QTableWidgetItem(intervention.get('trigger_reason', '')))
            
            # 干预类型
            self.intervention_table.setItem(row, 3, QTableWidgetItem(intervention.get('intervention_type', '')))
            
            # 状态
            status_item = QTableWidgetItem(intervention.get('status', ''))
            status = intervention.get('status', '')
            if status == 'completed':
                status_item.setBackground(QColor(144, 238, 144))  # 浅绿色
            elif status == 'failed':
                status_item.setBackground(QColor(255, 182, 193))  # 浅红色
            elif status == 'executing':
                status_item.setBackground(QColor(255, 255, 224))  # 浅黄色
            self.intervention_table.setItem(row, 4, status_item)
            
            # 参数建议
            suggested_params = intervention.get('suggested_params', {})
            params_text = ', '.join([f"{k}: {v}" for k, v in suggested_params.items()])
            self.intervention_table.setItem(row, 5, QTableWidgetItem(params_text))
            
            # 操作按钮
            if intervention.get('status') == 'completed':
                restart_btn = QPushButton("重启训练")
                restart_btn.clicked.connect(lambda checked, data=intervention: self.on_restart_training(data))
                self.intervention_table.setCellWidget(row, 6, restart_btn)
    
    def on_intervention_triggered(self, intervention_data: Dict[str, Any]):
        """干预触发事件"""
        self.status_updated.emit(f"检测到训练问题，触发干预: {intervention_data.get('trigger_reason', '')}")
        
        # 更新干预历史表格
        self.refresh_intervention_table()
    
    def on_training_restarted(self, restart_data: Dict[str, Any]):
        """训练重启事件"""
        self.status_updated.emit("训练已使用优化参数重启")
        
        # 发送重启训练请求
        self.restart_training_requested.emit(restart_data)
    
    def on_analysis_completed(self, analysis_data: Dict[str, Any]):
        """分析完成事件"""
        # 更新分析结果显示
        analysis_result = analysis_data.get('analysis_result', {})
        if 'combined_insights' in analysis_result:
            self.analysis_display.setText(analysis_result['combined_insights'])
        
        self.status_updated.emit("AI分析完成")
    
    def on_status_updated(self, status: str):
        """状态更新事件"""
        self.status_label.setText(status)
    
    def on_error_occurred(self, error: str):
        """错误事件"""
        self.status_label.setText(f"错误: {error}")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        
        # 显示错误对话框
        QMessageBox.warning(self, "智能训练控制器错误", error)
    
    def on_restart_training(self, intervention_data: Dict[str, Any]):
        """重启训练事件"""
        suggested_params = intervention_data.get('suggested_params', {})
        if suggested_params:
            reply = QMessageBox.question(
                self, 
                "确认重启训练", 
                f"是否使用以下优化参数重启训练？\n{json.dumps(suggested_params, ensure_ascii=False, indent=2)}",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.restart_training_requested.emit({
                    'intervention_id': intervention_data.get('intervention_id'),
                    'suggested_params': suggested_params
                })
    
    def save_config(self):
        """保存配置"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存配置", "", "JSON文件 (*.json)"
            )
            
            if file_path:
                config = self.get_current_config()
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                
                QMessageBox.information(self, "成功", f"配置已保存到: {file_path}")
                
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存配置失败: {str(e)}")
    
    def load_config(self):
        """加载配置"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "加载配置", "", "JSON文件 (*.json)"
            )
            
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                self.apply_config(config)
                QMessageBox.information(self, "成功", f"配置已从 {file_path} 加载")
                
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载配置失败: {str(e)}")
    
    def apply_config(self, config: Dict[str, Any]):
        """应用配置到UI"""
        try:
            # 基本设置
            if 'auto_intervention_enabled' in config:
                self.config_widgets['auto_intervention_enabled'].setChecked(config['auto_intervention_enabled'])
            
            if 'analysis_interval' in config:
                self.config_widgets['analysis_interval'].setValue(config['analysis_interval'])
            
            if 'max_interventions_per_session' in config:
                self.config_widgets['max_interventions_per_session'].setValue(config['max_interventions_per_session'])
            
            # 阈值设置
            if 'intervention_thresholds' in config:
                thresholds = config['intervention_thresholds']
                if 'overfitting_risk' in thresholds:
                    self.config_widgets['overfitting_risk'].setValue(thresholds['overfitting_risk'])
                if 'underfitting_risk' in thresholds:
                    self.config_widgets['underfitting_risk'].setValue(thresholds['underfitting_risk'])
                if 'stagnation_epochs' in thresholds:
                    self.config_widgets['stagnation_epochs'].setValue(thresholds['stagnation_epochs'])
                if 'divergence_threshold' in thresholds:
                    self.config_widgets['divergence_threshold'].setValue(thresholds['divergence_threshold'])
                if 'min_training_epochs' in thresholds:
                    self.config_widgets['min_training_epochs'].setValue(thresholds['min_training_epochs'])
            
            # 策略设置
            if 'parameter_tuning_strategy' in config:
                strategy_map = {"conservative": 0, "balanced": 1, "aggressive": 2}
                strategy_index = strategy_map.get(config['parameter_tuning_strategy'], 0)
                self.config_widgets['parameter_tuning_strategy'].setCurrentIndex(strategy_index)
                
        except Exception as e:
            QMessageBox.warning(self, "错误", f"应用配置失败: {str(e)}")
    
    def reset_config(self):
        """重置为默认配置"""
        reply = QMessageBox.question(
            self, 
            "确认重置", 
            "确定要重置为默认配置吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 重置为默认值
            self.config_widgets['auto_intervention_enabled'].setChecked(True)
            self.config_widgets['analysis_interval'].setValue(10)
            self.config_widgets['max_interventions_per_session'].setValue(3)
            self.config_widgets['overfitting_risk'].setValue(0.8)
            self.config_widgets['underfitting_risk'].setValue(0.7)
            self.config_widgets['stagnation_epochs'].setValue(5)
            self.config_widgets['divergence_threshold'].setValue(2.0)
            self.config_widgets['min_training_epochs'].setValue(3)
            self.config_widgets['parameter_tuning_strategy'].setCurrentIndex(0)
            
            QMessageBox.information(self, "成功", "配置已重置为默认值")
    
    def export_intervention_history(self):
        """导出干预历史"""
        try:
            # 外部/内部两种模式均支持
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "导出干预历史", "", "JSON文件 (*.json)"
            )
            
            if file_path:
                if self._use_external_controller and self._external_manager:
                    interventions = self._external_manager.get_intervention_history()
                else:
                    if not self.intelligent_controller:
                        return
                    interventions = self.intelligent_controller.get_intervention_history()
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(interventions, f, ensure_ascii=False, indent=2)
                
                QMessageBox.information(self, "成功", f"干预历史已导出到: {file_path}")
                
        except Exception as e:
            QMessageBox.warning(self, "错误", f"导出干预历史失败: {str(e)}")
    
    def generate_report(self):
        """生成会话报告"""
        try:
            if self._use_external_controller and self._external_manager:
                session_info = self._external_manager.get_current_session_info()
            else:
                if not self.intelligent_controller:
                    return
                session_info = self.intelligent_controller.get_current_session_info()
            if not session_info:
                QMessageBox.information(self, "提示", "没有可用的会话信息")
                return
            
            # 生成报告内容
            report_text = self._generate_report_content(session_info)
            self.report_display.setText(report_text)
            
            QMessageBox.information(self, "成功", "会话报告已生成")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"生成报告失败: {str(e)}")
    
    def _generate_report_content(self, session_info: Dict[str, Any]) -> str:
        """生成报告内容"""
        report = f"""
📊 智能训练会话报告
====================

📋 基本信息
-----------
会话ID: {session_info.get('session_id', 'N/A')}
状态: {session_info.get('status', 'N/A')}
开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_info.get('start_time', 0)))}
总轮数: {session_info.get('total_epochs', 0)}
已完成轮数: {session_info.get('completed_epochs', 0)}

📈 训练进度
-----------
完成率: {(session_info.get('completed_epochs', 0) / session_info.get('total_epochs', 1)) * 100:.1f}%

🔧 干预记录
-----------
总干预次数: {len(session_info.get('interventions', []))}
        """
        
        if session_info.get('end_time'):
            duration = session_info['end_time'] - session_info['start_time']
            report += f"\n训练时长: {duration/3600:.1f} 小时 ({duration/60:.1f} 分钟)"
        
        # 添加干预详情
        interventions = session_info.get('interventions', [])
        if interventions:
            report += "\n\n详细干预记录:\n"
            for i, intervention in enumerate(interventions, 1):
                report += f"\n{i}. 干预ID: {intervention.get('intervention_id', 'N/A')}"
                report += f"\n   时间: {time.strftime('%H:%M:%S', time.localtime(intervention.get('timestamp', 0)))}"
                report += f"\n   原因: {intervention.get('trigger_reason', 'N/A')}"
                report += f"\n   类型: {intervention.get('intervention_type', 'N/A')}"
                report += f"\n   状态: {intervention.get('status', 'N/A')}"
                
                suggested_params = intervention.get('suggested_params', {})
                if suggested_params:
                    report += f"\n   建议参数: {json.dumps(suggested_params, ensure_ascii=False, indent=2)}"
        
        return report
    
    def save_report(self):
        """保存报告"""
        try:
            report_text = self.report_display.toPlainText()
            if not report_text.strip():
                QMessageBox.information(self, "提示", "请先生成报告")
                return
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存报告", "", "文本文件 (*.txt)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                
                QMessageBox.information(self, "成功", f"报告已保存到: {file_path}")
                
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存报告失败: {str(e)}")
    
    def export_report(self):
        """导出报告"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "导出报告", "", "JSON文件 (*.json)"
            )
            
            if file_path:
                if self._use_external_controller and self._external_manager:
                    # 使用管理器的报告生成功能
                    self._external_manager.generate_session_report(file_path)
                else:
                    if not self.intelligent_controller:
                        return
                    self.intelligent_controller.save_session_report(file_path)
                
        except Exception as e:
            QMessageBox.warning(self, "错误", f"导出报告失败: {str(e)}")
    
    def update_training_progress(self, metrics: Dict[str, Any]):
        """更新训练进度"""
        if self._use_external_controller and self._external_manager:
            try:
                self._external_manager.update_training_progress(metrics)
            except Exception:
                pass
        elif self.intelligent_controller:
            self.intelligent_controller.update_training_progress(metrics)
        
        # 更新实时指标显示
        metrics_text = f"""
📊 实时训练指标
===============
轮数: {metrics.get('epoch', 'N/A')}
批次: {metrics.get('batch', 'N/A')}
阶段: {metrics.get('phase', 'N/A')}
训练损失: {metrics.get('train_loss', 'N/A'):.4f}
验证损失: {metrics.get('val_loss', 'N/A'):.4f}
训练准确率: {metrics.get('train_accuracy', 'N/A'):.4f}
验证准确率: {metrics.get('val_accuracy', 'N/A'):.4f}
学习率: {metrics.get('learning_rate', 'N/A'):.6f}
        """
        
        self.metrics_display.setText(metrics_text) 