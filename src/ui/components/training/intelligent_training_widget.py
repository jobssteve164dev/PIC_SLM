"""
智能训练组件

提供智能训练的UI界面和控制功能
主要功能：
- 智能训练启动和停止
- 实时显示训练状态和进度
- 显示配置调整历史
- 提供训练报告导出
"""

import os
import json
import time
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                           QLabel, QPushButton, QTextEdit, QProgressBar,
                           QTableWidget, QTableWidgetItem, QTabWidget,
                           QMessageBox, QFileDialog, QSplitter, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor

from src.training_components.intelligent_training_orchestrator import IntelligentTrainingOrchestrator


class IntelligentTrainingWidget(QWidget):
    """智能训练组件"""
    
    # 信号定义
    training_started = pyqtSignal(dict)      # 训练开始信号
    training_stopped = pyqtSignal(dict)      # 训练停止信号
    status_updated = pyqtSignal(str)         # 状态更新信号
    start_monitoring_requested = pyqtSignal(dict)  # 开始监控请求信号
    stop_monitoring_requested = pyqtSignal()       # 停止监控请求信号
    restart_training_requested = pyqtSignal(dict)  # 重启训练请求信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.training_tab = parent  # 直接保存父组件（TrainingTab）的引用
        self.main_window = None
        self.is_monitoring = False
        self.orchestrator = None
        self.current_config = {}
        
        # 查找主窗口引用
        if hasattr(parent, 'main_window'):
            self.main_window = parent.main_window
        elif hasattr(parent, 'parent') and hasattr(parent.parent(), 'main_window'):
            self.main_window = parent.parent().main_window
        
        # 智能训练编排器
        self.orchestrator = IntelligentTrainingOrchestrator()
        
        # 初始化UI
        self.init_ui()
        
        # 连接信号
        self.connect_signals()
        
        # 启动状态更新定时器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status_display)
        self.status_timer.start(1000)  # 每秒更新一次
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 创建标题
        title_label = QLabel("🤖 智能训练系统")
        title_label.setFont(QFont('微软雅黑', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # 右侧信息面板
        info_panel = self.create_info_panel()
        splitter.addWidget(info_panel)
        
        # 设置分割器比例
        splitter.setSizes([300, 500])
        
    def create_control_panel(self) -> QWidget:
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # 训练控制组
        control_group = QGroupBox("训练控制")
        control_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        control_layout = QVBoxLayout(control_group)
        
        # 启动按钮
        self.start_button = QPushButton("🚀 启动智能训练")
        self.start_button.setFont(QFont('微软雅黑', 10))
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.start_button.clicked.connect(self.start_intelligent_training)
        control_layout.addWidget(self.start_button)
        
        # 停止按钮
        self.stop_button = QPushButton("⏹️ 停止智能训练")
        self.stop_button.setFont(QFont('微软雅黑', 10))
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
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
        self.stop_button.clicked.connect(self.stop_intelligent_training)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        layout.addWidget(control_group)
        
        # 状态显示组
        status_group = QGroupBox("训练状态")
        status_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        status_layout = QVBoxLayout(status_group)
        
        # 状态标签
        self.status_label = QLabel("状态: 未启动")
        self.status_label.setFont(QFont('微软雅黑', 9))
        status_layout.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        
        # 会话信息
        self.session_info_label = QLabel("会话: 无")
        self.session_info_label.setFont(QFont('微软雅黑', 9))
        status_layout.addWidget(self.session_info_label)
        
        layout.addWidget(status_group)
        
        # 配置信息组
        config_group = QGroupBox("当前配置")
        config_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        config_layout = QVBoxLayout(config_group)
        
        # 配置显示
        self.config_display = QTextEdit()
        self.config_display.setMaximumHeight(150)
        self.config_display.setFont(QFont('Consolas', 8))
        self.config_display.setReadOnly(True)
        config_layout.addWidget(self.config_display)
        
        layout.addWidget(config_group)
        
        # 操作按钮组
        actions_group = QGroupBox("操作")
        actions_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        actions_layout = QVBoxLayout(actions_group)
        
        # 导出报告按钮
        self.export_button = QPushButton("📊 导出训练报告")
        self.export_button.setFont(QFont('微软雅黑', 9))
        self.export_button.clicked.connect(self.export_training_report)
        actions_layout.addWidget(self.export_button)
        
        # 清除历史按钮
        self.clear_button = QPushButton("🗑️ 清除历史记录")
        self.clear_button.setFont(QFont('微软雅黑', 9))
        self.clear_button.clicked.connect(self.clear_history)
        actions_layout.addWidget(self.clear_button)
        
        layout.addWidget(actions_group)
        
        # 添加弹性空间
        layout.addStretch()
        
        return panel
        
    def create_info_panel(self) -> QWidget:
        """创建信息面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # 实时日志标签页
        self.log_tab = self.create_log_tab()
        self.tab_widget.addTab(self.log_tab, "📝 实时日志")
        
        # 调整历史标签页
        self.history_tab = self.create_history_tab()
        self.tab_widget.addTab(self.history_tab, "📈 调整历史")
        
        # 训练迭代标签页
        self.iterations_tab = self.create_iterations_tab()
        self.tab_widget.addTab(self.iterations_tab, "🔄 训练迭代")
        
        return panel
        
    def create_log_tab(self) -> QWidget:
        """创建日志标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 日志显示
        self.log_display = QTextEdit()
        self.log_display.setFont(QFont('Consolas', 9))
        self.log_display.setReadOnly(True)
        layout.addWidget(self.log_display)
        
        # 清除日志按钮
        clear_log_button = QPushButton("清除日志")
        clear_log_button.clicked.connect(self.clear_log)
        layout.addWidget(clear_log_button)
        
        return widget
        
    def create_history_tab(self) -> QWidget:
        """创建调整历史标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 调整历史表格
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels([
            "时间", "参数", "原值", "新值", "原因", "状态"
        ])
        layout.addWidget(self.history_table)
        
        return widget
    
    def create_iterations_tab(self) -> QWidget:
        """创建训练迭代标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 迭代信息显示
        self.iterations_display = QTextEdit()
        self.iterations_display.setFont(QFont('Consolas', 9))
        self.iterations_display.setReadOnly(True)
        layout.addWidget(self.iterations_display)
        
        return widget
    
    def connect_signals(self):
        """连接信号"""
        # 连接编排器信号
        self.orchestrator.training_started.connect(self._on_training_started)
        self.orchestrator.training_completed.connect(self._on_training_completed)
        self.orchestrator.training_failed.connect(self._on_training_failed)
        self.orchestrator.config_generated.connect(self._on_config_generated)
        self.orchestrator.config_applied.connect(self._on_config_applied)
        self.orchestrator.iteration_completed.connect(self._on_iteration_completed)
        self.orchestrator.status_updated.connect(self._on_status_updated)
        self.orchestrator.error_occurred.connect(self._on_error_occurred)
        
        # 连接配置生成器的调整记录信号
        self.orchestrator.config_generator.adjustment_recorded.connect(self._on_adjustment_recorded)
        
    def start_intelligent_training(self):
        """启动智能训练"""
        try:
            # 启动前参考常规"开始训练"逻辑，自动刷新数据集目录
            training_tab = self.training_tab
            if not training_tab:
                QMessageBox.warning(self, "错误", "无法访问父组件 TrainingTab")
                return

            # 获取配置
            config = training_tab.get_config_from_manager()
            default_output_folder = config.get('default_output_folder', '')
            
            if not default_output_folder:
                QMessageBox.warning(self, "错误", "未配置默认输出文件夹，请先在设置中配置")
                return
            
            # 根据任务类型自动刷新数据集目录
            if training_tab.task_type == "classification":
                dataset_folder = os.path.join(default_output_folder, 'dataset')
                train_folder = os.path.join(dataset_folder, 'train')
                val_folder = os.path.join(dataset_folder, 'val')
                
                # 检查数据集结构
                if not os.path.exists(dataset_folder):
                    QMessageBox.warning(self, "错误", f"未找到分类数据集文件夹: {dataset_folder}\n\n请确保已正确配置数据集路径。")
                    return
                
                if not os.path.exists(train_folder) or not os.path.exists(val_folder):
                    QMessageBox.warning(self, "错误", f"分类数据集结构不完整:\n- 缺少训练集: {train_folder}\n- 缺少验证集: {val_folder}\n\n请确保数据集包含完整的train和val文件夹。")
                    return
                
                # 设置数据集路径
                training_tab.annotation_folder = dataset_folder
                if hasattr(training_tab, 'classification_widget'):
                    training_tab.classification_widget.set_folder_path(dataset_folder)
                training_tab.update_status(f"已自动刷新分类数据集目录: {dataset_folder}")
                self.add_log(f"✅ 分类数据集目录已刷新: {dataset_folder}")
                
            else:  # 目标检测
                detection_data_folder = os.path.join(default_output_folder, 'detection_data')
                train_images = os.path.join(detection_data_folder, 'images', 'train')
                val_images = os.path.join(detection_data_folder, 'images', 'val')
                train_labels = os.path.join(detection_data_folder, 'labels', 'train')
                val_labels = os.path.join(detection_data_folder, 'labels', 'val')
                
                # 检查目标检测数据集结构
                if not os.path.exists(detection_data_folder):
                    QMessageBox.warning(self, "错误", f"未找到目标检测数据集文件夹: {detection_data_folder}\n\n请确保已正确配置数据集路径。")
                    return
                
                missing_folders = []
                if not os.path.exists(train_images):
                    missing_folders.append("images/train")
                if not os.path.exists(val_images):
                    missing_folders.append("images/val")
                if not os.path.exists(train_labels):
                    missing_folders.append("labels/train")
                if not os.path.exists(val_labels):
                    missing_folders.append("labels/val")
                
                if missing_folders:
                    QMessageBox.warning(self, "错误", f"目标检测数据集结构不完整，缺少以下文件夹:\n{chr(10).join(missing_folders)}\n\n请确保数据集包含完整的images和labels文件夹结构。")
                    return
                
                # 设置数据集路径
                training_tab.annotation_folder = detection_data_folder
                if hasattr(training_tab, 'detection_widget'):
                    training_tab.detection_widget.set_folder_path(detection_data_folder)
                training_tab.update_status(f"已自动刷新目标检测数据集目录: {detection_data_folder}")
                self.add_log(f"✅ 目标检测数据集目录已刷新: {detection_data_folder}")
            
            # 检查训练准备状态
            if not training_tab.check_training_ready():
                QMessageBox.warning(self, "错误", "训练准备检查失败，请检查数据集配置")
                return
            
            self.add_log("✅ 数据集目录验证通过，可以开始智能训练")
            
            # 启动时总是从UI获取最新配置，而不是依赖可能过时的缓存
            self.current_config = self._get_current_training_config()

            # 检查配置是否有效
            if not self.current_config:
                error_msg = """请先配置训练参数！

请按以下步骤操作：
1. 在训练界面中设置数据集路径
2. 配置模型参数（模型类型、批次大小、学习率等）
3. 或者使用配置应用器选择预设配置

当前未检测到有效的训练配置。"""
                QMessageBox.warning(self, "配置缺失", error_msg)
                return
            
            # 确保data_dir使用正确的数据集路径
            if training_tab.task_type == "classification":
                correct_data_dir = os.path.join(default_output_folder, 'dataset')
            else:  # 目标检测
                correct_data_dir = os.path.join(default_output_folder, 'detection_data')
            
            # 更新配置中的data_dir
            self.current_config['data_dir'] = correct_data_dir
            self.add_log(f"✅ 已更新数据目录配置: {correct_data_dir}")
            
            # 检查关键配置项
            required_fields = ['data_dir', 'model_name', 'num_epochs', 'batch_size', 'learning_rate']
            missing_fields = [field for field in required_fields if not self.current_config.get(field)]
            
            if missing_fields:
                error_msg = f"""训练配置不完整！

缺少以下关键参数：
{', '.join(missing_fields)}

请检查训练界面中的参数设置。"""
                QMessageBox.warning(self, "配置不完整", error_msg)
                return
            
            # 发射开始监控请求信号
            self.start_monitoring_requested.emit(self.current_config)
            
            # 设置模型训练器和训练标签页
            self._setup_orchestrator()
            
            # 启动智能训练
            success = self.orchestrator.start_intelligent_training(self.current_config)
            
            if success:
                self.is_monitoring = True
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.add_log("🚀 智能训练已启动")
                self.add_log(f"📋 使用配置: {self.current_config.get('model_name', 'Unknown')} - {self.current_config.get('num_epochs', 0)} epochs")
            else:
                QMessageBox.critical(self, "错误", "启动智能训练失败")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动智能训练时出错: {str(e)}")
    
    def _get_current_training_config(self) -> Dict[str, Any]:
        """获取当前训练配置"""
        try:
            # 首先检查是否已经有缓存的配置
            if self.current_config:
                self.add_log("使用缓存的训练配置")
                return self.current_config
            
            # 尝试从训练标签页获取配置
            if hasattr(self.training_tab, 'main_window'):
                main_window = self.training_tab.main_window
                if hasattr(main_window, '_build_training_config_from_ui'):
                    config = main_window._build_training_config_from_ui()
                    if config:
                        self.add_log(f"从主窗口获取训练配置: {len(config)} 个参数")
                        return config
            
            # 尝试从父组件获取配置
            if hasattr(self.training_tab, '_build_training_config_from_ui'):
                config = self.training_tab._build_training_config_from_ui()
                if config:
                    self.add_log(f"从父组件获取训练配置: {len(config)} 个参数")
                    return config
            
            # 尝试从训练标签页的父组件获取配置
            if hasattr(self.training_tab, 'parent') and hasattr(self.training_tab.parent(), 'main_window'):
                main_window = self.training_tab.parent().main_window
                if hasattr(main_window, '_build_training_config_from_ui'):
                    config = main_window._build_training_config_from_ui()
                    if config:
                        self.add_log(f"从主窗口获取训练配置: {len(config)} 个参数")
                        return config
            
            self.add_log("⚠️ 未找到有效的训练配置")
            return {}
            
        except Exception as e:
            self.add_log(f"获取训练配置失败: {str(e)}")
            return {}
    
    def stop_intelligent_training(self):
        """停止智能训练"""
        try:
            # 发射停止监控请求信号
            self.stop_monitoring_requested.emit()
            
            self.orchestrator.stop_intelligent_training()
            self.is_monitoring = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.add_log("⏹️ 智能训练已停止")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"停止智能训练时出错: {str(e)}")
    
    def _setup_orchestrator(self):
        """设置编排器"""
        try:
            if hasattr(self.training_tab, 'main_window'):
                main_window = self.training_tab.main_window
                # 设置模型训练器
                if hasattr(main_window, 'worker') and hasattr(main_window.worker, 'model_trainer'):
                    self.orchestrator.set_model_trainer(main_window.worker.model_trainer)
            
            # 获取训练标签页（父组件本身就是训练标签页）
            if hasattr(self.training_tab, 'train_model'):
                self.orchestrator.set_training_tab(self.training_tab)
            
        except Exception as e:
            self.add_log(f"设置编排器时出错: {str(e)}")
    
    def set_training_config(self, config: Dict[str, Any]):
        """设置训练配置"""
        self.current_config = config.copy()
        self.update_config_display()
        self.add_log(f"📋 训练配置已更新: {len(config)} 个参数")
        
        # 如果智能训练正在运行，更新编排器的配置
        if self.is_monitoring and self.orchestrator:
            try:
                self.orchestrator.update_training_config(config)
                self.add_log("🔄 智能训练配置已同步更新")
            except Exception as e:
                self.add_log(f"⚠️ 同步配置到编排器失败: {str(e)}")
    
    def on_config_applied_from_selector(self, config: Dict[str, Any]):
        """当配置应用器应用配置时的回调"""
        self.set_training_config(config)
        self.add_log("✅ 已同步配置应用器的训练配置")
    
    def update_config_display(self):
        """更新配置显示"""
        if self.current_config:
            config_text = json.dumps(self.current_config, ensure_ascii=False, indent=2)
            self.config_display.setPlainText(config_text)
        else:
            self.config_display.setPlainText("无配置")
    
    def update_status_display(self):
        """更新状态显示"""
        try:
            if self.is_monitoring:
                session_info = self.orchestrator.get_current_session_info()
                if session_info:
                    self.status_label.setText(f"状态: {session_info.get('status', 'unknown')}")
                    self.session_info_label.setText(f"会话: {session_info.get('session_id', 'unknown')}")
                    
                    # 更新进度条
                    current_iter = session_info.get('current_iteration', 0)
                    max_iter = session_info.get('max_iterations', 1)
                    if max_iter > 0:
                        progress = int((current_iter / max_iter) * 100)
                        self.progress_bar.setValue(progress)
                        self.progress_bar.setVisible(True)
            else:
                self.status_label.setText("状态: 未启动")
                self.session_info_label.setText("会话: 无")
                self.progress_bar.setVisible(False)
                
        except Exception as e:
            pass  # 忽略更新错误
    
    def add_log(self, message: str):
        """添加日志"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.log_display.append(log_message)
        
        # 自动滚动到底部
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def update_history_table(self):
        """更新调整历史表格"""
        try:
            history = self.orchestrator.get_adjustment_history()
            
            self.history_table.setRowCount(len(history))
            
            for row, adjustment in enumerate(history):
                # 时间
                timestamp = time.strftime("%H:%M:%S", time.localtime(adjustment.get('timestamp', 0)))
                self.history_table.setItem(row, 0, QTableWidgetItem(timestamp))
                
                # 参数变更
                changes = adjustment.get('changes', {})
                if changes:
                    param_names = list(changes.keys())
                    self.history_table.setItem(row, 1, QTableWidgetItem(", ".join(param_names)))
                    
                    # 显示第一个参数的原值和新值
                    first_param = param_names[0]
                    change_info = changes[first_param]
                    self.history_table.setItem(row, 2, QTableWidgetItem(str(change_info.get('from', ''))))
                    self.history_table.setItem(row, 3, QTableWidgetItem(str(change_info.get('to', ''))))
                else:
                    self.history_table.setItem(row, 1, QTableWidgetItem("无变更"))
                    self.history_table.setItem(row, 2, QTableWidgetItem(""))
                    self.history_table.setItem(row, 3, QTableWidgetItem(""))
                
                # 原因
                reason = adjustment.get('reason', '')
                self.history_table.setItem(row, 4, QTableWidgetItem(reason))
                
                # 状态
                status = adjustment.get('status', 'unknown')
                self.history_table.setItem(row, 5, QTableWidgetItem(status))
            
            # 调整列宽
            self.history_table.resizeColumnsToContents()
            
        except Exception as e:
            self.add_log(f"更新历史表格失败: {str(e)}")
    
    def update_iterations_display(self):
        """更新迭代显示"""
        try:
            session_info = self.orchestrator.get_current_session_info()
            if session_info:
                iterations_text = f"当前迭代: {session_info.get('current_iteration', 0)}\n"
                iterations_text += f"最大迭代: {session_info.get('max_iterations', 0)}\n"
                iterations_text += f"状态: {session_info.get('status', 'unknown')}\n\n"
                
                best_metrics = session_info.get('best_metrics', {})
                if best_metrics:
                    iterations_text += "最佳结果:\n"
                    for key, value in best_metrics.items():
                        iterations_text += f"  {key}: {value}\n"
                
                self.iterations_display.setPlainText(iterations_text)
            else:
                self.iterations_display.setPlainText("无训练会话")
                
        except Exception as e:
            self.add_log(f"更新迭代显示失败: {str(e)}")
    
    def export_training_report(self):
        """导出训练报告"""
        try:
            report = self.orchestrator.export_training_report()
            if not report:
                QMessageBox.information(self, "提示", "没有可导出的训练报告")
                return
            
            # 选择保存路径
            file_path, _ = QFileDialog.getSaveFileName(
                self, "导出训练报告", 
                f"intelligent_training_report_{int(time.time())}.json",
                "JSON文件 (*.json)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                
                QMessageBox.information(self, "成功", f"训练报告已导出到: {file_path}")
                self.add_log(f"📊 训练报告已导出: {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出训练报告失败: {str(e)}")
    
    def clear_history(self):
        """清除历史记录"""
        try:
            reply = QMessageBox.question(
                self, "确认", "确定要清除所有历史记录吗？",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # 清除历史记录
                self.history_table.setRowCount(0)
                self.log_display.clear()
                self.iterations_display.clear()
                self.add_log("🗑️ 历史记录已清除")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"清除历史记录失败: {str(e)}")
    
    def clear_log(self):
        """清除日志"""
        self.log_display.clear()
        self.add_log("📝 日志已清除")
    
    # 信号回调方法
    def _on_training_started(self, data: Dict[str, Any]):
        """训练开始回调"""
        self.add_log(f"🚀 训练开始: {data.get('session_id', 'unknown')}")
        self.training_started.emit(data)
    
    def _on_training_completed(self, data: Dict[str, Any]):
        """训练完成回调"""
        self.add_log(f"✅ 训练完成: {data.get('total_iterations', 0)} 次迭代")
        self.is_monitoring = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.training_stopped.emit(data)
    
    def _on_training_failed(self, data: Dict[str, Any]):
        """训练失败回调"""
        error_msg = data.get('error', 'Unknown error')
        self.add_log(f"❌ 训练失败: {error_msg}")
    
    def _on_config_generated(self, data: Dict[str, Any]):
        """配置生成回调"""
        self.add_log("🔧 新配置已生成")
        self.update_history_table()
    
    def _on_config_applied(self, data: Dict[str, Any]):
        """配置应用回调"""
        success = data.get('success', False)
        if success:
            self.add_log("✅ 配置已应用")
        else:
            self.add_log("❌ 配置应用失败")
    
    def _on_iteration_completed(self, data: Dict[str, Any]):
        """迭代完成回调"""
        iteration = data.get('iteration', 0)
        metrics = data.get('metrics', {})
        val_acc = metrics.get('val_accuracy', 0)
        self.add_log(f"🔄 第 {iteration} 次迭代完成，验证准确率: {val_acc:.4f}")
        self.update_iterations_display()
    
    def _on_status_updated(self, message: str):
        """状态更新回调"""
        self.add_log(f"ℹ️ {message}")
        self.status_updated.emit(message)
    
    def _on_error_occurred(self, error: str):
        """错误发生回调"""
        self.add_log(f"❌ 错误: {error}")
        QMessageBox.warning(self, "错误", error)
    
    def _on_adjustment_recorded(self, adjustment: Dict[str, Any]):
        """调整记录回调"""
        self.add_log(f"📝 配置调整已记录: {adjustment.get('adjustment_id', 'unknown')}")
        # 更新历史表格
        self.update_history_table()