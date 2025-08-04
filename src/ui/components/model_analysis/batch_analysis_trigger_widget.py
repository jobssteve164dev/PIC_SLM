"""
Batch分析触发控件 - 根据训练batch数自定义AI分析触发频率

主要功能：
- 允许用户设置每隔多少个batch触发一次AI分析
- 实时显示当前batch计数和分析触发状态
- 提供手动触发分析的功能
- 集成到AI模型工厂中
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QLabel, QSpinBox, QGroupBox, QCheckBox, QProgressBar,
                           QTextEdit, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, pyqtSlot
from PyQt5.QtGui import QFont, QColor, QPalette
import time
from typing import Dict, Any, Optional


class BatchAnalysisTriggerWidget(QWidget):
    """Batch分析触发控件"""
    
    # 定义信号
    analysis_triggered = pyqtSignal(dict)  # 触发分析时发送信号
    status_updated = pyqtSignal(str)  # 状态更新信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.batch_trigger_interval = 10  # 默认每10个batch触发一次
        self.current_batch_count = 0
        self.total_analysis_count = 0
        self.is_auto_analysis_enabled = True
        self.last_analysis_time = 0
        self.analysis_cooldown = 30  # 分析冷却时间（秒）
        
        # 训练状态跟踪
        self.current_epoch = 0
        self.current_phase = ""
        self.is_training_active = False
        
        # 初始化UI
        self.init_ui()
        
        # 创建定时器用于更新显示
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(1000)  # 每秒更新一次
        
        # 初始化触发记录
        self._last_triggered_batch = 0
        
        # 配置将通过外部设置更新，不在这里加载
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 主控制组
        control_group = QGroupBox("🎯 Batch分析触发控制")
        control_layout = QVBoxLayout()
        
        # 自动分析开关
        auto_layout = QHBoxLayout()
        self.auto_analysis_checkbox = QCheckBox("启用自动分析")
        self.auto_analysis_checkbox.setChecked(self.is_auto_analysis_enabled)
        self.auto_analysis_checkbox.toggled.connect(self.on_auto_analysis_toggled)
        auto_layout.addWidget(self.auto_analysis_checkbox)
        auto_layout.addStretch()
        control_layout.addLayout(auto_layout)
        
        # Batch间隔设置
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("触发间隔:"))
        
        self.interval_spinbox = QSpinBox()
        self.interval_spinbox.setRange(1, 1000)
        self.interval_spinbox.setValue(self.batch_trigger_interval)
        self.interval_spinbox.setSuffix(" 个batch")
        self.interval_spinbox.valueChanged.connect(self.on_interval_changed)
        interval_layout.addWidget(self.interval_spinbox)
        
        interval_layout.addStretch()
        control_layout.addLayout(interval_layout)
        
        # 冷却时间设置
        cooldown_layout = QHBoxLayout()
        cooldown_layout.addWidget(QLabel("分析冷却:"))
        
        self.cooldown_spinbox = QSpinBox()
        self.cooldown_spinbox.setRange(5, 300)
        self.cooldown_spinbox.setValue(self.analysis_cooldown)
        self.cooldown_spinbox.setSuffix(" 秒")
        self.cooldown_spinbox.valueChanged.connect(self.on_cooldown_changed)
        cooldown_layout.addWidget(self.cooldown_spinbox)
        
        cooldown_layout.addStretch()
        control_layout.addLayout(cooldown_layout)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # 状态显示组
        status_group = QGroupBox("📊 当前状态")
        status_layout = QVBoxLayout()
        
        # 训练状态
        self.training_status_label = QLabel("训练状态: 未开始")
        self.training_status_label.setStyleSheet("font-weight: bold; color: #6c757d;")
        status_layout.addWidget(self.training_status_label)
        
        # Batch计数
        self.batch_count_label = QLabel("当前Batch: 0")
        self.batch_count_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.batch_count_label)
        
        # 距离下次分析的进度
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("距离下次分析:"))
        
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setRange(0, 100)
        self.analysis_progress.setValue(0)
        self.analysis_progress.setFormat("%p%")
        progress_layout.addWidget(self.analysis_progress)
        
        status_layout.addLayout(progress_layout)
        
        # 分析统计
        self.analysis_stats_label = QLabel("分析次数: 0")
        self.analysis_stats_label.setStyleSheet("color: #28a745;")
        status_layout.addWidget(self.analysis_stats_label)
        
        # 冷却状态
        self.cooldown_label = QLabel("")
        self.cooldown_label.setStyleSheet("color: #ffc107;")
        status_layout.addWidget(self.cooldown_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # 手动控制组
        manual_group = QGroupBox("🔄 手动控制")
        manual_layout = QVBoxLayout()
        
        # 手动触发按钮
        self.manual_trigger_btn = QPushButton("🚀 立即触发分析")
        self.manual_trigger_btn.clicked.connect(self.trigger_manual_analysis)
        self.manual_trigger_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        manual_layout.addWidget(self.manual_trigger_btn)
        
        # 重置按钮
        self.reset_btn = QPushButton("🔄 重置计数")
        self.reset_btn.clicked.connect(self.reset_counters)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #545b62;
            }
        """)
        manual_layout.addWidget(self.reset_btn)
        
        manual_group.setLayout(manual_layout)
        layout.addWidget(manual_group)
        
        # 日志显示
        log_group = QGroupBox("📝 分析日志")
        log_layout = QVBoxLayout()
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(120)
        self.log_display.setPlaceholderText("分析触发日志将在这里显示...")
        log_layout.addWidget(self.log_display)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # 设置控件样式
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        # 初始化显示
        self.update_display()
        
    def on_auto_analysis_toggled(self, enabled: bool):
        """自动分析开关切换"""
        self.is_auto_analysis_enabled = enabled
        self.add_log(f"自动分析已{'启用' if enabled else '禁用'}")
        self.status_updated.emit(f"自动分析已{'启用' if enabled else '禁用'}")
        
    def on_interval_changed(self, value: int):
        """触发间隔改变"""
        self.batch_trigger_interval = value
        self.add_log(f"分析触发间隔已设置为: {value} 个batch")
        self.status_updated.emit(f"分析触发间隔已设置为: {value} 个batch")
        
    def on_cooldown_changed(self, value: int):
        """冷却时间改变"""
        self.analysis_cooldown = value
        self.add_log(f"分析冷却时间已设置为: {value} 秒")
        self.status_updated.emit(f"分析冷却时间已设置为: {value} 秒")
        
    def update_training_progress(self, metrics: Dict[str, Any]):
        """更新训练进度（从训练线程接收）"""
        try:
            # 提取batch信息
            current_batch = metrics.get('batch', 0)
            total_batches = metrics.get('total_batches', 0)
            epoch = metrics.get('epoch', 0)
            phase = metrics.get('phase', '')
            
            # 更新状态
            self.current_batch_count = current_batch
            self.current_epoch = epoch
            self.current_phase = phase
            self.is_training_active = True
            
            # 检查是否需要触发分析
            if self.is_auto_analysis_enabled and self.should_trigger_analysis():
                self.trigger_analysis(metrics)
                
        except Exception as e:
            self.add_log(f"更新训练进度时出错: {str(e)}")
            
    def should_trigger_analysis(self) -> bool:
        """判断是否应该触发分析"""
        # 检查冷却时间
        current_time = time.time()
        if current_time - self.last_analysis_time < self.analysis_cooldown:
            return False
            
        # 检查batch间隔 - 修复：只有当batch数大于0且是间隔的倍数时才触发
        # 注意：训练线程每10个batch才发送一次状态更新，所以需要特殊处理
        if self.current_batch_count > 0 and self.current_batch_count % self.batch_trigger_interval == 0:
            # 记录上次触发的batch数，避免重复触发
            if not hasattr(self, '_last_triggered_batch'):
                self._last_triggered_batch = 0
                
            # 只有当当前batch数大于上次触发的batch数时才触发
            if self.current_batch_count > self._last_triggered_batch:
                self._last_triggered_batch = self.current_batch_count
                return True
            
        return False
        
    def trigger_analysis(self, metrics: Dict[str, Any]):
        """触发AI分析"""
        try:
            # 更新分析时间
            self.last_analysis_time = time.time()
            self.total_analysis_count += 1
            
            # 构建分析数据
            analysis_data = {
                'trigger_type': 'batch_interval',
                'batch_count': self.current_batch_count,
                'epoch': self.current_epoch,
                'phase': self.current_phase,
                'analysis_count': self.total_analysis_count,
                'trigger_time': self.last_analysis_time,
                'metrics': metrics.copy()
            }
            
            # 发送分析触发信号
            self.analysis_triggered.emit(analysis_data)
            
            # 更新日志
            self.add_log(f"🎯 自动触发第 {self.total_analysis_count} 次分析 (Batch {self.current_batch_count})")
            self.status_updated.emit(f"已触发第 {self.total_analysis_count} 次分析")
            
        except Exception as e:
            self.add_log(f"触发分析时出错: {str(e)}")
            
    def trigger_manual_analysis(self):
        """手动触发分析"""
        try:
            # 检查冷却时间
            current_time = time.time()
            if current_time - self.last_analysis_time < self.analysis_cooldown:
                remaining = int(self.analysis_cooldown - (current_time - self.last_analysis_time))
                self.add_log(f"⏳ 分析冷却中，还需等待 {remaining} 秒")
                return
                
            # 构建手动分析数据
            analysis_data = {
                'trigger_type': 'manual',
                'batch_count': self.current_batch_count,
                'epoch': self.current_epoch,
                'phase': self.current_phase,
                'analysis_count': self.total_analysis_count + 1,
                'trigger_time': current_time,
                'metrics': {
                    'batch': self.current_batch_count,
                    'epoch': self.current_epoch,
                    'phase': self.current_phase,
                    'manual_trigger': True
                }
            }
            
            # 更新计数
            self.last_analysis_time = current_time
            self.total_analysis_count += 1
            
            # 发送分析触发信号
            self.analysis_triggered.emit(analysis_data)
            
            # 更新日志
            self.add_log(f"🚀 手动触发第 {self.total_analysis_count} 次分析")
            self.status_updated.emit(f"已手动触发第 {self.total_analysis_count} 次分析")
            
        except Exception as e:
            self.add_log(f"手动触发分析时出错: {str(e)}")
            
    def reset_counters(self):
        """重置计数器"""
        self.current_batch_count = 0
        self.total_analysis_count = 0
        self.last_analysis_time = 0
        self._last_triggered_batch = 0  # 重置上次触发的batch记录
        self.add_log("🔄 已重置所有计数器")
        self.status_updated.emit("已重置所有计数器")
        
    def on_training_started(self, training_info: Dict[str, Any]):
        """训练开始时调用"""
        self.is_training_active = True
        self.current_batch_count = 0
        self.total_analysis_count = 0
        self.last_analysis_time = 0
        self._last_triggered_batch = 0  # 重置上次触发的batch记录
        self.add_log("🚀 训练已开始，分析触发器已激活")
        self.status_updated.emit("训练已开始，分析触发器已激活")
        
    def on_training_completed(self, results: Dict[str, Any]):
        """训练完成时调用"""
        self.is_training_active = False
        self.add_log(f"✅ 训练已完成，共触发 {self.total_analysis_count} 次分析")
        self.status_updated.emit(f"训练已完成，共触发 {self.total_analysis_count} 次分析")
        
    def on_training_stopped(self):
        """训练停止时调用"""
        self.is_training_active = False
        self.add_log("⏹️ 训练已停止")
        self.status_updated.emit("训练已停止")
        
    @pyqtSlot()
    def update_display(self):
        """更新显示"""
        try:
            # 更新训练状态
            if self.is_training_active:
                status_text = f"训练状态: 进行中 (Epoch {self.current_epoch}, {self.current_phase})"
                self.training_status_label.setText(status_text)
                self.training_status_label.setStyleSheet("font-weight: bold; color: #28a745;")
            else:
                self.training_status_label.setText("训练状态: 未开始")
                self.training_status_label.setStyleSheet("font-weight: bold; color: #6c757d;")
                
            # 更新batch计数
            self.batch_count_label.setText(f"当前Batch: {self.current_batch_count}")
            
            # 更新分析进度
            if self.batch_trigger_interval > 0:
                progress = (self.current_batch_count % self.batch_trigger_interval) / self.batch_trigger_interval * 100
                self.analysis_progress.setValue(int(progress))
            else:
                self.analysis_progress.setValue(0)
                
            # 更新分析统计
            self.analysis_stats_label.setText(f"分析次数: {self.total_analysis_count}")
            
            # 更新冷却状态
            current_time = time.time()
            if current_time - self.last_analysis_time < self.analysis_cooldown:
                remaining = int(self.analysis_cooldown - (current_time - self.last_analysis_time))
                self.cooldown_label.setText(f"⏳ 分析冷却中: {remaining} 秒")
                self.cooldown_label.setStyleSheet("color: #ffc107; font-weight: bold;")
                self.manual_trigger_btn.setEnabled(False)
            else:
                self.cooldown_label.setText("✅ 可以触发分析")
                self.cooldown_label.setStyleSheet("color: #28a745;")
                self.manual_trigger_btn.setEnabled(True)
                
        except Exception as e:
            print(f"更新显示时出错: {str(e)}")
            
    def add_log(self, message: str):
        """添加日志"""
        try:
            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}"
            
            # 添加到日志显示
            self.log_display.append(log_entry)
            
            # 滚动到底部
            cursor = self.log_display.textCursor()
            cursor.movePosition(cursor.End)
            self.log_display.setTextCursor(cursor)
            
            # 限制日志行数
            lines = self.log_display.toPlainText().split('\n')
            if len(lines) > 50:
                self.log_display.setPlainText('\n'.join(lines[-50:]))
                
        except Exception as e:
            print(f"添加日志时出错: {str(e)}")
            
    def get_current_config(self) -> Dict[str, Any]:
        """获取当前配置（用于调试和状态显示）"""
        return {
            'batch_trigger_interval': self.batch_trigger_interval,
            'analysis_cooldown': self.analysis_cooldown,
            'auto_analysis_enabled': self.is_auto_analysis_enabled,
            'total_analysis_count': self.total_analysis_count
        }
    
    def update_config_from_ai_settings(self, ai_config: Dict[str, Any]):
        """从AI设置更新配置"""
        try:
            # 从AI配置中提取Batch分析设置
            general_config = ai_config.get('general', {})
            batch_analysis_config = general_config.get('batch_analysis', {})
            
            # 更新配置
            self.is_auto_analysis_enabled = batch_analysis_config.get('enabled', True)
            self.batch_trigger_interval = batch_analysis_config.get('trigger_interval', 10)
            self.analysis_cooldown = batch_analysis_config.get('cooldown', 30)
            
            # 更新UI
            self.auto_analysis_checkbox.setChecked(self.is_auto_analysis_enabled)
            self.interval_spinbox.setValue(self.batch_trigger_interval)
            self.cooldown_spinbox.setValue(self.analysis_cooldown)
            
            self.add_log("✅ 配置已从AI设置更新")
            
        except Exception as e:
            self.add_log(f"从AI设置更新配置时出错: {str(e)}") 