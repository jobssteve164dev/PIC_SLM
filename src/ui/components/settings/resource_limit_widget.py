"""
系统资源限制组件
提供内存、CPU、硬盘占用的限制设置功能
"""

import os
import sys
import threading
import time
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, QProgressBar,
    QGridLayout, QComboBox, QMessageBox, QTextEdit, QTabWidget,
    QSlider, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QPalette

try:
    import psutil
except ImportError:
    psutil = None

try:
    import resource
except ImportError:
    resource = None


class ResourceMonitor(QThread):
    """资源监控线程"""
    
    # 定义信号
    resource_updated = pyqtSignal(dict)
    limit_exceeded = pyqtSignal(str, float, float)  # 资源类型, 当前值, 限制值
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.limits = {
            'memory_percent': 90.0,  # 内存使用百分比限制
            'cpu_percent': 80.0,     # CPU使用百分比限制
            'disk_usage_percent': 85.0,  # 磁盘使用百分比限制
        }
        self.enabled = True
        self.check_interval = 2.0  # 检查间隔（秒）
        
    def set_limits(self, limits: Dict[str, float]):
        """设置资源限制"""
        self.limits.update(limits)
        
    def set_enabled(self, enabled: bool):
        """启用/禁用监控"""
        self.enabled = enabled
        
    def set_check_interval(self, interval: float):
        """设置检查间隔"""
        self.check_interval = max(0.5, interval)
        
    def run(self):
        """监控线程主循环"""
        if not psutil:
            return
            
        self.running = True
        
        while self.running:
            try:
                if self.enabled:
                    # 获取系统资源信息
                    resource_info = self._get_resource_info()
                    
                    # 发送资源更新信号
                    self.resource_updated.emit(resource_info)
                    
                    # 检查是否超过限制
                    self._check_limits(resource_info)
                
                # 等待下次检查
                time.sleep(self.check_interval)
                
            except Exception as e:
                print(f"资源监控线程错误: {e}")
                time.sleep(1.0)
                
    def stop(self):
        """停止监控"""
        self.running = False
        
    def _get_resource_info(self) -> Dict[str, Any]:
        """获取系统资源信息"""
        info = {}
        
        try:
            # CPU信息
            info['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            info['cpu_count'] = psutil.cpu_count()
            info['cpu_freq'] = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
            
            # 内存信息
            memory = psutil.virtual_memory()
            info['memory'] = {
                'total': memory.total,
                'used': memory.used,
                'free': memory.free,
                'percent': memory.percent
            }
            
            # 磁盘信息
            disk_usage = psutil.disk_usage('/')
            info['disk'] = {
                'total': disk_usage.total,
                'used': disk_usage.used,
                'free': disk_usage.free,
                'percent': (disk_usage.used / disk_usage.total) * 100
            }
            
            # 网络信息
            net_io = psutil.net_io_counters()
            info['network'] = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
            # 进程信息
            info['process_count'] = len(psutil.pids())
            
        except Exception as e:
            print(f"获取资源信息错误: {e}")
            
        return info
        
    def _check_limits(self, resource_info: Dict[str, Any]):
        """检查资源限制"""
        try:
            # 检查内存限制
            if 'memory' in resource_info:
                memory_percent = resource_info['memory']['percent']
                if memory_percent > self.limits.get('memory_percent', 90.0):
                    self.limit_exceeded.emit('memory', memory_percent, self.limits['memory_percent'])
                    
            # 检查CPU限制
            cpu_percent = resource_info.get('cpu_percent', 0)
            if cpu_percent > self.limits.get('cpu_percent', 80.0):
                self.limit_exceeded.emit('cpu', cpu_percent, self.limits['cpu_percent'])
                
            # 检查磁盘限制
            if 'disk' in resource_info:
                disk_percent = resource_info['disk']['percent']
                if disk_percent > self.limits.get('disk_usage_percent', 85.0):
                    self.limit_exceeded.emit('disk', disk_percent, self.limits['disk_usage_percent'])
                    
        except Exception as e:
            print(f"检查资源限制错误: {e}")


class ResourceLimitWidget(QWidget):
    """系统资源限制设置组件"""
    
    # 定义信号
    limits_changed = pyqtSignal(dict)
    monitoring_toggled = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 检查依赖
        self.psutil_available = psutil is not None
        self.resource_available = resource is not None
        
        # 资源监控器
        self.resource_monitor = None
        if self.psutil_available:
            self.resource_monitor = ResourceMonitor()
            self.resource_monitor.resource_updated.connect(self.on_resource_updated)
            self.resource_monitor.limit_exceeded.connect(self.on_limit_exceeded)
            
        # 当前资源信息
        self.current_resources = {}
        
        # 初始化UI
        self.init_ui()
        
        # 启动定时器更新显示
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(1000)  # 每秒更新一次显示
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 添加标题
        title_label = QLabel("系统资源使用限制")
        title_label.setFont(QFont('微软雅黑', 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 检查依赖
        if not self.psutil_available:
            warning_label = QLabel("⚠️ 缺少psutil模块，无法监控系统资源")
            warning_label.setStyleSheet("color: orange; font-weight: bold;")
            layout.addWidget(warning_label)
            
        # 创建选项卡
        self.tabs = QTabWidget()
        
        # 资源限制设置选项卡
        self.create_limits_tab()
        self.tabs.addTab(self.limits_tab, "资源限制")
        
        # 实时监控选项卡
        self.create_monitor_tab()
        self.tabs.addTab(self.monitor_tab, "实时监控")
        
        # 高级设置选项卡
        self.create_advanced_tab()
        self.tabs.addTab(self.advanced_tab, "高级设置")
        
        layout.addWidget(self.tabs)
        
        # 控制按钮
        self.create_control_buttons(layout)
        
    def create_limits_tab(self):
        """创建资源限制设置选项卡"""
        self.limits_tab = QWidget()
        layout = QVBoxLayout(self.limits_tab)
        
        # 内存限制组
        memory_group = QGroupBox("内存使用限制")
        memory_layout = QGridLayout(memory_group)
        
        # 启用内存限制
        self.memory_limit_enabled = QCheckBox("启用内存使用限制")
        self.memory_limit_enabled.setChecked(True)
        self.memory_limit_enabled.setToolTip("启用后将监控内存使用率，超过限制时发出警告")
        memory_layout.addWidget(self.memory_limit_enabled, 0, 0, 1, 2)
        
        # 内存使用百分比限制
        memory_layout.addWidget(QLabel("内存使用率限制 (%):"), 1, 0)
        self.memory_percent_limit = QSpinBox()
        self.memory_percent_limit.setRange(50, 95)
        self.memory_percent_limit.setValue(90)
        self.memory_percent_limit.setSuffix("%")
        self.memory_percent_limit.setToolTip("当内存使用率超过此百分比时发出警告")
        memory_layout.addWidget(self.memory_percent_limit, 1, 1)
        
        # 内存使用绝对限制（GB）
        memory_layout.addWidget(QLabel("内存使用绝对限制 (GB):"), 2, 0)
        self.memory_absolute_limit = QDoubleSpinBox()
        self.memory_absolute_limit.setRange(0.5, 64.0)
        self.memory_absolute_limit.setValue(8.0)
        self.memory_absolute_limit.setSingleStep(0.5)
        self.memory_absolute_limit.setSuffix(" GB")
        self.memory_absolute_limit.setToolTip("程序最大可使用的内存量")
        memory_layout.addWidget(self.memory_absolute_limit, 2, 1)
        
        layout.addWidget(memory_group)
        
        # CPU限制组
        cpu_group = QGroupBox("CPU使用限制")
        cpu_layout = QGridLayout(cpu_group)
        
        # 启用CPU限制
        self.cpu_limit_enabled = QCheckBox("启用CPU使用限制")
        self.cpu_limit_enabled.setChecked(True)
        self.cpu_limit_enabled.setToolTip("启用后将监控CPU使用率，超过限制时发出警告")
        cpu_layout.addWidget(self.cpu_limit_enabled, 0, 0, 1, 2)
        
        # CPU使用百分比限制
        cpu_layout.addWidget(QLabel("CPU使用率限制 (%):"), 1, 0)
        self.cpu_percent_limit = QSpinBox()
        self.cpu_percent_limit.setRange(30, 95)
        self.cpu_percent_limit.setValue(80)
        self.cpu_percent_limit.setSuffix("%")
        self.cpu_percent_limit.setToolTip("当CPU使用率超过此百分比时发出警告")
        cpu_layout.addWidget(self.cpu_percent_limit, 1, 1)
        
        # CPU核心数限制
        cpu_layout.addWidget(QLabel("最大使用CPU核心数:"), 2, 0)
        self.cpu_cores_limit = QSpinBox()
        self.cpu_cores_limit.setRange(1, psutil.cpu_count() if self.psutil_available else 8)
        self.cpu_cores_limit.setValue(psutil.cpu_count() if self.psutil_available else 4)
        self.cpu_cores_limit.setToolTip("程序最多可使用的CPU核心数")
        cpu_layout.addWidget(self.cpu_cores_limit, 2, 1)
        
        layout.addWidget(cpu_group)
        
        # 磁盘限制组
        disk_group = QGroupBox("磁盘使用限制")
        disk_layout = QGridLayout(disk_group)
        
        # 启用磁盘限制
        self.disk_limit_enabled = QCheckBox("启用磁盘使用监控")
        self.disk_limit_enabled.setChecked(True)
        self.disk_limit_enabled.setToolTip("启用后将监控磁盘使用率，超过限制时发出警告")
        disk_layout.addWidget(self.disk_limit_enabled, 0, 0, 1, 2)
        
        # 磁盘使用百分比限制
        disk_layout.addWidget(QLabel("磁盘使用率限制 (%):"), 1, 0)
        self.disk_percent_limit = QSpinBox()
        self.disk_percent_limit.setRange(70, 95)
        self.disk_percent_limit.setValue(85)
        self.disk_percent_limit.setSuffix("%")
        self.disk_percent_limit.setToolTip("当磁盘使用率超过此百分比时发出警告")
        disk_layout.addWidget(self.disk_percent_limit, 1, 1)
        
        # 临时文件大小限制
        disk_layout.addWidget(QLabel("临时文件大小限制 (GB):"), 2, 0)
        self.temp_files_limit = QDoubleSpinBox()
        self.temp_files_limit.setRange(1.0, 100.0)
        self.temp_files_limit.setValue(10.0)
        self.temp_files_limit.setSingleStep(1.0)
        self.temp_files_limit.setSuffix(" GB")
        self.temp_files_limit.setToolTip("程序可创建的临时文件总大小限制")
        disk_layout.addWidget(self.temp_files_limit, 2, 1)
        
        layout.addWidget(disk_group)
        
        layout.addStretch()
        
    def create_monitor_tab(self):
        """创建实时监控选项卡"""
        self.monitor_tab = QWidget()
        layout = QVBoxLayout(self.monitor_tab)
        
        # 监控控制
        control_layout = QHBoxLayout()
        
        self.start_monitor_btn = QPushButton("开始监控")
        self.start_monitor_btn.clicked.connect(self.start_monitoring)
        self.start_monitor_btn.setEnabled(self.psutil_available)
        control_layout.addWidget(self.start_monitor_btn)
        
        self.stop_monitor_btn = QPushButton("停止监控")
        self.stop_monitor_btn.clicked.connect(self.stop_monitoring)
        self.stop_monitor_btn.setEnabled(False)
        control_layout.addWidget(self.stop_monitor_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # 实时资源显示
        monitor_group = QGroupBox("实时资源使用情况")
        monitor_layout = QGridLayout(monitor_group)
        
        # CPU监控
        monitor_layout.addWidget(QLabel("CPU使用率:"), 0, 0)
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setRange(0, 100)
        self.cpu_progress.setValue(0)
        monitor_layout.addWidget(self.cpu_progress, 0, 1)
        self.cpu_value_label = QLabel("0%")
        monitor_layout.addWidget(self.cpu_value_label, 0, 2)
        
        # 内存监控
        monitor_layout.addWidget(QLabel("内存使用率:"), 1, 0)
        self.memory_progress = QProgressBar()
        self.memory_progress.setRange(0, 100)
        self.memory_progress.setValue(0)
        monitor_layout.addWidget(self.memory_progress, 1, 1)
        self.memory_value_label = QLabel("0%")
        monitor_layout.addWidget(self.memory_value_label, 1, 2)
        
        # 磁盘监控
        monitor_layout.addWidget(QLabel("磁盘使用率:"), 2, 0)
        self.disk_progress = QProgressBar()
        self.disk_progress.setRange(0, 100)
        self.disk_progress.setValue(0)
        monitor_layout.addWidget(self.disk_progress, 2, 1)
        self.disk_value_label = QLabel("0%")
        monitor_layout.addWidget(self.disk_value_label, 2, 2)
        
        layout.addWidget(monitor_group)
        
        # 详细信息显示
        details_group = QGroupBox("详细信息")
        details_layout = QVBoxLayout(details_group)
        
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(200)
        self.details_text.setReadOnly(True)
        details_layout.addWidget(self.details_text)
        
        layout.addWidget(details_group)
        
        layout.addStretch()
        
    def create_advanced_tab(self):
        """创建高级设置选项卡"""
        self.advanced_tab = QWidget()
        layout = QVBoxLayout(self.advanced_tab)
        
        # 监控设置组
        monitor_settings_group = QGroupBox("监控设置")
        monitor_settings_layout = QGridLayout(monitor_settings_group)
        
        # 检查间隔
        monitor_settings_layout.addWidget(QLabel("检查间隔 (秒):"), 0, 0)
        self.check_interval_spin = QDoubleSpinBox()
        self.check_interval_spin.setRange(0.5, 10.0)
        self.check_interval_spin.setValue(2.0)
        self.check_interval_spin.setSingleStep(0.5)
        self.check_interval_spin.setToolTip("资源监控的检查间隔，较小的值提供更及时的监控但消耗更多资源")
        monitor_settings_layout.addWidget(self.check_interval_spin, 0, 1)
        
        # 警报设置
        monitor_settings_layout.addWidget(QLabel("警报方式:"), 1, 0)
        self.alert_method = QComboBox()
        self.alert_method.addItems(["弹窗提醒", "状态栏提醒", "日志记录", "全部启用"])
        self.alert_method.setCurrentText("弹窗提醒")
        monitor_settings_layout.addWidget(self.alert_method, 1, 1)
        
        # 自动处理
        self.auto_cleanup_enabled = QCheckBox("启用自动清理")
        self.auto_cleanup_enabled.setToolTip("当资源使用超限时自动尝试清理临时文件和缓存")
        monitor_settings_layout.addWidget(self.auto_cleanup_enabled, 2, 0, 1, 2)
        
        layout.addWidget(monitor_settings_group)
        
        # 系统限制设置组（仅Linux/Unix）
        if self.resource_available and hasattr(resource, 'setrlimit'):
            system_limits_group = QGroupBox("系统级限制设置")
            system_limits_layout = QGridLayout(system_limits_group)
            
            # 进程数限制
            system_limits_layout.addWidget(QLabel("最大进程数:"), 0, 0)
            self.max_processes = QSpinBox()
            self.max_processes.setRange(10, 1000)
            self.max_processes.setValue(100)
            self.max_processes.setToolTip("程序可创建的最大进程数")
            system_limits_layout.addWidget(self.max_processes, 0, 1)
            
            # 文件描述符限制
            system_limits_layout.addWidget(QLabel("最大文件描述符:"), 1, 0)
            self.max_file_descriptors = QSpinBox()
            self.max_file_descriptors.setRange(100, 10000)
            self.max_file_descriptors.setValue(1024)
            self.max_file_descriptors.setToolTip("程序可打开的最大文件描述符数")
            system_limits_layout.addWidget(self.max_file_descriptors, 1, 1)
            
            # 应用系统限制按钮
            apply_system_limits_btn = QPushButton("应用系统限制")
            apply_system_limits_btn.clicked.connect(self.apply_system_limits)
            apply_system_limits_btn.setToolTip("将限制应用到当前进程（需要管理员权限）")
            system_limits_layout.addWidget(apply_system_limits_btn, 2, 0, 1, 2)
            
            layout.addWidget(system_limits_group)
        
        layout.addStretch()
        
    def create_control_buttons(self, layout):
        """创建控制按钮"""
        button_layout = QHBoxLayout()
        
        # 应用设置按钮
        apply_btn = QPushButton("应用设置")
        apply_btn.clicked.connect(self.apply_limits)
        apply_btn.setMinimumHeight(35)
        button_layout.addWidget(apply_btn)
        
        # 重置按钮
        reset_btn = QPushButton("重置为默认")
        reset_btn.clicked.connect(self.reset_to_defaults)
        reset_btn.setMinimumHeight(35)
        button_layout.addWidget(reset_btn)
        
        # 导出配置按钮
        export_btn = QPushButton("导出配置")
        export_btn.clicked.connect(self.export_config)
        export_btn.setMinimumHeight(35)
        button_layout.addWidget(export_btn)
        
        # 导入配置按钮
        import_btn = QPushButton("导入配置")
        import_btn.clicked.connect(self.import_config)
        import_btn.setMinimumHeight(35)
        button_layout.addWidget(import_btn)
        
        layout.addLayout(button_layout)
        
    def start_monitoring(self):
        """开始资源监控"""
        if not self.psutil_available or not self.resource_monitor:
            QMessageBox.warning(self, "警告", "psutil模块不可用，无法启动监控")
            return
            
        # 设置监控参数
        limits = self.get_current_limits()
        self.resource_monitor.set_limits(limits)
        self.resource_monitor.set_check_interval(self.check_interval_spin.value())
        
        # 启动监控线程
        self.resource_monitor.start()
        
        # 更新按钮状态
        self.start_monitor_btn.setEnabled(False)
        self.stop_monitor_btn.setEnabled(True)
        
        self.monitoring_toggled.emit(True)
        
    def stop_monitoring(self):
        """停止资源监控"""
        if self.resource_monitor and self.resource_monitor.running:
            self.resource_monitor.stop()
            self.resource_monitor.wait(3000)  # 等待最多3秒
            
        # 更新按钮状态
        self.start_monitor_btn.setEnabled(True)
        self.stop_monitor_btn.setEnabled(False)
        
        self.monitoring_toggled.emit(False)
        
    def get_current_limits(self) -> Dict[str, float]:
        """获取当前设置的限制"""
        return {
            'memory_percent': self.memory_percent_limit.value(),
            'cpu_percent': self.cpu_percent_limit.value(),
            'disk_usage_percent': self.disk_percent_limit.value(),
        }
        
    def apply_limits(self):
        """应用资源限制设置"""
        limits = self.get_current_limits()
        
        # 如果监控正在运行，更新监控器的限制
        if self.resource_monitor and self.resource_monitor.running:
            self.resource_monitor.set_limits(limits)
            
        # 发送限制变化信号
        self.limits_changed.emit(limits)
        
        QMessageBox.information(self, "成功", "资源限制设置已应用")
        
    def reset_to_defaults(self):
        """重置为默认设置"""
        self.memory_percent_limit.setValue(90)
        self.cpu_percent_limit.setValue(80)
        self.disk_percent_limit.setValue(85)
        self.memory_absolute_limit.setValue(8.0)
        self.cpu_cores_limit.setValue(psutil.cpu_count() if self.psutil_available else 4)
        self.temp_files_limit.setValue(10.0)
        self.check_interval_spin.setValue(2.0)
        self.alert_method.setCurrentText("弹窗提醒")
        
        # 重置复选框
        self.memory_limit_enabled.setChecked(True)
        self.cpu_limit_enabled.setChecked(True)
        self.disk_limit_enabled.setChecked(True)
        self.auto_cleanup_enabled.setChecked(False)
        
    def export_config(self):
        """导出配置"""
        # TODO: 实现配置导出功能
        QMessageBox.information(self, "提示", "配置导出功能将在后续版本中实现")
        
    def import_config(self):
        """导入配置"""
        # TODO: 实现配置导入功能
        QMessageBox.information(self, "提示", "配置导入功能将在后续版本中实现")
        
    def apply_system_limits(self):
        """应用系统级限制"""
        if not self.resource_available:
            QMessageBox.warning(self, "警告", "系统不支持资源限制功能")
            return
            
        try:
            # 设置进程数限制
            if hasattr(resource, 'RLIMIT_NPROC'):
                current_soft, current_hard = resource.getrlimit(resource.RLIMIT_NPROC)
                new_limit = min(self.max_processes.value(), current_hard)
                resource.setrlimit(resource.RLIMIT_NPROC, (new_limit, current_hard))
                
            # 设置文件描述符限制
            if hasattr(resource, 'RLIMIT_NOFILE'):
                current_soft, current_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                new_limit = min(self.max_file_descriptors.value(), current_hard)
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, current_hard))
                
            QMessageBox.information(self, "成功", "系统级限制已应用")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"应用系统限制失败：{str(e)}")
            
    @pyqtSlot(dict)
    def on_resource_updated(self, resource_info: Dict[str, Any]):
        """处理资源更新"""
        self.current_resources = resource_info
        
    @pyqtSlot(str, float, float)
    def on_limit_exceeded(self, resource_type: str, current_value: float, limit_value: float):
        """处理资源超限"""
        alert_method = self.alert_method.currentText()
        
        message = f"{resource_type.upper()}使用率超限！\n当前: {current_value:.1f}%\n限制: {limit_value:.1f}%"
        
        if alert_method in ["弹窗提醒", "全部启用"]:
            QMessageBox.warning(self, "资源使用警告", message)
            
        if alert_method in ["日志记录", "全部启用"]:
            print(f"[资源警告] {message}")
            
        # 自动清理
        if self.auto_cleanup_enabled.isChecked():
            self.perform_auto_cleanup()
            
    def perform_auto_cleanup(self):
        """执行自动清理"""
        try:
            # 清理Python垃圾回收
            import gc
            gc.collect()
            
            # TODO: 添加更多清理逻辑
            # - 清理临时文件
            # - 清理缓存
            # - 释放不必要的内存
            
        except Exception as e:
            print(f"自动清理失败: {e}")
            
    def update_display(self):
        """更新显示"""
        if not self.current_resources:
            return
            
        try:
            # 更新进度条
            cpu_percent = self.current_resources.get('cpu_percent', 0)
            self.cpu_progress.setValue(int(cpu_percent))
            self.cpu_value_label.setText(f"{cpu_percent:.1f}%")
            
            memory_info = self.current_resources.get('memory', {})
            memory_percent = memory_info.get('percent', 0)
            self.memory_progress.setValue(int(memory_percent))
            self.memory_value_label.setText(f"{memory_percent:.1f}%")
            
            disk_info = self.current_resources.get('disk', {})
            disk_percent = disk_info.get('percent', 0)
            self.disk_progress.setValue(int(disk_percent))
            self.disk_value_label.setText(f"{disk_percent:.1f}%")
            
            # 更新详细信息
            details = []
            details.append(f"CPU: {cpu_percent:.1f}% ({self.current_resources.get('cpu_count', 0)} 核心)")
            
            if memory_info:
                memory_gb = memory_info.get('used', 0) / (1024**3)
                total_gb = memory_info.get('total', 0) / (1024**3)
                details.append(f"内存: {memory_gb:.1f}GB / {total_gb:.1f}GB ({memory_percent:.1f}%)")
                
            if disk_info:
                disk_gb = disk_info.get('used', 0) / (1024**3)
                total_disk_gb = disk_info.get('total', 0) / (1024**3)
                details.append(f"磁盘: {disk_gb:.1f}GB / {total_disk_gb:.1f}GB ({disk_percent:.1f}%)")
                
            process_count = self.current_resources.get('process_count', 0)
            details.append(f"进程数: {process_count}")
            
            self.details_text.setPlainText('\n'.join(details))
            
        except Exception as e:
            print(f"更新显示错误: {e}")
            
    def get_resource_limits_config(self) -> Dict[str, Any]:
        """获取资源限制配置"""
        return {
            'memory_limit_enabled': self.memory_limit_enabled.isChecked(),
            'memory_percent_limit': self.memory_percent_limit.value(),
            'memory_absolute_limit_gb': self.memory_absolute_limit.value(),
            'cpu_limit_enabled': self.cpu_limit_enabled.isChecked(),
            'cpu_percent_limit': self.cpu_percent_limit.value(),
            'cpu_cores_limit': self.cpu_cores_limit.value(),
            'disk_limit_enabled': self.disk_limit_enabled.isChecked(),
            'disk_percent_limit': self.disk_percent_limit.value(),
            'temp_files_limit_gb': self.temp_files_limit.value(),
            'check_interval': self.check_interval_spin.value(),
            'alert_method': self.alert_method.currentText(),
            'auto_cleanup_enabled': self.auto_cleanup_enabled.isChecked(),
        }
        
    def set_resource_limits_config(self, config: Dict[str, Any]):
        """设置资源限制配置"""
        try:
            self.memory_limit_enabled.setChecked(config.get('memory_limit_enabled', True))
            self.memory_percent_limit.setValue(config.get('memory_percent_limit', 90))
            self.memory_absolute_limit.setValue(config.get('memory_absolute_limit_gb', 8.0))
            
            self.cpu_limit_enabled.setChecked(config.get('cpu_limit_enabled', True))
            self.cpu_percent_limit.setValue(config.get('cpu_percent_limit', 80))
            self.cpu_cores_limit.setValue(config.get('cpu_cores_limit', 4))
            
            self.disk_limit_enabled.setChecked(config.get('disk_limit_enabled', True))
            self.disk_percent_limit.setValue(config.get('disk_percent_limit', 85))
            self.temp_files_limit.setValue(config.get('temp_files_limit_gb', 10.0))
            
            self.check_interval_spin.setValue(config.get('check_interval', 2.0))
            self.alert_method.setCurrentText(config.get('alert_method', '弹窗提醒'))
            self.auto_cleanup_enabled.setChecked(config.get('auto_cleanup_enabled', False))
            
        except Exception as e:
            print(f"设置资源限制配置错误: {e}")
            
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止监控线程
        if self.resource_monitor and self.resource_monitor.running:
            self.resource_monitor.stop()
            self.resource_monitor.wait(3000)
            
        # 停止定时器
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
            
        super().closeEvent(event) 