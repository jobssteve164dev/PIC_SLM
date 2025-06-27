"""
系统资源限制组件
提供内存、CPU、硬盘占用的限制设置功能
现在集成了真正的资源限制器
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

# 导入真正的资源限制器
from ....utils.resource_limiter import (
    ResourceLimiter, ResourceLimits, ResourceLimitException,
    initialize_resource_limiter, get_resource_limiter
)


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
    
    def _get_disk_usage_info(self) -> Dict[str, Any]:
        """获取磁盘使用信息，支持多驱动器"""
        try:
            if os.name == 'nt':  # Windows
                # 获取所有驱动器
                drives = []
                for drive_letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    drive_path = f'{drive_letter}:\\'
                    if os.path.exists(drive_path):
                        try:
                            usage = psutil.disk_usage(drive_path)
                            drives.append({
                                'drive': drive_letter,
                                'path': drive_path,
                                'total': usage.total,
                                'used': usage.used,
                                'free': usage.free,
                                'percent': (usage.used / usage.total) * 100 if usage.total > 0 else 0
                            })
                        except (OSError, PermissionError):
                            continue
                
                if drives:
                    # 返回主驱动器（通常是C盘）的信息，同时包含所有驱动器信息
                    main_drive = next((d for d in drives if d['drive'] == 'C'), drives[0])
                    return {
                        'total': main_drive['total'],
                        'used': main_drive['used'],
                        'free': main_drive['free'],
                        'percent': main_drive['percent'],
                        'main_drive': main_drive['drive'],
                        'all_drives': drives
                    }
            else:  # Unix/Linux
                # 使用根目录
                usage = psutil.disk_usage('/')
                return {
                    'total': usage.total,
                    'used': usage.used,
                    'free': usage.free,
                    'percent': (usage.used / usage.total) * 100 if usage.total > 0 else 0,
                    'main_drive': '/',
                    'all_drives': [{'drive': '/', 'path': '/', 'total': usage.total, 
                                   'used': usage.used, 'free': usage.free, 
                                   'percent': (usage.used / usage.total) * 100}]
                }
                
        except Exception as e:
            print(f"获取磁盘信息错误: {e}")
            return {
                'total': 0,
                'used': 0,
                'free': 0,
                'percent': 0,
                'main_drive': 'Unknown',
                'all_drives': []
            }
        
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
            disk_info = self._get_disk_usage_info()
            info['disk'] = disk_info
            
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
        
        # 真正的资源限制器
        self.resource_limiter = None
        
        # 资源监控器（兼容原有功能）
        self.resource_monitor = None
        if self.psutil_available:
            self.resource_monitor = ResourceMonitor()
            self.resource_monitor.resource_updated.connect(self.on_resource_updated)
            self.resource_monitor.limit_exceeded.connect(self.on_limit_exceeded)
            
        # 当前资源信息
        self.current_resources = {}
        
        # 资源使用历史记录（最近100个数据点）
        self.resource_history = {
            'cpu': [],
            'memory': [],
            'disk': [],
            'timestamps': []
        }
        self.max_history_length = 100
        
        # 警告状态跟踪
        self.warning_states = {
            'memory': False,
            'cpu': False,
            'disk': False
        }
        
        # 初始化UI
        self.init_ui()
        
        # 设置右键菜单
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        # 启动定时器更新显示
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(1000)  # 每秒更新一次显示
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
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
        
        # 不再添加独立的控制按钮，使用统一的设置保存逻辑
        
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
        
        # 手动清理按钮
        manual_cleanup_btn = QPushButton("手动清理")
        manual_cleanup_btn.clicked.connect(self.perform_auto_cleanup)
        manual_cleanup_btn.setToolTip("立即执行系统清理操作")
        control_layout.addWidget(manual_cleanup_btn)
        
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
        
        # 真正的资源限制设置组
        real_limits_group = QGroupBox("强制资源限制")
        real_limits_layout = QGridLayout(real_limits_group)
        
        # 启用强制限制
        self.enforce_limits_enabled = QCheckBox("启用强制资源限制")
        self.enforce_limits_enabled.setChecked(False)
        self.enforce_limits_enabled.setToolTip("启用后将真正限制程序的资源使用，而不仅仅是监控")
        real_limits_layout.addWidget(self.enforce_limits_enabled, 0, 0, 1, 2)
        
        # 强制限制说明
        limits_info = QLabel("⚠️ 强制限制将实际限制程序资源使用，可能影响性能")
        limits_info.setStyleSheet("color: orange; font-size: 12px;")
        limits_info.setWordWrap(True)
        real_limits_layout.addWidget(limits_info, 1, 0, 1, 2)
        
        # Windows Job Object说明
        if os.name == 'nt':
            job_info = QLabel("✅ Windows系统支持Job Object内存限制")
            job_info.setStyleSheet("color: green; font-size: 12px;")
        else:
            job_info = QLabel("ℹ️ Unix/Linux系统支持基础进程限制")
            job_info.setStyleSheet("color: blue; font-size: 12px;")
        real_limits_layout.addWidget(job_info, 2, 0, 1, 2)
        
        layout.addWidget(real_limits_group)
        
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
        

        
    def start_monitoring(self):
        """开始资源监控"""
        if not self.psutil_available:
            QMessageBox.warning(self, "警告", "psutil模块不可用，无法启动监控")
            return
            
        try:
            # 启动真正的资源限制器
            if self.enforce_limits_enabled.isChecked():
                limits = ResourceLimits(
                    max_memory_gb=self.memory_absolute_limit.value(),
                    max_cpu_percent=self.cpu_percent_limit.value(),
                    max_disk_usage_gb=self.temp_files_limit.value(),
                    max_processes=4,
                    max_threads=self.cpu_cores_limit.value(),
                    check_interval=self.check_interval_spin.value(),
                    enforce_limits=True,
                    auto_cleanup=self.auto_cleanup_enabled.isChecked()
                )
                
                self.resource_limiter = initialize_resource_limiter(limits)
                
                # 添加回调处理资源超限
                self.resource_limiter.add_callback('memory_limit', self._on_real_limit_exceeded)
                self.resource_limiter.add_callback('cpu_limit', self._on_real_limit_exceeded)
                self.resource_limiter.add_callback('disk_limit', self._on_real_limit_exceeded)
                self.resource_limiter.add_callback('process_limit', self._on_real_limit_exceeded)
                
                self.resource_limiter.start_monitoring()
                
                QMessageBox.information(self, "成功", "强制资源限制已启动！")
            
            # 启动传统监控器（用于显示）
            if self.resource_monitor:
                limits = self.get_current_limits()
                self.resource_monitor.set_limits(limits)
                self.resource_monitor.set_check_interval(self.check_interval_spin.value())
                self.resource_monitor.start()
            
            # 更新按钮状态
            self.start_monitor_btn.setEnabled(False)
            self.stop_monitor_btn.setEnabled(True)
            
            self.monitoring_toggled.emit(True)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动监控失败: {str(e)}")
        
    def stop_monitoring(self):
        """停止资源监控"""
        try:
            # 停止真正的资源限制器
            if self.resource_limiter:
                self.resource_limiter.stop_monitoring()
                self.resource_limiter = None
                
            # 停止传统监控器
            if self.resource_monitor and self.resource_monitor.running:
                self.resource_monitor.stop()
                self.resource_monitor.wait(3000)  # 等待最多3秒
                
            # 更新按钮状态
            self.start_monitor_btn.setEnabled(True)
            self.stop_monitor_btn.setEnabled(False)
            
            self.monitoring_toggled.emit(False)
            
        except Exception as e:
            print(f"停止监控错误: {e}")
    
    def _on_real_limit_exceeded(self, event_type: str, current_value: float, limit_value: float):
        """处理真正的资源限制超限"""
        try:
            resource_name = {"memory_limit": "内存", "cpu_limit": "CPU", 
                           "disk_limit": "磁盘", "process_limit": "进程"}
            resource_name = resource_name.get(event_type, event_type)
            
            # 创建警告对话框
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("强制资源限制触发")
            
            message = f"🚨 {resource_name}使用超过强制限制！\n"
            message += f"当前值: {current_value:.2f}\n"
            message += f"限制值: {limit_value:.2f}\n\n"
            message += "系统已自动执行限制措施。"
            
            msg_box.setText(message)
            msg_box.addButton("确定", QMessageBox.AcceptRole)
            
            emergency_btn = msg_box.addButton("紧急清理", QMessageBox.ActionRole)
            stop_btn = msg_box.addButton("停止所有操作", QMessageBox.DestructiveRole)
            
            result = msg_box.exec_()
            
            # 处理用户选择
            clicked_button = msg_box.clickedButton()
            if clicked_button == emergency_btn:
                if self.resource_limiter:
                    self.resource_limiter.emergency_cleanup()
            elif clicked_button == stop_btn:
                if self.resource_limiter:
                    self.resource_limiter.request_stop()
                    QMessageBox.information(self, "停止请求", "已请求停止所有正在进行的操作")
                    
        except Exception as e:
            print(f"处理资源限制超限错误: {e}")
        
    def get_current_limits(self) -> Dict[str, float]:
        """获取当前设置的限制"""
        return {
            'memory_percent': self.memory_percent_limit.value(),
            'cpu_percent': self.cpu_percent_limit.value(),
            'disk_usage_percent': self.disk_percent_limit.value(),
        }
        
    def apply_limits(self):
        """应用资源限制设置（内部调用，不显示消息框）"""
        limits = self.get_current_limits()
        
        # 如果监控正在运行，更新监控器的限制
        if self.resource_monitor and self.resource_monitor.running:
            self.resource_monitor.set_limits(limits)
            
        # 发送限制变化信号
        self.limits_changed.emit(limits)
        
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
        
        # 记录历史数据
        self._record_resource_history(resource_info)
        
    @pyqtSlot(str, float, float)
    def on_limit_exceeded(self, resource_type: str, current_value: float, limit_value: float):
        """处理资源超限"""
        # 避免重复警告（5分钟内不重复弹窗）
        current_time = time.time()
        last_warning_key = f"{resource_type}_last_warning"
        
        if not hasattr(self, 'last_warnings'):
            self.last_warnings = {}
            
        if (last_warning_key in self.last_warnings and 
            current_time - self.last_warnings[last_warning_key] < 300):  # 5分钟
            return
            
        self.last_warnings[last_warning_key] = current_time
        
        alert_method = self.alert_method.currentText()
        resource_name = {"memory": "内存", "cpu": "CPU", "disk": "磁盘"}.get(resource_type, resource_type.upper())
        
        message = f"{resource_name}使用率超限！\n当前: {current_value:.1f}%\n限制: {limit_value:.1f}%"
        
        # 记录警告状态
        self.warning_states[resource_type] = True
        
        if alert_method in ["弹窗提醒", "全部启用"]:
            # 创建更详细的警告对话框
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("资源使用警告")
            msg_box.setText(message)
            
            # 添加建议操作
            suggestions = self._get_resource_suggestions(resource_type, current_value)
            if suggestions:
                msg_box.setDetailedText(f"建议操作:\n{suggestions}")
                
            msg_box.addButton("确定", QMessageBox.AcceptRole)
            if self.auto_cleanup_enabled.isChecked():
                cleanup_btn = msg_box.addButton("立即清理", QMessageBox.ActionRole)
                
            result = msg_box.exec_()
            
            # 如果用户选择立即清理
            if (self.auto_cleanup_enabled.isChecked() and 
                msg_box.clickedButton().text() == "立即清理"):
                self.perform_auto_cleanup()
            
        if alert_method in ["日志记录", "全部启用"]:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] [资源警告] {message}")
            
        # 自动清理
        if self.auto_cleanup_enabled.isChecked():
            self.perform_auto_cleanup()
    
    def _get_resource_suggestions(self, resource_type: str, current_value: float) -> str:
        """获取资源优化建议"""
        suggestions = []
        
        if resource_type == "memory":
            suggestions.extend([
                "• 关闭不必要的程序和浏览器标签页",
                "• 清理系统垃圾文件和缓存",
                "• 重启程序释放内存泄漏",
                "• 考虑增加物理内存"
            ])
        elif resource_type == "cpu":
            suggestions.extend([
                "• 关闭高CPU占用的程序",
                "• 检查是否有病毒或恶意软件",
                "• 降低程序运行优先级",
                "• 等待当前任务完成"
            ])
        elif resource_type == "disk":
            suggestions.extend([
                "• 清理磁盘空间，删除不必要的文件",
                "• 清空回收站和临时文件",
                "• 移动大文件到其他磁盘",
                "• 使用磁盘清理工具"
            ])
            
        return "\n".join(suggestions)
            
    def perform_auto_cleanup(self):
        """执行自动清理"""
        cleanup_results = []
        
        try:
            # 1. 清理Python垃圾回收
            import gc
            collected = gc.collect()
            cleanup_results.append(f"垃圾回收: 清理了 {collected} 个对象")
            
            # 2. 清理临时文件
            temp_cleaned = self._cleanup_temp_files()
            if temp_cleaned > 0:
                cleanup_results.append(f"临时文件: 清理了 {temp_cleaned} 个文件")
            
            # 3. 清理缓存
            cache_cleaned = self._cleanup_cache()
            if cache_cleaned > 0:
                cleanup_results.append(f"缓存清理: 释放了 {cache_cleaned:.1f} MB")
            
            # 4. 清理图像处理缓存
            self._cleanup_image_cache()
            cleanup_results.append("图像缓存: 已清理")
            
            # 5. 强制内存整理
            if hasattr(gc, 'set_threshold'):
                gc.set_threshold(700, 10, 10)  # 更激进的垃圾回收
                
            # 显示清理结果
            if cleanup_results:
                result_text = "\n".join(cleanup_results)
                print(f"[自动清理] 清理完成:\n{result_text}")
                
                # 如果是弹窗提醒模式，显示清理结果
                if self.alert_method.currentText() in ["弹窗提醒", "全部启用"]:
                    QMessageBox.information(self, "自动清理完成", f"清理结果:\n{result_text}")
            
        except Exception as e:
            error_msg = f"自动清理失败: {e}"
            print(error_msg)
            if self.alert_method.currentText() in ["弹窗提醒", "全部启用"]:
                QMessageBox.warning(self, "清理失败", error_msg)
    
    def _cleanup_temp_files(self) -> int:
        """清理临时文件"""
        import tempfile
        import glob
        
        cleaned_count = 0
        temp_dirs = [tempfile.gettempdir()]
        
        # 添加用户临时目录
        if os.name == 'nt':  # Windows
            temp_dirs.extend([
                os.path.join(os.environ.get('USERPROFILE', ''), 'AppData', 'Local', 'Temp'),
                os.path.join(os.environ.get('TEMP', ''))
            ])
        else:  # Unix/Linux
            temp_dirs.extend(['/tmp', '/var/tmp'])
        
        for temp_dir in temp_dirs:
            if not os.path.exists(temp_dir):
                continue
                
            try:
                # 清理Python临时文件
                for pattern in ['*.tmp', '*.temp', 'tmp*', 'temp*']:
                    for file_path in glob.glob(os.path.join(temp_dir, pattern)):
                        try:
                            if os.path.isfile(file_path):
                                # 只清理超过1小时的文件
                                if time.time() - os.path.getmtime(file_path) > 3600:
                                    os.remove(file_path)
                                    cleaned_count += 1
                        except (OSError, PermissionError):
                            continue
                            
            except Exception:
                continue
                
        return cleaned_count
    
    def _cleanup_cache(self) -> float:
        """清理缓存，返回清理的MB数"""
        cleaned_mb = 0.0
        
        try:
            # 清理matplotlib缓存
            import matplotlib
            if hasattr(matplotlib, 'get_cachedir'):
                cache_dir = matplotlib.get_cachedir()
                if os.path.exists(cache_dir):
                    cleaned_mb += self._clean_directory(cache_dir, max_age_hours=24)
            
            # 清理PIL/Pillow缓存
            try:
                from PIL import Image
                if hasattr(Image, '_getdecoder'):
                    # 清理PIL解码器缓存
                    Image._getdecoder.cache_clear() if hasattr(Image._getdecoder, 'cache_clear') else None
            except:
                pass
            
            # 清理OpenCV缓存
            try:
                import cv2
                # OpenCV没有直接的缓存清理方法，但可以重置一些全局状态
                pass
            except:
                pass
                
        except Exception as e:
            print(f"缓存清理错误: {e}")
            
        return cleaned_mb
    
    def _cleanup_image_cache(self):
        """清理图像处理相关的缓存"""
        try:
            # 清理高级采样管理器的缓存
            from src.image_processing.advanced_sampling import AdvancedSamplingManager
            from src.image_processing.advanced_oversampling import AdvancedOversamplingManager
            
            # 这些类有clear_cache方法
            sampling_manager = AdvancedSamplingManager()
            sampling_manager.clear_cache()
            
            oversampling_manager = AdvancedOversamplingManager()
            oversampling_manager.clear_cache()
            
        except Exception as e:
            print(f"图像缓存清理错误: {e}")
    
    def _clean_directory(self, directory: str, max_age_hours: int = 24) -> float:
        """清理目录中的旧文件，返回清理的MB数"""
        cleaned_mb = 0.0
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        if current_time - os.path.getmtime(file_path) > max_age_seconds:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            cleaned_mb += file_size / (1024 * 1024)
                    except (OSError, PermissionError):
                        continue
        except Exception:
            pass
            
        return cleaned_mb
    
    def show_context_menu(self, position):
        """显示右键菜单"""
        from PyQt5.QtWidgets import QMenu, QAction
        
        menu = QMenu(self)
        
        # 重置为默认值
        reset_action = QAction("重置为默认值", self)
        reset_action.triggered.connect(self.reset_to_defaults)
        menu.addAction(reset_action)
        
        # 手动清理
        cleanup_action = QAction("立即清理系统", self)
        cleanup_action.triggered.connect(self.perform_auto_cleanup)
        menu.addAction(cleanup_action)
        
        menu.addSeparator()
        
        # 开始/停止监控
        if self.resource_monitor and self.resource_monitor.running:
            stop_action = QAction("停止监控", self)
            stop_action.triggered.connect(self.stop_monitoring)
            menu.addAction(stop_action)
        else:
            start_action = QAction("开始监控", self)
            start_action.triggered.connect(self.start_monitoring)
            start_action.setEnabled(self.psutil_available)
            menu.addAction(start_action)
        
        # 显示菜单
        menu.exec_(self.mapToGlobal(position))
    
    def _record_resource_history(self, resource_info: Dict[str, Any]):
        """记录资源使用历史"""
        import time
        
        current_time = time.time()
        
        # 记录CPU使用率
        cpu_percent = resource_info.get('cpu_percent', 0)
        self.resource_history['cpu'].append(cpu_percent)
        
        # 记录内存使用率
        memory_info = resource_info.get('memory', {})
        memory_percent = memory_info.get('percent', 0)
        self.resource_history['memory'].append(memory_percent)
        
        # 记录磁盘使用率
        disk_info = resource_info.get('disk', {})
        disk_percent = disk_info.get('percent', 0)
        self.resource_history['disk'].append(disk_percent)
        
        # 记录时间戳
        self.resource_history['timestamps'].append(current_time)
        
        # 保持历史记录长度
        for key in self.resource_history:
            if len(self.resource_history[key]) > self.max_history_length:
                self.resource_history[key] = self.resource_history[key][-self.max_history_length:]
    
    def get_resource_trends(self) -> Dict[str, str]:
        """获取资源使用趋势"""
        trends = {}
        
        for resource_type in ['cpu', 'memory', 'disk']:
            history = self.resource_history[resource_type]
            if len(history) < 10:  # 需要至少10个数据点
                trends[resource_type] = "数据不足"
                continue
                
            # 计算最近10个数据点的平均值和之前10个数据点的平均值
            recent_avg = sum(history[-10:]) / 10
            previous_avg = sum(history[-20:-10]) / 10 if len(history) >= 20 else recent_avg
            
            if recent_avg > previous_avg + 5:
                trends[resource_type] = "上升"
            elif recent_avg < previous_avg - 5:
                trends[resource_type] = "下降"
            else:
                trends[resource_type] = "稳定"
                
        return trends
            
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
            
            # CPU信息
            cpu_count = self.current_resources.get('cpu_count', 0)
            cpu_freq = self.current_resources.get('cpu_freq', {})
            cpu_current = cpu_freq.get('current', 0) / 1000 if cpu_freq.get('current') else 0
            cpu_max = cpu_freq.get('max', 0) / 1000 if cpu_freq.get('max') else 0
            
            details.append(f"CPU: {cpu_percent:.1f}% ({cpu_count} 核心)")
            if cpu_current > 0:
                details.append(f"CPU频率: {cpu_current:.1f}GHz / {cpu_max:.1f}GHz")
            
            # 内存信息
            if memory_info:
                memory_gb = memory_info.get('used', 0) / (1024**3)
                total_gb = memory_info.get('total', 0) / (1024**3)
                free_gb = memory_info.get('free', 0) / (1024**3)
                details.append(f"内存: {memory_gb:.1f}GB / {total_gb:.1f}GB ({memory_percent:.1f}%)")
                details.append(f"可用内存: {free_gb:.1f}GB")
                
            # 磁盘信息
            if disk_info:
                disk_gb = disk_info.get('used', 0) / (1024**3)
                total_disk_gb = disk_info.get('total', 0) / (1024**3)
                free_disk_gb = disk_info.get('free', 0) / (1024**3)
                main_drive = disk_info.get('main_drive', 'Unknown')
                
                details.append(f"磁盘 ({main_drive}): {disk_gb:.1f}GB / {total_disk_gb:.1f}GB ({disk_percent:.1f}%)")
                details.append(f"可用磁盘: {free_disk_gb:.1f}GB")
                
                # 显示所有驱动器信息（Windows）
                all_drives = disk_info.get('all_drives', [])
                if len(all_drives) > 1:
                    details.append("所有驱动器:")
                    for drive in all_drives:
                        drive_used = drive['used'] / (1024**3)
                        drive_total = drive['total'] / (1024**3)
                        drive_percent = drive['percent']
                        details.append(f"  {drive['drive']}: {drive_used:.1f}GB / {drive_total:.1f}GB ({drive_percent:.1f}%)")
                
            # 网络信息
            network_info = self.current_resources.get('network', {})
            if network_info:
                bytes_sent = network_info.get('bytes_sent', 0) / (1024**2)  # MB
                bytes_recv = network_info.get('bytes_recv', 0) / (1024**2)  # MB
                details.append(f"网络: 发送 {bytes_sent:.1f}MB, 接收 {bytes_recv:.1f}MB")
            
            # 进程信息
            process_count = self.current_resources.get('process_count', 0)
            details.append(f"进程数: {process_count}")
            
            # 资源使用趋势
            trends = self.get_resource_trends()
            if any(trend != "数据不足" for trend in trends.values()):
                details.append("")  # 空行分隔
                details.append("资源使用趋势:")
                for resource_type, trend in trends.items():
                    if trend != "数据不足":
                        resource_name = {"cpu": "CPU", "memory": "内存", "disk": "磁盘"}[resource_type]
                        trend_icon = {"上升": "↗", "下降": "↘", "稳定": "→"}[trend]
                        details.append(f"  {resource_name}: {trend} {trend_icon}")
            
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
            'enforce_limits_enabled': self.enforce_limits_enabled.isChecked(),
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
            self.enforce_limits_enabled.setChecked(config.get('enforce_limits_enabled', False))
            
        except Exception as e:
            print(f"设置资源限制配置错误: {e}")
    
    def validate_configuration(self) -> tuple[bool, list]:
        """验证资源限制配置的有效性"""
        errors = []
        
        try:
            # 检查内存限制设置
            if self.memory_limit_enabled.isChecked():
                memory_percent = self.memory_percent_limit.value()
                memory_absolute = self.memory_absolute_limit.value()
                
                if memory_percent < 50:
                    errors.append("内存使用率限制不应低于50%")
                if memory_absolute < 0.5:
                    errors.append("内存绝对限制不应低于0.5GB")
                    
            # 检查CPU限制设置
            if self.cpu_limit_enabled.isChecked():
                cpu_percent = self.cpu_percent_limit.value()
                cpu_cores = self.cpu_cores_limit.value()
                
                if cpu_percent < 30:
                    errors.append("CPU使用率限制不应低于30%")
                if cpu_cores < 1:
                    errors.append("CPU核心数限制不应低于1")
                    
            # 检查磁盘限制设置
            if self.disk_limit_enabled.isChecked():
                disk_percent = self.disk_percent_limit.value()
                temp_files = self.temp_files_limit.value()
                
                if disk_percent < 70:
                    errors.append("磁盘使用率限制不应低于70%")
                if temp_files < 1.0:
                    errors.append("临时文件大小限制不应低于1GB")
                    
            # 检查监控间隔
            check_interval = self.check_interval_spin.value()
            if check_interval < 0.5:
                errors.append("检查间隔不应低于0.5秒")
            elif check_interval > 10.0:
                errors.append("检查间隔不应超过10秒")
                
        except Exception as e:
            errors.append(f"配置验证时出错: {str(e)}")
            
        return len(errors) == 0, errors
    
    def get_status_summary(self) -> str:
        """获取资源限制状态摘要"""
        try:
            status_parts = []
            
            # 监控状态
            if self.resource_monitor and self.resource_monitor.running:
                status_parts.append("监控: 运行中")
            else:
                status_parts.append("监控: 已停止")
                
            # 启用的限制
            enabled_limits = []
            if self.memory_limit_enabled.isChecked():
                enabled_limits.append(f"内存({self.memory_percent_limit.value()}%)")
            if self.cpu_limit_enabled.isChecked():
                enabled_limits.append(f"CPU({self.cpu_percent_limit.value()}%)")
            if self.disk_limit_enabled.isChecked():
                enabled_limits.append(f"磁盘({self.disk_percent_limit.value()}%)")
                
            if enabled_limits:
                status_parts.append(f"限制: {', '.join(enabled_limits)}")
            else:
                status_parts.append("限制: 无")
                
            # 警告状态
            active_warnings = [k for k, v in self.warning_states.items() if v]
            if active_warnings:
                warning_names = [{"memory": "内存", "cpu": "CPU", "disk": "磁盘"}[w] for w in active_warnings]
                status_parts.append(f"警告: {', '.join(warning_names)}")
                
            return " | ".join(status_parts)
            
        except Exception as e:
            return f"状态获取失败: {str(e)}"
            
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