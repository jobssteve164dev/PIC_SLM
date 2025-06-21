"""
日志查看器组件

功能特性：
- 实时显示日志内容
- 支持日志级别过滤
- 支持日志搜索
- 支持日志导出
- 显示日志统计信息
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QComboBox, 
    QLineEdit, QPushButton, QLabel, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
    QCheckBox, QSpinBox, QProgressBar, QTabWidget, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QTextCursor, QColor, QPalette
from src.utils.logger import LoggerManager, get_logger


class LogReaderThread(QThread):
    """日志读取线程"""
    
    log_updated = pyqtSignal(list)  # 发送新的日志条目
    
    def __init__(self, log_file_path: str, parent=None):
        super().__init__(parent)
        self.log_file_path = log_file_path
        self.last_position = 0
        self.running = True
        
    def run(self):
        """读取日志文件的新内容"""
        while self.running:
            try:
                if os.path.exists(self.log_file_path):
                    with open(self.log_file_path, 'r', encoding='utf-8') as f:
                        f.seek(self.last_position)
                        new_lines = f.readlines()
                        if new_lines:
                            # 解析结构化日志
                            log_entries = []
                            for line in new_lines:
                                try:
                                    if line.strip():
                                        entry = json.loads(line.strip())
                                        log_entries.append(entry)
                                except json.JSONDecodeError:
                                    # 如果不是JSON格式，作为普通文本处理
                                    log_entries.append({
                                        'timestamp': datetime.now().isoformat(),
                                        'level': 'INFO',
                                        'message': line.strip(),
                                        'logger': 'unknown'
                                    })
                            
                            if log_entries:
                                self.log_updated.emit(log_entries)
                            
                        self.last_position = f.tell()
                
                self.msleep(1000)  # 每秒检查一次
                
            except Exception as e:
                print(f"读取日志文件时出错: {e}")
                self.msleep(5000)  # 出错时等待5秒再重试
    
    def stop(self):
        """停止线程"""
        self.running = False


class LogViewerWidget(QWidget):
    """日志查看器主组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger(__name__, "log_viewer")
        self.log_entries = []
        self.filtered_entries = []
        self.log_reader_threads = {}
        self.refresh_timer = None
        
        self.init_ui()
        self.setup_log_monitoring()
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        layout.addWidget(self.tab_widget)
        
        # 日志查看器标签页
        self.log_viewer_tab = self.create_log_viewer_tab()
        self.tab_widget.addTab(self.log_viewer_tab, "日志查看器")
        
        # 日志统计标签页
        self.log_stats_tab = self.create_log_stats_tab()
        self.tab_widget.addTab(self.log_stats_tab, "日志统计")
        
        # 日志管理标签页
        self.log_management_tab = self.create_log_management_tab()
        self.tab_widget.addTab(self.log_management_tab, "日志管理")
        
        # 设置最小尺寸
        self.setMinimumSize(800, 600)
        
    def create_log_viewer_tab(self) -> QWidget:
        """创建日志查看器标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 控制面板
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # 主要内容区域 - 使用垂直分割器
        main_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(main_splitter)
        
        # 上半部分：日志显示和详情的水平分割
        top_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(top_splitter)
        
        # 日志显示区域
        log_container = QFrame()
        log_container.setFrameStyle(QFrame.StyledPanel)
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(2, 2, 2, 2)
        
        log_label = QLabel("日志内容:")
        log_label.setStyleSheet("font-weight: bold; color: #333;")
        log_layout.addWidget(log_label)
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 9))
        self.log_display.setLineWrapMode(QTextEdit.NoWrap)
        # 设置最小和推荐尺寸
        self.log_display.setMinimumSize(400, 200)
        self.log_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_layout.addWidget(self.log_display)
        
        top_splitter.addWidget(log_container)
        
        # 日志详情面板
        self.log_details = self.create_log_details_panel()
        top_splitter.addWidget(self.log_details)
        
        # 设置水平分割器比例 (日志显示:详情 = 7:3)
        top_splitter.setStretchFactor(0, 7)
        top_splitter.setStretchFactor(1, 3)
        top_splitter.setSizes([560, 240])  # 初始大小
        
        # 下半部分：统计信息面板（可折叠）
        stats_container = self.create_quick_stats_panel()
        main_splitter.addWidget(stats_container)
        
        # 设置垂直分割器比例 (主要内容:统计 = 8:2)
        main_splitter.setStretchFactor(0, 8)
        main_splitter.setStretchFactor(1, 2)
        main_splitter.setSizes([480, 120])  # 初始大小
        
        return widget
        
    def create_control_panel(self) -> QWidget:
        """创建控制面板"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # 使用网格布局来更好地适应不同窗口大小
        main_layout = QVBoxLayout(panel)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(5)
        
        # 第一行：过滤器
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(10)
        
        # 日志级别过滤
        filter_layout.addWidget(QLabel("级别:"))
        self.level_filter = QComboBox()
        self.level_filter.addItems(["全部", "DEBUG", "INFO", "WARNING", "ERROR"])
        self.level_filter.setMinimumWidth(80)
        self.level_filter.currentTextChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.level_filter)
        
        # 组件过滤
        filter_layout.addWidget(QLabel("组件:"))
        self.component_filter = QComboBox()
        self.component_filter.addItem("全部")
        self.component_filter.setMinimumWidth(120)
        self.component_filter.currentTextChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.component_filter)
        
        # 搜索框
        filter_layout.addWidget(QLabel("搜索:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入关键词搜索...")
        self.search_input.setMinimumWidth(200)
        self.search_input.textChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.search_input)
        
        filter_layout.addStretch()
        main_layout.addLayout(filter_layout)
        
        # 第二行：控制按钮
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)
        
        # 实时更新开关
        self.auto_update_checkbox = QCheckBox("实时更新")
        self.auto_update_checkbox.setChecked(True)
        self.auto_update_checkbox.toggled.connect(self.toggle_auto_update)
        control_layout.addWidget(self.auto_update_checkbox)
        
        # 按钮组
        button_group = QHBoxLayout()
        button_group.setSpacing(5)
        
        clear_button = QPushButton("清空")
        clear_button.setMaximumWidth(60)
        clear_button.clicked.connect(self.clear_logs)
        button_group.addWidget(clear_button)
        
        export_button = QPushButton("导出")
        export_button.setMaximumWidth(60)
        export_button.clicked.connect(self.export_logs)
        button_group.addWidget(export_button)
        
        refresh_button = QPushButton("刷新")
        refresh_button.setMaximumWidth(60)
        refresh_button.clicked.connect(self.refresh_stats)
        button_group.addWidget(refresh_button)
        
        control_layout.addLayout(button_group)
        control_layout.addStretch()
        
        main_layout.addLayout(control_layout)
        
        return panel
        
    def create_log_details_panel(self) -> QWidget:
        """创建日志详情面板"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setMinimumWidth(200)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        detail_label = QLabel("日志详情:")
        detail_label.setStyleSheet("font-weight: bold; color: #333;")
        layout.addWidget(detail_label)
        
        self.details_display = QTextEdit()
        self.details_display.setReadOnly(True)
        self.details_display.setFont(QFont("Consolas", 8))
        self.details_display.setMaximumHeight(300)  # 限制最大高度
        layout.addWidget(self.details_display)
        
        return panel
        
    def create_quick_stats_panel(self) -> QWidget:
        """创建快速统计面板"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setMaximumHeight(150)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        stats_label = QLabel("实时统计:")
        stats_label.setStyleSheet("font-weight: bold; color: #333;")
        layout.addWidget(stats_label)
        
        # 统计信息显示
        self.quick_stats_display = QTextEdit()
        self.quick_stats_display.setReadOnly(True)
        self.quick_stats_display.setFont(QFont("Arial", 8))
        self.quick_stats_display.setMaximumHeight(100)
        layout.addWidget(self.quick_stats_display)
        
        return panel
        
    def create_log_stats_tab(self) -> QWidget:
        """创建日志统计标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 统计信息表格
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(3)
        self.stats_table.setHorizontalHeaderLabels(["指标", "数值", "描述"])
        
        # 设置表格列宽自适应
        header = self.stats_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        
        # 设置表格行高自适应
        self.stats_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        
        layout.addWidget(self.stats_table)
        
        # 刷新按钮
        button_layout = QHBoxLayout()
        refresh_button = QPushButton("刷新统计")
        refresh_button.clicked.connect(self.refresh_stats)
        button_layout.addWidget(refresh_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        return widget
        
    def create_log_management_tab(self) -> QWidget:
        """创建日志管理标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 日志文件列表
        file_label = QLabel("日志文件:")
        file_label.setStyleSheet("font-weight: bold; color: #333;")
        layout.addWidget(file_label)
        
        self.log_files_table = QTableWidget()
        self.log_files_table.setColumnCount(4)
        self.log_files_table.setHorizontalHeaderLabels(["文件名", "大小(MB)", "修改时间", "操作"])
        
        # 设置表格列宽
        header = self.log_files_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        
        # 设置表格行高
        self.log_files_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        
        layout.addWidget(self.log_files_table)
        
        # 管理按钮
        button_layout = QHBoxLayout()
        
        refresh_files_button = QPushButton("刷新文件列表")
        refresh_files_button.clicked.connect(self.refresh_log_files)
        button_layout.addWidget(refresh_files_button)
        
        open_log_dir_button = QPushButton("打开日志目录")
        open_log_dir_button.clicked.connect(self.open_log_directory)
        button_layout.addWidget(open_log_dir_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        return widget
        
    def setup_log_monitoring(self):
        """设置日志监控"""
        logger_manager = LoggerManager()
        log_dir = logger_manager.config.log_dir
        
        # 监控主要日志文件
        log_files = ['main.log', 'errors.log', 'performance.log']
        
        for log_file in log_files:
            log_path = os.path.join(log_dir, log_file)
            if os.path.exists(log_path):
                thread = LogReaderThread(log_path)
                thread.log_updated.connect(self.add_log_entries)
                thread.start()
                self.log_reader_threads[log_file] = thread
                
        # 定时刷新组件列表
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_component_filter)
        self.refresh_timer.start(5000)  # 每5秒刷新一次
        
    @pyqtSlot(list)
    def add_log_entries(self, entries: List[Dict]):
        """添加新的日志条目"""
        self.log_entries.extend(entries)
        
        # 更新组件过滤器
        components = set()
        for entry in self.log_entries:
            if 'component' in entry:
                components.add(entry['component'])
                
        current_components = [self.component_filter.itemText(i) 
                             for i in range(1, self.component_filter.count())]
        
        for component in components:
            if component not in current_components:
                self.component_filter.addItem(component)
        
        # 应用过滤器并更新显示
        if self.auto_update_checkbox.isChecked():
            self.apply_filters()
            
    def apply_filters(self):
        """应用过滤器"""
        level_filter = self.level_filter.currentText()
        component_filter = self.component_filter.currentText()
        search_text = self.search_input.text().lower()
        
        self.filtered_entries = []
        
        for entry in self.log_entries:
            # 级别过滤
            if level_filter != "全部" and entry.get('level') != level_filter:
                continue
                
            # 组件过滤
            if component_filter != "全部" and entry.get('component') != component_filter:
                continue
                
            # 搜索过滤
            if search_text:
                searchable_text = f"{entry.get('message', '')} {entry.get('logger', '')}".lower()
                if search_text not in searchable_text:
                    continue
                    
            self.filtered_entries.append(entry)
            
        self.update_log_display()
        self.update_quick_stats()
        
    def update_log_display(self):
        """更新日志显示"""
        self.log_display.clear()
        
        for entry in self.filtered_entries[-1000:]:  # 只显示最近1000条
            timestamp = entry.get('timestamp', '')
            level = entry.get('level', 'INFO')
            logger_name = entry.get('logger', '')
            message = entry.get('message', '')
            component = entry.get('component', '')
            
            # 格式化日志行
            if component:
                log_line = f"[{timestamp}] [{level}] {logger_name}({component}): {message}"
            else:
                log_line = f"[{timestamp}] [{level}] {logger_name}: {message}"
                
            # 根据级别设置颜色
            if level == 'ERROR':
                color = 'red'
            elif level == 'WARNING':
                color = 'orange'
            elif level == 'DEBUG':
                color = 'gray'
            else:
                color = 'black'
                
            self.log_display.append(f'<span style="color: {color};">{log_line}</span>')
            
        # 滚动到底部（如果启用了自动更新）
        if self.auto_update_checkbox.isChecked():
            cursor = self.log_display.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.log_display.setTextCursor(cursor)
        
    def update_quick_stats(self):
        """更新快速统计信息"""
        if not hasattr(self, 'quick_stats_display'):
            return
            
        if not self.filtered_entries:
            self.quick_stats_display.setText("暂无日志数据")
            return
            
        # 统计各级别日志数量
        level_counts = {}
        component_counts = {}
        
        for entry in self.filtered_entries:
            level = entry.get('level', 'UNKNOWN')
            component = entry.get('component', '未知组件')
            
            level_counts[level] = level_counts.get(level, 0) + 1
            component_counts[component] = component_counts.get(component, 0) + 1
        
        # 生成统计文本
        stats_text = f"总计: {len(self.filtered_entries)} 条日志\n\n"
        
        # 级别统计
        stats_text += "按级别统计:\n"
        for level in ['ERROR', 'WARNING', 'INFO', 'DEBUG']:
            count = level_counts.get(level, 0)
            if count > 0:
                stats_text += f"  {level}: {count} 条\n"
        
        # 组件统计（只显示前5个）
        if component_counts:
            stats_text += "\n主要组件:\n"
            sorted_components = sorted(component_counts.items(), key=lambda x: x[1], reverse=True)
            for component, count in sorted_components[:5]:
                stats_text += f"  {component}: {count} 条\n"
        
        self.quick_stats_display.setText(stats_text)
    
    def toggle_auto_update(self, enabled: bool):
        """切换自动更新"""
        if enabled:
            self.apply_filters()
            
    def clear_logs(self):
        """清空日志显示"""
        self.log_display.clear()
        self.details_display.clear()
        self.log_entries.clear()
        self.filtered_entries.clear()
        
    def export_logs(self):
        """导出日志"""
        if not self.filtered_entries:
            QMessageBox.information(self, "提示", "没有日志可以导出")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出日志", f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "文本文件 (*.txt);;JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(self.filtered_entries, f, ensure_ascii=False, indent=2)
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        for entry in self.filtered_entries:
                            timestamp = entry.get('timestamp', '')
                            level = entry.get('level', 'INFO')
                            logger_name = entry.get('logger', '')
                            message = entry.get('message', '')
                            component = entry.get('component', '')
                            
                            if component:
                                line = f"[{timestamp}] [{level}] {logger_name}({component}): {message}\n"
                            else:
                                line = f"[{timestamp}] [{level}] {logger_name}: {message}\n"
                            f.write(line)
                            
                QMessageBox.information(self, "成功", f"日志已导出到: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出日志失败: {str(e)}")
                
    def refresh_stats(self):
        """刷新统计信息"""
        try:
            logger_manager = LoggerManager()
            stats = logger_manager.get_log_stats()
            
            # 计算日志级别和组件统计
            level_counts = {}
            component_counts = {}
            
            for entry in self.log_entries:
                level = entry.get('level', 'INFO')
                component = entry.get('component', 'unknown')
                
                level_counts[level] = level_counts.get(level, 0) + 1
                component_counts[component] = component_counts.get(component, 0) + 1
                
            # 填充统计表格
            stats_data = [
                ("日志目录", stats.get('log_directory', '未知'), "日志文件存储目录"),
                ("日志文件数量", str(len(stats.get('log_files', []))), "当前日志文件总数"),
                ("总大小", f"{stats.get('total_size_mb', 0):.2f} MB", "所有日志文件总大小"),
                ("内存中日志条数", str(len(self.log_entries)), "当前加载的日志条目数"),
                ("当前显示条数", str(len(self.filtered_entries)), "经过过滤后显示的日志条目数"),
                ("ERROR数量", str(level_counts.get('ERROR', 0)), "错误级别日志数量"),
                ("WARNING数量", str(level_counts.get('WARNING', 0)), "警告级别日志数量"),
                ("INFO数量", str(level_counts.get('INFO', 0)), "信息级别日志数量"),
                ("DEBUG数量", str(level_counts.get('DEBUG', 0)), "调试级别日志数量"),
            ]
            
            # 添加主要组件统计（前5个）
            if component_counts:
                sorted_components = sorted(component_counts.items(), key=lambda x: x[1], reverse=True)
                for i, (component, count) in enumerate(sorted_components[:5]):
                    if component != 'unknown':
                        stats_data.append((f"组件 {i+1}", f"{component} ({count})", f"活跃组件及其日志数量"))
            
            self.stats_table.setRowCount(len(stats_data))
            for i, (metric, value, description) in enumerate(stats_data):
                self.stats_table.setItem(i, 0, QTableWidgetItem(metric))
                self.stats_table.setItem(i, 1, QTableWidgetItem(value))
                self.stats_table.setItem(i, 2, QTableWidgetItem(description))
            
            # 同时更新快速统计
            self.update_quick_stats()
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"刷新统计信息失败: {str(e)}")
            
    def refresh_log_files(self):
        """刷新日志文件列表"""
        try:
            logger_manager = LoggerManager()
            stats = logger_manager.get_log_stats()
            
            self.log_files_table.setRowCount(len(stats['log_files']))
            
            for i, file_info in enumerate(stats['log_files']):
                self.log_files_table.setItem(i, 0, QTableWidgetItem(file_info['name']))
                self.log_files_table.setItem(i, 1, QTableWidgetItem(str(file_info['size_mb'])))
                self.log_files_table.setItem(i, 2, QTableWidgetItem(file_info['modified']))
                
                # 操作按钮
                open_button = QPushButton("打开")
                open_button.clicked.connect(
                    lambda checked, name=file_info['name']: self.open_log_file(name)
                )
                self.log_files_table.setCellWidget(i, 3, open_button)
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"刷新文件列表失败: {str(e)}")
            
    def open_log_file(self, filename: str):
        """打开指定的日志文件"""
        try:
            logger_manager = LoggerManager()
            log_path = os.path.join(logger_manager.config.log_dir, filename)
            
            import subprocess
            import sys
            
            if sys.platform == 'win32':
                subprocess.Popen(['notepad', log_path])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', log_path])
            else:
                subprocess.Popen(['xdg-open', log_path])
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开日志文件失败: {str(e)}")
            
    def open_log_directory(self):
        """打开日志目录"""
        try:
            logger_manager = LoggerManager()
            log_dir = logger_manager.config.log_dir
            
            import subprocess
            import sys
            
            if sys.platform == 'win32':
                subprocess.Popen(['explorer', log_dir])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', log_dir])
            else:
                subprocess.Popen(['xdg-open', log_dir])
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开日志目录失败: {str(e)}")
            
    def refresh_component_filter(self):
        """刷新组件过滤器"""
        # 这个方法由定时器调用，已经在add_log_entries中实现
        pass
        
    def closeEvent(self, event):
        """关闭事件处理"""
        # 停止所有日志读取线程
        for thread in self.log_reader_threads.values():
            thread.stop()
            thread.wait()
            
        if hasattr(self, 'refresh_timer'):
            self.refresh_timer.stop()
            
        super().closeEvent(event) 