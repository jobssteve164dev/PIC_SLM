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
    QCheckBox, QSpinBox, QProgressBar, QTabWidget
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
        
        self.init_ui()
        self.setup_log_monitoring()
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
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
        
    def create_log_viewer_tab(self) -> QWidget:
        """创建日志查看器标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 控制面板
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # 主要内容区域
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # 日志显示区域
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 9))
        splitter.addWidget(self.log_display)
        
        # 日志详情面板
        self.log_details = self.create_log_details_panel()
        splitter.addWidget(self.log_details)
        
        # 设置分割器比例
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        return widget
        
    def create_control_panel(self) -> QWidget:
        """创建控制面板"""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        
        # 日志级别过滤
        layout.addWidget(QLabel("级别:"))
        self.level_filter = QComboBox()
        self.level_filter.addItems(["全部", "DEBUG", "INFO", "WARNING", "ERROR"])
        self.level_filter.currentTextChanged.connect(self.apply_filters)
        layout.addWidget(self.level_filter)
        
        # 组件过滤
        layout.addWidget(QLabel("组件:"))
        self.component_filter = QComboBox()
        self.component_filter.addItem("全部")
        self.component_filter.currentTextChanged.connect(self.apply_filters)
        layout.addWidget(self.component_filter)
        
        # 搜索框
        layout.addWidget(QLabel("搜索:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入关键词搜索...")
        self.search_input.textChanged.connect(self.apply_filters)
        layout.addWidget(self.search_input)
        
        # 实时更新开关
        self.auto_update_checkbox = QCheckBox("实时更新")
        self.auto_update_checkbox.setChecked(True)
        self.auto_update_checkbox.toggled.connect(self.toggle_auto_update)
        layout.addWidget(self.auto_update_checkbox)
        
        # 清空按钮
        clear_button = QPushButton("清空")
        clear_button.clicked.connect(self.clear_logs)
        layout.addWidget(clear_button)
        
        # 导出按钮
        export_button = QPushButton("导出")
        export_button.clicked.connect(self.export_logs)
        layout.addWidget(export_button)
        
        layout.addStretch()
        return panel
        
    def create_log_details_panel(self) -> QWidget:
        """创建日志详情面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        layout.addWidget(QLabel("日志详情:"))
        
        self.details_display = QTextEdit()
        self.details_display.setReadOnly(True)
        self.details_display.setFont(QFont("Consolas", 9))
        layout.addWidget(self.details_display)
        
        return panel
        
    def create_log_stats_tab(self) -> QWidget:
        """创建日志统计标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 统计信息表格
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(3)
        self.stats_table.setHorizontalHeaderLabels(["指标", "数值", "描述"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.stats_table)
        
        # 刷新按钮
        refresh_button = QPushButton("刷新统计")
        refresh_button.clicked.connect(self.refresh_stats)
        layout.addWidget(refresh_button)
        
        return widget
        
    def create_log_management_tab(self) -> QWidget:
        """创建日志管理标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 日志文件列表
        layout.addWidget(QLabel("日志文件:"))
        self.log_files_table = QTableWidget()
        self.log_files_table.setColumnCount(4)
        self.log_files_table.setHorizontalHeaderLabels(["文件名", "大小(MB)", "修改时间", "操作"])
        self.log_files_table.horizontalHeader().setStretchLastSection(True)
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
            
        # 滚动到底部
        cursor = self.log_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_display.setTextCursor(cursor)
        
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
            
            # 计算日志级别统计
            level_counts = {}
            component_counts = {}
            
            for entry in self.log_entries:
                level = entry.get('level', 'INFO')
                component = entry.get('component', 'unknown')
                
                level_counts[level] = level_counts.get(level, 0) + 1
                component_counts[component] = component_counts.get(component, 0) + 1
                
            # 填充统计表格
            stats_data = [
                ("日志目录", stats['log_directory'], "日志文件存储目录"),
                ("日志文件数量", str(len(stats['log_files'])), "当前日志文件总数"),
                ("总大小", f"{stats['total_size_mb']} MB", "所有日志文件总大小"),
                ("内存中日志条数", str(len(self.log_entries)), "当前加载的日志条目数"),
                ("ERROR数量", str(level_counts.get('ERROR', 0)), "错误级别日志数量"),
                ("WARNING数量", str(level_counts.get('WARNING', 0)), "警告级别日志数量"),
                ("INFO数量", str(level_counts.get('INFO', 0)), "信息级别日志数量"),
                ("DEBUG数量", str(level_counts.get('DEBUG', 0)), "调试级别日志数量"),
            ]
            
            self.stats_table.setRowCount(len(stats_data))
            for i, (metric, value, description) in enumerate(stats_data):
                self.stats_table.setItem(i, 0, QTableWidgetItem(metric))
                self.stats_table.setItem(i, 1, QTableWidgetItem(value))
                self.stats_table.setItem(i, 2, QTableWidgetItem(description))
                
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