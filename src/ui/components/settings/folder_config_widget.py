"""
文件夹配置组件 - 负责管理默认文件夹路径设置
"""

import os
from typing import Dict, Any
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                           QGridLayout, QLabel, QLineEdit, QPushButton, QFileDialog)
from PyQt5.QtCore import pyqtSignal


class FolderConfigWidget(QWidget):
    """文件夹配置组件"""
    
    # 定义信号
    folder_changed = pyqtSignal(str, str)  # 文件夹类型, 文件夹路径
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.folder_configs = {}
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建默认文件夹组
        folders_group = QGroupBox("默认文件夹")
        folders_layout = QGridLayout()
        folders_layout.setContentsMargins(10, 20, 10, 10)
        
        # 定义文件夹配置项
        self.folder_items = [
            ('source', '默认源文件夹', 'default_source_folder'),
            ('output', '默认输出文件夹', 'default_output_folder')
        ]
        
        self.folder_edits = {}
        
        for row, (key, label, config_key) in enumerate(self.folder_items):
            # 创建标签
            label_widget = QLabel(f"{label}:")
            folders_layout.addWidget(label_widget, row, 0)
            
            # 创建路径输入框
            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setPlaceholderText(label)
            self.folder_edits[key] = edit
            folders_layout.addWidget(edit, row, 1)
            
            # 创建浏览按钮
            browse_btn = QPushButton("浏览...")
            browse_btn.clicked.connect(lambda checked, k=key, l=label: self.select_folder(k, l))
            folders_layout.addWidget(browse_btn, row, 2)
        
        folders_group.setLayout(folders_layout)
        layout.addWidget(folders_group)
    
    def select_folder(self, key: str, label: str):
        """选择文件夹"""
        folder = QFileDialog.getExistingDirectory(self, f"选择{label}")
        if folder:
            # 标准化路径格式
            folder = os.path.normpath(folder)
            self.folder_edits[key].setText(folder)
            self.folder_configs[key] = folder
            
            # 发送信号通知文件夹变化
            self.folder_changed.emit(key, folder)
    
    def get_folder_config(self) -> Dict[str, str]:
        """获取文件夹配置"""
        config = {}
        for key, (_, _, config_key) in zip(self.folder_edits.keys(), self.folder_items):
            config[config_key] = self.folder_edits[key].text()
        return config
    
    def set_folder_config(self, config: Dict[str, Any]):
        """设置文件夹配置"""
        mapping = {
            'source': 'default_source_folder',
            'output': 'default_output_folder'
        }
        
        for key, config_key in mapping.items():
            value = config.get(config_key, '')
            if key in self.folder_edits:
                self.folder_edits[key].setText(value)
                self.folder_configs[key] = value
    
    def clear_config(self):
        """清空配置"""
        for edit in self.folder_edits.values():
            edit.clear()
        self.folder_configs.clear()
    
    def validate_paths(self) -> Dict[str, bool]:
        """验证路径有效性"""
        validation_result = {}
        for key, edit in self.folder_edits.items():
            path = edit.text()
            validation_result[key] = os.path.exists(path) if path else True
        return validation_result 