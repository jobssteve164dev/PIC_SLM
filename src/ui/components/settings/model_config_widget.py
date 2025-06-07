"""
模型配置组件 - 负责管理模型相关的文件和文件夹设置
"""

import os
from typing import Dict, Any
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QGridLayout, 
                           QLabel, QLineEdit, QPushButton, QFileDialog)
from PyQt5.QtCore import pyqtSignal


class ModelConfigWidget(QWidget):
    """模型配置组件"""
    
    # 定义信号
    config_changed = pyqtSignal(str, str)  # 配置类型, 配置值
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_configs = {}
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建模型文件组
        model_group = QGroupBox("默认模型文件")
        model_layout = QGridLayout()
        model_layout.setContentsMargins(10, 20, 10, 10)
        
        # 定义配置项 (key, label, config_key, is_file, file_filter)
        self.config_items = [
            ('model_file', '默认模型文件', 'default_model_file', True, "模型文件 (*.pth *.pt *.h5)"),
            ('class_info_file', '默认类别信息', 'default_class_info_file', True, "JSON文件 (*.json)"),
            ('model_eval_dir', '默认模型评估文件夹', 'default_model_eval_dir', False, ""),
            ('model_save_dir', '默认模型保存文件夹', 'default_model_save_dir', False, ""),
            ('tensorboard_log_dir', '默认TensorBoard日志文件夹', 'default_tensorboard_log_dir', False, ""),
            ('dataset_dir', '默认数据集评估文件夹', 'default_dataset_dir', False, ""),
            ('param_save_dir', '默认训练参数保存文件夹', 'default_param_save_dir', False, "")
        ]
        
        self.config_edits = {}
        
        for row, (key, label, config_key, is_file, file_filter) in enumerate(self.config_items):
            # 创建标签
            label_widget = QLabel(f"{label}:")
            model_layout.addWidget(label_widget, row, 0)
            
            # 创建路径输入框
            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setPlaceholderText(label)
            self.config_edits[key] = edit
            model_layout.addWidget(edit, row, 1)
            
            # 创建浏览按钮
            browse_btn = QPushButton("浏览...")
            if is_file:
                browse_btn.clicked.connect(
                    lambda checked, k=key, l=label, f=file_filter: self.select_file(k, l, f)
                )
            else:
                browse_btn.clicked.connect(
                    lambda checked, k=key, l=label: self.select_folder(k, l)
                )
            model_layout.addWidget(browse_btn, row, 2)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
    
    def select_file(self, key: str, label: str, file_filter: str):
        """选择文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, f"选择{label}", "", file_filter)
        if file_path:
            # 标准化路径格式
            file_path = os.path.normpath(file_path)
            self.config_edits[key].setText(file_path)
            self.model_configs[key] = file_path
            
            # 发送信号通知配置变化
            self.config_changed.emit(key, file_path)
    
    def select_folder(self, key: str, label: str):
        """选择文件夹"""
        folder = QFileDialog.getExistingDirectory(self, f"选择{label}")
        if folder:
            # 标准化路径格式
            folder = os.path.normpath(folder)
            self.config_edits[key].setText(folder)
            self.model_configs[key] = folder
            
            # 发送信号通知配置变化
            self.config_changed.emit(key, folder)
    
    def get_model_config(self) -> Dict[str, str]:
        """获取模型配置"""
        config = {}
        for key, (_, _, config_key, _, _) in zip(self.config_edits.keys(), self.config_items):
            config[config_key] = self.config_edits[key].text()
        return config
    
    def set_model_config(self, config: Dict[str, Any]):
        """设置模型配置"""
        mapping = {
            'model_file': 'default_model_file',
            'class_info_file': 'default_class_info_file',
            'model_eval_dir': 'default_model_eval_dir',
            'model_save_dir': 'default_model_save_dir',
            'tensorboard_log_dir': 'default_tensorboard_log_dir',
            'dataset_dir': 'default_dataset_dir',
            'param_save_dir': 'default_param_save_dir'
        }
        
        for key, config_key in mapping.items():
            value = config.get(config_key, '')
            if key in self.config_edits:
                self.config_edits[key].setText(value)
                self.model_configs[key] = value
    
    def clear_config(self):
        """清空配置"""
        for edit in self.config_edits.values():
            edit.clear()
        self.model_configs.clear()
    
    def validate_paths(self) -> Dict[str, bool]:
        """验证路径有效性"""
        validation_result = {}
        for key, edit in self.config_edits.items():
            path = edit.text()
            validation_result[key] = os.path.exists(path) if path else True
        return validation_result 