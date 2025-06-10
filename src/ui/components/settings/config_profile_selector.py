"""
配置文件选择器组件 - 用于切换不同的配置文件
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, 
                           QLabel, QPushButton, QMessageBox, QGroupBox, QFrame)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont


class ConfigProfileSelector(QWidget):
    """配置文件选择器组件"""
    
    # 信号定义
    profile_changed = pyqtSignal(str, dict)  # (profile_name, config_data)
    profile_loaded = pyqtSignal(dict)        # 配置文件加载完成
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 配置文件目录
        self.config_dir = "setting"
        
        # 当前选中的配置文件
        self.current_profile = None
        self.current_config = {}
        
        # 配置文件列表
        self.profile_list = []
        
        # 初始化UI
        self.init_ui()
        
        # 加载配置文件列表
        self.refresh_profile_list()
    
    def init_ui(self):
        """初始化用户界面"""
        # 创建主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)
        
        # 创建组框
        group_box = QGroupBox("配置文件管理")
        group_box.setFont(QFont('微软雅黑', 10, QFont.Bold))
        group_layout = QVBoxLayout(group_box)
        
        # 创建选择器布局
        selector_layout = QHBoxLayout()
        
        # 配置文件标签
        profile_label = QLabel("当前配置:")
        profile_label.setMinimumWidth(80)
        selector_layout.addWidget(profile_label)
        
        # 配置文件下拉框
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(300)
        self.profile_combo.currentTextChanged.connect(self.on_profile_selected)
        selector_layout.addWidget(self.profile_combo)
        
        # 刷新按钮
        refresh_btn = QPushButton("刷新")
        refresh_btn.setMaximumWidth(60)
        refresh_btn.clicked.connect(self.refresh_profile_list)
        selector_layout.addWidget(refresh_btn)
        
        group_layout.addLayout(selector_layout)
        
        # 创建信息显示区域
        info_layout = QVBoxLayout()
        
        # 配置文件信息标签
        self.profile_info_label = QLabel("请选择配置文件")
        self.profile_info_label.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 11px;
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                margin: 2px 0px;
            }
        """)
        self.profile_info_label.setWordWrap(True)
        self.profile_info_label.setMinimumHeight(60)
        info_layout.addWidget(self.profile_info_label)
        
        group_layout.addLayout(info_layout)
        
        # 创建操作按钮布局
        button_layout = QHBoxLayout()
        
        # 应用配置按钮
        self.apply_btn = QPushButton("应用配置")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.apply_current_profile)
        button_layout.addWidget(self.apply_btn)
        
        # 添加弹性空间
        button_layout.addStretch()
        
        group_layout.addLayout(button_layout)
        
        layout.addWidget(group_box)
        
        # 添加分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
    
    def refresh_profile_list(self):
        """刷新配置文件列表"""
        try:
            self.profile_list = []
            self.profile_combo.clear()
            
            # 检查配置目录是否存在
            if not os.path.exists(self.config_dir):
                self.profile_info_label.setText("配置目录不存在")
                return
            
            # 扫描配置文件
            for filename in os.listdir(self.config_dir):
                if filename.endswith('.json') and 'config' in filename.lower():
                    file_path = os.path.join(self.config_dir, filename)
                    try:
                        # 验证JSON文件有效性
                        with open(file_path, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                            
                        # 提取配置信息
                        profile_info = self.extract_profile_info(filename, config_data)
                        self.profile_list.append(profile_info)
                        
                    except Exception as e:
                        print(f"读取配置文件失败 {filename}: {str(e)}")
                        continue
            
            # 按文件名排序
            self.profile_list.sort(key=lambda x: x['filename'])
            
            # 填充下拉框
            self.profile_combo.addItem("-- 请选择配置文件 --")
            for profile in self.profile_list:
                display_name = f"{profile['display_name']} ({profile['filename']})"
                self.profile_combo.addItem(display_name)
            
            self.profile_info_label.setText(f"发现 {len(self.profile_list)} 个配置文件")
            
        except Exception as e:
            self.profile_info_label.setText(f"刷新配置文件列表失败: {str(e)}")
            QMessageBox.warning(self, "错误", f"刷新配置文件列表失败:\n{str(e)}")
    
    def extract_profile_info(self, filename: str, config_data: dict) -> dict:
        """提取配置文件信息"""
        profile_info = {
            'filename': filename,
            'file_path': os.path.join(self.config_dir, filename),
            'display_name': filename.replace('.json', ''),
            'description': '',
            'version': '',
            'created_time': '',
            'config_data': config_data
        }
        
        # 尝试从元数据中提取信息
        if 'metadata' in config_data:
            metadata = config_data['metadata']
            profile_info['description'] = metadata.get('description', '')
            profile_info['version'] = metadata.get('version', '')
            profile_info['created_time'] = metadata.get('export_time', '')
            
            # 如果有更友好的名称，使用它
            if 'name' in metadata:
                profile_info['display_name'] = metadata['name']
        
        return profile_info
    
    def on_profile_selected(self, selected_text: str):
        """处理配置文件选择事件"""
        if selected_text == "-- 请选择配置文件 --" or not selected_text:
            self.current_profile = None
            self.current_config = {}
            self.apply_btn.setEnabled(False)
            self.profile_info_label.setText("请选择配置文件")
            return
        
        try:
            # 从选择的文本中解析文件名
            filename = selected_text.split('(')[-1].rstrip(')')
            
            # 查找对应的配置文件信息
            selected_profile = None
            for profile in self.profile_list:
                if profile['filename'] == filename:
                    selected_profile = profile
                    break
            
            if not selected_profile:
                raise ValueError(f"未找到配置文件: {filename}")
            
            # 加载配置文件
            self.load_profile(selected_profile)
            
        except Exception as e:
            self.profile_info_label.setText(f"加载配置文件失败: {str(e)}")
            QMessageBox.warning(self, "错误", f"加载配置文件失败:\n{str(e)}")
    
    def load_profile(self, profile_info: dict):
        """加载指定的配置文件"""
        try:
            # 重新读取文件以确保数据最新
            with open(profile_info['file_path'], 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            self.current_profile = profile_info['filename']
            self.current_config = config_data
            
            # 更新信息显示
            self.update_profile_info_display(profile_info, config_data)
            
            # 启用应用按钮
            self.apply_btn.setEnabled(True)
            
            # 发出配置文件改变信号
            self.profile_changed.emit(self.current_profile, self.current_config)
            
        except Exception as e:
            raise Exception(f"读取配置文件失败: {str(e)}")
    
    def update_profile_info_display(self, profile_info: dict, config_data: dict):
        """更新配置文件信息显示"""
        info_lines = [
            f"配置文件: {profile_info['display_name']}",
            f"文件名: {profile_info['filename']}"
        ]
        
        if profile_info['description']:
            info_lines.append(f"描述: {profile_info['description']}")
        
        if profile_info['version']:
            info_lines.append(f"版本: {profile_info['version']}")
        
        if profile_info['created_time']:
            info_lines.append(f"创建时间: {profile_info['created_time']}")
        
        # 显示配置摘要
        if 'config' in config_data:
            config = config_data['config']
            classes = config.get('default_classes', [])
            if classes:
                info_lines.append(f"包含类别: {len(classes)} 个")
        
        self.profile_info_label.setText("\n".join(info_lines))
    
    def apply_current_profile(self):
        """应用当前选择的配置文件"""
        if not self.current_profile or not self.current_config:
            QMessageBox.warning(self, "警告", "请先选择配置文件")
            return
        
        try:
            # 发出配置加载信号
            self.profile_loaded.emit(self.current_config)
            
            QMessageBox.information(self, "成功", f"已应用配置文件: {self.current_profile}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"应用配置文件失败:\n{str(e)}")
    
    def get_current_profile(self) -> Tuple[Optional[str], dict]:
        """获取当前选择的配置文件"""
        return self.current_profile, self.current_config
    
    def set_current_profile(self, filename: str):
        """设置当前配置文件"""
        for i, profile in enumerate(self.profile_list):
            if profile['filename'] == filename:
                # 设置下拉框选择
                display_name = f"{profile['display_name']} ({profile['filename']})"
                index = self.profile_combo.findText(display_name)
                if index >= 0:
                    self.profile_combo.setCurrentIndex(index)
                return True
        return False
    
    def get_available_profiles(self) -> List[dict]:
        """获取所有可用的配置文件列表"""
        return self.profile_list.copy() 