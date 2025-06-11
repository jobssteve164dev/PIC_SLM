"""
训练配置选择器组件

该组件用于读取train_config文件夹中的训练配置文件，
提供下拉框选择界面，并应用选择的配置参数
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QComboBox, QPushButton, QGroupBox, QMessageBox,
                           QTextEdit, QSplitter, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import os
import json
import glob
from datetime import datetime


class TrainingConfigSelector(QWidget):
    """训练配置选择器组件"""
    
    # 定义信号
    config_selected = pyqtSignal(dict)  # 配置被选择时发出，携带配置字典
    config_applied = pyqtSignal(dict)   # 配置被应用时发出，携带配置字典
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config_folder = ""
        self.available_configs = {}
        self.current_config = None
        self.init_ui()
        self.refresh_configs()
    
    def init_ui(self):
        """初始化用户界面"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 创建配置选择组
        config_group = QGroupBox("训练参数配置选择")
        config_layout = QVBoxLayout()
        
        # 配置选择行
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("选择配置:"))
        
        self.config_combo = QComboBox()
        self.config_combo.setMinimumWidth(300)
        self.config_combo.setToolTip("选择已保存的训练配置文件")
        self.config_combo.currentTextChanged.connect(self.on_config_changed)
        select_layout.addWidget(self.config_combo)
        
        # 刷新按钮
        refresh_btn = QPushButton("刷新")
        refresh_btn.setFixedWidth(60)
        refresh_btn.setToolTip("重新扫描train_config文件夹，刷新配置列表")
        refresh_btn.clicked.connect(self.refresh_configs)
        select_layout.addWidget(refresh_btn)
        
        # 应用按钮
        apply_btn = QPushButton("应用配置")
        apply_btn.setFixedWidth(80)
        apply_btn.setToolTip("将选择的配置应用到训练界面")
        apply_btn.clicked.connect(self.apply_current_config)
        select_layout.addWidget(apply_btn)
        
        select_layout.addStretch()
        config_layout.addLayout(select_layout)
        
        # 创建分割器用于显示配置详情
        splitter = QSplitter(Qt.Horizontal)
        
        # 配置预览区域
        preview_frame = QFrame()
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(5, 5, 5, 5)
        
        preview_label = QLabel("配置预览:")
        preview_label.setFont(QFont('微软雅黑', 9, QFont.Bold))
        preview_layout.addWidget(preview_label)
        
        self.config_preview = QTextEdit()
        self.config_preview.setReadOnly(True)
        self.config_preview.setMaximumHeight(200)
        self.config_preview.setPlaceholderText("选择配置后将显示详细参数...")
        self.config_preview.setToolTip("显示选择配置的详细参数信息")
        preview_layout.addWidget(self.config_preview)
        
        # 配置信息区域
        info_frame = QFrame()
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(5, 5, 5, 5)
        
        info_label = QLabel("配置信息:")
        info_label.setFont(QFont('微软雅黑', 9, QFont.Bold))
        info_layout.addWidget(info_label)
        
        self.config_info = QTextEdit()
        self.config_info.setReadOnly(True)
        self.config_info.setMaximumHeight(200)
        self.config_info.setPlaceholderText("选择配置后将显示基本信息...")
        self.config_info.setToolTip("显示配置的基本信息，如模型名称、创建时间等")
        info_layout.addWidget(self.config_info)
        
        splitter.addWidget(preview_frame)
        splitter.addWidget(info_frame)
        splitter.setSizes([350, 250])
        
        config_layout.addWidget(splitter)
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #666; font-size: 12px;")
        main_layout.addWidget(self.status_label)
    
    def refresh_configs(self):
        """刷新配置列表"""
        try:
            # 获取train_config文件夹路径
            self.config_folder = self.get_config_folder_path()
            
            if not os.path.exists(self.config_folder):
                self.status_label.setText(f"配置文件夹不存在: {self.config_folder}")
                self.config_combo.clear()
                return
            
            # 清空现有配置
            self.available_configs.clear()
            self.config_combo.clear()
            
            # 搜索所有JSON配置文件
            config_pattern = os.path.join(self.config_folder, "*_config.json")
            config_files = glob.glob(config_pattern)
            
            if not config_files:
                self.status_label.setText("未找到训练配置文件")
                self.config_combo.addItem("-- 无可用配置 --")
                return
            
            # 按修改时间排序（最新的在前面）
            config_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # 添加默认选项
            self.config_combo.addItem("-- 请选择配置 --")
            
            # 读取配置文件
            for config_file in config_files:
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    
                    # 生成显示名称
                    filename = os.path.basename(config_file)
                    display_name = self.generate_display_name(config_data, filename)
                    
                    # 存储配置
                    self.available_configs[display_name] = {
                        'file_path': config_file,
                        'data': config_data,
                        'filename': filename
                    }
                    
                    # 添加到下拉框
                    self.config_combo.addItem(display_name)
                    
                except Exception as e:
                    print(f"读取配置文件失败 {config_file}: {str(e)}")
                    continue
            
            # 更新状态
            config_count = len(self.available_configs)
            self.status_label.setText(f"找到 {config_count} 个配置文件")
            
        except Exception as e:
            error_msg = f"刷新配置列表失败: {str(e)}"
            self.status_label.setText(error_msg)
            QMessageBox.warning(self, "错误", error_msg)
    
    def get_config_folder_path(self):
        """获取配置文件夹路径"""
        # 从当前文件位置推算train_config文件夹路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
        return os.path.join(project_root, 'train_config')
    
    def generate_display_name(self, config_data, filename):
        """生成配置显示名称"""
        try:
            # 提取基本信息
            model_name = config_data.get('model_name', 'Unknown')
            timestamp = config_data.get('timestamp', '')
            batch_size = config_data.get('batch_size', '')
            learning_rate = config_data.get('learning_rate', '')
            epochs = config_data.get('num_epochs', '')
            
            # 格式化时间戳
            if timestamp:
                try:
                    # 尝试解析时间戳格式
                    if '-' in timestamp and len(timestamp) >= 13:
                        # 格式如：20250611-072935
                        date_part = timestamp[:8]
                        time_part = timestamp[9:15]
                        formatted_time = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                    else:
                        formatted_time = timestamp
                except:
                    formatted_time = timestamp
            else:
                formatted_time = "未知时间"
            
            # 生成显示名称
            display_parts = []
            display_parts.append(f"[{model_name}]")
            
            if batch_size:
                display_parts.append(f"批次:{batch_size}")
            if learning_rate:
                display_parts.append(f"学习率:{learning_rate}")
            if epochs:
                display_parts.append(f"轮数:{epochs}")
                
            display_parts.append(f"({formatted_time})")
            
            return " ".join(display_parts)
            
        except Exception as e:
            # 如果生成显示名称失败，使用文件名
            return filename.replace('_config.json', '')
    
    def on_config_changed(self, config_name):
        """配置选择改变时的处理"""
        if config_name == "-- 请选择配置 --" or config_name == "-- 无可用配置 --":
            self.current_config = None
            self.config_preview.clear()
            self.config_info.clear()
            return
        
        if config_name not in self.available_configs:
            return
        
        # 获取选择的配置
        config_info = self.available_configs[config_name]
        self.current_config = config_info['data']
        
        # 更新预览
        self.update_config_preview()
        self.update_config_info(config_info)
        
        # 发出信号
        self.config_selected.emit(self.current_config)
    
    def update_config_preview(self):
        """更新配置预览"""
        if not self.current_config:
            return
        
        try:
            # 格式化JSON显示
            preview_text = json.dumps(self.current_config, indent=2, ensure_ascii=False)
            self.config_preview.setPlainText(preview_text)
        except Exception as e:
            self.config_preview.setPlainText(f"预览失败: {str(e)}")
    
    def update_config_info(self, config_info):
        """更新配置信息"""
        if not config_info:
            return
        
        try:
            config_data = config_info['data']
            
            # 生成信息文本
            info_lines = []
            info_lines.append(f"文件名: {config_info['filename']}")
            info_lines.append(f"文件路径: {config_info['file_path']}")
            info_lines.append("")
            
            # 基本训练参数
            info_lines.append("=== 基本参数 ===")
            info_lines.append(f"模型: {config_data.get('model_name', '未设置')}")
            info_lines.append(f"任务类型: {config_data.get('task_type', '未设置')}")
            info_lines.append(f"训练轮数: {config_data.get('num_epochs', '未设置')}")
            info_lines.append(f"批次大小: {config_data.get('batch_size', '未设置')}")
            info_lines.append(f"学习率: {config_data.get('learning_rate', '未设置')}")
            info_lines.append(f"优化器: {config_data.get('optimizer', '未设置')}")
            info_lines.append("")
            
            # 高级参数
            info_lines.append("=== 高级参数 ===")
            info_lines.append(f"权重衰减: {config_data.get('weight_decay', '未设置')}")
            info_lines.append(f"学习率调度: {config_data.get('lr_scheduler', '未设置')}")
            info_lines.append(f"数据增强: {config_data.get('use_augmentation', '未设置')}")
            info_lines.append(f"早停策略: {config_data.get('early_stopping', '未设置')}")
            info_lines.append(f"混合精度: {config_data.get('mixed_precision', '未设置')}")
            info_lines.append(f"使用预训练: {config_data.get('use_pretrained', '未设置')}")
            
            # 路径信息
            if config_data.get('data_dir'):
                info_lines.append("")
                info_lines.append("=== 路径配置 ===")
                info_lines.append(f"数据目录: {config_data.get('data_dir', '未设置')}")
                info_lines.append(f"模型保存: {config_data.get('model_save_dir', '未设置')}")
            
            # 时间信息
            if config_data.get('timestamp'):
                info_lines.append("")
                info_lines.append("=== 时间信息 ===")
                info_lines.append(f"创建时间: {config_data.get('timestamp', '未设置')}")
            
            self.config_info.setPlainText("\n".join(info_lines))
            
        except Exception as e:
            self.config_info.setPlainText(f"信息显示失败: {str(e)}")
    
    def apply_current_config(self):
        """应用当前选择的配置"""
        if not self.current_config:
            QMessageBox.information(self, "提示", "请先选择一个配置！")
            return
        
        try:
            # 确认对话框
            reply = QMessageBox.question(
                self, "确认应用", 
                "确定要应用选择的训练配置吗？\n这将覆盖当前的训练参数设置。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # 发出应用配置信号
                self.config_applied.emit(self.current_config)
                self.status_label.setText("配置已应用")
                QMessageBox.information(self, "成功", "训练配置已成功应用！")
            
        except Exception as e:
            error_msg = f"应用配置失败: {str(e)}"
            self.status_label.setText(error_msg)
            QMessageBox.critical(self, "错误", error_msg)
    
    def get_current_config(self):
        """获取当前选择的配置"""
        return self.current_config
    
    def get_config_count(self):
        """获取可用配置数量"""
        return len(self.available_configs) 