from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QGroupBox, QListWidget, QListWidgetItem,
                           QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import os
import json


class ParamsComparisonWidget(QWidget):
    """训练参数对比组件，负责加载和比较不同模型的训练参数"""
    
    status_updated = pyqtSignal(str)
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.model_dir = ""
        self.model_configs = []
        
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 模型目录选择部分
        dir_layout = QHBoxLayout()
        self.model_dir_label = QLabel("参数目录:")
        self.model_dir_edit = QLabel()
        self.model_dir_edit.setStyleSheet("QLabel { border: 1px solid gray; padding: 5px; }")
        self.model_dir_edit.setText("请选择包含参数配置的目录")
        
        self.model_dir_button = QPushButton("浏览...")
        self.model_dir_button.clicked.connect(self.browse_param_dir)
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.clicked.connect(self.load_model_configs)
        
        dir_layout.addWidget(self.model_dir_label)
        dir_layout.addWidget(self.model_dir_edit)
        dir_layout.addWidget(self.model_dir_button)
        dir_layout.addWidget(self.refresh_button)
        
        layout.addLayout(dir_layout)
        
        # 模型列表和参数表格布局
        content_layout = QHBoxLayout()
        
        # 左侧模型列表
        model_group = QGroupBox("参数列表")
        model_layout = QVBoxLayout()
        
        # 修改变量名，避免与模型评估界面的model_list冲突
        self.params_model_list = QListWidget()
        self.params_model_list.setSelectionMode(QListWidget.MultiSelection)
        
        model_buttons = QHBoxLayout()
        self.select_all_button = QPushButton("全选")
        self.select_all_button.clicked.connect(self.select_all_models)
        self.deselect_all_button = QPushButton("取消全选")
        self.deselect_all_button.clicked.connect(self.deselect_all_models)
        self.compare_button = QPushButton("参数对比")
        self.compare_button.clicked.connect(self.compare_params)
        
        model_buttons.addWidget(self.select_all_button)
        model_buttons.addWidget(self.deselect_all_button)
        model_buttons.addWidget(self.compare_button)
        
        model_layout.addWidget(self.params_model_list)
        model_layout.addLayout(model_buttons)
        
        model_group.setLayout(model_layout)
        
        # 右侧参数表格
        params_group = QGroupBox("参数对比")
        params_layout = QVBoxLayout()
        
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(1)  # 初始只有参数名列
        self.params_table.setHorizontalHeaderLabels(["参数名"])
        self.params_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.params_table.verticalHeader().setVisible(False)
        
        params_layout.addWidget(self.params_table)
        
        params_group.setLayout(params_layout)
        
        # 添加到主布局
        content_layout.addWidget(model_group, 1)
        content_layout.addWidget(params_group, 3)
        
        layout.addLayout(content_layout)
        
    def browse_param_dir(self):
        """浏览参数目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择参数目录")
        if dir_path:
            # 标准化路径格式
            dir_path = os.path.normpath(dir_path)
            self.model_dir = dir_path
            self.model_dir_edit.setText(dir_path)
            self.load_model_configs()
            
            # 如果有主窗口并且有设置标签页，则更新设置
            if self.main_window and hasattr(self.main_window, 'settings_tab'):
                self.main_window.settings_tab.default_param_save_dir_edit.setText(dir_path)
            
    def load_model_configs(self):
        """加载模型配置文件"""
        self.params_model_list.clear()
        self.model_configs = []
        
        if not self.model_dir or not os.path.exists(self.model_dir):
            self.status_updated.emit("参数目录不存在")
            return
            
        # 显示加载状态
        self.status_updated.emit(f"正在加载参数文件...")
            
        # 查找模型目录下所有的配置文件
        config_files = []
        for file in os.listdir(self.model_dir):
            if file.endswith('_config.json'):
                config_files.append(file)
                
        if not config_files:
            self.status_updated.emit(f"在 {self.model_dir} 中未找到参数配置文件")
            return
            
        # 加载配置文件
        for config_file in config_files:
            try:
                # 检查文件是否是有效的参数配置，而不是其他模型文件
                file_path = os.path.join(self.model_dir, config_file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        
                        # 验证这是一个训练参数配置文件，而不是其他类型的JSON文件
                        if 'model_name' not in config or 'task_type' not in config:
                            self.status_updated.emit(f"跳过非参数配置文件: {config_file}")
                            continue
                            
                        self.model_configs.append({
                            'filename': config_file,
                            'config': config
                        })
                        
                        # 创建显示项目
                        model_name = config.get('model_name', 'Unknown')
                        model_note = config.get('model_note', '')
                        task_type = config.get('task_type', 'Unknown')
                        timestamp = config.get('timestamp', '')
                        
                        # 显示名称格式：模型名称 - 任务类型 - 时间戳 (备注)
                        display_name = f"{model_name} - {task_type}"
                        if timestamp:
                            display_name += f" - {timestamp}"
                        if model_note:
                            display_name += f" ({model_note})"
                            
                        item = QListWidgetItem(display_name)
                        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                        item.setCheckState(Qt.Unchecked)
                        self.params_model_list.addItem(item)
                except json.JSONDecodeError:
                    self.status_updated.emit(f"跳过无效的JSON文件: {config_file}")
                    continue
            except Exception as e:
                self.status_updated.emit(f"加载配置文件 {config_file} 时出错: {str(e)}")
                
        self.status_updated.emit(f"已加载 {len(self.model_configs)} 个参数配置")
        
    def select_all_models(self):
        """选择所有模型"""
        for i in range(self.params_model_list.count()):
            item = self.params_model_list.item(i)
            item.setCheckState(Qt.Checked)
            
    def deselect_all_models(self):
        """取消选择所有模型"""
        for i in range(self.params_model_list.count()):
            item = self.params_model_list.item(i)
            item.setCheckState(Qt.Unchecked)
            
    def compare_params(self):
        """比较所选模型的训练参数"""
        selected_indices = []
        
        # 获取所有选中的模型索引
        for i in range(self.params_model_list.count()):
            item = self.params_model_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_indices.append(i)
                
        if not selected_indices:
            QMessageBox.warning(self, "警告", "请至少选择一个模型进行比较")
            return
            
        # 准备表格显示
        selected_configs = [self.model_configs[i] for i in selected_indices]
        self.update_params_table(selected_configs)
        
    def update_params_table(self, selected_configs):
        """更新参数对比表格"""
        if not selected_configs:
            return
            
        # 清空表格
        self.params_table.clear()
        
        # 设置列数为参数名+每个模型一列
        self.params_table.setColumnCount(1 + len(selected_configs))
        
        # 设置列标题
        headers = ["参数名"]
        for config in selected_configs:
            model_name = config['config'].get('model_name', 'Unknown')
            model_note = config['config'].get('model_note', '')
            if model_note:
                headers.append(f"{model_name} ({model_note})")
            else:
                headers.append(model_name)
                
        self.params_table.setHorizontalHeaderLabels(headers)
        
        # 收集所有可能的参数名称
        all_params = set()
        for config in selected_configs:
            all_params.update(config['config'].keys())
            
        # 排序参数名称，使得重要参数排在前面
        important_params = [
            # 基础训练参数
            'task_type', 'model_name', 'model_note', 'data_dir', 
            'num_epochs', 'batch_size', 'learning_rate', 'optimizer',
            'dropout_rate', 'weight_decay', 'activation_function',
            'use_pretrained', 'pretrained_path', 'metrics', 'use_tensorboard',
            
            # 优化器高级参数（阶段一新增）
            'beta1', 'beta2', 'eps', 'momentum', 'nesterov',
            
            # 学习率预热参数（阶段一新增）
            'warmup_steps', 'warmup_ratio', 'warmup_method',
            
            # 高级学习率调度参数（阶段一新增）
            'min_lr', 'step_size', 'gamma', 'T_max',
            
            # 标签平滑参数（阶段一新增）
            'label_smoothing',
            
            # 第二阶段高级超参数
            'model_ema', 'model_ema_decay',
            'gradient_accumulation_steps',
            'cutmix_prob', 'mixup_alpha',
            'loss_scale', 'static_loss_scale',
            
            # 检测任务特定参数
            'iou_threshold', 'conf_threshold', 'resolution', 'nms_threshold', 'use_fpn'
        ]
        
        sorted_params = []
        # 先添加重要参数（如果存在）
        for param in important_params:
            if param in all_params:
                sorted_params.append(param)
                all_params.remove(param)
        
        # 再添加剩余参数（按字母排序）
        sorted_params.extend(sorted(all_params))
        
        # 设置行数
        self.params_table.setRowCount(len(sorted_params))
        
        # 填充表格内容
        for row, param in enumerate(sorted_params):
            # 设置参数名
            self.params_table.setItem(row, 0, QTableWidgetItem(param))
            
            # 设置每个模型的参数值
            for col, config in enumerate(selected_configs, start=1):
                value = config['config'].get(param, '')
                
                # 格式化值，使其更易读
                if isinstance(value, bool):
                    value = "是" if value else "否"
                elif isinstance(value, list):
                    value = ", ".join(map(str, value))
                elif isinstance(value, dict):
                    value = json.dumps(value, ensure_ascii=False)
                    
                self.params_table.setItem(row, col, QTableWidgetItem(str(value)))
                
        # 调整表格列宽
        self.params_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for i in range(1, self.params_table.columnCount()):
            self.params_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
    
    def apply_config(self, config):
        """应用配置"""
        if not config:
            return
            
        # 设置参数目录
        if 'default_param_save_dir' in config:
            param_dir = config['default_param_save_dir']
            if os.path.exists(param_dir):
                # 同时也应用到参数对比界面的参数目录
                self.model_dir = param_dir
                self.model_dir_edit.setText(param_dir)
                self.load_model_configs()
    
    def showEvent(self, event):
        """当组件显示时刷新参数列表"""
        super().showEvent(event)
        # 如果目录已设置，刷新参数列表
        if self.model_dir:
            self.load_model_configs() 