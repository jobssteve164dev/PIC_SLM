from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QComboBox, QSpinBox, QGroupBox, QGridLayout,
                           QSizePolicy, QLineEdit, QCheckBox, QListWidget, QInputDialog,
                           QMessageBox, QTabWidget, QScrollArea, QTableWidget, QTableWidgetItem,
                           QDoubleSpinBox, QSpacerItem, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
import os
import json
import time
from .base_tab import BaseTab

class SettingsTab(BaseTab):
    """设置标签页，负责应用设置管理"""
    
    # 定义信号
    settings_saved = pyqtSignal(dict)
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent, main_window)
        self.config = {}
        self.default_classes = []
        self.class_weights = {}  # 添加类别权重字典
        self.init_ui()
        
        # BaseTab已经连接了标签页切换信号，这里不需要重复连接
        
        # 添加特殊的延迟重建布局定时器
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.timeout.connect(self._fix_layout)
        
    def init_ui(self):
        """初始化UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self.scroll_content)
        # 确保没有多余的边距造成空白
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 添加标题
        title_label = QLabel("应用设置")
        title_label.setFont(QFont('微软雅黑', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 创建设置选项卡
        settings_tabs = QTabWidget()
        settings_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 创建常规设置选项卡
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)
        general_layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建默认文件夹组
        folders_group = QGroupBox("默认文件夹")
        folders_layout = QGridLayout()
        folders_layout.setContentsMargins(10, 20, 10, 10)
        
        # 源文件夹
        self.default_source_edit = QLineEdit()
        self.default_source_edit.setReadOnly(True)
        self.default_source_edit.setPlaceholderText("默认源图片文件夹")
        
        source_btn = QPushButton("浏览...")
        source_btn.clicked.connect(self.select_default_source_folder)
        
        folders_layout.addWidget(QLabel("默认源文件夹:"), 0, 0)
        folders_layout.addWidget(self.default_source_edit, 0, 1)
        folders_layout.addWidget(source_btn, 0, 2)
        
        # 输出文件夹
        self.default_output_edit = QLineEdit()
        self.default_output_edit.setReadOnly(True)
        self.default_output_edit.setPlaceholderText("默认输出文件夹")
        
        output_btn = QPushButton("浏览...")
        output_btn.clicked.connect(self.select_default_output_folder)
        
        folders_layout.addWidget(QLabel("默认输出文件夹:"), 1, 0)
        folders_layout.addWidget(self.default_output_edit, 1, 1)
        folders_layout.addWidget(output_btn, 1, 2)
        
        folders_group.setLayout(folders_layout)
        general_layout.addWidget(folders_group)
        
        # 创建默认类别组
        classes_group = QGroupBox("默认缺陷类别与权重配置")
        classes_layout = QVBoxLayout()
        classes_layout.setContentsMargins(10, 20, 10, 10)
        
        # 添加权重策略选择
        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(QLabel("权重策略:"))
        
        self.weight_strategy_combo = QComboBox()
        self.weight_strategy_combo.addItems([
            "balanced (平衡权重)",
            "inverse (逆频率权重)", 
            "log_inverse (对数逆频率权重)",
            "custom (自定义权重)",
            "none (无权重)"
        ])
        self.weight_strategy_combo.setCurrentText("balanced (平衡权重)")
        self.weight_strategy_combo.currentTextChanged.connect(self.on_weight_strategy_changed)
        strategy_layout.addWidget(self.weight_strategy_combo)
        strategy_layout.addStretch()
        
        classes_layout.addLayout(strategy_layout)
        
        # 添加说明标签
        info_label = QLabel("说明: balanced自动平衡权重, inverse逆频率权重, log_inverse对数逆频率权重, custom使用自定义权重, none不使用权重")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 11px;")
        classes_layout.addWidget(info_label)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        classes_layout.addWidget(line)
        
        # 添加类别权重表格
        self.class_weight_table = QTableWidget()
        self.class_weight_table.setColumnCount(2)
        self.class_weight_table.setHorizontalHeaderLabels(["类别名称", "权重值"])
        self.class_weight_table.horizontalHeader().setStretchLastSection(True)
        self.class_weight_table.setMinimumHeight(200)
        self.class_weight_table.setAlternatingRowColors(True)
        classes_layout.addWidget(self.class_weight_table)
        
        # 添加按钮组
        btn_layout = QHBoxLayout()
        
        add_class_btn = QPushButton("添加类别")
        add_class_btn.clicked.connect(self.settings_add_defect_class)
        btn_layout.addWidget(add_class_btn)
        
        remove_class_btn = QPushButton("删除类别")
        remove_class_btn.clicked.connect(self.settings_remove_defect_class)
        btn_layout.addWidget(remove_class_btn)
        
        # 添加重置权重按钮
        reset_weights_btn = QPushButton("重置权重")
        reset_weights_btn.clicked.connect(self.reset_class_weights)
        btn_layout.addWidget(reset_weights_btn)
        
        btn_layout.addSpacerItem(QSpacerItem(20, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        # 添加保存到文件按钮
        save_to_file_btn = QPushButton("保存到文件")
        save_to_file_btn.clicked.connect(self.save_classes_to_file)
        btn_layout.addWidget(save_to_file_btn)
        
        # 添加从文件加载按钮
        load_from_file_btn = QPushButton("从文件加载")
        load_from_file_btn.clicked.connect(self.load_classes_from_file)
        btn_layout.addWidget(load_from_file_btn)
        
        classes_layout.addLayout(btn_layout)
        classes_group.setLayout(classes_layout)
        general_layout.addWidget(classes_group)
        
        # 添加常规设置选项卡
        settings_tabs.addTab(general_tab, "常规设置")
        
        # 创建高级设置选项卡
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        advanced_layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建模型文件组
        model_group = QGroupBox("默认模型文件")
        model_layout = QGridLayout()
        model_layout.setContentsMargins(10, 20, 10, 10)
        
        # 模型文件
        self.default_model_edit = QLineEdit()
        self.default_model_edit.setReadOnly(True)
        self.default_model_edit.setPlaceholderText("默认模型文件")
        
        model_btn = QPushButton("浏览...")
        model_btn.clicked.connect(self.select_default_model_file)
        
        model_layout.addWidget(QLabel("默认模型文件:"), 0, 0)
        model_layout.addWidget(self.default_model_edit, 0, 1)
        model_layout.addWidget(model_btn, 0, 2)
        
        # 类别信息文件
        self.default_class_info_edit = QLineEdit()
        self.default_class_info_edit.setReadOnly(True)
        self.default_class_info_edit.setPlaceholderText("默认类别信息文件")
        
        class_info_btn = QPushButton("浏览...")
        class_info_btn.clicked.connect(self.select_default_class_info_file)
        
        model_layout.addWidget(QLabel("默认类别信息:"), 1, 0)
        model_layout.addWidget(self.default_class_info_edit, 1, 1)
        model_layout.addWidget(class_info_btn, 1, 2)
        
        # 模型评估文件夹
        self.default_model_eval_dir_edit = QLineEdit()
        self.default_model_eval_dir_edit.setReadOnly(True)
        self.default_model_eval_dir_edit.setPlaceholderText("默认模型评估文件夹")
        
        model_eval_dir_btn = QPushButton("浏览...")
        model_eval_dir_btn.clicked.connect(self.select_default_model_eval_dir)
        
        model_layout.addWidget(QLabel("默认模型评估文件夹:"), 2, 0)
        model_layout.addWidget(self.default_model_eval_dir_edit, 2, 1)
        model_layout.addWidget(model_eval_dir_btn, 2, 2)
        
        # 模型保存文件夹
        self.default_model_save_dir_edit = QLineEdit()
        self.default_model_save_dir_edit.setReadOnly(True)
        self.default_model_save_dir_edit.setPlaceholderText("默认模型保存文件夹")
        
        model_save_dir_btn = QPushButton("浏览...")
        model_save_dir_btn.clicked.connect(self.select_default_model_save_dir)
        
        model_layout.addWidget(QLabel("默认模型保存文件夹:"), 3, 0)
        model_layout.addWidget(self.default_model_save_dir_edit, 3, 1)
        model_layout.addWidget(model_save_dir_btn, 3, 2)
        
        # TensorBoard日志文件夹
        self.default_tensorboard_log_dir_edit = QLineEdit()
        self.default_tensorboard_log_dir_edit.setReadOnly(True)
        self.default_tensorboard_log_dir_edit.setPlaceholderText("默认TensorBoard日志文件夹")
        
        tensorboard_log_dir_btn = QPushButton("浏览...")
        tensorboard_log_dir_btn.clicked.connect(self.select_default_tensorboard_log_dir)
        
        model_layout.addWidget(QLabel("默认TensorBoard日志文件夹:"), 4, 0)
        model_layout.addWidget(self.default_tensorboard_log_dir_edit, 4, 1)
        model_layout.addWidget(tensorboard_log_dir_btn, 4, 2)
        
        # 数据集评估文件夹
        self.default_dataset_dir_edit = QLineEdit()
        self.default_dataset_dir_edit.setReadOnly(True)
        self.default_dataset_dir_edit.setPlaceholderText("默认数据集评估文件夹")
        
        dataset_dir_btn = QPushButton("浏览...")
        dataset_dir_btn.clicked.connect(self.select_default_dataset_dir)
        
        model_layout.addWidget(QLabel("默认数据集评估文件夹:"), 5, 0)
        model_layout.addWidget(self.default_dataset_dir_edit, 5, 1)
        model_layout.addWidget(dataset_dir_btn, 5, 2)
        
        # 训练参数保存文件夹
        self.default_param_save_dir_edit = QLineEdit()
        self.default_param_save_dir_edit.setReadOnly(True)
        self.default_param_save_dir_edit.setPlaceholderText("默认训练参数保存文件夹")
        
        param_save_dir_btn = QPushButton("浏览...")
        param_save_dir_btn.clicked.connect(self.select_default_param_save_dir)
        
        model_layout.addWidget(QLabel("默认训练参数保存文件夹:"), 6, 0)
        model_layout.addWidget(self.default_param_save_dir_edit, 6, 1)
        model_layout.addWidget(param_save_dir_btn, 6, 2)
        
        model_group.setLayout(model_layout)
        advanced_layout.addWidget(model_group)
        
        # 添加高级设置选项卡
        settings_tabs.addTab(advanced_tab, "高级设置")
        
        main_layout.addWidget(settings_tabs)
        
        # 添加按钮组
        button_layout = QHBoxLayout()
        
        # 添加保存设置按钮
        save_btn = QPushButton("保存设置")
        save_btn.clicked.connect(self.save_settings)
        save_btn.setMinimumHeight(40)
        button_layout.addWidget(save_btn)
        
        # 添加保存配置到文件按钮
        save_config_to_file_btn = QPushButton("保存配置到文件")
        save_config_to_file_btn.clicked.connect(self.save_config_to_file)
        save_config_to_file_btn.setMinimumHeight(40)
        button_layout.addWidget(save_config_to_file_btn)
        
        # 添加从文件加载配置按钮
        load_config_from_file_btn = QPushButton("从文件加载配置")
        load_config_from_file_btn.clicked.connect(self.load_config_from_file)
        load_config_from_file_btn.setMinimumHeight(40)
        button_layout.addWidget(load_config_from_file_btn)
        
        main_layout.addLayout(button_layout)
        
        # 添加弹性空间
        main_layout.addStretch(1)
        
        # 设置滚动内容大小策略
        self.scroll_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 设置滚动区域
        # 确保内容从顶部开始显示，而不是居中显示
        if hasattr(self, 'layout') and self.layout.count() > 0:
            scroll_area = self.layout.itemAt(0).widget()
            if isinstance(scroll_area, QScrollArea):
                # 设置边框和滚动条
                scroll_area.setFrameShape(QScrollArea.NoFrame)
                scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                # 滚动到顶部
                scroll_area.verticalScrollBar().setValue(0)
        
        # 加载当前设置
        self.load_current_settings()
    
    def select_default_source_folder(self):
        """选择默认源文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择默认源文件夹")
        if folder:
            # 标准化路径格式
            folder = os.path.normpath(folder)
            self.default_source_edit.setText(folder)
    
    def select_default_output_folder(self):
        """选择默认输出文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择默认输出文件夹")
        if folder:
            # 标准化路径格式
            folder = os.path.normpath(folder)
            self.default_output_edit.setText(folder)
    
    def settings_add_defect_class(self):
        """添加默认缺陷类别"""
        class_name, ok = QInputDialog.getText(self, "添加缺陷类别", "请输入缺陷类别名称:")
        if ok and class_name:
            # 检查是否已存在
            if class_name in self.default_classes:
                QMessageBox.warning(self, "警告", f"类别 '{class_name}' 已存在!")
                return
                
            self.default_classes.append(class_name)
            # 为新类别设置默认权重
            self.class_weights[class_name] = 1.0
            
            # 添加到表格
            row_count = self.class_weight_table.rowCount()
            self.class_weight_table.insertRow(row_count)
            self.class_weight_table.setItem(row_count, 0, QTableWidgetItem(class_name))
            
            # 创建权重输入框
            weight_spinbox = QDoubleSpinBox()
            weight_spinbox.setMinimum(0.01)
            weight_spinbox.setMaximum(100.0)
            weight_spinbox.setSingleStep(0.1)
            weight_spinbox.setDecimals(2)
            weight_spinbox.setValue(1.0)
            weight_spinbox.valueChanged.connect(lambda value, name=class_name: self.on_weight_changed(name, value))
            
            self.class_weight_table.setCellWidget(row_count, 1, weight_spinbox)
            self.update_weight_widgets_state()
    
    def settings_remove_defect_class(self):
        """删除默认缺陷类别"""
        current_row = self.class_weight_table.currentRow()
        if current_row >= 0:
            class_name_item = self.class_weight_table.item(current_row, 0)
            if class_name_item:
                class_name = class_name_item.text()
                
                # 从列表和权重字典中移除
                if class_name in self.default_classes:
                    self.default_classes.remove(class_name)
                if class_name in self.class_weights:
                    del self.class_weights[class_name]
                
                # 从表格中移除
                self.class_weight_table.removeRow(current_row)
    
    def on_weight_changed(self, class_name, value):
        """处理权重值变化"""
        self.class_weights[class_name] = value
    
    def on_weight_strategy_changed(self):
        """处理权重策略选择变化"""
        strategy = self.weight_strategy_combo.currentText()
        self.update_weight_widgets_state()
        
        # 如果选择了自定义权重策略，显示提示
        if "custom" in strategy.lower():
            QMessageBox.information(
                self, 
                "自定义权重", 
                "您选择了自定义权重策略。\n请在下表中设置每个类别的权重值。\n较高的权重值会让模型更关注该类别的样本。"
            )
    
    def update_weight_widgets_state(self):
        """根据权重策略更新权重输入框的状态"""
        strategy = self.weight_strategy_combo.currentText()
        is_custom = "custom" in strategy.lower()
        
        # 启用或禁用权重输入框
        for row in range(self.class_weight_table.rowCount()):
            weight_widget = self.class_weight_table.cellWidget(row, 1)
            if weight_widget:
                weight_widget.setEnabled(is_custom)
    
    def reset_class_weights(self):
        """重置类别权重"""
        reply = QMessageBox.question(
            self, 
            "重置权重", 
            "确定要重置所有类别权重为1.0吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 重置所有权重为1.0
            for class_name in self.default_classes:
                self.class_weights[class_name] = 1.0
            
            # 更新表格中的权重显示
            for row in range(self.class_weight_table.rowCount()):
                weight_widget = self.class_weight_table.cellWidget(row, 1)
                if weight_widget:
                    weight_widget.setValue(1.0)
                    
            QMessageBox.information(self, "完成", "已重置所有类别权重为1.0")
    
    def select_default_model_file(self):
        """选择默认模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择默认模型文件", "", "模型文件 (*.pth *.pt *.h5)")
        if file_path:
            # 标准化路径格式
            file_path = os.path.normpath(file_path)
            self.default_model_edit.setText(file_path)
    
    def select_default_class_info_file(self):
        """选择默认类别信息文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择默认类别信息文件", "", "JSON文件 (*.json)")
        if file_path:
            # 标准化路径格式
            file_path = os.path.normpath(file_path)
            self.default_class_info_edit.setText(file_path)
    
    def select_default_model_eval_dir(self):
        """选择默认模型评估文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择默认模型评估文件夹")
        if folder:
            # 标准化路径格式
            folder = os.path.normpath(folder)
            self.default_model_eval_dir_edit.setText(folder)
    
    def select_default_model_save_dir(self):
        """选择默认模型保存文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择默认模型保存文件夹")
        if folder:
            # 标准化路径格式
            folder = os.path.normpath(folder)
            self.default_model_save_dir_edit.setText(folder)
    
    def select_default_tensorboard_log_dir(self):
        """选择默认TensorBoard日志文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择默认TensorBoard日志文件夹")
        if folder:
            # 标准化路径格式
            folder = os.path.normpath(folder)
            self.default_tensorboard_log_dir_edit.setText(folder)
    
    def select_default_dataset_dir(self):
        """选择默认数据集评估文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择默认数据集评估文件夹")
        if folder:
            # 标准化路径格式
            folder = os.path.normpath(folder)
            self.default_dataset_dir_edit.setText(folder)
    
    def select_default_param_save_dir(self):
        """选择默认训练参数保存文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择默认训练参数保存文件夹")
        if folder:
            # 标准化路径格式
            folder = os.path.normpath(folder)
            self.default_param_save_dir_edit.setText(folder)
    
    def load_current_settings(self):
        """加载当前设置"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config.json')
            print(f"SettingsTab: 尝试从以下路径加载配置: {config_path}")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                    self.apply_config_to_ui()
                    print(f"SettingsTab: 已加载配置: {self.config}")
            else:
                print(f"SettingsTab: 配置文件不存在: {config_path}")
        except Exception as e:
            print(f"SettingsTab: 加载配置失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def apply_config_to_ui(self):
        """将配置应用到UI"""
        if not self.config:
            return
            
        # 设置默认文件夹
        self.default_source_edit.setText(self.config.get('default_source_folder', ''))
        self.default_output_edit.setText(self.config.get('default_output_folder', ''))
        
        # 设置默认模型文件和类别信息
        self.default_model_edit.setText(self.config.get('default_model_file', ''))
        self.default_class_info_edit.setText(self.config.get('default_class_info_file', ''))
        self.default_model_eval_dir_edit.setText(self.config.get('default_model_eval_dir', ''))
        self.default_model_save_dir_edit.setText(self.config.get('default_model_save_dir', ''))
        self.default_tensorboard_log_dir_edit.setText(self.config.get('default_tensorboard_log_dir', ''))
        self.default_dataset_dir_edit.setText(self.config.get('default_dataset_dir', ''))
        self.default_param_save_dir_edit.setText(self.config.get('default_param_save_dir', ''))
        
        # 设置默认类别和权重
        self.default_classes = self.config.get('default_classes', [])
        self.class_weights = self.config.get('class_weights', {})
        
        # 设置权重策略
        weight_strategy = self.config.get('weight_strategy', 'balanced')
        strategy_mapping = {
            'balanced': 'balanced (平衡权重)',
            'inverse': 'inverse (逆频率权重)',
            'log_inverse': 'log_inverse (对数逆频率权重)',
            'custom': 'custom (自定义权重)',
            'none': 'none (无权重)'
        }
        strategy_text = strategy_mapping.get(weight_strategy, 'balanced (平衡权重)')
        self.weight_strategy_combo.setCurrentText(strategy_text)
        
        # 清空并重新填充类别权重表格
        self.class_weight_table.setRowCount(0)
        
        for class_name in self.default_classes:
            row_count = self.class_weight_table.rowCount()
            self.class_weight_table.insertRow(row_count)
            self.class_weight_table.setItem(row_count, 0, QTableWidgetItem(class_name))
            
            # 创建权重输入框
            weight_value = self.class_weights.get(class_name, 1.0)
            weight_spinbox = QDoubleSpinBox()
            weight_spinbox.setMinimum(0.01)
            weight_spinbox.setMaximum(100.0)
            weight_spinbox.setSingleStep(0.1)
            weight_spinbox.setDecimals(2)
            weight_spinbox.setValue(weight_value)
            weight_spinbox.valueChanged.connect(lambda value, name=class_name: self.on_weight_changed(name, value))
            
            self.class_weight_table.setCellWidget(row_count, 1, weight_spinbox)
        
        # 更新权重输入框状态
        self.update_weight_widgets_state()
    
    def save_settings(self):
        """保存设置"""
        try:
            # 从表格收集最新的类别权重数据
            self.collect_weights_from_table()
            
            # 获取权重策略
            strategy_text = self.weight_strategy_combo.currentText()
            weight_strategy = strategy_text.split(' ')[0]  # 提取策略名称部分
            
            # 创建配置字典
            config = {
                'default_source_folder': self.default_source_edit.text(),
                'default_output_folder': self.default_output_edit.text(),
                'default_model_file': self.default_model_edit.text(),
                'default_class_info_file': self.default_class_info_edit.text(),
                'default_model_eval_dir': self.default_model_eval_dir_edit.text(),
                'default_model_save_dir': self.default_model_save_dir_edit.text(),
                'default_tensorboard_log_dir': self.default_tensorboard_log_dir_edit.text(),
                'default_dataset_dir': self.default_dataset_dir_edit.text(),
                'default_param_save_dir': self.default_param_save_dir_edit.text(),
                'default_classes': self.default_classes,
                'class_weights': self.class_weights,
                'weight_strategy': weight_strategy,
                'use_class_weights': weight_strategy != 'none'
            }
            
            # 保存配置到文件
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config.json')
            print(f"SettingsTab: 尝试保存配置到: {config_path}")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            
            self.config = config
            self.settings_saved.emit(config)
            QMessageBox.information(self, "成功", "设置已保存")
            
            print(f"SettingsTab: 已保存配置: {config}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存设置失败: {str(e)}")
            print(f"SettingsTab: 保存设置失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def collect_weights_from_table(self):
        """从表格收集权重数据"""
        self.class_weights.clear()
        self.default_classes.clear()
        
        for row in range(self.class_weight_table.rowCount()):
            class_name_item = self.class_weight_table.item(row, 0)
            if class_name_item:
                class_name = class_name_item.text()
                self.default_classes.append(class_name)
                
                # 获取权重值
                weight_widget = self.class_weight_table.cellWidget(row, 1)
                if weight_widget:
                    weight_value = weight_widget.value()
                    self.class_weights[class_name] = weight_value
    
    def on_tab_changed(self, index):
        """处理标签页切换事件，设置标签页需要特殊处理"""
        # 调用基类方法
        super().on_tab_changed(index)
        
        # 添加特殊处理：当切换到设置标签页时，尝试激活完全重建布局
        if self.main_window and hasattr(self.main_window, 'tabs'):
            current_widget = self.main_window.tabs.widget(index)
            if current_widget == self:
                print("切换到设置标签页，启动布局修复机制")
                # 使用定时器延迟启动我们的特殊布局修复
                self._rebuild_timer.start(250)
                
                # 使用多个定时器在不同时间点尝试修复，提高成功率
                QTimer.singleShot(350, self._fix_layout)
                QTimer.singleShot(500, self._fix_layout)
    
    def refresh_layout(self):
        """强制刷新整个标签页的布局 - 这个方法可以被删除，因为BaseTab已经实现了相同的功能"""
        # 调用基类的实现
        super().refresh_layout()
        
        # 如果需要，可以在这里添加设置标签页特有的刷新逻辑 
    
    def _fix_layout(self):
        """特殊方法：尝试通过强制措施修复设置标签页的布局问题"""
        try:
            # 强制滚动到顶部
            if hasattr(self, 'layout') and self.layout.count() > 0:
                scroll_area = self.layout.itemAt(0).widget()
                if isinstance(scroll_area, QScrollArea):
                    # 设置滚动条策略确保内容显示
                    scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                    scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                    
                    # 尝试调整视图，确保内容从顶部开始显示
                    if scroll_area.verticalScrollBar():
                        scroll_area.verticalScrollBar().setValue(0)
                        
                    # 尝试调整内部部件的大小
                    content_widget = scroll_area.widget()
                    if content_widget:
                        # 确保内容部件比可视区域稍大，以触发正确的滚动行为
                        viewport_height = scroll_area.viewport().height()
                        if viewport_height > 0:
                            content_widget.setMinimumHeight(viewport_height)
                        
                        # 强制重新计算滚动区域的布局
                        content_widget.updateGeometry()
                        scroll_area.updateGeometry()
            
            # 触发整个标签页和主窗口的刷新
            self.update()
            if self.main_window:
                # 尝试调整主窗口大小，这常常能触发Qt重新计算所有布局
                size = self.main_window.size()
                self.main_window.resize(size.width() + 1, size.height())
                QTimer.singleShot(50, lambda: self.main_window.resize(size))
        except Exception as e:
            print(f"尝试修复设置标签页布局时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def save_classes_to_file(self):
        """保存类别配置到文件"""
        try:
            # 从表格收集最新数据
            self.collect_weights_from_table()
            
            # 获取权重策略
            strategy_text = self.weight_strategy_combo.currentText()
            weight_strategy = strategy_text.split(' ')[0]
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "保存类别配置文件", 
                "defect_classes_config.json", 
                "JSON文件 (*.json)"
            )
            
            if file_path:
                # 创建包含权重信息的配置数据
                classes_config = {
                    "classes": self.default_classes,
                    "class_weights": self.class_weights,
                    "weight_strategy": weight_strategy,
                    "use_class_weights": weight_strategy != 'none',
                    "description": "缺陷类别配置文件，包含类别名称、权重信息和权重策略",
                    "version": "2.0"
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(classes_config, f, ensure_ascii=False, indent=4)
                
                QMessageBox.information(
                    self, 
                    "保存成功", 
                    f"类别配置已保存到:\n{file_path}\n\n"
                    f"包含 {len(self.default_classes)} 个类别\n"
                    f"权重策略: {weight_strategy}"
                )
                
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存类别配置文件时出错:\n{str(e)}")

    def load_classes_from_file(self):
        """从文件加载类别配置"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "加载类别配置文件", 
                "", 
                "文本文件 (*.txt);;JSON文件 (*.json);;所有文件 (*)"
            )
            
            if not file_path:
                return
            
            # 询问是否替换现有类别
            if self.default_classes:
                reply = QMessageBox.question(
                    self, 
                    "加载确认", 
                    "是否替换现有的类别配置？\n选择'是'将替换所有现有类别，选择'否'将添加到现有类别。",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
                )
                
                if reply == QMessageBox.Cancel:
                    return
                    
                replace_existing = reply == QMessageBox.Yes
            else:
                replace_existing = True
            
            loaded_classes = []
            loaded_weights = {}
            loaded_strategy = 'balanced'
            
            # 根据文件扩展名处理不同格式
            if file_path.lower().endswith('.json'):
                # JSON文件格式
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查新版本格式（包含权重信息）
                if isinstance(data, dict) and 'classes' in data:
                    loaded_classes = data.get('classes', [])
                    loaded_weights = data.get('class_weights', {})
                    loaded_strategy = data.get('weight_strategy', 'balanced')
                    
                    QMessageBox.information(
                        self, 
                        "加载信息", 
                        f"检测到包含权重信息的配置文件\n"
                        f"类别数量: {len(loaded_classes)}\n"
                        f"权重策略: {loaded_strategy}"
                    )
                # 检查旧版本格式或简单列表格式
                elif isinstance(data, list):
                    loaded_classes = data
                    # 为所有类别设置默认权重
                    loaded_weights = {class_name: 1.0 for class_name in loaded_classes}
                else:
                    raise ValueError("不支持的JSON文件格式")
            else:
                # 文本文件格式（每行一个类别）
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_classes = [line.strip() for line in f if line.strip()]
                # 为所有类别设置默认权重
                loaded_weights = {class_name: 1.0 for class_name in loaded_classes}
            
            if not loaded_classes:
                QMessageBox.warning(self, "警告", "文件中没有找到有效的类别信息")
                return
            
            # 根据用户选择处理现有类别
            if replace_existing:
                # 替换现有类别
                self.default_classes.clear()
                self.class_weights.clear()
                self.class_weight_table.setRowCount(0)
            
            # 添加新类别
            added_count = 0
            for class_name in loaded_classes:
                if class_name not in self.default_classes:
                    self.default_classes.append(class_name)
                    # 使用加载的权重或默认权重
                    weight_value = loaded_weights.get(class_name, 1.0)
                    self.class_weights[class_name] = weight_value
                    
                    # 添加到表格
                    row_count = self.class_weight_table.rowCount()
                    self.class_weight_table.insertRow(row_count)
                    self.class_weight_table.setItem(row_count, 0, QTableWidgetItem(class_name))
                    
                    # 创建权重输入框
                    weight_spinbox = QDoubleSpinBox()
                    weight_spinbox.setMinimum(0.01)
                    weight_spinbox.setMaximum(100.0)
                    weight_spinbox.setSingleStep(0.1)
                    weight_spinbox.setDecimals(2)
                    weight_spinbox.setValue(weight_value)
                    weight_spinbox.valueChanged.connect(lambda value, name=class_name: self.on_weight_changed(name, value))
                    
                    self.class_weight_table.setCellWidget(row_count, 1, weight_spinbox)
                    added_count += 1
            
            # 设置权重策略
            strategy_mapping = {
                'balanced': 'balanced (平衡权重)',
                'inverse': 'inverse (逆频率权重)',
                'log_inverse': 'log_inverse (对数逆频率权重)',
                'custom': 'custom (自定义权重)',
                'none': 'none (无权重)'
            }
            strategy_text = strategy_mapping.get(loaded_strategy, 'balanced (平衡权重)')
            self.weight_strategy_combo.setCurrentText(strategy_text)
            
            # 更新权重输入框状态
            self.update_weight_widgets_state()
            
            action_text = "替换" if replace_existing else "添加"
            QMessageBox.information(
                self, 
                "加载成功", 
                f"成功{action_text}了 {added_count} 个类别\n"
                f"权重策略: {loaded_strategy}\n"
                f"当前总类别数: {len(self.default_classes)}"
            )
                
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"加载类别配置文件时出错:\n{str(e)}")
    
    def save_config_to_file(self):
        """保存配置到文件"""
        try:
            # 从表格收集最新数据
            self.collect_weights_from_table()
            
            # 获取权重策略
            strategy_text = self.weight_strategy_combo.currentText()
            weight_strategy = strategy_text.split(' ')[0]
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "保存配置文件", 
                "app_config.json", 
                "JSON文件 (*.json)"
            )
            
            if file_path:
                # 创建完整的配置数据
                config = {
                    'default_source_folder': self.default_source_edit.text(),
                    'default_output_folder': self.default_output_edit.text(),
                    'default_model_file': self.default_model_edit.text(),
                    'default_class_info_file': self.default_class_info_edit.text(),
                    'default_model_eval_dir': self.default_model_eval_dir_edit.text(),
                    'default_model_save_dir': self.default_model_save_dir_edit.text(),
                    'default_tensorboard_log_dir': self.default_tensorboard_log_dir_edit.text(),
                    'default_dataset_dir': self.default_dataset_dir_edit.text(),
                    'default_param_save_dir': self.default_param_save_dir_edit.text(),
                    'default_classes': self.default_classes,
                    'class_weights': self.class_weights,
                    'weight_strategy': weight_strategy,
                    'use_class_weights': weight_strategy != 'none',
                    'version': '2.0',
                    'export_time': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=4)
                
                QMessageBox.information(
                    self, 
                    "保存成功", 
                    f"配置已保存到:\n{file_path}\n\n"
                    f"包含 {len(self.default_classes)} 个类别\n"
                    f"权重策略: {weight_strategy}"
                )
                
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存配置文件时出错:\n{str(e)}")

    def load_config_from_file(self):
        """从文件加载配置"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "加载配置文件", 
                "", 
                "JSON文件 (*.json);;所有文件 (*)"
            )
            
            if not file_path:
                return
                
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # 检查配置文件版本
            version = config.get('version', '1.0')
            has_weight_config = 'class_weights' in config and 'weight_strategy' in config
            
            if has_weight_config:
                QMessageBox.information(
                    self, 
                    "配置信息", 
                    f"检测到版本 {version} 的配置文件\n"
                    f"包含权重配置信息\n"
                    f"类别数量: {len(config.get('default_classes', []))}\n"
                    f"权重策略: {config.get('weight_strategy', 'balanced')}"
                )
            
            # 确认是否要应用配置
            reply = QMessageBox.question(
                self, 
                "确认加载", 
                "确定要应用这个配置文件吗？\n这将覆盖当前的所有设置。",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
            
            # 应用文件夹配置
            self.default_source_edit.setText(config.get('default_source_folder', ''))
            self.default_output_edit.setText(config.get('default_output_folder', ''))
            self.default_model_edit.setText(config.get('default_model_file', ''))
            self.default_class_info_edit.setText(config.get('default_class_info_file', ''))
            self.default_model_eval_dir_edit.setText(config.get('default_model_eval_dir', ''))
            self.default_model_save_dir_edit.setText(config.get('default_model_save_dir', ''))
            self.default_tensorboard_log_dir_edit.setText(config.get('default_tensorboard_log_dir', ''))
            self.default_dataset_dir_edit.setText(config.get('default_dataset_dir', ''))
            self.default_param_save_dir_edit.setText(config.get('default_param_save_dir', ''))
            
            # 应用类别和权重配置
            self.default_classes = config.get('default_classes', [])
            self.class_weights = config.get('class_weights', {})
            
            # 设置权重策略
            weight_strategy = config.get('weight_strategy', 'balanced')
            strategy_mapping = {
                'balanced': 'balanced (平衡权重)',
                'inverse': 'inverse (逆频率权重)',
                'log_inverse': 'log_inverse (对数逆频率权重)',
                'custom': 'custom (自定义权重)',
                'none': 'none (无权重)'
            }
            strategy_text = strategy_mapping.get(weight_strategy, 'balanced (平衡权重)')
            self.weight_strategy_combo.setCurrentText(strategy_text)
            
            # 重新填充类别权重表格
            self.class_weight_table.setRowCount(0)
            for class_name in self.default_classes:
                row_count = self.class_weight_table.rowCount()
                self.class_weight_table.insertRow(row_count)
                self.class_weight_table.setItem(row_count, 0, QTableWidgetItem(class_name))
                
                # 创建权重输入框
                weight_value = self.class_weights.get(class_name, 1.0)
                weight_spinbox = QDoubleSpinBox()
                weight_spinbox.setMinimum(0.01)
                weight_spinbox.setMaximum(100.0)
                weight_spinbox.setSingleStep(0.1)
                weight_spinbox.setDecimals(2)
                weight_spinbox.setValue(weight_value)
                weight_spinbox.valueChanged.connect(lambda value, name=class_name: self.on_weight_changed(name, value))
                
                self.class_weight_table.setCellWidget(row_count, 1, weight_spinbox)
            
            # 更新权重输入框状态
            self.update_weight_widgets_state()
            
            # 保存更新后的配置
            self.config = config
            
            QMessageBox.information(
                self, 
                "加载成功", 
                f"配置文件已成功加载\n"
                f"类别数量: {len(self.default_classes)}\n"
                f"权重策略: {weight_strategy}"
            )
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "加载失败", 
                f"加载配置文件失败:\n{str(e)}"
            ) 