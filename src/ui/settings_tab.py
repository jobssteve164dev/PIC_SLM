from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QComboBox, QSpinBox, QGroupBox, QGridLayout,
                           QSizePolicy, QLineEdit, QCheckBox, QListWidget, QInputDialog,
                           QMessageBox, QTabWidget, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
import os
import json
from .base_tab import BaseTab

class SettingsTab(BaseTab):
    """设置标签页，负责应用设置管理"""
    
    # 定义信号
    settings_saved = pyqtSignal(dict)
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent, main_window)
        self.config = {}
        self.default_classes = []
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
        classes_group = QGroupBox("默认缺陷类别")
        classes_layout = QVBoxLayout()
        classes_layout.setContentsMargins(10, 20, 10, 10)
        
        # 添加类别列表
        self.default_class_list = QListWidget()
        self.default_class_list.setMinimumHeight(150)
        classes_layout.addWidget(self.default_class_list)
        
        # 添加按钮组
        btn_layout = QHBoxLayout()
        
        add_class_btn = QPushButton("添加类别")
        add_class_btn.clicked.connect(self.settings_add_defect_class)
        btn_layout.addWidget(add_class_btn)
        
        remove_class_btn = QPushButton("删除类别")
        remove_class_btn.clicked.connect(self.settings_remove_defect_class)
        btn_layout.addWidget(remove_class_btn)
        
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
            self.default_class_list.addItem(class_name)
    
    def settings_remove_defect_class(self):
        """删除默认缺陷类别"""
        current_item = self.default_class_list.currentItem()
        if current_item:
            class_name = current_item.text()
            self.default_classes.remove(class_name)
            self.default_class_list.takeItem(self.default_class_list.row(current_item))
    
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
        
        # 设置默认类别
        self.default_classes = self.config.get('default_classes', [])
        self.default_class_list.clear()
        for class_name in self.default_classes:
            self.default_class_list.addItem(class_name)
    
    def save_settings(self):
        """保存设置"""
        try:
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
                'default_classes': self.default_classes
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
        """保存缺陷类别到文件"""
        if not self.default_classes:
            QMessageBox.warning(self, "警告", "没有缺陷类别可以保存!")
            return
            
        # 选择保存文件
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "保存缺陷类别", 
            "defect_classes.json", 
            "JSON文件 (*.json);;文本文件 (*.txt);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            # 根据文件扩展名决定保存格式
            if file_path.endswith('.json'):
                # 保存为JSON格式
                data = {
                    "defect_classes": self.default_classes,
                    "total_count": len(self.default_classes),
                    "created_by": "图片模型训练系统",
                    "version": "1.0"
                }
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            else:
                # 保存为纯文本格式
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("缺陷类别列表\n")
                    f.write("=" * 20 + "\n")
                    for i, class_name in enumerate(self.default_classes, 1):
                        f.write(f"{i}. {class_name}\n")
                    f.write(f"\n总计: {len(self.default_classes)} 个类别\n")
                    
            QMessageBox.information(
                self, 
                "成功", 
                f"缺陷类别已成功保存到文件:\n{file_path}\n\n共保存了 {len(self.default_classes)} 个类别"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "错误", 
                f"保存文件失败:\n{str(e)}"
            ) 
    
    def load_classes_from_file(self):
        """从文件加载缺陷类别"""
        # 选择要加载的文件
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "加载缺陷类别", 
            "", 
            "JSON文件 (*.json);;文本文件 (*.txt);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            classes_to_load = []
            
            # 根据文件扩展名决定加载格式
            if file_path.endswith('.json'):
                # 从JSON格式加载
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 支持多种JSON格式
                if isinstance(data, dict):
                    if 'defect_classes' in data:
                        # 我们自己保存的格式
                        classes_to_load = data['defect_classes']
                    elif 'classes' in data:
                        # 其他可能的格式
                        classes_to_load = data['classes']
                    else:
                        # 如果字典包含其他键值对，尝试找到包含类别列表的键
                        for key, value in data.items():
                            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                                classes_to_load = value
                                break
                elif isinstance(data, list):
                    # 直接是列表格式
                    classes_to_load = data
            else:
                # 从文本文件加载，每行一个类别
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    line = line.strip()
                    # 跳过空行、标题行和分隔符行
                    if line and not line.startswith('=') and not line.startswith('缺陷类别') and not line.startswith('总计'):
                        # 处理编号格式 "1. 类别名"
                        if '. ' in line and line[0].isdigit():
                            class_name = line.split('. ', 1)[1]
                        else:
                            class_name = line
                        
                        if class_name and class_name not in classes_to_load:
                            classes_to_load.append(class_name)
            
            if not classes_to_load:
                QMessageBox.warning(self, "警告", "文件中没有找到有效的缺陷类别!")
                return
                
            # 询问用户是要替换还是追加
            reply = QMessageBox.question(
                self, 
                "加载方式", 
                f"从文件中找到 {len(classes_to_load)} 个类别:\n" + 
                "\n".join(f"• {cls}" for cls in classes_to_load[:5]) + 
                ("\n..." if len(classes_to_load) > 5 else "") +
                f"\n\n您想要:\n• 替换当前类别 (是)\n• 追加到当前类别 (否)", 
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Cancel:
                return
            elif reply == QMessageBox.Yes:
                # 替换当前类别
                self.default_classes.clear()
                self.default_class_list.clear()
                
            # 添加新类别（去重）
            added_count = 0
            for class_name in classes_to_load:
                if class_name not in self.default_classes:
                    self.default_classes.append(class_name)
                    self.default_class_list.addItem(class_name)
                    added_count += 1
                    
            action = "替换" if reply == QMessageBox.Yes else "追加"
            QMessageBox.information(
                self, 
                "成功", 
                f"已成功{action}缺陷类别!\n\n"
                f"从文件加载: {len(classes_to_load)} 个类别\n"
                f"实际添加: {added_count} 个类别\n"
                f"当前总计: {len(self.default_classes)} 个类别"
            )
            
        except json.JSONDecodeError as e:
            QMessageBox.critical(
                self, 
                "错误", 
                f"JSON文件格式错误:\n{str(e)}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, 
                "错误", 
                f"加载文件失败:\n{str(e)}"
            ) 
    
    def save_config_to_file(self):
        """保存完整配置到文件"""
        # 创建当前配置字典
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
            'default_classes': self.default_classes
        }
        
        # 选择保存文件
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "保存配置文件", 
            "settings_config.json", 
            "JSON文件 (*.json);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            # 添加元数据
            config_with_metadata = {
                "config": config,
                "metadata": {
                    "created_by": "图片模型训练系统",
                    "version": "1.0",
                    "description": "应用设置配置文件",
                    "export_time": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_with_metadata, f, ensure_ascii=False, indent=4)
                
            QMessageBox.information(
                self, 
                "成功", 
                f"配置已成功保存到文件:\n{file_path}\n\n"
                f"包含内容:\n"
                f"• 默认源文件夹\n"
                f"• 默认输出文件夹\n"
                f"• 默认模型文件\n"
                f"• 默认类别信息文件\n"
                f"• 默认模型评估文件夹\n"
                f"• 默认模型保存文件夹\n"
                f"• 默认TensorBoard日志文件夹\n"
                f"• 默认数据集评估文件夹\n"
                f"• 默认训练参数保存文件夹\n"
                f"• 缺陷类别设置 ({len(self.default_classes)} 个类别)"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "错误", 
                f"保存配置文件失败:\n{str(e)}"
            )
    
    def load_config_from_file(self):
        """从文件加载完整配置"""
        # 选择要加载的文件
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "加载配置文件", 
            "", 
            "JSON文件 (*.json);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 支持不同的配置文件格式
            config = None
            if isinstance(data, dict):
                if 'config' in data:
                    # 带元数据的格式
                    config = data['config']
                    metadata = data.get('metadata', {})
                    created_by = metadata.get('created_by', '未知')
                    export_time = metadata.get('export_time', '未知')
                else:
                    # 直接配置格式
                    config = data
                    created_by = "未知"
                    export_time = "未知"
            
            if not config:
                QMessageBox.warning(self, "警告", "文件格式不正确，无法找到有效的配置数据!")
                return
            
            # 显示配置预览并询问用户是否加载
            preview_text = f"配置文件信息:\n"
            if 'metadata' in data:
                preview_text += f"创建者: {created_by}\n"
                preview_text += f"导出时间: {export_time}\n\n"
            
            preview_text += f"配置内容:\n"
            preview_text += f"• 默认源文件夹: {config.get('default_source_folder', '未设置')}\n"
            preview_text += f"• 默认输出文件夹: {config.get('default_output_folder', '未设置')}\n"
            preview_text += f"• 默认模型文件: {config.get('default_model_file', '未设置')}\n"
            preview_text += f"• 默认类别信息文件: {config.get('default_class_info_file', '未设置')}\n"
            preview_text += f"• 默认模型评估文件夹: {config.get('default_model_eval_dir', '未设置')}\n"
            preview_text += f"• 默认模型保存文件夹: {config.get('default_model_save_dir', '未设置')}\n"
            preview_text += f"• 默认TensorBoard日志文件夹: {config.get('default_tensorboard_log_dir', '未设置')}\n"
            preview_text += f"• 默认数据集评估文件夹: {config.get('default_dataset_dir', '未设置')}\n"
            preview_text += f"• 默认训练参数保存文件夹: {config.get('default_param_save_dir', '未设置')}\n"
            preview_text += f"• 缺陷类别数量: {len(config.get('default_classes', []))} 个\n"
            
            if config.get('default_classes'):
                preview_text += f"• 缺陷类别: {', '.join(config['default_classes'][:3])}"
                if len(config['default_classes']) > 3:
                    preview_text += f" 等{len(config['default_classes'])}个"
                preview_text += "\n"
            
            reply = QMessageBox.question(
                self, 
                "确认加载配置", 
                f"{preview_text}\n是否要加载此配置？\n\n注意：这将覆盖当前的所有设置！", 
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
            
            # 应用配置到UI
            self.default_source_edit.setText(config.get('default_source_folder', ''))
            self.default_output_edit.setText(config.get('default_output_folder', ''))
            self.default_model_edit.setText(config.get('default_model_file', ''))
            self.default_class_info_edit.setText(config.get('default_class_info_file', ''))
            self.default_model_eval_dir_edit.setText(config.get('default_model_eval_dir', ''))
            self.default_model_save_dir_edit.setText(config.get('default_model_save_dir', ''))
            self.default_tensorboard_log_dir_edit.setText(config.get('default_tensorboard_log_dir', ''))
            self.default_dataset_dir_edit.setText(config.get('default_dataset_dir', ''))
            self.default_param_save_dir_edit.setText(config.get('default_param_save_dir', ''))
            
            # 更新缺陷类别
            self.default_classes = config.get('default_classes', [])
            self.default_class_list.clear()
            for class_name in self.default_classes:
                self.default_class_list.addItem(class_name)
            
            QMessageBox.information(
                self, 
                "成功", 
                f"配置已成功加载!\n\n"
                f"已加载:\n"
                f"• 默认源文件夹\n"
                f"• 默认输出文件夹\n"
                f"• 默认模型文件\n"
                f"• 默认类别信息文件\n"
                f"• 默认模型评估文件夹\n"
                f"• 默认模型保存文件夹\n"
                f"• 默认TensorBoard日志文件夹\n"
                f"• 默认数据集评估文件夹\n"
                f"• 默认训练参数保存文件夹\n"
                f"• {len(self.default_classes)} 个缺陷类别\n\n"
                f"请点击'保存设置'按钮来保存这些更改。"
            )
            
        except json.JSONDecodeError as e:
            QMessageBox.critical(
                self, 
                "错误", 
                f"JSON文件格式错误:\n{str(e)}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, 
                "错误", 
                f"加载配置文件失败:\n{str(e)}"
            ) 