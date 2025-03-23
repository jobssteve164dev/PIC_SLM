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
        
        # 训练参数保存文件夹
        self.default_param_save_dir_edit = QLineEdit()
        self.default_param_save_dir_edit.setReadOnly(True)
        self.default_param_save_dir_edit.setPlaceholderText("默认训练参数保存文件夹")
        
        param_save_dir_btn = QPushButton("浏览...")
        param_save_dir_btn.clicked.connect(self.select_default_param_save_dir)
        
        model_layout.addWidget(QLabel("默认训练参数保存文件夹:"), 5, 0)
        model_layout.addWidget(self.default_param_save_dir_edit, 5, 1)
        model_layout.addWidget(param_save_dir_btn, 5, 2)
        
        model_group.setLayout(model_layout)
        advanced_layout.addWidget(model_group)
        
        # 添加高级设置选项卡
        settings_tabs.addTab(advanced_tab, "高级设置")
        
        main_layout.addWidget(settings_tabs)
        
        # 添加保存按钮
        save_btn = QPushButton("保存设置")
        save_btn.clicked.connect(self.save_settings)
        save_btn.setMinimumHeight(40)
        main_layout.addWidget(save_btn)
        
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
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
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
                'default_param_save_dir': self.default_param_save_dir_edit.text(),
                'default_classes': self.default_classes
            }
            
            # 保存配置到文件
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
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