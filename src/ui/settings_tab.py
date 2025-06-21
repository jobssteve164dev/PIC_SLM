"""
重构后的设置标签页 - 使用拆分后的组件
"""

from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                           QTabWidget, QWidget, QMessageBox, QFileDialog, QCheckBox, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
import os
import time
from .base_tab import BaseTab
from .components.settings import (ConfigManager, FolderConfigWidget, 
                                ClassWeightWidget, ModelConfigWidget, WeightStrategy,
                                ConfigProfileSelector)


class SettingsTab(BaseTab):
    """重构后的设置标签页，使用组件化设计"""
    
    # 定义信号
    settings_saved = pyqtSignal(dict)
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent, main_window)
        
        # 初始化管理器和配置
        self.config_manager = ConfigManager()
        self.config = {}
        
        # 初始化UI
        self.init_ui()
        
        # 连接信号
        self._connect_signals()
        
        # 添加特殊的延迟重建布局定时器
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.timeout.connect(self._fix_layout)
        
        # 加载当前设置
        self.load_current_settings()
        
    def init_ui(self):
        """初始化UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self.scroll_content)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 添加标题
        title_label = QLabel("应用设置")
        title_label.setFont(QFont('微软雅黑', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 添加配置文件选择器组件
        self.config_profile_selector = ConfigProfileSelector()
        main_layout.addWidget(self.config_profile_selector)
        
        # 创建设置选项卡
        self.settings_tabs = QTabWidget()
        
        # 创建常规设置选项卡
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)
        general_layout.setContentsMargins(10, 10, 10, 10)
        
        # 添加文件夹配置组件
        self.folder_config_widget = FolderConfigWidget()
        general_layout.addWidget(self.folder_config_widget)
        
        # 添加类别权重配置组件
        self.class_weight_widget = ClassWeightWidget()
        general_layout.addWidget(self.class_weight_widget)
        
        # 添加系统托盘设置组
        self._create_system_tray_group(general_layout)
        
        # 添加弹性空间
        general_layout.addStretch()
        
        # 添加常规设置选项卡
        self.settings_tabs.addTab(general_tab, "常规设置")
        
        # 创建高级设置选项卡
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        advanced_layout.setContentsMargins(10, 10, 10, 10)
        
        # 添加模型配置组件
        self.model_config_widget = ModelConfigWidget()
        advanced_layout.addWidget(self.model_config_widget)
        
        # 添加高级设置选项卡
        self.settings_tabs.addTab(advanced_tab, "高级设置")
        
        main_layout.addWidget(self.settings_tabs)
        
        # 添加按钮组
        self._create_button_layout(main_layout)
        
        # 添加弹性空间
        main_layout.addStretch(1)
    
    def _create_system_tray_group(self, parent_layout):
        """创建系统托盘设置组"""
        tray_group = QGroupBox("系统托盘设置")
        tray_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        tray_layout = QVBoxLayout(tray_group)
        
        # 最小化到托盘选项
        self.minimize_to_tray_checkbox = QCheckBox("最小化到系统托盘")
        self.minimize_to_tray_checkbox.setChecked(True)
        self.minimize_to_tray_checkbox.setToolTip(
            "勾选后，点击最小化按钮或关闭按钮将程序隐藏到系统托盘而不是退出程序。\n"
            "双击托盘图标或右键菜单可以重新显示窗口。"
        )
        self.minimize_to_tray_checkbox.toggled.connect(self.on_minimize_to_tray_toggled)
        tray_layout.addWidget(self.minimize_to_tray_checkbox)
        
        parent_layout.addWidget(tray_group)
    
    def _create_button_layout(self, parent_layout):
        """创建按钮布局"""
        button_layout = QHBoxLayout()
        
        # 保存设置按钮
        save_btn = QPushButton("保存设置")
        save_btn.clicked.connect(self.save_settings)
        save_btn.setMinimumHeight(40)
        button_layout.addWidget(save_btn)
        
        # 保存配置到文件按钮
        save_config_to_file_btn = QPushButton("保存配置到文件")
        save_config_to_file_btn.clicked.connect(self.save_config_to_file)
        save_config_to_file_btn.setMinimumHeight(40)
        button_layout.addWidget(save_config_to_file_btn)
        
        # 从文件加载配置按钮
        load_config_from_file_btn = QPushButton("从文件加载配置")
        load_config_from_file_btn.clicked.connect(self.load_config_from_file)
        load_config_from_file_btn.setMinimumHeight(40)
        button_layout.addWidget(load_config_from_file_btn)
        
        parent_layout.addLayout(button_layout)
    
    def _connect_signals(self):
        """连接所有组件的信号"""
        # 连接文件夹配置变化信号
        self.folder_config_widget.folder_changed.connect(self.on_folder_changed)
        
        # 连接模型配置变化信号  
        self.model_config_widget.config_changed.connect(self.on_model_config_changed)
        
        # 连接类别权重配置变化信号
        self.class_weight_widget.classes_changed.connect(self.on_classes_changed)
        self.class_weight_widget.weights_changed.connect(self.on_weights_changed)
        self.class_weight_widget.strategy_changed.connect(self.on_strategy_changed)
        
        # 连接配置文件选择器信号
        self.config_profile_selector.profile_changed.connect(self.on_profile_changed)
        self.config_profile_selector.profile_loaded.connect(self.on_profile_loaded)
    
    def on_folder_changed(self, folder_type: str, folder_path: str):
        """处理文件夹变化"""
        print(f"文件夹变化: {folder_type} -> {folder_path}")
    
    def on_model_config_changed(self, config_type: str, config_value: str):
        """处理模型配置变化"""
        print(f"模型配置变化: {config_type} -> {config_value}")
    
    def on_classes_changed(self, classes: list):
        """处理类别变化"""
        print(f"类别变化: {classes}")
    
    def on_weights_changed(self, weights: dict):
        """处理权重变化"""
        print(f"权重变化: {weights}")
    
    def on_strategy_changed(self, strategy: WeightStrategy):
        """处理策略变化"""
        print(f"策略变化: {strategy.value}")
    
    def on_minimize_to_tray_toggled(self, checked: bool):
        """处理最小化到托盘选项变化"""
        print(f"最小化到托盘选项变化: {checked}")
        # 通知主窗口更新托盘设置
        if hasattr(self.main_window, 'set_minimize_to_tray_enabled'):
            self.main_window.set_minimize_to_tray_enabled(checked)
    
    def on_profile_changed(self, profile_name: str, config_data: dict):
        """处理配置文件改变"""
        print(f"配置文件改变: {profile_name}")
        # 这里可以添加预览逻辑，但不自动应用
    
    def on_profile_loaded(self, config_data: dict):
        """处理配置文件加载"""
        try:
            print(f"应用配置文件数据: {config_data}")
            
            # 提取配置数据
            if 'config' in config_data:
                config = config_data['config']
                
                # 更新当前配置
                self.config = config
                
                # 应用配置到UI组件
                self._apply_config_to_ui()
                
                print("配置文件应用成功")
            else:
                print("配置文件格式不正确，缺少config字段")
                
        except Exception as e:
            print(f"应用配置文件失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_current_settings(self):
        """加载当前设置"""
        try:
            print("SettingsTab.load_current_settings: 开始加载配置...")
            self.config = self.config_manager.load_config()
            print(f"SettingsTab.load_current_settings: 已加载配置 = {self.config}")
            self._apply_config_to_ui()
            print("SettingsTab.load_current_settings: 配置加载完成")
        except Exception as e:
            print(f"SettingsTab: 加载配置失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _apply_config_to_ui(self):
        """将配置应用到UI组件"""
        if not self.config:
            print("SettingsTab._apply_config_to_ui: 配置为空，跳过应用")
            return
        
        print(f"SettingsTab._apply_config_to_ui: 开始应用配置 = {self.config}")
        
        # 应用文件夹配置
        print(f"SettingsTab._apply_config_to_ui: 应用文件夹配置...")
        print(f"  源文件夹: {self.config.get('default_source_folder', 'NOT_SET')}")
        print(f"  输出文件夹: {self.config.get('default_output_folder', 'NOT_SET')}")
        self.folder_config_widget.set_folder_config(self.config)
        
        # 应用模型配置
        print(f"SettingsTab._apply_config_to_ui: 应用模型配置...")
        self.model_config_widget.set_model_config(self.config)
        
        # 应用类别权重配置
        print(f"SettingsTab._apply_config_to_ui: 应用类别权重配置...")
        classes = self.config.get('default_classes', [])
        weights = self.config.get('class_weights', {})
        strategy_value = self.config.get('weight_strategy', 'balanced')
        strategy = WeightStrategy.from_value(strategy_value)
        
        self.class_weight_widget.set_classes_config(classes, weights, strategy)
        
        # 应用系统托盘配置
        print(f"SettingsTab._apply_config_to_ui: 应用系统托盘配置...")
        minimize_to_tray = self.config.get('minimize_to_tray', True)
        self.minimize_to_tray_checkbox.setChecked(minimize_to_tray)
        
        print("SettingsTab._apply_config_to_ui: 配置应用完成")
    
    def _collect_current_config(self) -> dict:
        """收集当前所有组件的配置"""
        # 获取文件夹配置
        folder_config = self.folder_config_widget.get_folder_config()
        
        # 获取模型配置
        model_config = self.model_config_widget.get_model_config()
        
        # 获取类别权重配置
        classes, weights, strategy = self.class_weight_widget.get_classes_config()
        
        # 获取系统托盘配置
        minimize_to_tray = self.minimize_to_tray_checkbox.isChecked()
        
        # 创建完整配置
        config = self.config_manager.create_config_dict(
            default_source_folder=folder_config.get('default_source_folder', ''),
            default_output_folder=folder_config.get('default_output_folder', ''),
            default_model_file=model_config.get('default_model_file', ''),
            default_class_info_file=model_config.get('default_class_info_file', ''),
            default_model_eval_dir=model_config.get('default_model_eval_dir', ''),
            default_model_save_dir=model_config.get('default_model_save_dir', ''),
            default_tensorboard_log_dir=model_config.get('default_tensorboard_log_dir', ''),
            default_dataset_dir=model_config.get('default_dataset_dir', ''),
            default_param_save_dir=model_config.get('default_param_save_dir', ''),
            default_classes=classes,
            class_weights=weights,
            weight_strategy=strategy
        )
        
        # 添加系统托盘配置
        config['minimize_to_tray'] = minimize_to_tray
        
        return config
    
    def save_settings(self):
        """保存设置"""
        try:
            # 收集当前配置
            config = self._collect_current_config()
            print(f"SettingsTab.save_settings: 收集到的配置 = {config}")
            
            # 验证配置
            warnings = self.config_manager.validate_config(config)
            if warnings:
                warning_text = "\n".join(warnings)
                reply = QMessageBox.question(
                    self, 
                    "配置警告", 
                    f"发现以下配置问题:\n\n{warning_text}\n\n是否仍要保存？",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
            
            # 保存配置
            success = self.config_manager.save_config(config)
            print(f"SettingsTab.save_settings: 配置保存结果 = {success}")
            
            if success:
                self.config = config
                print(f"SettingsTab.save_settings: 准备发送settings_saved信号，配置内容 = {config}")
                self.settings_saved.emit(config)
                print("SettingsTab.save_settings: settings_saved信号已发送")
                QMessageBox.information(self, "成功", "设置已保存")
            else:
                QMessageBox.critical(self, "错误", "保存设置失败")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存设置失败: {str(e)}")
            print(f"SettingsTab: 保存设置失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def save_config_to_file(self):
        """保存配置到文件"""
        try:
            # 收集当前配置
            config = self._collect_current_config()
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "保存配置文件", 
                "app_config.json", 
                "JSON文件 (*.json)"
            )
            
            if file_path:
                success = self.config_manager.save_config_to_file(config, file_path)
                
                if success:
                    classes = config.get('default_classes', [])
                    strategy = config.get('weight_strategy', 'balanced')
                    QMessageBox.information(
                        self, 
                        "保存成功", 
                        f"配置已保存到:\n{file_path}\n\n"
                        f"包含 {len(classes)} 个类别\n"
                        f"权重策略: {strategy}"
                    )
                else:
                    QMessageBox.critical(self, "保存失败", "保存配置文件时出错")
                
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
            
            # 加载配置
            config = self.config_manager.load_config_from_file(file_path)
            if config is None:
                QMessageBox.critical(self, "加载失败", "无法加载配置文件")
                return
            
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
            
            # 应用配置
            self.config = config
            self._apply_config_to_ui()
            
            # 验证配置
            warnings = self.config_manager.validate_config(config)
            if warnings:
                warning_text = "\n".join(warnings)
                QMessageBox.warning(
                    self, 
                    "配置警告", 
                    f"加载的配置存在以下问题:\n\n{warning_text}\n\n"
                    f"建议检查相关路径和文件。"
                )
            
            classes = config.get('default_classes', [])
            weight_strategy = config.get('weight_strategy', 'balanced')
            QMessageBox.information(
                self, 
                "加载成功", 
                f"配置文件已成功加载\n"
                f"类别数量: {len(classes)}\n"
                f"权重策略: {weight_strategy}"
            )
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "加载失败", 
                f"加载配置文件失败:\n{str(e)}"
            )
    
    def on_tab_changed(self, index):
        """处理标签页切换事件"""
        # 调用基类方法
        super().on_tab_changed(index)
        
        # 添加特殊处理：当切换到设置标签页时，重新加载配置并修复布局
        if self.main_window and hasattr(self.main_window, 'tabs'):
            current_widget = self.main_window.tabs.widget(index)
            if current_widget == self:
                print("切换到设置标签页，重新加载配置并启动布局修复机制")
                
                # 强制重新加载配置以确保显示最新的设置
                self.load_current_settings()
                
                # 使用定时器延迟启动我们的特殊布局修复
                self._rebuild_timer.start(250)
                
                # 使用多个定时器在不同时间点尝试修复，提高成功率
                QTimer.singleShot(350, self._fix_layout)
                QTimer.singleShot(500, self._fix_layout)
    
    def _fix_layout(self):
        """特殊方法：尝试通过强制措施修复设置标签页的布局问题"""
        try:
            # 强制滚动到顶部
            if hasattr(self, 'layout') and self.layout.count() > 0:
                scroll_area = self.layout.itemAt(0).widget()
                if hasattr(scroll_area, 'verticalScrollBar'):
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
    
    def get_current_config(self) -> dict:
        """获取当前配置（供外部调用）"""
        return self._collect_current_config()
    
    def clear_all_settings(self):
        """清空所有设置"""
        reply = QMessageBox.question(
            self, 
            "清空设置", 
            "确定要清空所有设置吗？\n这个操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.folder_config_widget.clear_config()
            self.model_config_widget.clear_config()
            self.class_weight_widget.clear_config()
            
            QMessageBox.information(self, "完成", "已清空所有设置")
    
    def validate_current_config(self) -> bool:
        """验证当前配置的有效性"""
        try:
            config = self._collect_current_config()
            warnings = self.config_manager.validate_config(config)
            
            if warnings:
                warning_text = "\n".join(warnings)
                QMessageBox.warning(
                    self, 
                    "配置验证", 
                    f"当前配置存在以下问题:\n\n{warning_text}"
                )
                return False
            else:
                QMessageBox.information(self, "配置验证", "当前配置验证通过")
                return True
                
        except Exception as e:
            QMessageBox.critical(self, "验证失败", f"配置验证失败:\n{str(e)}")
            return False 