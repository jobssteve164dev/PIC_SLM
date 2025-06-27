from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QListWidget, QListWidgetItem, QGroupBox, QGridLayout,
                           QSizePolicy, QLineEdit, QInputDialog, QMessageBox, QRadioButton,
                           QButtonGroup, QStackedWidget, QComboBox, QScrollArea, QFrame, QSplitter,
                           QCheckBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF, QSizeF
from PyQt5.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QColor, QBrush, QCursor, QKeySequence
import os
import glob
import xml.etree.ElementTree as ET
from datetime import datetime
from .base_tab import BaseTab
import json
from PyQt5.QtWidgets import QApplication
from .components.annotation import ClassificationWidget, DetectionWidget

class AnnotationTab(BaseTab):
    """标注标签页，负责图像标注功能"""
    
    # 定义信号
    annotation_started = pyqtSignal(str)
    open_validation_folder_signal = pyqtSignal(str)
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent, main_window)
        self.defect_classes = []  # 缺陷类别列表
        self.detection_classes = []  # 目标检测类别列表
        self.processed_folder = ""  # 处理后的图片文件夹
        self.detection_folder = ""  # 目标检测图像文件夹
        self.annotation_folder = ""  # 标注输出文件夹
        
        # 先初始化UI
        self.init_ui()
        
        # 使用新的智能配置系统
        config = self.get_config_from_manager()
        if config:
            self.apply_config(config)
        
    def init_ui(self):
        """初始化UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self.scroll_content)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 添加标题
        title_label = QLabel("图像标注")
        title_label.setFont(QFont('微软雅黑', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 添加标注模式选择
        mode_group = QGroupBox("标注模式")
        mode_layout = QHBoxLayout()
        
        # 创建单选按钮
        self.classification_radio = QRadioButton("图片分类")
        self.detection_radio = QRadioButton("目标检测")
        
        # 创建按钮组
        self.mode_button_group = QButtonGroup()
        self.mode_button_group.addButton(self.classification_radio, 0)
        self.mode_button_group.addButton(self.detection_radio, 1)
        self.classification_radio.setChecked(True)  # 默认选择图片分类
        
        # 添加到布局
        mode_layout.addWidget(self.classification_radio)
        mode_layout.addWidget(self.detection_radio)
        mode_layout.addStretch()
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)
        
        # 创建堆叠部件用于切换不同的标注界面
        self.stacked_widget = QStackedWidget()
        
        # 创建图片分类标注界面
        self.classification_widget = ClassificationWidget(self)
        # 连接分类组件的信号
        self.classification_widget.annotation_started.connect(self.on_annotation_started)
        self.classification_widget.open_validation_folder_signal.connect(self.on_open_validation_folder)
        
        # 创建目标检测标注界面
        self.detection_widget = DetectionWidget(self, self.main_window)
        
        # 添加到堆叠部件
        self.stacked_widget.addWidget(self.classification_widget)
        self.stacked_widget.addWidget(self.detection_widget)
        
        # 添加到主布局
        main_layout.addWidget(self.stacked_widget)
        
        # 连接信号
        self.mode_button_group.buttonClicked.connect(self.on_mode_changed)
        
    def on_mode_changed(self, button):
        """标注模式改变时调用"""
        try:
            if button == self.classification_radio:
                self.stacked_widget.setCurrentIndex(0)
            else:
                # 切换到目标检测模式
                self.stacked_widget.setCurrentIndex(1)
                
                # 确保在切换到目标检测模式时应用路径和类别
                if hasattr(self, 'processed_folder') and self.processed_folder:
                    self.detection_widget.set_detection_folder(self.processed_folder)
                    
                if hasattr(self, 'detection_classes') and self.detection_classes:
                    self.detection_widget.set_detection_classes(self.detection_classes)
                    
        except Exception as e:
            print(f"切换标注模式时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 出错时恢复到分类模式
            if hasattr(self, 'stacked_widget') and hasattr(self, 'classification_radio'):
                self.stacked_widget.setCurrentIndex(0)
                self.classification_radio.setChecked(True)
                
    def on_annotation_started(self, folder):
        """处理标注开始信号"""
        self.annotation_started.emit(folder)
        self.update_status("开始图像标注...")
        
    def on_open_validation_folder(self, folder):
        """处理打开验证集文件夹信号"""
        self.open_validation_folder_signal.emit(folder)
        self.update_status("正在打开验证集文件夹...")

    def update_status(self, message):
        """更新状态"""
        try:
            if hasattr(self, 'main_window') and self.main_window is not None:
                self.main_window.update_status(message)
        except Exception as e:
            print(f"更新主窗口状态时出错: {str(e)}")

    def _do_apply_config(self, config):
        """实现具体的配置应用逻辑 - 智能配置系统"""
        print(f"AnnotationTab: 智能应用配置，包含 {len(config)} 个配置项")
        
        # 应用配置到子组件
        self.apply_config_to_components(config)
        
        print("AnnotationTab: 智能配置应用完成")
            
    def apply_config_to_components(self, config):
        """应用配置到子组件"""
        try:
            # 加载默认输出文件夹路径
            if 'default_output_folder' in config and config['default_output_folder']:
                print(f"发现默认输出文件夹配置: {config['default_output_folder']}")
                
                # 设置分类标注相关路径
                self.processed_folder = config['default_output_folder']
                if hasattr(self, 'classification_widget'):
                    self.classification_widget.set_processed_folder(config['default_output_folder'])
                    
                # 设置目标检测相关路径
                if hasattr(self, 'detection_widget'):
                    self.detection_widget.set_detection_folder(config['default_output_folder'])
            
            # 加载默认类别
            if 'default_classes' in config and config['default_classes']:
                print(f"加载默认类别: {config['default_classes']}")
                # 更新类别列表
                self.defect_classes = config['default_classes'].copy()
                self.detection_classes = config['default_classes'].copy()
                
                # 应用到子组件
                if hasattr(self, 'classification_widget'):
                    self.classification_widget.set_defect_classes(config['default_classes'])
                    
                if hasattr(self, 'detection_widget'):
                    self.detection_widget.set_detection_classes(config['default_classes'])
                    
                print(f"已更新所有组件的类别列表")
                
        except Exception as e:
            print(f"应用配置到子组件时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def open_validation_folder(self):
        """打开验证集文件夹"""
        self.open_validation_folder_signal.emit(self.processed_folder)

    def generate_detection_dataset(self):
        """生成目标检测训练数据集文件结构"""
        if hasattr(self, 'detection_widget'):
            self.detection_widget.generate_detection_dataset()
            
    # 为了保持向后兼容性，添加一些属性访问器
    @property
    def image_files(self):
        """获取图像文件列表"""
        if hasattr(self, 'detection_widget'):
            return getattr(self.detection_widget, 'image_files', [])
        return []
        
    @property
    def current_index(self):
        """获取当前图像索引"""
        if hasattr(self, 'detection_widget'):
            return getattr(self.detection_widget, 'current_index', -1)
        return -1
        
    def load_image_files(self, folder):
        """加载图像文件列表（向后兼容）"""
        if hasattr(self, 'detection_widget'):
            self.detection_widget.load_image_files(folder)
            
    def check_annotation_ready(self):
        """检查分类标注是否准备就绪"""
        if hasattr(self, 'classification_widget'):
            self.classification_widget.check_annotation_ready()
            
    def check_detection_ready(self):
        """检查检测标注是否准备就绪"""
        if hasattr(self, 'detection_widget'):
            self.detection_widget.check_detection_ready()