from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QComboBox, QSpinBox, QGroupBox, QGridLayout,
                           QSizePolicy, QLineEdit, QCheckBox, QDoubleSpinBox, QRadioButton,
                           QButtonGroup, QToolTip, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QFont, QIcon
import os
from .base_tab import BaseTab

class DataProcessingTab(BaseTab):
    """数据处理标签页，负责图像预处理功能"""
    
    # 定义信号
    image_preprocessing_started = pyqtSignal(dict)
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent, main_window)
        self.source_folder = ""
        self.output_folder = ""
        self.resize_width = 224
        self.resize_height = 224
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self.scroll_content)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 添加标题
        title_label = QLabel("图像预处理")
        title_label.setFont(QFont('微软雅黑', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 创建源文件夹选择组
        source_group = QGroupBox("源图片文件夹")
        source_layout = QGridLayout()
        
        self.source_path_edit = QLineEdit()
        self.source_path_edit.setReadOnly(True)
        self.source_path_edit.setPlaceholderText("请选择包含原始图片的文件夹")
        
        source_btn = QPushButton("浏览...")
        source_btn.clicked.connect(self.select_source_folder)
        
        source_layout.addWidget(QLabel("源文件夹:"), 0, 0)
        source_layout.addWidget(self.source_path_edit, 0, 1)
        source_layout.addWidget(source_btn, 0, 2)
        
        source_group.setLayout(source_layout)
        main_layout.addWidget(source_group)
        
        # 创建输出文件夹选择组
        output_group = QGroupBox("输出文件夹")
        output_layout = QGridLayout()
        
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setReadOnly(True)
        self.output_path_edit.setPlaceholderText("请选择处理后图片的保存文件夹")
        
        output_btn = QPushButton("浏览...")
        output_btn.clicked.connect(self.select_output_folder)
        
        output_layout.addWidget(QLabel("输出文件夹:"), 0, 0)
        output_layout.addWidget(self.output_path_edit, 0, 1)
        output_layout.addWidget(output_btn, 0, 2)
        
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)
        
        # 创建预处理选项组
        options_group = QGroupBox("预处理选项")
        options_layout = QGridLayout()
        
        # 调整图像大小
        options_layout.addWidget(QLabel("调整图像大小:"), 0, 0)
        self.size_combo = QComboBox()
        self.size_combo.addItems(["224x224", "256x256", "299x299", "320x320", "384x384", "512x512", "自定义"])
        self.size_combo.currentTextChanged.connect(self.on_size_changed)
        options_layout.addWidget(self.size_combo, 0, 1)
        
        # 锁定长宽比
        self.keep_aspect_ratio = QCheckBox("锁定长宽比")
        self.keep_aspect_ratio.setChecked(False)
        self.keep_aspect_ratio.stateChanged.connect(self.on_aspect_ratio_changed)
        options_layout.addWidget(self.keep_aspect_ratio, 0, 2)
        
        # 自定义宽度和高度
        options_layout.addWidget(QLabel("宽:"), 1, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(32, 1024)
        self.width_spin.setValue(self.resize_width)
        self.width_spin.valueChanged.connect(self.on_width_changed)
        options_layout.addWidget(self.width_spin, 1, 1)
        
        options_layout.addWidget(QLabel("高:"), 2, 0)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(32, 1024)
        self.height_spin.setValue(self.resize_height)
        self.height_spin.valueChanged.connect(self.on_height_changed)
        options_layout.addWidget(self.height_spin, 2, 1)
        
        # 添加水平分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        options_layout.addWidget(separator, 3, 0, 1, 3)
        
        # 数据增强标题
        augment_title = QLabel("数据增强选项")
        augment_title.setFont(QFont('微软雅黑', 10, QFont.Bold))
        options_layout.addWidget(augment_title, 4, 0, 1, 3)
        
        # 增强模式选择
        options_layout.addWidget(QLabel("增强模式:"), 5, 0)
        
        # 创建单选按钮组
        mode_container = QWidget()
        mode_container_layout = QVBoxLayout(mode_container)
        mode_container_layout.setContentsMargins(0, 0, 0, 0)
        mode_container_layout.setSpacing(2)
        
        self.mode_group = QButtonGroup(self)
        self.combined_mode_radio = QRadioButton("组合模式")
        self.combined_mode_radio.setChecked(True)  # 默认选择组合模式
        combined_desc = QLabel("将所有勾选的增强方式组合应用到每张图片上（输出图片数量 = 原图数量）")
        combined_desc.setWordWrap(True)
        combined_desc.setStyleSheet("color: gray; font-size: 9pt;")
        
        self.separate_mode_radio = QRadioButton("独立模式")
        separate_desc = QLabel("为每种勾选的增强方式单独生成一张图片（输出图片数量 = 原图数量 × 勾选的增强数量）")
        separate_desc.setWordWrap(True)
        separate_desc.setStyleSheet("color: gray; font-size: 9pt;")
        
        self.mode_group.addButton(self.combined_mode_radio)
        self.mode_group.addButton(self.separate_mode_radio)
        
        mode_container_layout.addWidget(self.combined_mode_radio)
        mode_container_layout.addWidget(combined_desc)
        mode_container_layout.addSpacing(3)
        mode_container_layout.addWidget(self.separate_mode_radio)
        mode_container_layout.addWidget(separate_desc)
        
        options_layout.addWidget(mode_container, 5, 1, 1, 2)
        
        # 增强方法选择
        options_layout.addWidget(QLabel("增强方法:"), 6, 0, Qt.AlignTop)
        
        # 创建两列增强选项布局
        augment_container = QWidget()
        augment_container_layout = QHBoxLayout(augment_container)
        augment_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # 左侧增强选项
        augment_layout_left = QVBoxLayout()
        self.flip_horizontal_check = QCheckBox("水平翻转")
        self.flip_vertical_check = QCheckBox("垂直翻转")
        self.rotate_check = QCheckBox("随机旋转")
        self.random_crop_check = QCheckBox("随机裁剪")
        self.random_scale_check = QCheckBox("随机缩放")
        
        augment_layout_left.addWidget(self.flip_horizontal_check)
        augment_layout_left.addWidget(self.flip_vertical_check)
        augment_layout_left.addWidget(self.rotate_check)
        augment_layout_left.addWidget(self.random_crop_check)
        augment_layout_left.addWidget(self.random_scale_check)
        
        # 右侧增强选项
        augment_layout_right = QVBoxLayout()
        self.brightness_check = QCheckBox("亮度调整")
        self.contrast_check = QCheckBox("对比度调整")
        self.noise_check = QCheckBox("高斯噪声")
        self.blur_check = QCheckBox("高斯模糊")
        self.hue_check = QCheckBox("色相调整")
        
        augment_layout_right.addWidget(self.brightness_check)
        augment_layout_right.addWidget(self.contrast_check)
        augment_layout_right.addWidget(self.noise_check)
        augment_layout_right.addWidget(self.blur_check)
        augment_layout_right.addWidget(self.hue_check)
        
        # 将两列布局添加到容器中
        augment_container_layout.addLayout(augment_layout_left)
        augment_container_layout.addLayout(augment_layout_right)
        
        options_layout.addWidget(augment_container, 6, 1, 1, 2)
        
        # 增强强度控制
        options_layout.addWidget(QLabel("增强强度:"), 7, 0)
        self.aug_intensity = QDoubleSpinBox()
        self.aug_intensity.setRange(0.1, 1.0)
        self.aug_intensity.setValue(0.5)
        self.aug_intensity.setSingleStep(0.1)
        options_layout.addWidget(self.aug_intensity, 7, 1)
        intensity_info = QLabel("(数值越大，增强效果越明显)")
        intensity_info.setStyleSheet("color: gray; font-size: 9pt;")
        options_layout.addWidget(intensity_info, 7, 2)
        
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)
        
        # 创建预处理按钮
        self.preprocess_btn = QPushButton("开始预处理")
        self.preprocess_btn.clicked.connect(self.preprocess_images)
        self.preprocess_btn.setEnabled(False)
        main_layout.addWidget(self.preprocess_btn)
        
        # 添加弹性空间
        main_layout.addStretch()
    
    def select_source_folder(self):
        """选择源图片文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择源图片文件夹")
        if folder:
            self.source_folder = folder
            self.source_path_edit.setText(folder)
            self.check_preprocess_ready()
    
    def select_output_folder(self):
        """选择输出文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder:
            self.output_folder = folder
            self.output_path_edit.setText(folder)
            self.check_preprocess_ready()
    
    def check_preprocess_ready(self):
        """检查是否可以开始预处理"""
        self.preprocess_btn.setEnabled(bool(self.source_folder and self.output_folder))
    
    def on_size_changed(self, size_text):
        """当尺寸选择改变时"""
        if size_text == "自定义":
            self.width_spin.setEnabled(True)
            self.height_spin.setEnabled(True)
        else:
            self.width_spin.setEnabled(False)
            self.height_spin.setEnabled(False)
            try:
                width, height = size_text.split('x')
                self.resize_width = int(width)
                self.resize_height = int(height)
                self.width_spin.setValue(self.resize_width)
                self.height_spin.setValue(self.resize_height)
            except:
                pass
    
    def on_width_changed(self, new_width):
        """当宽度改变时"""
        self.resize_width = new_width
        # 如果不是自定义模式，则切换到自定义模式
        if self.size_combo.currentText() != "自定义":
            self.size_combo.blockSignals(True)
            self.size_combo.setCurrentText("自定义")
            self.size_combo.blockSignals(False)
            self.width_spin.setEnabled(True)
            self.height_spin.setEnabled(True)
        
        # 如果锁定长宽比，则更新高度
        if self.keep_aspect_ratio.isChecked() and hasattr(self, 'aspect_ratio'):
            self.height_spin.blockSignals(True)
            self.resize_height = int(new_width / self.aspect_ratio)
            self.height_spin.setValue(self.resize_height)
            self.height_spin.blockSignals(False)
    
    def on_height_changed(self, new_height):
        """当高度改变时"""
        self.resize_height = new_height
        # 如果不是自定义模式，则切换到自定义模式
        if self.size_combo.currentText() != "自定义":
            self.size_combo.blockSignals(True)
            self.size_combo.setCurrentText("自定义")
            self.size_combo.blockSignals(False)
            self.width_spin.setEnabled(True)
            self.height_spin.setEnabled(True)
        
        # 如果锁定长宽比，则更新宽度
        if self.keep_aspect_ratio.isChecked() and hasattr(self, 'aspect_ratio'):
            self.width_spin.blockSignals(True)
            self.resize_width = int(new_height * self.aspect_ratio)
            self.width_spin.setValue(self.resize_width)
            self.width_spin.blockSignals(False)
    
    def on_aspect_ratio_changed(self, state):
        """当长宽比锁定状态改变时"""
        if state == Qt.Checked:
            # 计算并存储当前的长宽比
            self.aspect_ratio = self.resize_width / self.resize_height
    
    def preprocess_images(self):
        """开始预处理图像"""
        # 收集预处理参数
        params = {
            'source_folder': self.source_folder,
            'output_folder': self.output_folder,
            'resize_width': self.resize_width,
            'resize_height': self.resize_height,
            'keep_aspect_ratio': self.keep_aspect_ratio.isChecked(),
            'augmentation_mode': 'combined' if self.combined_mode_radio.isChecked() else 'separate',
            'flip_horizontal': self.flip_horizontal_check.isChecked(),
            'flip_vertical': self.flip_vertical_check.isChecked(),
            'rotate': self.rotate_check.isChecked(),
            'random_crop': self.random_crop_check.isChecked(),
            'random_scale': self.random_scale_check.isChecked(),
            'brightness': self.brightness_check.isChecked(),
            'contrast': self.contrast_check.isChecked(),
            'noise': self.noise_check.isChecked(),
            'blur': self.blur_check.isChecked(),
            'hue': self.hue_check.isChecked(),
            'augmentation_intensity': self.aug_intensity.value()
        }
        
        # 发出预处理开始信号
        self.image_preprocessing_started.emit(params)
        self.update_status("开始图像预处理...")
        self.preprocess_btn.setEnabled(False) 