from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QComboBox, QSpinBox, QGroupBox, QGridLayout,
                           QSizePolicy, QLineEdit, QCheckBox, QDoubleSpinBox, QRadioButton,
                           QButtonGroup, QToolTip, QFrame, QListWidget, QInputDialog, QMessageBox,
                           QApplication, QProgressBar, QTextEdit)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QTimer, QThread
from PyQt5.QtGui import QFont, QIcon
import os
import sys
import subprocess
import platform
from .base_tab import BaseTab
import json
import shutil
from pathlib import Path

# 导入统一的配置路径工具
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))
from utils.config_path import get_config_file_path

class DataProcessingTab(BaseTab):
    """数据处理标签页，负责图像预处理功能"""
    
    # 定义信号
    image_preprocessing_started = pyqtSignal(dict)
    create_class_folders_signal = pyqtSignal(str, list)  # 添加创建类别文件夹信号
    folder_organize_started = pyqtSignal(dict)  # 添加文件夹整理信号
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent, main_window)
        self.source_folder = ""
        self.output_folder = ""
        self.resize_width = 224
        self.resize_height = 224
        self.defect_classes = []  # 初始化类别列表
        # 文件夹整理相关变量
        self.organize_source_folder = ""
        self.organize_target_folder = ""
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
        
        # 添加类别管理组 - 新增
        class_group = QGroupBox("图片类别管理")
        class_layout = QVBoxLayout()
        
        # 添加类别文件夹检查开关
        self.check_class_folders = QCheckBox("启用类别文件夹检查和预处理")
        self.check_class_folders.setChecked(True)  # 默认启用
        self.check_class_folders.setToolTip("目标检测任务不需要创建类别文件夹，可以关闭此选项")
        class_layout.addWidget(self.check_class_folders)
        
        # 添加类别提示
        class_tip = QLabel("请添加需要分类的图片类别，这些类别将用于创建源文件夹中的子文件夹结构。")
        class_tip.setWordWrap(True)
        class_tip.setStyleSheet("color: #666666; font-size: 9pt;")
        class_layout.addWidget(class_tip)
        
        # 添加类别列表
        self.class_list = QListWidget()
        self.class_list.setMinimumHeight(120)
        class_layout.addWidget(self.class_list)
        
        # 添加类别按钮组
        btn_layout = QHBoxLayout()
        
        # 添加类别按钮
        add_class_btn = QPushButton("添加类别")
        add_class_btn.clicked.connect(self.add_defect_class)
        btn_layout.addWidget(add_class_btn)
        
        # 删除类别按钮
        remove_class_btn = QPushButton("删除类别")
        remove_class_btn.clicked.connect(self.remove_defect_class)
        btn_layout.addWidget(remove_class_btn)
        
        # 加载默认类别按钮
        load_default_classes_btn = QPushButton("加载默认类别")
        load_default_classes_btn.clicked.connect(self.load_default_classes)
        btn_layout.addWidget(load_default_classes_btn)
        
        class_layout.addLayout(btn_layout)
        
        # 创建类别文件夹按钮
        create_folders_btn = QPushButton("在源文件夹中创建类别文件夹")
        create_folders_btn.setMinimumHeight(30)
        create_folders_btn.clicked.connect(self.create_class_folders)
        self.create_folders_btn = create_folders_btn  # 保存为实例变量
        class_layout.addWidget(create_folders_btn)
        
        # 添加说明
        folder_info = QLabel("创建类别文件夹后，请将原始图片放入相应文件夹中，再开始预处理。")
        folder_info.setWordWrap(True)
        folder_info.setStyleSheet("color: #666666; font-size: 9pt;")
        class_layout.addWidget(folder_info)
        
        class_group.setLayout(class_layout)
        main_layout.addWidget(class_group)
        
        # 连接勾选框信号
        self.check_class_folders.stateChanged.connect(self.on_check_class_folders_changed)
        
        # 创建文件夹整理组
        organize_group = QGroupBox("文件夹图片整理")
        organize_layout = QVBoxLayout()
        
        # 添加说明
        organize_tip = QLabel("将源文件夹中的多个子文件夹的图片移动或复制到目标文件夹中，支持批量整理操作。")
        organize_tip.setWordWrap(True)
        organize_tip.setStyleSheet("color: #666666; font-size: 9pt;")
        organize_layout.addWidget(organize_tip)
        
        # 源文件夹选择
        organize_source_layout = QHBoxLayout()
        organize_source_layout.addWidget(QLabel("源文件夹:"))
        self.organize_source_edit = QLineEdit()
        self.organize_source_edit.setReadOnly(True)
        self.organize_source_edit.setPlaceholderText("请选择包含多个子文件夹的源文件夹")
        organize_source_btn = QPushButton("浏览...")
        organize_source_btn.clicked.connect(self.select_organize_source_folder)
        organize_source_layout.addWidget(self.organize_source_edit)
        organize_source_layout.addWidget(organize_source_btn)
        organize_layout.addLayout(organize_source_layout)
        
        # 目标文件夹选择
        organize_target_layout = QHBoxLayout()
        organize_target_layout.addWidget(QLabel("目标文件夹:"))
        self.organize_target_edit = QLineEdit()
        self.organize_target_edit.setReadOnly(True)
        self.organize_target_edit.setPlaceholderText("请选择图片整理后的目标文件夹")
        organize_target_btn = QPushButton("浏览...")
        organize_target_btn.clicked.connect(self.select_organize_target_folder)
        organize_target_layout.addWidget(self.organize_target_edit)
        organize_target_layout.addWidget(organize_target_btn)
        organize_layout.addLayout(organize_target_layout)
        
        # 操作选项
        organize_options_layout = QHBoxLayout()
        self.organize_operation_group = QButtonGroup(self)
        self.organize_copy_radio = QRadioButton("复制")
        self.organize_move_radio = QRadioButton("移动")
        self.organize_copy_radio.setChecked(True)  # 默认选择复制
        self.organize_operation_group.addButton(self.organize_copy_radio, 0)
        self.organize_operation_group.addButton(self.organize_move_radio, 1)
        
        organize_options_layout.addWidget(QLabel("操作类型:"))
        organize_options_layout.addWidget(self.organize_copy_radio)
        organize_options_layout.addWidget(self.organize_move_radio)
        organize_options_layout.addStretch()
        
        # 添加保持文件夹结构选项
        self.keep_folder_structure = QCheckBox("保持子文件夹结构")
        self.keep_folder_structure.setChecked(True)
        self.keep_folder_structure.setToolTip("勾选后将在目标文件夹中重建源文件夹的子文件夹结构")
        organize_options_layout.addWidget(self.keep_folder_structure)
        
        organize_layout.addLayout(organize_options_layout)
        
        # 文件类型筛选
        organize_filter_layout = QHBoxLayout()
        organize_filter_layout.addWidget(QLabel("文件类型:"))
        self.organize_file_types = QComboBox()
        self.organize_file_types.addItems([
            "所有图片 (*.jpg *.jpeg *.png *.bmp *.tiff *.gif)",
            "JPEG (*.jpg *.jpeg)",
            "PNG (*.png)",
            "BMP (*.bmp)",
            "TIFF (*.tiff)",
            "所有文件 (*.*)"
        ])
        organize_filter_layout.addWidget(self.organize_file_types)
        organize_filter_layout.addStretch()
        organize_layout.addLayout(organize_filter_layout)
        
        # 进度条
        self.organize_progress = QProgressBar()
        self.organize_progress.setVisible(False)
        organize_layout.addWidget(self.organize_progress)
        
        # 日志显示
        self.organize_log = QTextEdit()
        self.organize_log.setMaximumHeight(100)
        self.organize_log.setVisible(False)
        organize_layout.addWidget(self.organize_log)
        
        # 按钮组
        organize_btn_layout = QHBoxLayout()
        self.organize_start_btn = QPushButton("开始整理")
        self.organize_start_btn.clicked.connect(self.start_folder_organize)
        self.organize_start_btn.setEnabled(False)
        self.organize_start_btn.setMinimumHeight(35)
        
        self.organize_preview_btn = QPushButton("预览操作")
        self.organize_preview_btn.clicked.connect(self.preview_folder_organize)
        self.organize_preview_btn.setEnabled(False)
        
        organize_btn_layout.addWidget(self.organize_preview_btn)
        organize_btn_layout.addWidget(self.organize_start_btn)
        organize_btn_layout.addStretch()
        
        organize_layout.addLayout(organize_btn_layout)
        
        organize_group.setLayout(organize_layout)
        main_layout.addWidget(organize_group)
        
        # 创建输出文件夹选择组
        output_group = QGroupBox("输出文件夹")
        output_layout = QGridLayout()
        
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setReadOnly(True)
        self.output_path_edit.setPlaceholderText("请选择处理后图片的保存文件夹")
        
        output_btn = QPushButton("浏览...")
        output_btn.clicked.connect(self.select_output_folder)
        
        # 添加打开输出文件夹按钮
        self.open_output_btn = QPushButton("打开文件夹")
        self.open_output_btn.clicked.connect(self.open_output_folder)
        self.open_output_btn.setEnabled(False)  # 初始状态禁用
        self.open_output_btn.setToolTip("打开输出文件夹查看预处理结果")
        
        output_layout.addWidget(QLabel("输出文件夹:"), 0, 0)
        output_layout.addWidget(self.output_path_edit, 0, 1)
        output_layout.addWidget(output_btn, 0, 2)
        output_layout.addWidget(self.open_output_btn, 0, 3)
        
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
        
        # 添加训练验证集比例控制
        options_layout.addWidget(QLabel("训练集比例:"), 8, 0)
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.5, 0.9)
        self.train_ratio_spin.setValue(0.8)
        self.train_ratio_spin.setSingleStep(0.05)
        options_layout.addWidget(self.train_ratio_spin, 8, 1)
        ratio_info = QLabel("(训练集占总数据的比例)")
        ratio_info.setStyleSheet("color: gray; font-size: 9pt;")
        options_layout.addWidget(ratio_info, 8, 2)
        
        # 添加保持类别平衡选项
        self.balance_classes_check = QCheckBox("保持类别平衡")
        self.balance_classes_check.setChecked(True)
        options_layout.addWidget(self.balance_classes_check, 9, 0, 1, 3)
        balance_info = QLabel("(确保训练集和验证集中包含所有类别，且每个类别的样本数量均衡)")
        balance_info.setWordWrap(True)
        balance_info.setStyleSheet("color: gray; font-size: 9pt;")
        options_layout.addWidget(balance_info, 10, 0, 1, 3)
        
        # 添加采样平衡选项
        self.use_sampling_check = QCheckBox("启用智能采样平衡")
        self.use_sampling_check.setChecked(False)
        self.use_sampling_check.stateChanged.connect(self.on_sampling_changed)
        options_layout.addWidget(self.use_sampling_check, 11, 0, 1, 3)
        sampling_info = QLabel("(自动分析类别分布，使用过采样/欠采样方法平衡样本数量)")
        sampling_info.setWordWrap(True)
        sampling_info.setStyleSheet("color: gray; font-size: 9pt;")
        options_layout.addWidget(sampling_info, 12, 0, 1, 3)
        
        # 采样策略选择
        options_layout.addWidget(QLabel("采样策略:"), 13, 0)
        self.sampling_strategy_combo = QComboBox()
        self.sampling_strategy_combo.addItems([
            "auto - 自动选择",
            "oversample - 过采样",
            "undersample - 欠采样", 
            "median - 中位数采样",
            "custom - 自定义"
        ])
        self.sampling_strategy_combo.setEnabled(False)
        options_layout.addWidget(self.sampling_strategy_combo, 13, 1, 1, 2)
        
        # 过采样方法选择
        options_layout.addWidget(QLabel("过采样方法:"), 14, 0)
        self.oversample_method_combo = QComboBox()
        self.oversample_method_combo.addItems([
            "augmentation - 传统数据增强",
            "duplication - 重复采样",
            "mixup - Mixup混合采样",
            "cutmix - CutMix裁剪混合",
            "interpolation - 特征插值采样",
            "adaptive - 自适应采样",
            "smart - 智能增强采样"
        ])
        self.oversample_method_combo.setEnabled(False)
        options_layout.addWidget(self.oversample_method_combo, 14, 1, 1, 2)
        
        # 欠采样方法选择
        options_layout.addWidget(QLabel("欠采样方法:"), 15, 0)
        self.undersample_method_combo = QComboBox()
        self.undersample_method_combo.addItems([
            "random - 随机采样",
            "cluster - 聚类采样（基于K-means）",
            "diversity - 多样性采样（最大化差异）",
            "quality - 质量采样（选择高质量图像）"
        ])
        self.undersample_method_combo.setEnabled(False)
        options_layout.addWidget(self.undersample_method_combo, 15, 1, 1, 2)
        
        # 自定义目标样本数（仅在custom策略时启用）
        options_layout.addWidget(QLabel("目标样本数:"), 16, 0)
        self.target_samples_spin = QSpinBox()
        self.target_samples_spin.setRange(10, 10000)
        self.target_samples_spin.setValue(100)
        self.target_samples_spin.setEnabled(False)
        options_layout.addWidget(self.target_samples_spin, 16, 1)
        target_info = QLabel("(仅在自定义策略时生效)")
        target_info.setStyleSheet("color: gray; font-size: 9pt;")
        options_layout.addWidget(target_info, 16, 2)
        
        # 高级增强选项
        self.use_enhanced_augmentation_check = QCheckBox("启用高级随机增强")
        self.use_enhanced_augmentation_check.setChecked(True)
        self.use_enhanced_augmentation_check.setEnabled(False)
        options_layout.addWidget(self.use_enhanced_augmentation_check, 17, 0, 1, 3)
        enhanced_info = QLabel("(使用更多连续随机参数，大幅降低重复概率，仅在数据增强过采样时生效)")
        enhanced_info.setWordWrap(True)
        enhanced_info.setStyleSheet("color: gray; font-size: 9pt;")
        options_layout.addWidget(enhanced_info, 18, 0, 1, 3)
        
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)
        
        # 创建预处理按钮
        button_layout = QHBoxLayout()
        
        self.preprocess_btn = QPushButton("开始预处理")
        self.preprocess_btn.clicked.connect(self.preprocess_images)
        self.preprocess_btn.setEnabled(False)
        self.preprocess_btn.setMinimumWidth(200)  # 设置最小宽度
        self.preprocess_btn.setMinimumHeight(40)  # 设置最小高度
        
        # 添加停止预处理按钮
        self.stop_preprocess_btn = QPushButton("停止预处理")
        self.stop_preprocess_btn.clicked.connect(self.stop_preprocessing)
        self.stop_preprocess_btn.setEnabled(False)
        self.stop_preprocess_btn.setMinimumWidth(120)
        self.stop_preprocess_btn.setMinimumHeight(40)
        
        button_layout.addWidget(self.preprocess_btn)
        button_layout.addWidget(self.stop_preprocess_btn)
        button_layout.addStretch()  # 添加弹性空间
        
        main_layout.addLayout(button_layout)
    
    def select_source_folder(self):
        """选择源图片文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择源图片文件夹")
        print(f"选择源文件夹: {folder}")
        if folder:
            self.source_folder = folder
            self.source_path_edit.setText(folder)
            print(f"设置源文件夹路径: {self.source_folder}")
            self.check_preprocess_ready()
    
    def select_output_folder(self):
        """选择输出文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        print(f"选择输出文件夹: {folder}")
        if folder:
            self.output_folder = folder
            self.output_path_edit.setText(folder)
            print(f"设置输出文件夹路径: {self.output_folder}")
            # 启用打开输出文件夹按钮
            self.open_output_btn.setEnabled(True)
            self.check_preprocess_ready()
    
    def check_preprocess_ready(self):
        """检查是否可以开始预处理"""
        print(f"检查预处理准备状态: source_folder='{self.source_folder}', output_folder='{self.output_folder}'")
        is_ready = bool(self.source_folder and self.output_folder)
        self.preprocess_btn.setEnabled(is_ready)
        # 强制更新按钮状态
        self.preprocess_btn.repaint()
        self.preprocess_btn.update()
        # 刷新整个标签页
        self.repaint()
        self.update()
        print(f"预处理按钮状态: {'启用' if is_ready else '禁用'}")
        print(f"预处理按钮isEnabled()状态: {self.preprocess_btn.isEnabled()}")
        return is_ready
    
    def add_defect_class(self):
        """添加缺陷类别"""
        class_name, ok = QInputDialog.getText(self, "添加类别", "请输入图片类别名称:")
        if ok and class_name:
            # 检查是否已存在
            if class_name in self.defect_classes:
                QMessageBox.warning(self, "警告", f"类别 '{class_name}' 已存在!")
                return
                
            self.defect_classes.append(class_name)
            self.class_list.addItem(class_name)
    
    def remove_defect_class(self):
        """删除缺陷类别"""
        current_item = self.class_list.currentItem()
        if current_item:
            class_name = current_item.text()
            self.defect_classes.remove(class_name)
            self.class_list.takeItem(self.class_list.row(current_item))
    
    def load_default_classes(self):
        """从配置加载默认缺陷类别"""
        try:
            config_file = get_config_file_path()
            print(f"DataProcessingTab.load_default_classes: 尝试从以下路径加载配置: {config_file}")
            
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if 'default_classes' in config and config['default_classes']:
                        default_classes = config['default_classes']
                        # 清空当前类别
                        self.defect_classes = []
                        self.class_list.clear()
                        
                        # 添加默认类别
                        for class_name in default_classes:
                            self.defect_classes.append(class_name)
                            self.class_list.addItem(class_name)
                        
                        print(f"DataProcessingTab.load_default_classes: 已加载 {len(default_classes)} 个默认类别: {default_classes}")
                        self.update_status(f"已加载 {len(default_classes)} 个默认类别")
                    else:
                        print("DataProcessingTab.load_default_classes: 配置文件中未找到default_classes字段")
                        self.update_status("未找到默认类别")
            else:
                print(f"DataProcessingTab.load_default_classes: 配置文件不存在: {config_file}")
                self.update_status("未找到配置文件")
                
            # 作为备选，尝试从主窗口获取ConfigLoader实例
            if not self.defect_classes and hasattr(self.main_window, 'config_loader'):
                default_classes = self.main_window.config_loader.get_defect_classes()
                if default_classes:
                    # 清空当前类别
                    self.defect_classes = []
                    self.class_list.clear()
                    
                    # 添加默认类别
                    for class_name in default_classes:
                        self.defect_classes.append(class_name)
                        self.class_list.addItem(class_name)
                    
                    print(f"DataProcessingTab.load_default_classes: 从ConfigLoader加载了 {len(default_classes)} 个默认类别")
                    self.update_status(f"已加载 {len(default_classes)} 个默认类别")
        except Exception as e:
            print(f"DataProcessingTab.load_default_classes: 加载默认类别时出错: {str(e)}")
            self.update_status(f"加载默认类别时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_class_folders(self):
        """在源文件夹中创建类别文件夹"""
        if not self.source_folder:
            QMessageBox.warning(self, "警告", "请先选择源图片文件夹!")
            return
            
        if not self.defect_classes:
            QMessageBox.warning(self, "警告", "请先添加至少一个图片类别!")
            return
            
        # 确认是否创建文件夹
        reply = QMessageBox.question(self, "确认创建文件夹", 
                                   f"将在 {self.source_folder} 中创建 {len(self.defect_classes)} 个类别文件夹，是否继续?",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No:
            return
            
        # 发出创建文件夹信号
        self.create_class_folders_signal.emit(self.source_folder, self.defect_classes)
        
        # 提示用户完成后续操作
        QMessageBox.information(self, "文件夹创建成功", 
                              "类别文件夹已创建，请将原始图片分别放入对应的类别文件夹中，然后开始预处理。")
    
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
    
    def on_check_class_folders_changed(self, state):
        """当类别文件夹检查开关状态改变时调用"""
        # 更新创建文件夹按钮的状态
        self.create_folders_btn.setEnabled(state == Qt.Checked)
        # 更新类别列表和相关按钮的状态
        self.class_list.setEnabled(state == Qt.Checked)
        for button in self.findChildren(QPushButton):
            if button.text() in ["添加类别", "删除类别", "加载默认类别"]:
                button.setEnabled(state == Qt.Checked)
                
    def on_sampling_changed(self, state):
        """当采样选项状态改变时调用"""
        is_enabled = state == 2  # Qt.Checked
        
        # 启用/禁用采样相关控件
        self.sampling_strategy_combo.setEnabled(is_enabled)
        self.oversample_method_combo.setEnabled(is_enabled)
        self.undersample_method_combo.setEnabled(is_enabled)
        self.use_enhanced_augmentation_check.setEnabled(is_enabled)
        
        # 检查是否为自定义策略
        if is_enabled:
            self.on_sampling_strategy_changed()
            
        # 如果启用采样，禁用传统类别平衡
        if is_enabled:
            self.balance_classes_check.setChecked(False)
            self.balance_classes_check.setEnabled(False)
        else:
            self.balance_classes_check.setEnabled(True)
            self.target_samples_spin.setEnabled(False)
            self.use_enhanced_augmentation_check.setEnabled(False)
            
        # 连接策略变化信号
        if not hasattr(self, '_strategy_connected'):
            self.sampling_strategy_combo.currentTextChanged.connect(self.on_sampling_strategy_changed)
            self._strategy_connected = True
            
    def on_sampling_strategy_changed(self):
        """当采样策略改变时调用"""
        strategy = self.sampling_strategy_combo.currentText()
        is_custom = strategy.startswith("custom")
        self.target_samples_spin.setEnabled(is_custom and self.use_sampling_check.isChecked())
    
    def preprocess_images(self):
        """开始预处理图像"""
        # 检查是否需要处理类别文件夹
        if self.check_class_folders.isChecked() and self.defect_classes and self.balance_classes_check.isChecked():
            # 检查源文件夹是否包含所有类别子文件夹
            missing_folders = []
            for class_name in self.defect_classes:
                class_folder = os.path.join(self.source_folder, class_name)
                if not os.path.exists(class_folder) or not os.path.isdir(class_folder):
                    missing_folders.append(class_name)
            
            if missing_folders:
                reply = QMessageBox.question(self, "缺少类别文件夹", 
                                           f"以下类别文件夹不存在: {', '.join(missing_folders)}。\n"
                                           "是否创建这些文件夹并继续？",
                                           QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    # 创建缺少的文件夹
                    for class_name in missing_folders:
                        os.makedirs(os.path.join(self.source_folder, class_name), exist_ok=True)
                else:
                    return
        
        # 收集预处理参数
        params = {
            'source_folder': self.source_folder,
            'target_folder': self.output_folder,
            'width': self.resize_width,
            'height': self.resize_height,
            'format': 'jpg',
            'brightness_value': 0,  # 重命名亮度调整值
            'contrast_value': 0,    # 重命名对比度调整值
            'train_ratio': self.train_ratio_spin.value(),
            'augmentation_level': '基础',
            'dataset_folder': os.path.join(self.output_folder, 'dataset'),
            'keep_aspect_ratio': self.keep_aspect_ratio.isChecked(),
            'augmentation_mode': 'combined' if self.combined_mode_radio.isChecked() else 'separate',
            'flip_horizontal': self.flip_horizontal_check.isChecked(),
            'flip_vertical': self.flip_vertical_check.isChecked(),
            'rotate': self.rotate_check.isChecked(),
            'random_crop': self.random_crop_check.isChecked(),
            'random_scale': self.random_scale_check.isChecked(),
            'brightness': self.brightness_check.isChecked(),  # 这是增强方法开关
            'contrast': self.contrast_check.isChecked(),      # 这是增强方法开关
            'noise': self.noise_check.isChecked(),
            'blur': self.blur_check.isChecked(),
            'hue': self.hue_check.isChecked(),
            'augmentation_intensity': self.aug_intensity.value(),
            'balance_classes': self.balance_classes_check.isChecked() and self.check_class_folders.isChecked(),  # 只在启用类别文件夹检查时才启用类别平衡
            'class_names': self.defect_classes if self.check_class_folders.isChecked() else [],  # 只在启用类别文件夹检查时才传递类别名称
            'check_class_folders': self.check_class_folders.isChecked(),  # 添加类别文件夹检查状态
            # 采样相关参数
            'use_sampling': self.use_sampling_check.isChecked() and self.check_class_folders.isChecked(),
            'sampling_strategy': self.sampling_strategy_combo.currentText().split(' - ')[0] if self.use_sampling_check.isChecked() else 'auto',
            'oversample_method': self.oversample_method_combo.currentText().split(' - ')[0] if self.use_sampling_check.isChecked() else 'augmentation',
            'undersample_method': self.undersample_method_combo.currentText().split(' - ')[0] if self.use_sampling_check.isChecked() else 'random',
            'target_samples_per_class': self.target_samples_spin.value() if self.use_sampling_check.isChecked() else 100,
            'use_enhanced_augmentation': self.use_enhanced_augmentation_check.isChecked() if self.use_sampling_check.isChecked() else False
        }
        
        # 发出预处理开始信号
        self.image_preprocessing_started.emit(params)
        self.update_status("开始图像预处理...")
        self.preprocess_btn.setEnabled(False)
        self.stop_preprocess_btn.setEnabled(True)  # 启用停止按钮
        
        # 添加安全定时器，确保即使信号连接有问题，按钮也会在一定时间后重新启用
        QTimer.singleShot(120000, self._ensure_button_enabled)  # 2分钟后强制重新启用按钮
        print("已设置安全定时器，2分钟后将强制重新启用按钮")
        
    def _ensure_button_enabled(self):
        """确保按钮被重新启用的安全方法"""
        if hasattr(self, 'preprocess_btn') and not self.preprocess_btn.isEnabled():
            print("安全定时器触发：强制重新启用预处理按钮")
            self.preprocess_btn.setEnabled(True)
            self.preprocess_btn.update()
            self.repaint()
            self.update()
            QApplication.processEvents()
    
    def enable_preprocess_button(self):
        """重新启用预处理按钮"""
        print("DataProcessingTab.enable_preprocess_button被调用，重新启用预处理按钮")
        if hasattr(self, 'preprocess_btn'):
            # 确保预处理条件满足
            if self.source_folder and self.output_folder:
                self.preprocess_btn.setEnabled(True)
                # 强制重绘按钮以确保视觉状态更新
                self.preprocess_btn.repaint()
                self.preprocess_btn.update()
                # 更新整个标签页以确保所有元素都被正确重绘
                self.repaint()
                self.update()
                print(f"预处理按钮状态: {'启用' if self.preprocess_btn.isEnabled() else '禁用'}")
            else:
                print("警告: 源文件夹或输出文件夹为空，无法启用预处理按钮")
        else:
            print("错误: 找不到预处理按钮对象")
        
        # 禁用停止按钮
        if hasattr(self, 'stop_preprocess_btn'):
            self.stop_preprocess_btn.setEnabled(False)
        
        # 确保打开文件夹按钮可用（如果输出文件夹存在）
        if hasattr(self, 'open_output_btn') and self.output_folder and os.path.exists(self.output_folder):
            self.open_output_btn.setEnabled(True)
        
        self.update_status("预处理完成，可以再次开始新的预处理。")
        # 注意：弹窗提示已移至MainWindow.preprocessing_finished方法中
    
    def _do_apply_config(self, config):
        """实现具体的配置应用逻辑 - 智能配置系统"""
        print(f"DataProcessingTab: 智能应用配置，包含 {len(config)} 个配置项")
        
        # 应用文件夹路径配置
        if 'default_source_folder' in config and config['default_source_folder']:
            print(f"DataProcessingTab: 应用源文件夹配置: {config['default_source_folder']}")
            self.source_folder = config['default_source_folder']
            if hasattr(self, 'source_path_edit'):
                self.source_path_edit.setText(config['default_source_folder'])
            
        if 'default_output_folder' in config and config['default_output_folder']:
            print(f"DataProcessingTab: 应用输出文件夹配置: {config['default_output_folder']}")
            self.output_folder = config['default_output_folder']
            if hasattr(self, 'output_path_edit'):
                self.output_path_edit.setText(config['default_output_folder'])
            # 如果输出文件夹存在，启用打开文件夹按钮
            if hasattr(self, 'open_output_btn') and os.path.exists(config['default_output_folder']):
                self.open_output_btn.setEnabled(True)
        
        # 应用类别配置
        if 'default_classes' in config and config['default_classes']:
            print(f"DataProcessingTab: 找到default_classes字段: {config['default_classes']}")
            self.defect_classes = config['default_classes'].copy()
            if hasattr(self, 'class_list'):
                self.class_list.clear()
                for class_name in self.defect_classes:
                    self.class_list.addItem(class_name)
            print(f"DataProcessingTab: 已加载{len(self.defect_classes)}个类别")
        elif 'classes' in config and config['classes']:
            print(f"DataProcessingTab: 找到classes字段: {config['classes']}")
            self.defect_classes = config['classes'].copy()
            if hasattr(self, 'class_list'):
                self.class_list.clear()
                for class_name in self.defect_classes:
                    self.class_list.addItem(class_name)
            print(f"DataProcessingTab: 已加载{len(self.defect_classes)}个类别")
        
        # 配置应用完成后，检查预处理准备状态
        if hasattr(self, 'check_preprocess_ready'):
            self.check_preprocess_ready()
        print("DataProcessingTab: 智能配置应用完成") 

    def stop_preprocessing(self):
        """停止预处理"""
        print("DataProcessingTab: 停止预处理")
        self.update_status("正在停止预处理...")
        self.preprocess_btn.setEnabled(False)
        self.stop_preprocess_btn.setEnabled(False)
        
        # 通知主窗口停止预处理线程
        if hasattr(self.main_window, 'preprocessing_thread') and self.main_window.preprocessing_thread:
            self.main_window.preprocessing_thread.stop_preprocessing()
            print("已发送停止预处理信号到线程")
    
    def open_output_folder(self):
        """打开输出文件夹"""
        if not self.output_folder or not os.path.exists(self.output_folder):
            QMessageBox.warning(self, "警告", "输出文件夹不存在或未设置！")
            return
            
        try:
            system = platform.system()
            if system == "Windows":
                # Windows系统使用explorer
                os.startfile(self.output_folder)
            elif system == "Darwin":
                # macOS系统使用open
                subprocess.run(["open", self.output_folder])
            else:
                # Linux系统使用xdg-open
                subprocess.run(["xdg-open", self.output_folder])
            print(f"已打开输出文件夹: {self.output_folder}")
            self.update_status(f"已打开文件夹: {self.output_folder}")
        except Exception as e:
            error_msg = f"无法打开文件夹: {str(e)}"
            print(error_msg)
            QMessageBox.critical(self, "错误", error_msg)
    
    # 文件夹整理相关方法
    def select_organize_source_folder(self):
        """选择文件夹整理的源文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择包含多个子文件夹的源文件夹")
        if folder:
            self.organize_source_folder = folder
            self.organize_source_edit.setText(folder)
            self.check_organize_ready()
    
    def select_organize_target_folder(self):
        """选择文件夹整理的目标文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择图片整理后的目标文件夹")
        if folder:
            self.organize_target_folder = folder
            self.organize_target_edit.setText(folder)
            self.check_organize_ready()
    
    def check_organize_ready(self):
        """检查文件夹整理是否准备就绪"""
        is_ready = bool(self.organize_source_folder and self.organize_target_folder)
        self.organize_start_btn.setEnabled(is_ready)
        self.organize_preview_btn.setEnabled(is_ready)
        return is_ready
    
    def get_file_extensions(self):
        """根据选择的文件类型获取文件扩展名列表"""
        file_type = self.organize_file_types.currentText()
        if "所有图片" in file_type:
            return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        elif "JPEG" in file_type:
            return ['.jpg', '.jpeg']
        elif "PNG" in file_type:
            return ['.png']
        elif "BMP" in file_type:
            return ['.bmp']
        elif "TIFF" in file_type:
            return ['.tiff']
        elif "所有文件" in file_type:
            return []  # 空列表表示所有文件
        else:
            return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    
    def scan_source_folder(self):
        """扫描源文件夹，获取所有符合条件的文件"""
        if not self.organize_source_folder or not os.path.exists(self.organize_source_folder):
            return []
        
        extensions = self.get_file_extensions()
        files_to_process = []
        
        try:
            for root, dirs, files in os.walk(self.organize_source_folder):
                for file in files:
                    if not extensions:  # 所有文件
                        files_to_process.append((root, file))
                    else:
                        file_ext = os.path.splitext(file)[1].lower()
                        if file_ext in extensions:
                            files_to_process.append((root, file))
        except Exception as e:
            print(f"扫描源文件夹时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"扫描源文件夹时出错: {str(e)}")
            return []
        
        return files_to_process
    
    def preview_folder_organize(self):
        """预览文件夹整理操作"""
        files_to_process = self.scan_source_folder()
        
        if not files_to_process:
            QMessageBox.information(self, "提示", "源文件夹中没有找到符合条件的文件。")
            return
        
        # 显示预览信息
        operation = "复制" if self.organize_copy_radio.isChecked() else "移动"
        keep_structure = self.keep_folder_structure.isChecked()
        
        preview_text = f"操作预览:\n\n"
        preview_text += f"操作类型: {operation}\n"
        preview_text += f"源文件夹: {self.organize_source_folder}\n"
        preview_text += f"目标文件夹: {self.organize_target_folder}\n"
        preview_text += f"保持文件夹结构: {'是' if keep_structure else '否'}\n"
        preview_text += f"文件类型: {self.organize_file_types.currentText()}\n"
        preview_text += f"找到文件数量: {len(files_to_process)}\n\n"
        
        # 显示前10个文件的路径示例
        preview_text += "文件示例:\n"
        for i, (root, file) in enumerate(files_to_process[:10]):
            rel_path = os.path.relpath(root, self.organize_source_folder)
            if keep_structure and rel_path != '.':
                target_path = os.path.join(self.organize_target_folder, rel_path, file)
            else:
                target_path = os.path.join(self.organize_target_folder, file)
            preview_text += f"{i+1}. {os.path.join(root, file)} -> {target_path}\n"
        
        if len(files_to_process) > 10:
            preview_text += f"... 还有 {len(files_to_process) - 10} 个文件\n"
        
        # 显示预览对话框
        msg = QMessageBox(self)
        msg.setWindowTitle("操作预览")
        msg.setText("文件夹整理操作预览")
        msg.setDetailedText(preview_text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
    
    def start_folder_organize(self):
        """开始文件夹整理"""
        if not self.check_organize_ready():
            QMessageBox.warning(self, "警告", "请先选择源文件夹和目标文件夹。")
            return
        
        # 确认操作
        operation = "复制" if self.organize_copy_radio.isChecked() else "移动"
        reply = QMessageBox.question(self, "确认操作", 
                                   f"确定要{operation}文件夹中的图片吗？\n\n"
                                   f"源文件夹: {self.organize_source_folder}\n"
                                   f"目标文件夹: {self.organize_target_folder}\n\n"
                                   f"注意：移动操作将删除原文件！",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply != QMessageBox.Yes:
            return
        
        # 准备参数
        params = {
            'source_folder': self.organize_source_folder,
            'target_folder': self.organize_target_folder,
            'operation': 'copy' if self.organize_copy_radio.isChecked() else 'move',
            'keep_structure': self.keep_folder_structure.isChecked(),
            'file_extensions': self.get_file_extensions()
        }
        
        # 显示进度条和日志
        self.organize_progress.setVisible(True)
        self.organize_log.setVisible(True)
        self.organize_progress.setValue(0)
        self.organize_log.clear()
        
        # 禁用按钮
        self.organize_start_btn.setEnabled(False)
        self.organize_preview_btn.setEnabled(False)
        
        # 启动文件夹整理线程
        self.organize_thread = FolderOrganizeThread(params)
        self.organize_thread.progress_updated.connect(self.update_organize_progress)
        self.organize_thread.log_updated.connect(self.update_organize_log)
        self.organize_thread.finished.connect(self.organize_finished)
        self.organize_thread.error_occurred.connect(self.organize_error)
        self.organize_thread.start()
        
        self.update_status("开始文件夹整理...")
    
    def update_organize_progress(self, value):
        """更新文件夹整理进度"""
        self.organize_progress.setValue(value)
    
    def update_organize_log(self, message):
        """更新文件夹整理日志"""
        self.organize_log.append(message)
        # 自动滚动到底部
        cursor = self.organize_log.textCursor()
        cursor.movePosition(cursor.End)
        self.organize_log.setTextCursor(cursor)
    
    def organize_finished(self, result):
        """文件夹整理完成"""
        self.organize_progress.setValue(100)
        self.organize_start_btn.setEnabled(True)
        self.organize_preview_btn.setEnabled(True)
        
        success_count = result.get('success_count', 0)
        error_count = result.get('error_count', 0)
        
        self.update_status(f"文件夹整理完成！成功: {success_count}, 失败: {error_count}")
        
        # 显示完成信息
        QMessageBox.information(self, "完成", 
                              f"文件夹整理完成！\n\n"
                              f"成功处理: {success_count} 个文件\n"
                              f"处理失败: {error_count} 个文件\n\n"
                              f"详细信息请查看日志。")
    
    def organize_error(self, error_message):
        """文件夹整理出错"""
        self.organize_start_btn.setEnabled(True)
        self.organize_preview_btn.setEnabled(True)
        self.update_status(f"文件夹整理出错: {error_message}")
        QMessageBox.critical(self, "错误", f"文件夹整理出错: {error_message}")


class FolderOrganizeThread(QThread):
    """文件夹整理线程"""
    
    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.is_running = True
    
    def run(self):
        """执行文件夹整理"""
        try:
            source_folder = self.params['source_folder']
            target_folder = self.params['target_folder']
            operation = self.params['operation']
            keep_structure = self.params['keep_structure']
            file_extensions = self.params['file_extensions']
            
            # 确保目标文件夹存在
            os.makedirs(target_folder, exist_ok=True)
            
            # 扫描所有文件
            files_to_process = []
            for root, dirs, files in os.walk(source_folder):
                for file in files:
                    if not file_extensions:  # 所有文件
                        files_to_process.append((root, file))
                    else:
                        file_ext = os.path.splitext(file)[1].lower()
                        if file_ext in file_extensions:
                            files_to_process.append((root, file))
            
            if not files_to_process:
                self.log_updated.emit("没有找到符合条件的文件。")
                self.finished.emit({'success_count': 0, 'error_count': 0})
                return
            
            self.log_updated.emit(f"找到 {len(files_to_process)} 个文件需要处理。")
            
            success_count = 0
            error_count = 0
            
            for i, (root, file) in enumerate(files_to_process):
                if not self.is_running:
                    break
                
                try:
                    source_file = os.path.join(root, file)
                    
                    # 确定目标路径
                    if keep_structure:
                        rel_path = os.path.relpath(root, source_folder)
                        if rel_path == '.':
                            target_dir = target_folder
                        else:
                            target_dir = os.path.join(target_folder, rel_path)
                    else:
                        target_dir = target_folder
                    
                    # 确保目标目录存在
                    os.makedirs(target_dir, exist_ok=True)
                    
                    target_file = os.path.join(target_dir, file)
                    
                    # 处理重名文件
                    if os.path.exists(target_file):
                        base_name, ext = os.path.splitext(file)
                        counter = 1
                        while os.path.exists(target_file):
                            new_name = f"{base_name}_{counter}{ext}"
                            target_file = os.path.join(target_dir, new_name)
                            counter += 1
                    
                    # 执行操作
                    if operation == 'copy':
                        shutil.copy2(source_file, target_file)
                        action = "复制"
                    else:  # move
                        shutil.move(source_file, target_file)
                        action = "移动"
                    
                    success_count += 1
                    self.log_updated.emit(f"{action}成功: {file}")
                    
                except Exception as e:
                    error_count += 1
                    self.log_updated.emit(f"处理失败: {file} - {str(e)}")
                
                # 更新进度
                progress = int((i + 1) * 100 / len(files_to_process))
                self.progress_updated.emit(progress)
            
            self.log_updated.emit(f"处理完成！成功: {success_count}, 失败: {error_count}")
            self.finished.emit({'success_count': success_count, 'error_count': error_count})
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def stop(self):
        """停止处理"""
        self.is_running = False
 