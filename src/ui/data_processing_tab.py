from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QComboBox, QSpinBox, QGroupBox, QGridLayout,
                           QSizePolicy, QLineEdit, QCheckBox, QDoubleSpinBox, QRadioButton,
                           QButtonGroup, QToolTip, QFrame, QListWidget, QInputDialog, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QFont, QIcon
import os
from .base_tab import BaseTab
import json

class DataProcessingTab(BaseTab):
    """数据处理标签页，负责图像预处理功能"""
    
    # 定义信号
    image_preprocessing_started = pyqtSignal(dict)
    create_class_folders_signal = pyqtSignal(str, list)  # 添加创建类别文件夹信号
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent, main_window)
        self.source_folder = ""
        self.output_folder = ""
        self.resize_width = 224
        self.resize_height = 224
        self.defect_classes = []  # 初始化类别列表
        self.init_ui()
        
        # 尝试从配置文件中加载类别信息
        config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if 'default_classes' in config and config['default_classes']:
                        self.defect_classes = config['default_classes'].copy()
                        self.class_list.clear()
                        for class_name in self.defect_classes:
                            self.class_list.addItem(class_name)
                        print(f"DataProcessingTab.__init__: 从配置文件加载了{len(self.defect_classes)}个类别")
            except Exception as e:
                print(f"DataProcessingTab.__init__: 加载配置文件出错: {str(e)}")
                import traceback
                traceback.print_exc()
    
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
        
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)
        
        # 创建预处理按钮
        self.preprocess_btn = QPushButton("开始预处理")
        self.preprocess_btn.clicked.connect(self.preprocess_images)
        self.preprocess_btn.setEnabled(False)
        self.preprocess_btn.setMinimumWidth(200)  # 设置最小宽度
        self.preprocess_btn.setMinimumHeight(40)  # 设置最小高度
        main_layout.addWidget(self.preprocess_btn)
        
        # 添加弹性空间
        main_layout.addStretch()
    
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
            self.check_preprocess_ready()
    
    def check_preprocess_ready(self):
        """检查是否可以开始预处理"""
        print(f"检查预处理准备状态: source_folder='{self.source_folder}', output_folder='{self.output_folder}'")
        is_ready = bool(self.source_folder and self.output_folder)
        print(f"预处理按钮状态: {'启用' if is_ready else '禁用'}")
        self.preprocess_btn.setEnabled(is_ready)
    
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
            config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
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
            'check_class_folders': self.check_class_folders.isChecked()  # 添加类别文件夹检查状态
        }
        
        # 发出预处理开始信号
        self.image_preprocessing_started.emit(params)
        self.update_status("开始图像预处理...")
        self.preprocess_btn.setEnabled(False)
        
    def enable_preprocess_button(self):
        """重新启用预处理按钮"""
        self.preprocess_btn.setEnabled(True)
        self.update_status("预处理完成，可以再次开始新的预处理。")
        # 注意：弹窗提示已移至MainWindow.preprocessing_finished方法中
    
    def apply_config(self, config):
        """应用配置信息"""
        print(f"DataProcessingTab.apply_config被调用，配置内容: {config}")
        if config:
            # 应用类别配置
            if 'default_classes' in config and config['default_classes']:
                print(f"DataProcessingTab: 找到default_classes字段: {config['default_classes']}")
                self.defect_classes = config['default_classes'].copy()
                self.class_list.clear()
                for class_name in self.defect_classes:
                    self.class_list.addItem(class_name)
                print(f"DataProcessingTab: 已加载{len(self.defect_classes)}个类别到类别列表中，类别列表项数: {self.class_list.count()}")
            elif 'classes' in config and config['classes']:
                print(f"DataProcessingTab: 找到classes字段: {config['classes']}")
                self.defect_classes = config['classes'].copy()
                self.class_list.clear()
                for class_name in self.defect_classes:
                    self.class_list.addItem(class_name)
                print(f"DataProcessingTab: 已加载{len(self.defect_classes)}个类别到类别列表中，类别列表项数: {self.class_list.count()}")
            else:
                print("DataProcessingTab: 配置文件中未找到有效的类别信息") 