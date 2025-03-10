from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QPushButton,
                           QLabel, QFileDialog, QMessageBox, QProgressBar,
                           QHBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox,
                           QTabWidget, QListWidget, QListWidgetItem, QGridLayout,
                           QGroupBox, QRadioButton, QButtonGroup, QScrollArea,
                           QSizePolicy, QFrame, QSlider)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QPixmap, QImage
import os
import subprocess
import sys

class MainWindow(QMainWindow):
    # 定义信号
    data_processing_started = pyqtSignal()
    training_started = pyqtSignal()
    prediction_started = pyqtSignal()
    image_preprocessing_started = pyqtSignal(dict)
    annotation_started = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.data_folder = None
        self.image_path = None
        self.processed_folder = None
        self.annotation_folder = None
        self.current_image_index = 0
        self.image_list = []

    def init_ui(self):
        self.setWindowTitle('半导体芯片缺陷检测系统')
        self.setGeometry(100, 100, 1200, 900)

        # 创建中心部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 创建标题
        title_label = QLabel('半导体芯片缺陷检测系统')
        title_label.setFont(QFont('Arial', 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # 创建选项卡
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # 创建数据处理选项卡
        data_tab = QWidget()
        self.tab_widget.addTab(data_tab, "数据处理")
        self.create_data_processing_tab(data_tab)

        # 创建标注选项卡
        annotation_tab = QWidget()
        self.tab_widget.addTab(annotation_tab, "图片标注")
        self.create_annotation_tab(annotation_tab)

        # 创建训练选项卡
        train_tab = QWidget()
        self.tab_widget.addTab(train_tab, "模型训练")
        self.create_training_tab(train_tab)

        # 创建预测选项卡
        predict_tab = QWidget()
        self.tab_widget.addTab(predict_tab, "缺陷预测")
        self.create_prediction_tab(predict_tab)

        # 进度条
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        # 状态标签
        self.status_label = QLabel('就绪')
        self.status_label.setStyleSheet('color: green;')
        main_layout.addWidget(self.status_label)

    def create_data_processing_tab(self, parent):
        layout = QVBoxLayout(parent)
        
        # 1. 源文件夹选择
        source_group = QGroupBox("1. 选择源图片文件夹")
        source_layout = QVBoxLayout()
        
        source_select_layout = QHBoxLayout()
        self.select_source_btn = QPushButton('选择源文件夹')
        self.source_path_label = QLabel('未选择文件夹')
        source_select_layout.addWidget(self.select_source_btn)
        source_select_layout.addWidget(self.source_path_label)
        source_layout.addLayout(source_select_layout)
        
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)
        
        # 2. 图片预处理选项
        preprocess_group = QGroupBox("2. 图片预处理")
        preprocess_layout = QVBoxLayout()
        
        # 输出文件夹
        output_layout = QHBoxLayout()
        self.select_output_btn = QPushButton('选择输出根目录')
        self.output_path_label = QLabel('未选择目录')
        output_layout.addWidget(self.select_output_btn)
        output_layout.addWidget(self.output_path_label)
        preprocess_layout.addLayout(output_layout)
        
        # 预处理选项
        options_layout = QGridLayout()
        
        # 图片尺寸
        options_layout.addWidget(QLabel('目标尺寸:'), 0, 0)
        self.size_combo = QComboBox()
        self.size_combo.addItems(['224x224', '256x256', '299x299', '384x384', '512x512', '自定义'])
        options_layout.addWidget(self.size_combo, 0, 1)
        
        # 自定义尺寸
        self.custom_size_layout = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(32, 1024)
        self.width_spin.setValue(224)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(32, 1024)
        self.height_spin.setValue(224)
        self.custom_size_layout.addWidget(QLabel('宽:'))
        self.custom_size_layout.addWidget(self.width_spin)
        self.custom_size_layout.addWidget(QLabel('高:'))
        self.custom_size_layout.addWidget(self.height_spin)
        options_layout.addLayout(self.custom_size_layout, 1, 1)
        
        # 图片格式
        options_layout.addWidget(QLabel('目标格式:'), 2, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(['JPG', 'PNG', 'BMP'])
        options_layout.addWidget(self.format_combo, 2, 1)
        
        # 亮度调整
        options_layout.addWidget(QLabel('亮度调整:'), 3, 0)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-50, 50)
        self.brightness_slider.setValue(0)
        options_layout.addWidget(self.brightness_slider, 3, 1)
        
        # 对比度调整
        options_layout.addWidget(QLabel('对比度调整:'), 4, 0)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-50, 50)
        self.contrast_slider.setValue(0)
        options_layout.addWidget(self.contrast_slider, 4, 1)
        
        preprocess_layout.addLayout(options_layout)
        
        # 3. 数据集划分选项
        dataset_options_layout = QGridLayout()
        
        # 训练集比例选择
        dataset_options_layout.addWidget(QLabel('训练集比例:'), 0, 0)
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.1, 0.9)
        self.train_ratio_spin.setValue(0.8)
        self.train_ratio_spin.setSingleStep(0.1)
        dataset_options_layout.addWidget(self.train_ratio_spin, 0, 1)
        
        # 数据增强选项
        dataset_options_layout.addWidget(QLabel('数据增强:'), 1, 0)
        self.augmentation_combo = QComboBox()
        self.augmentation_combo.addItems(['基础', '中等', '强化'])
        dataset_options_layout.addWidget(self.augmentation_combo, 1, 1)
        
        preprocess_layout.addLayout(dataset_options_layout)
        
        # 处理按钮
        self.preprocess_btn = QPushButton('开始处理数据')
        self.preprocess_btn.setEnabled(False)
        preprocess_layout.addWidget(self.preprocess_btn)
        
        preprocess_group.setLayout(preprocess_layout)
        layout.addWidget(preprocess_group)
        
        # 处理结果信息
        result_group = QGroupBox("3. 处理结果")
        result_layout = QVBoxLayout()
        
        self.processed_info_label = QLabel('未处理任何数据')
        self.processed_info_label.setWordWrap(True)
        result_layout.addWidget(self.processed_info_label)
        
        # 进入标注按钮
        self.goto_annotation_btn = QPushButton('进入图片标注')
        self.goto_annotation_btn.setEnabled(False)
        result_layout.addWidget(self.goto_annotation_btn)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        # 绑定事件
        self.select_source_btn.clicked.connect(self.select_source_folder)
        self.select_output_btn.clicked.connect(self.select_output_folder)
        self.size_combo.currentTextChanged.connect(self.on_size_changed)
        self.preprocess_btn.clicked.connect(self.preprocess_images)
        self.goto_annotation_btn.clicked.connect(self.goto_annotation_tab)
        
        # 初始化UI状态
        self.on_size_changed(self.size_combo.currentText())

    def create_annotation_tab(self, parent):
        layout = QVBoxLayout(parent)
        
        # 选择预处理后的图片文件夹
        folder_layout = QHBoxLayout()
        self.select_processed_btn = QPushButton('选择预处理后的图片文件夹')
        self.processed_folder_label = QLabel('未选择文件夹')
        folder_layout.addWidget(self.select_processed_btn)
        folder_layout.addWidget(self.processed_folder_label)
        layout.addLayout(folder_layout)
        
        # 标注工具选择
        tools_group = QGroupBox("标注工具")
        tools_layout = QVBoxLayout()
        
        self.labelimg_radio = QRadioButton("LabelImg (矩形标注)")
        self.labelimg_radio.setChecked(True)
        self.labelme_radio = QRadioButton("LabelMe (多边形标注)")
        
        tools_layout.addWidget(self.labelimg_radio)
        tools_layout.addWidget(self.labelme_radio)
        
        tools_group.setLayout(tools_layout)
        layout.addWidget(tools_group)
        
        # 标注类别管理
        classes_group = QGroupBox("缺陷类别管理")
        classes_layout = QVBoxLayout()
        
        # 类别列表
        self.class_list = QListWidget()
        classes_layout.addWidget(self.class_list)
        
        # 添加/删除类别按钮
        class_btn_layout = QHBoxLayout()
        self.add_class_btn = QPushButton("添加类别")
        self.remove_class_btn = QPushButton("删除类别")
        class_btn_layout.addWidget(self.add_class_btn)
        class_btn_layout.addWidget(self.remove_class_btn)
        classes_layout.addLayout(class_btn_layout)
        
        classes_group.setLayout(classes_layout)
        layout.addWidget(classes_group)
        
        # 启动标注工具按钮
        self.start_annotation_btn = QPushButton("启动标注工具")
        self.start_annotation_btn.setEnabled(False)
        layout.addWidget(self.start_annotation_btn)
        
        # 绑定事件
        self.select_processed_btn.clicked.connect(self.select_processed_folder)
        self.add_class_btn.clicked.connect(self.add_defect_class)
        self.remove_class_btn.clicked.connect(self.remove_defect_class)
        self.start_annotation_btn.clicked.connect(self.start_annotation)

    def create_training_tab(self, parent):
        layout = QVBoxLayout(parent)
        
        # 选择标注数据文件夹
        folder_layout = QHBoxLayout()
        self.select_annotation_btn = QPushButton('选择标注数据文件夹')
        self.annotation_folder_label = QLabel('未选择文件夹')
        folder_layout.addWidget(self.select_annotation_btn)
        folder_layout.addWidget(self.annotation_folder_label)
        layout.addLayout(folder_layout)
        
        # 模型选择
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel('选择模型:'))
        self.model_combo = QComboBox()
        self.model_combo.addItems(['ResNet18', 'ResNet50', 'EfficientNet-B0', 'EfficientNet-B4'])
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        # 训练参数
        params_layout = QGridLayout()
        
        # 批次大小
        params_layout.addWidget(QLabel('批次大小:'), 0, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(32)
        params_layout.addWidget(self.batch_size_spin, 0, 1)

        # 学习率
        params_layout.addWidget(QLabel('学习率:'), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setDecimals(4)
        params_layout.addWidget(self.lr_spin, 1, 1)

        # 训练轮数
        params_layout.addWidget(QLabel('训练轮数:'), 2, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(20)
        params_layout.addWidget(self.epochs_spin, 2, 1)
        
        layout.addLayout(params_layout)

        # 训练控制按钮
        buttons_layout = QHBoxLayout()
        self.train_model_btn = QPushButton('开始训练')
        self.train_model_btn.setEnabled(False)
        self.stop_train_btn = QPushButton('停止训练')
        self.stop_train_btn.setEnabled(False)
        buttons_layout.addWidget(self.train_model_btn)
        buttons_layout.addWidget(self.stop_train_btn)
        layout.addLayout(buttons_layout)
        
        # 绑定事件
        self.select_annotation_btn.clicked.connect(self.select_annotation_folder)
        self.train_model_btn.clicked.connect(self.train_model)
        self.stop_train_btn.clicked.connect(self.stop_training)

    def create_prediction_tab(self, parent):
        layout = QVBoxLayout(parent)

        # 选择图片
        select_image_layout = QHBoxLayout()
        self.select_image_btn = QPushButton('选择图片')
        self.image_path_label = QLabel('未选择图片')
        select_image_layout.addWidget(self.select_image_btn)
        select_image_layout.addWidget(self.image_path_label)
        layout.addLayout(select_image_layout)

        # 预测结果显示区域
        result_group = QGroupBox("预测结果")
        result_layout = QVBoxLayout()
        
        # 图片显示
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        result_layout.addWidget(self.image_label)
        
        # 预测结果列表
        self.result_list = QListWidget()
        result_layout.addWidget(self.result_list)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        # 预测按钮
        self.predict_btn = QPushButton('开始预测')
        self.predict_btn.setEnabled(False)
        layout.addWidget(self.predict_btn)
        
        # 绑定事件
        self.select_image_btn.clicked.connect(self.select_image)
        self.predict_btn.clicked.connect(self.predict)

    def select_source_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择源图片文件夹")
        if folder:
            self.data_folder = folder
            self.source_path_label.setText(folder)
            self.update_status('已选择源图片文件夹')
            self.check_preprocess_ready()

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出根目录")
        if folder:
            self.output_folder = folder
            self.output_path_label.setText(folder)
            self.update_status('已选择输出根目录')
            self.check_preprocess_ready()

    def check_preprocess_ready(self):
        if hasattr(self, 'data_folder') and self.data_folder and hasattr(self, 'output_folder') and self.output_folder:
            self.preprocess_btn.setEnabled(True)
        else:
            self.preprocess_btn.setEnabled(False)

    def on_size_changed(self, size_text):
        if size_text == '自定义':
            self.width_spin.setEnabled(True)
            self.height_spin.setEnabled(True)
        else:
            self.width_spin.setEnabled(False)
            self.height_spin.setEnabled(False)
            if 'x' in size_text:
                width, height = size_text.split('x')
                self.width_spin.setValue(int(width))
                self.height_spin.setValue(int(height))

    def preprocess_images(self):
        if not self.data_folder or not self.output_folder:
            QMessageBox.warning(self, '警告', '请先选择源文件夹和输出根目录')
            return
            
        # 创建预处理目录和数据集目录
        self.preprocessed_folder = os.path.join(self.output_folder, 'preprocessed')
        self.dataset_folder = os.path.join(self.output_folder, 'dataset')
        os.makedirs(self.preprocessed_folder, exist_ok=True)
        os.makedirs(self.dataset_folder, exist_ok=True)
            
        # 获取预处理参数
        if self.size_combo.currentText() == '自定义':
            width = self.width_spin.value()
            height = self.height_spin.value()
        else:
            width, height = map(int, self.size_combo.currentText().split('x'))
            
        params = {
            'source_folder': self.data_folder,
            'target_folder': self.preprocessed_folder,
            'width': width,
            'height': height,
            'format': self.format_combo.currentText().lower(),
            'brightness': self.brightness_slider.value(),
            'contrast': self.contrast_slider.value(),
            'train_ratio': self.train_ratio_spin.value(),
            'augmentation_level': self.augmentation_combo.currentText(),
            'dataset_folder': self.dataset_folder
        }
        
        self.update_status('正在预处理图片...')
        self.image_preprocessing_started.emit(params)
        
        # 更新处理结果信息
        self.processed_folder = self.preprocessed_folder
        self.processed_info_label.setText(f'预处理完成。\n预处理图片保存在: {self.preprocessed_folder}\n数据集保存在: {self.dataset_folder}')
        self.goto_annotation_btn.setEnabled(True)
        
        # 更新标注界面的文件夹路径
        self.processed_folder_label.setText(self.preprocessed_folder)

    def select_processed_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择预处理后的图片文件夹")
        if folder:
            self.processed_folder = folder
            self.processed_folder_label.setText(folder)
            self.update_status('已选择预处理后的图片文件夹')
            self.start_annotation_btn.setEnabled(True)

    def add_defect_class(self):
        class_name, ok = QFileDialog.getSaveFileName(self, "添加缺陷类别", "", "文本文件 (*.txt)")
        if ok and class_name:
            class_name = os.path.basename(class_name).replace('.txt', '')
            item = QListWidgetItem(class_name)
            self.class_list.addItem(item)

    def remove_defect_class(self):
        selected_items = self.class_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, '警告', '请先选择要删除的类别')
            return
            
        for item in selected_items:
            self.class_list.takeItem(self.class_list.row(item))

    def start_annotation(self):
        if not self.processed_folder:
            QMessageBox.warning(self, '警告', '请先选择预处理后的图片文件夹')
            return
            
        # 保存类别文件
        classes = []
        for i in range(self.class_list.count()):
            classes.append(self.class_list.item(i).text())
            
        if not classes:
            QMessageBox.warning(self, '警告', '请先添加缺陷类别')
            return
            
        # 确定使用哪个标注工具
        if self.labelimg_radio.isChecked():
            tool = 'labelimg'
        else:
            tool = 'labelme'
            
        self.update_status(f'正在启动{tool}标注工具...')
        self.annotation_started.emit(self.processed_folder)

    def select_annotation_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择标注数据文件夹")
        if folder:
            self.annotation_folder = folder
            self.annotation_folder_label.setText(folder)
            self.update_status('已选择标注数据文件夹')
            self.train_model_btn.setEnabled(True)

    def process_data(self):
        if not self.processed_folder:
            QMessageBox.warning(self, '警告', '请先完成图片预处理')
            return
        self.update_status('正在处理数据...')
        self.data_processing_started.emit()

    def train_model(self):
        if not self.annotation_folder:
            QMessageBox.warning(self, '警告', '请先选择标注数据文件夹')
            return
        self.update_status('正在训练模型...')
        self.train_model_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)
        self.training_started.emit()

    def stop_training(self):
        self.update_status('正在停止训练...')
        self.train_model_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg)")
        if file_name:
            self.image_path = file_name
            self.image_path_label.setText(file_name)
            self.predict_btn.setEnabled(True)
            self.update_status('已选择图片')
            
            # 显示图片
            pixmap = QPixmap(file_name)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio)
                self.image_label.setPixmap(pixmap)

    def predict(self):
        if not self.image_path:
            QMessageBox.warning(self, '警告', '请先选择图片')
            return
        self.update_status('正在预测...')
        self.prediction_started.emit()

    def update_status(self, message):
        self.status_label.setText(message)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def update_prediction_result(self, result):
        self.result_list.clear()
        for pred in result['predictions']:
            item_text = f"{pred['class_name']}: {pred['probability']:.2f}%"
            self.result_list.addItem(item_text)

    def goto_annotation_tab(self):
        # 切换到标注选项卡
        self.tab_widget.setCurrentIndex(1)
        # 自动设置标注界面的文件夹路径
        self.processed_folder = self.preprocessed_folder
        self.processed_folder_label.setText(self.preprocessed_folder)
        self.start_annotation_btn.setEnabled(True)

    def apply_config(self, config):
        """应用配置"""
        try:
            # 应用UI配置
            ui_config = config.get('ui', {})
            width = ui_config.get('window_width', 1200)
            height = ui_config.get('window_height', 900)
            self.setGeometry(100, 100, width, height)
            
            # 应用预处理配置
            preprocess_config = config.get('preprocessing', {})
            default_size = preprocess_config.get('default_size', '224x224')
            default_format = preprocess_config.get('default_format', 'jpg').upper()
            default_train_ratio = preprocess_config.get('default_train_ratio', 0.8)
            default_augmentation = preprocess_config.get('default_augmentation', '基础')
            
            # 设置默认值
            index = self.size_combo.findText(default_size)
            if index >= 0:
                self.size_combo.setCurrentIndex(index)
            
            index = self.format_combo.findText(default_format)
            if index >= 0:
                self.format_combo.setCurrentIndex(index)
            
            self.train_ratio_spin.setValue(default_train_ratio)
            
            index = self.augmentation_combo.findText(default_augmentation)
            if index >= 0:
                self.augmentation_combo.setCurrentIndex(index)
            
            # 应用训练配置
            training_config = config.get('training', {})
            default_model = training_config.get('default_model', 'ResNet50')
            default_batch_size = training_config.get('default_batch_size', 32)
            default_lr = training_config.get('default_learning_rate', 0.001)
            default_epochs = training_config.get('default_epochs', 20)
            
            index = self.model_combo.findText(default_model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
            
            self.batch_size_spin.setValue(default_batch_size)
            self.lr_spin.setValue(default_lr)
            self.epochs_spin.setValue(default_epochs)
            
            # 应用缺陷类别配置
            defect_classes = config.get('defect_classes', [])
            self.class_list.clear()
            for class_name in defect_classes:
                self.class_list.addItem(class_name)
            
            # 应用路径配置
            paths_config = config.get('paths', {})
            # 这里可以设置默认路径，但通常不直接设置，而是在用户选择时提供默认路径
            
        except Exception as e:
            print(f"应用配置时出错: {str(e)}") 