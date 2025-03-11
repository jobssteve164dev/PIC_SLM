from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QPushButton,
                           QLabel, QFileDialog, QMessageBox, QProgressBar,
                           QHBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox,
                           QTabWidget, QListWidget, QListWidgetItem, QGridLayout,
                           QGroupBox, QRadioButton, QButtonGroup, QScrollArea,
                           QSizePolicy, QFrame, QSlider, QLineEdit, QInputDialog,
                           QCheckBox, QStackedWidget, QToolTip, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QPixmap, QImage
import os
import subprocess
import sys
from .annotation_widget import AnnotationWidget
from .training_visualization import TrainingVisualizationWidget, TensorBoardWidget

class MainWindow(QMainWindow):
    # 定义信号
    data_processing_started = pyqtSignal()
    training_started = pyqtSignal()
    prediction_started = pyqtSignal()
    image_preprocessing_started = pyqtSignal(dict)
    annotation_started = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        
        # 设置工具提示字体和样式
        QToolTip.setFont(QFont('微软雅黑', 9))
        
        # 使用样式表增强工具提示的可见性
        self.setStyleSheet("""
            QToolTip {
                background-color: #FFFFCC;
                color: #000000;
                border: 1px solid #76797C;
                padding: 5px;
                opacity: 200;
            }
        """)
        
        # 设置窗口标题和大小
        self.setWindowTitle('图片模型训练系统')
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化基础变量
        self.batch_size_spin = None
        self.lr_spin = None
        self.epochs_spin = None
        self.optimizer_combo = None
        self.lr_scheduler_combo = None
        self.weight_decay_spin = None
        self.pretrained_checkbox = None
        self.mixed_precision_checkbox = None
        self.early_stopping_checkbox = None
        self.patience_spin = None
        self.class_weight_combo = None
        self.tensorboard_checkbox = None
        self.accuracy_checkbox = None
        self.precision_checkbox = None
        self.recall_checkbox = None
        self.f1_checkbox = None
        self.confusion_matrix_checkbox = None
        self.detection_batch_size_spin = None
        self.detection_lr_spin = None
        self.detection_epochs_spin = None
        self.detection_optimizer_combo = None
        self.detection_lr_scheduler_combo = None
        self.detection_weight_decay_spin = None
        self.iou_threshold_spin = None
        self.nms_threshold_spin = None
        self.conf_threshold_spin = None
        self.anchor_size_combo = None
        self.use_fpn_checkbox = None
        self.map_checkbox = None
        self.map50_checkbox = None
        self.map75_checkbox = None
        self.precision_curve_checkbox = None
        self.detection_speed_checkbox = None
        self.training_status_label = None
        self.training_progress_bar = None
        self.train_model_btn = None
        self.stop_train_btn = None
        self.model_stack = None
        self.params_stack = None
        self.training_classification_radio = None
        self.training_detection_radio = None
        self.classification_model_combo = None
        self.detection_model_combo = None
        
        # 初始化UI
        self.init_ui()
        
        # 加载当前设置
        self.load_current_settings()
        
        self.data_folder = None
        self.image_path = None
        self.processed_folder = None
        self.annotation_folder = None
        self.current_image_index = 0
        self.image_list = []

    def init_ui(self):
        self.setWindowTitle('图片模型训练系统')
        self.setGeometry(100, 100, 1200, 900)

        # 创建中心部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 创建标题
        title_label = QLabel('图片模型训练系统')
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
        
        # 创建模型评估选项卡（原TensorBoard标签页）
        evaluation_tab = QWidget()
        self.tab_widget.addTab(evaluation_tab, "模型评估")
        self.create_evaluation_tab(evaluation_tab)

        # 创建预测选项卡
        predict_tab = QWidget()
        self.tab_widget.addTab(predict_tab, "缺陷预测")
        self.create_prediction_tab(predict_tab)
        
        # 创建设置选项卡
        settings_tab = QWidget()
        self.tab_widget.addTab(settings_tab, "设置")
        self.create_settings_tab(settings_tab)

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
        
        # 创建任务类型选择组
        task_group = QGroupBox("任务类型")
        task_layout = QHBoxLayout()
        
        self.classification_radio = QRadioButton("图像分类")
        self.classification_radio.setChecked(True)
        self.detection_radio = QRadioButton("目标检测")
        
        task_layout.addWidget(self.classification_radio)
        task_layout.addWidget(self.detection_radio)
        
        task_group.setLayout(task_layout)
        layout.addWidget(task_group)
        
        # 创建文件夹设置组
        folders_group = QGroupBox("文件夹设置")
        folders_layout = QVBoxLayout()
        
        # 待标注的图片文件夹
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("待标注的图片文件夹:"))
        self.processed_folder_label = QLabel('未选择文件夹')
        folder_layout.addWidget(self.processed_folder_label)
        self.select_processed_btn = QPushButton('选择文件夹')
        folder_layout.addWidget(self.select_processed_btn)
        folders_layout.addLayout(folder_layout)
        
        # 添加提示信息
        source_tip = QLabel("提示: 可以从数据处理流程获取，也可以手动选择")
        source_tip.setStyleSheet("color: gray; font-size: 10px;")
        folders_layout.addWidget(source_tip)
        
        # 标注输出目录
        annotation_folder_layout = QHBoxLayout()
        annotation_folder_layout.addWidget(QLabel("标注结果保存目录:"))
        self.annotation_folder_label = QLabel('未选择文件夹')
        annotation_folder_layout.addWidget(self.annotation_folder_label)
        self.select_annotation_folder_btn = QPushButton('选择文件夹')
        annotation_folder_layout.addWidget(self.select_annotation_folder_btn)
        folders_layout.addLayout(annotation_folder_layout)
        
        # 添加提示信息
        output_tip = QLabel("提示: 可以从数据处理流程获取，也可以手动选择")
        output_tip.setStyleSheet("color: gray; font-size: 10px;")
        folders_layout.addWidget(output_tip)
        
        folders_group.setLayout(folders_layout)
        layout.addWidget(folders_group)
        
        # 标注工具选择
        tools_group = QGroupBox("标注工具")
        tools_layout = QVBoxLayout()
        
        self.labelimg_radio = QRadioButton("内置标注工具 (矩形标注)")
        self.labelimg_radio.setChecked(True)
        self.labelme_radio = QRadioButton("外部LabelMe (多边形标注)")
        
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
        
        # 创建堆叠部件，用于切换不同任务类型的标注界面
        self.annotation_task_stack = QStackedWidget()
        
        # 分类任务标注界面
        classification_page = QWidget()
        classification_layout = QVBoxLayout(classification_page)
        
        # 分类任务说明
        classification_info = QLabel("分类任务标注说明：\n"
                                   "1. 选择待标注的图片文件夹和标注结果保存目录\n"
                                   "2. 在标注结果目录下会自动创建train和val子目录\n"
                                   "3. 在train和val目录下会为每个类别创建子文件夹\n"
                                   "4. 手动将图片拖放到对应类别的文件夹中完成标注")
        classification_info.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        classification_layout.addWidget(classification_info)
        
        # 创建分类标注工具按钮
        self.create_folders_btn = QPushButton("创建分类数据集目录结构")
        self.create_folders_btn.setEnabled(False)
        self.open_folders_btn = QPushButton("打开标注结果目录")
        self.open_folders_btn.setEnabled(False)
        
        classification_layout.addWidget(self.create_folders_btn)
        classification_layout.addWidget(self.open_folders_btn)
        
        # 目标检测任务标注界面
        detection_page = QWidget()
        detection_layout = QVBoxLayout(detection_page)
        
        # 创建堆叠部件，用于切换内置标注工具和启动外部工具按钮
        self.detection_tool_stack = QStackedWidget()
        
        # 创建启动外部标注工具的按钮页面
        external_page = QWidget()
        external_layout = QVBoxLayout(external_page)
        self.start_annotation_btn = QPushButton("启动外部标注工具")
        self.start_annotation_btn.setEnabled(False)
        external_layout.addWidget(self.start_annotation_btn)
        
        # 创建内置标注工具页面
        self.annotation_widget = AnnotationWidget()
        
        # 添加到目标检测工具堆叠部件
        self.detection_tool_stack.addWidget(self.annotation_widget)  # 索引0：内置标注工具
        self.detection_tool_stack.addWidget(external_page)  # 索引1：外部标注工具按钮
        
        # 默认显示内置标注工具
        self.detection_tool_stack.setCurrentIndex(0)
        
        detection_layout.addWidget(self.detection_tool_stack)
        
        # 添加到任务类型堆叠部件
        self.annotation_task_stack.addWidget(classification_page)  # 索引0：分类任务
        self.annotation_task_stack.addWidget(detection_page)  # 索引1：目标检测任务
        
        # 默认显示分类任务
        self.annotation_task_stack.setCurrentIndex(0)
        
        # 添加到主布局
        layout.addWidget(self.annotation_task_stack)
        
        # 绑定事件
        self.select_processed_btn.clicked.connect(self.select_processed_folder)
        self.select_annotation_folder_btn.clicked.connect(self.select_annotation_folder)
        self.add_class_btn.clicked.connect(self.add_defect_class)
        self.remove_class_btn.clicked.connect(self.remove_defect_class)
        self.start_annotation_btn.clicked.connect(self.start_annotation)
        self.create_folders_btn.clicked.connect(self.create_classification_folders)
        self.open_folders_btn.clicked.connect(self.open_annotation_folder)
        
        # 绑定任务类型选择事件
        self.classification_radio.toggled.connect(self.toggle_annotation_task)
        
        # 绑定标注工具选择事件
        self.labelimg_radio.toggled.connect(self.toggle_detection_tool)
        
        # 连接内置标注工具的信号
        self.annotation_widget.status_updated.connect(self.update_status)

    def create_training_tab(self, parent):
        """创建模型训练选项卡"""
        layout = QVBoxLayout(parent)
        
        # 添加帮助按钮
        help_btn = QPushButton("训练帮助")
        help_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        help_btn.clicked.connect(self.show_training_help)
        layout.addWidget(help_btn)
        
        # 添加说明标签
        description_label = QLabel(
            "在此页面配置模型训练参数。训练过程中的实时指标和可视化将显示在\"模型评估\"选项卡中。"
        )
        description_label.setWordWrap(True)
        description_label.setStyleSheet("color: #666; margin: 10px 0;")
        layout.addWidget(description_label)
        
        # 任务类型选择
        task_group = QGroupBox("任务类型")
        task_layout = QVBoxLayout()
        task_description = QLabel("选择要执行的任务类型：")
        task_description.setStyleSheet("color: #666;")
        task_layout.addWidget(task_description)
        
        radio_layout = QHBoxLayout()
        self.training_classification_radio = QRadioButton("图像分类")
        self.training_classification_radio.setChecked(True)
        self.training_classification_radio.setToolTip(
            "图像分类任务：将图像分为预定义的类别。\n"
            "适用于：缺陷类型识别、产品分类等场景。\n"
            "输出：每个图像属于哪个类别的概率分布。"
        )
        self.training_detection_radio = QRadioButton("目标检测")
        self.training_detection_radio.setToolTip(
            "目标检测任务：在图像中定位和识别多个目标。\n"
            "适用于：缺陷位置标注、多目标检测等场景。\n"
            "输出：每个目标的类别和边界框坐标。"
        )
        radio_layout.addWidget(self.training_classification_radio)
        radio_layout.addWidget(self.training_detection_radio)
        task_layout.addLayout(radio_layout)
        task_group.setLayout(task_layout)
        layout.addWidget(task_group)
        
        # 数据选择
        data_group = QGroupBox("数据选择")
        data_layout = QVBoxLayout()
        data_description = QLabel("选择包含已标注数据的文件夹：")
        data_description.setStyleSheet("color: #666;")
        data_layout.addWidget(data_description)
        
        folder_layout = QHBoxLayout()
        self.select_annotation_btn = QPushButton("选择标注数据文件夹")
        self.select_annotation_btn.setToolTip(
            "选择包含已标注数据的文件夹。\n"
            "对于分类任务，应包含按类别组织的图像。\n"
            "对于检测任务，应包含图像和对应的标注文件。"
        )
        self.select_annotation_btn.clicked.connect(self.select_annotation_folder)
        folder_layout.addWidget(self.select_annotation_btn)
        
        self.training_annotation_folder_label = QLabel("未选择文件夹")
        self.training_annotation_folder_label.setStyleSheet("color: #666;")
        folder_layout.addWidget(self.training_annotation_folder_label)
        
        data_layout.addLayout(folder_layout)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # 模型选择
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout()
        model_description = QLabel("选择要使用的模型架构：")
        model_description.setStyleSheet("color: #666;")
        model_layout.addWidget(model_description)
        
        self.model_stack = QStackedWidget()
        
        # 分类模型选择
        classification_model_layout = QHBoxLayout()
        self.classification_model_combo = QComboBox()
        self.classification_model_combo.addItems([
            "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
            "VGG16", "VGG19", "DenseNet121", "DenseNet169", "DenseNet201",
            "EfficientNet-B0", "EfficientNet-B1", "EfficientNet-B2",
            "MobileNetV2", "MobileNetV3", "Swin Transformer"
        ])
        self.classification_model_combo.setToolTip(
            "选择用于图像分类的模型架构：\n"
            "- ResNet系列：经典的残差网络，性能稳定\n"
            "- VGG系列：结构简单，易于理解\n"
            "- DenseNet系列：特征重用，参数量少\n"
            "- EfficientNet系列：计算效率高\n"
            "- MobileNet系列：轻量级，适合移动端\n"
            "- Swin Transformer：基于Transformer的视觉模型"
        )
        classification_model_layout.addWidget(self.classification_model_combo)
        classification_model_widget = QWidget()
        classification_model_widget.setLayout(classification_model_layout)
        self.model_stack.addWidget(classification_model_widget)
        
        # 检测模型选择
        detection_model_layout = QHBoxLayout()
        self.detection_model_combo = QComboBox()
        self.detection_model_combo.addItems([
            "YOLOv5", "YOLOv7", "YOLOv8", "Faster R-CNN", "SSD",
            "RetinaNet", "DETR", "Swin Transformer-DETR"
        ])
        self.detection_model_combo.setToolTip(
            "选择用于目标检测的模型架构：\n"
            "- YOLO系列：速度快，精度高\n"
            "- Faster R-CNN：经典两阶段检测器\n"
            "- SSD：单阶段检测器，平衡速度和精度\n"
            "- RetinaNet：解决类别不平衡问题\n"
            "- DETR：基于Transformer的端到端检测器"
        )
        detection_model_layout.addWidget(self.detection_model_combo)
        detection_model_widget = QWidget()
        detection_model_widget.setLayout(detection_model_layout)
        self.model_stack.addWidget(detection_model_widget)
        
        # 设置默认显示的模型选择页面
        self.model_stack.setCurrentIndex(0)  # 默认显示分类模型选择
        
        model_layout.addWidget(self.model_stack)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 参数设置
        params_group = QGroupBox("训练参数")
        params_layout = QVBoxLayout()
        params_description = QLabel("配置模型训练参数：")
        params_description.setStyleSheet("color: #666;")
        params_layout.addWidget(params_description)
        
        self.params_stack = QStackedWidget()
        
        # 分类参数页面
        classification_params = QWidget()
        classification_params_layout = QVBoxLayout()
        
        # 基本训练参数
        basic_group = QGroupBox("基本训练参数")
        basic_layout = QGridLayout()
        
        # 批次大小
        batch_size_label = QLabel("批次大小：")
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 256)
        self.batch_size_spin.setValue(32)
        self.batch_size_spin.setToolTip(
            "每次训练迭代处理的图像数量。\n"
            "较大的批次大小：\n"
            "- 优点：训练更稳定，速度更快\n"
            "- 缺点：需要更多显存，可能影响泛化性能\n"
            "较小的批次大小：\n"
            "- 优点：泛化性能更好，显存占用更少\n"
            "- 缺点：训练不稳定，速度较慢"
        )
        basic_layout.addWidget(batch_size_label, 0, 0)
        basic_layout.addWidget(self.batch_size_spin, 0, 1)
        
        # 学习率
        lr_label = QLabel("学习率：")
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setToolTip(
            "控制模型参数更新的步长。\n"
            "较大的学习率：\n"
            "- 优点：收敛更快\n"
            "- 缺点：可能错过最优解\n"
            "较小的学习率：\n"
            "- 优点：训练更稳定，可能找到更好的解\n"
            "- 缺点：收敛更慢"
        )
        basic_layout.addWidget(lr_label, 1, 0)
        basic_layout.addWidget(self.lr_spin, 1, 1)
        
        # 训练轮数
        epochs_label = QLabel("训练轮数：")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(20)
        self.epochs_spin.setToolTip(
            "整个训练数据集被完整训练的次数。\n"
            "轮数过少：\n"
            "- 模型可能欠拟合\n"
            "轮数过多：\n"
            "- 可能导致过拟合\n"
            "- 训练时间更长"
        )
        basic_layout.addWidget(epochs_label, 2, 0)
        basic_layout.addWidget(self.epochs_spin, 2, 1)
        
        # 优化器
        optimizer_label = QLabel("优化器：")
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD", "RMSprop", "AdamW"])
        self.optimizer_combo.setToolTip(
            "选择用于更新模型参数的优化算法：\n"
            "- Adam：自适应学习率，收敛快\n"
            "- SGD：随机梯度下降，基础优化器\n"
            "- RMSprop：自适应学习率，适合非平稳目标\n"
            "- AdamW：Adam的改进版本，更好的权重衰减"
        )
        basic_layout.addWidget(optimizer_label, 3, 0)
        basic_layout.addWidget(self.optimizer_combo, 3, 1)
        
        # 学习率调度器
        lr_scheduler_label = QLabel("学习率调度：")
        self.lr_scheduler_combo = QComboBox()
        self.lr_scheduler_combo.addItems(["固定", "Step", "Cosine", "Linear"])
        self.lr_scheduler_combo.setToolTip(
            "选择学习率的调整策略：\n"
            "- 固定：保持学习率不变\n"
            "- Step：按固定步长降低学习率\n"
            "- Cosine：余弦退火，周期性调整\n"
            "- Linear：线性降低学习率"
        )
        basic_layout.addWidget(lr_scheduler_label, 4, 0)
        basic_layout.addWidget(self.lr_scheduler_combo, 4, 1)
        
        # 权重衰减
        weight_decay_label = QLabel("权重衰减：")
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0, 0.1)
        self.weight_decay_spin.setValue(0.0001)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.weight_decay_spin.setToolTip(
            "L2正则化系数，用于防止过拟合。\n"
            "较大的权重衰减：\n"
            "- 更强的正则化效果\n"
            "- 可能影响模型性能\n"
            "较小的权重衰减：\n"
            "- 更弱的正则化效果\n"
            "- 模型可能过拟合"
        )
        basic_layout.addWidget(weight_decay_label, 5, 0)
        basic_layout.addWidget(self.weight_decay_spin, 5, 1)
        
        basic_group.setLayout(basic_layout)
        classification_params_layout.addWidget(basic_group)
        
        # 高级训练参数
        advanced_group = QGroupBox("高级训练参数")
        advanced_layout = QGridLayout()
        
        # 使用预训练模型
        self.pretrained_checkbox = QCheckBox("使用预训练模型")
        self.pretrained_checkbox.setChecked(True)
        self.pretrained_checkbox.setToolTip(
            "是否使用预训练权重初始化模型。\n"
            "优点：\n"
            "- 加快训练速度\n"
            "- 提高模型性能\n"
            "- 减少过拟合风险"
        )
        advanced_layout.addWidget(self.pretrained_checkbox, 0, 0)
        
        # 混合精度训练
        self.mixed_precision_checkbox = QCheckBox("使用混合精度训练")
        self.mixed_precision_checkbox.setChecked(True)
        self.mixed_precision_checkbox.setToolTip(
            "使用FP16（半精度）进行训练。\n"
            "优点：\n"
            "- 减少显存占用\n"
            "- 加快训练速度\n"
            "缺点：\n"
            "- 可能影响模型精度"
        )
        advanced_layout.addWidget(self.mixed_precision_checkbox, 0, 1)
        
        # 早停
        self.early_stopping_checkbox = QCheckBox("使用早停")
        self.early_stopping_checkbox.setChecked(True)
        self.early_stopping_checkbox.setToolTip(
            "当验证集性能不再提升时停止训练。\n"
            "优点：\n"
            "- 防止过拟合\n"
            "- 节省训练时间"
        )
        advanced_layout.addWidget(self.early_stopping_checkbox, 1, 0)
        
        # 早停耐心值
        patience_label = QLabel("早停耐心值：")
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 50)
        self.patience_spin.setValue(5)
        self.patience_spin.setToolTip(
            "在触发早停前等待的轮数。\n"
            "较大的耐心值：\n"
            "- 给模型更多机会改善\n"
            "- 训练时间更长\n"
            "较小的耐心值：\n"
            "- 更快停止训练\n"
            "- 可能过早停止"
        )
        advanced_layout.addWidget(patience_label, 1, 1)
        advanced_layout.addWidget(self.patience_spin, 1, 2)
        
        # 类别权重
        class_weight_label = QLabel("类别权重：")
        self.class_weight_combo = QComboBox()
        self.class_weight_combo.addItems(["均衡", "自动", "自定义"])
        self.class_weight_combo.setToolTip(
            "处理类别不平衡问题的方法：\n"
            "- 均衡：所有类别权重相等\n"
            "- 自动：根据类别频率自动计算权重\n"
            "- 自定义：手动设置每个类别的权重"
        )
        advanced_layout.addWidget(class_weight_label, 2, 0)
        advanced_layout.addWidget(self.class_weight_combo, 2, 1)
        
        advanced_group.setLayout(advanced_layout)
        classification_params_layout.addWidget(advanced_group)
        
        # 评估指标
        metrics_group = QGroupBox("评估指标")
        metrics_layout = QGridLayout()
        
        # 准确率
        self.accuracy_checkbox = QCheckBox("准确率")
        self.accuracy_checkbox.setChecked(True)
        self.accuracy_checkbox.setToolTip(
            "正确预测的样本数占总样本数的比例。\n"
            "优点：\n"
            "- 直观易懂\n"
            "缺点：\n"
            "- 在类别不平衡时可能具有误导性"
        )
        metrics_layout.addWidget(self.accuracy_checkbox, 0, 0)
        
        # 精确率
        self.precision_checkbox = QCheckBox("精确率")
        self.precision_checkbox.setChecked(True)
        self.precision_checkbox.setToolTip(
            "正确预测为正类的样本数占预测为正类的样本总数的比例。\n"
            "适用场景：\n"
            "- 假阳性代价高的任务"
        )
        metrics_layout.addWidget(self.precision_checkbox, 0, 1)
        
        # 召回率
        self.recall_checkbox = QCheckBox("召回率")
        self.recall_checkbox.setChecked(True)
        self.recall_checkbox.setToolTip(
            "正确预测为正类的样本数占实际为正类的样本总数的比例。\n"
            "适用场景：\n"
            "- 假阴性代价高的任务"
        )
        metrics_layout.addWidget(self.recall_checkbox, 1, 0)
        
        # F1分数
        self.f1_checkbox = QCheckBox("F1分数")
        self.f1_checkbox.setChecked(True)
        self.f1_checkbox.setToolTip(
            "精确率和召回率的调和平均数。\n"
            "优点：\n"
            "- 平衡精确率和召回率\n"
            "- 适合类别不平衡问题"
        )
        metrics_layout.addWidget(self.f1_checkbox, 1, 1)
        
        # 混淆矩阵
        self.confusion_matrix_checkbox = QCheckBox("混淆矩阵")
        self.confusion_matrix_checkbox.setChecked(True)
        self.confusion_matrix_checkbox.setToolTip(
            "展示模型在各个类别上的预测结果。\n"
            "用途：\n"
            "- 分析模型错误类型\n"
            "- 识别易混淆的类别"
        )
        metrics_layout.addWidget(self.confusion_matrix_checkbox, 2, 0)
        
        metrics_group.setLayout(metrics_layout)
        classification_params_layout.addWidget(metrics_group)
        
        classification_params.setLayout(classification_params_layout)
        self.params_stack.addWidget(classification_params)
        
        # 检测参数页面
        detection_params = QWidget()
        detection_params_layout = QVBoxLayout()
        
        # 基本训练参数
        detection_basic_group = QGroupBox("基本训练参数")
        detection_basic_layout = QGridLayout()
        
        # 批次大小
        detection_batch_size_label = QLabel("批次大小：")
        self.detection_batch_size_spin = QSpinBox()
        self.detection_batch_size_spin.setRange(1, 128)
        self.detection_batch_size_spin.setValue(16)
        self.detection_batch_size_spin.setToolTip(
            "每次训练迭代处理的图像数量。\n"
            "注意：\n"
            "- 检测任务通常使用较小的批次大小\n"
            "- 因为每张图像可能包含多个目标"
        )
        detection_basic_layout.addWidget(detection_batch_size_label, 0, 0)
        detection_basic_layout.addWidget(self.detection_batch_size_spin, 0, 1)
        
        # 学习率
        detection_lr_label = QLabel("学习率：")
        self.detection_lr_spin = QDoubleSpinBox()
        self.detection_lr_spin.setRange(0.00001, 0.1)
        self.detection_lr_spin.setValue(0.001)
        self.detection_lr_spin.setSingleStep(0.0001)
        self.detection_lr_spin.setToolTip(
            "控制模型参数更新的步长。\n"
            "建议：\n"
            "- 检测任务通常使用较小的学习率\n"
            "- 以确保定位和分类的稳定性"
        )
        detection_basic_layout.addWidget(detection_lr_label, 1, 0)
        detection_basic_layout.addWidget(self.detection_lr_spin, 1, 1)
        
        # 训练轮数
        detection_epochs_label = QLabel("训练轮数：")
        self.detection_epochs_spin = QSpinBox()
        self.detection_epochs_spin.setRange(1, 1000)
        self.detection_epochs_spin.setValue(50)
        self.detection_epochs_spin.setToolTip(
            "整个训练数据集被完整训练的次数。\n"
            "注意：\n"
            "- 检测任务通常需要更多轮数\n"
            "- 因为需要同时学习定位和分类"
        )
        detection_basic_layout.addWidget(detection_epochs_label, 2, 0)
        detection_basic_layout.addWidget(self.detection_epochs_spin, 2, 1)
        
        # 优化器
        detection_optimizer_label = QLabel("优化器：")
        self.detection_optimizer_combo = QComboBox()
        self.detection_optimizer_combo.addItems(["Adam", "SGD", "AdamW"])
        self.detection_optimizer_combo.setToolTip(
            "选择用于更新模型参数的优化算法：\n"
            "- Adam：自适应学习率，收敛快\n"
            "- SGD：随机梯度下降，基础优化器\n"
            "- AdamW：Adam的改进版本，更好的权重衰减"
        )
        detection_basic_layout.addWidget(detection_optimizer_label, 3, 0)
        detection_basic_layout.addWidget(self.detection_optimizer_combo, 3, 1)
        
        # 学习率调度器
        detection_lr_scheduler_label = QLabel("学习率调度：")
        self.detection_lr_scheduler_combo = QComboBox()
        self.detection_lr_scheduler_combo.addItems(["固定", "Step", "Cosine", "Linear"])
        self.detection_lr_scheduler_combo.setToolTip(
            "选择学习率的调整策略：\n"
            "- 固定：保持学习率不变\n"
            "- Step：按固定步长降低学习率\n"
            "- Cosine：余弦退火，周期性调整\n"
            "- Linear：线性降低学习率"
        )
        detection_basic_layout.addWidget(detection_lr_scheduler_label, 4, 0)
        detection_basic_layout.addWidget(self.detection_lr_scheduler_combo, 4, 1)
        
        # 权重衰减
        detection_weight_decay_label = QLabel("权重衰减：")
        self.detection_weight_decay_spin = QDoubleSpinBox()
        self.detection_weight_decay_spin.setRange(0, 0.1)
        self.detection_weight_decay_spin.setValue(0.0001)
        self.detection_weight_decay_spin.setSingleStep(0.0001)
        self.detection_weight_decay_spin.setToolTip(
            "L2正则化系数，用于防止过拟合。\n"
            "建议：\n"
            "- 检测任务通常使用较小的权重衰减\n"
            "- 以保持模型的定位能力"
        )
        detection_basic_layout.addWidget(detection_weight_decay_label, 5, 0)
        detection_basic_layout.addWidget(self.detection_weight_decay_spin, 5, 1)
        
        detection_basic_group.setLayout(detection_basic_layout)
        detection_params_layout.addWidget(detection_basic_group)
        
        # 检测特有参数
        detection_specific_group = QGroupBox("检测特有参数")
        detection_specific_layout = QGridLayout()
        
        # IoU阈值
        iou_threshold_label = QLabel("IoU阈值：")
        self.iou_threshold_spin = QDoubleSpinBox()
        self.iou_threshold_spin.setRange(0.1, 0.9)
        self.iou_threshold_spin.setValue(0.5)
        self.iou_threshold_spin.setSingleStep(0.05)
        self.iou_threshold_spin.setToolTip(
            "用于判断预测框和真实框是否匹配的阈值。\n"
            "较大的阈值：\n"
            "- 更严格的匹配标准\n"
            "- 可能降低召回率\n"
            "较小的阈值：\n"
            "- 更宽松的匹配标准\n"
            "- 可能提高召回率"
        )
        detection_specific_layout.addWidget(iou_threshold_label, 0, 0)
        detection_specific_layout.addWidget(self.iou_threshold_spin, 0, 1)
        
        # NMS阈值
        nms_threshold_label = QLabel("NMS阈值：")
        self.nms_threshold_spin = QDoubleSpinBox()
        self.nms_threshold_spin.setRange(0.1, 0.9)
        self.nms_threshold_spin.setValue(0.45)
        self.nms_threshold_spin.setSingleStep(0.05)
        self.nms_threshold_spin.setToolTip(
            "非极大值抑制的阈值。\n"
            "较大的阈值：\n"
            "- 保留更多重叠框\n"
            "- 可能产生重复检测\n"
            "较小的阈值：\n"
            "- 更严格的框筛选\n"
            "- 可能漏检目标"
        )
        detection_specific_layout.addWidget(nms_threshold_label, 1, 0)
        detection_specific_layout.addWidget(self.nms_threshold_spin, 1, 1)
        
        # 置信度阈值
        conf_threshold_label = QLabel("置信度阈值：")
        self.conf_threshold_spin = QDoubleSpinBox()
        self.conf_threshold_spin.setRange(0.1, 0.9)
        self.conf_threshold_spin.setValue(0.25)
        self.conf_threshold_spin.setSingleStep(0.05)
        self.conf_threshold_spin.setToolTip(
            "检测框的置信度阈值。\n"
            "较大的阈值：\n"
            "- 更可靠的检测结果\n"
            "- 可能漏检目标\n"
            "较小的阈值：\n"
            "- 更多的检测结果\n"
            "- 可能包含误检"
        )
        detection_specific_layout.addWidget(conf_threshold_label, 2, 0)
        detection_specific_layout.addWidget(self.conf_threshold_spin, 2, 1)
        
        # 锚框大小
        anchor_size_label = QLabel("锚框大小：")
        self.anchor_size_combo = QComboBox()
        self.anchor_size_combo.addItems(["自动", "小", "中", "大"])
        self.anchor_size_combo.setToolTip(
            "选择锚框的大小设置：\n"
            "- 自动：根据数据集自动计算\n"
            "- 小：适合检测小目标\n"
            "- 中：适合检测中等目标\n"
            "- 大：适合检测大目标"
        )
        detection_specific_layout.addWidget(anchor_size_label, 3, 0)
        detection_specific_layout.addWidget(self.anchor_size_combo, 3, 1)
        
        # 使用FPN
        self.use_fpn_checkbox = QCheckBox("使用FPN")
        self.use_fpn_checkbox.setChecked(True)
        self.use_fpn_checkbox.setToolTip(
            "是否使用特征金字塔网络。\n"
            "优点：\n"
            "- 提高多尺度检测能力\n"
            "- 改善小目标检测效果"
        )
        detection_specific_layout.addWidget(self.use_fpn_checkbox, 4, 0)
        
        detection_specific_group.setLayout(detection_specific_layout)
        detection_params_layout.addWidget(detection_specific_group)
        
        # 评估指标
        detection_metrics_group = QGroupBox("评估指标")
        detection_metrics_layout = QGridLayout()
        
        # mAP
        self.map_checkbox = QCheckBox("mAP")
        self.map_checkbox.setChecked(True)
        self.map_checkbox.setToolTip(
            "所有类别的平均精确率。\n"
            "优点：\n"
            "- 综合评估检测性能\n"
            "- 考虑精确率和召回率"
        )
        detection_metrics_layout.addWidget(self.map_checkbox, 0, 0)
        
        # mAP50
        self.map50_checkbox = QCheckBox("mAP50")
        self.map50_checkbox.setChecked(True)
        self.map50_checkbox.setToolTip(
            "IoU阈值为0.5时的mAP。\n"
            "用途：\n"
            "- 评估一般检测性能\n"
            "- 常用的评估指标"
        )
        detection_metrics_layout.addWidget(self.map50_checkbox, 0, 1)
        
        # mAP75
        self.map75_checkbox = QCheckBox("mAP75")
        self.map75_checkbox.setChecked(True)
        self.map75_checkbox.setToolTip(
            "IoU阈值为0.75时的mAP。\n"
            "用途：\n"
            "- 评估定位精度\n"
            "- 更严格的评估标准"
        )
        detection_metrics_layout.addWidget(self.map75_checkbox, 1, 0)
        
        # 精确率曲线
        self.precision_curve_checkbox = QCheckBox("精确率曲线")
        self.precision_curve_checkbox.setChecked(True)
        self.precision_curve_checkbox.setToolTip(
            "不同置信度阈值下的精确率变化曲线。\n"
            "用途：\n"
            "- 分析模型在不同置信度下的表现\n"
            "- 帮助选择最佳置信度阈值"
        )
        detection_metrics_layout.addWidget(self.precision_curve_checkbox, 1, 1)
        
        # 检测速度
        self.detection_speed_checkbox = QCheckBox("检测速度")
        self.detection_speed_checkbox.setChecked(True)
        self.detection_speed_checkbox.setToolTip(
            "模型处理单张图像的时间。\n"
            "用途：\n"
            "- 评估模型实时性能\n"
            "- 帮助选择适合部署的模型"
        )
        detection_metrics_layout.addWidget(self.detection_speed_checkbox, 2, 0)
        
        detection_metrics_group.setLayout(detection_metrics_layout)
        detection_params_layout.addWidget(detection_metrics_group)
        
        detection_params.setLayout(detection_params_layout)
        self.params_stack.addWidget(detection_params)
        
        # 设置默认显示的参数设置页面
        self.params_stack.setCurrentIndex(0)  # 默认显示分类参数设置
        
        params_layout.addWidget(self.params_stack)
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 训练控制
        control_group = QGroupBox("训练控制")
        control_layout = QVBoxLayout()
        
        # 训练状态
        status_layout = QHBoxLayout()
        self.training_status_label = QLabel("就绪")
        self.training_status_label.setStyleSheet("color: #666;")
        status_layout.addWidget(self.training_status_label)
        
        self.training_progress_bar = QProgressBar()
        self.training_progress_bar.setRange(0, 100)
        self.training_progress_bar.setValue(0)
        status_layout.addWidget(self.training_progress_bar)
        
        control_layout.addLayout(status_layout)
        
        # 训练按钮
        button_layout = QHBoxLayout()
        self.train_model_btn = QPushButton("开始训练")
        self.train_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.train_model_btn.clicked.connect(self.train_model)
        self.train_model_btn.setEnabled(False)  # 初始状态禁用，直到选择了标注文件夹
        button_layout.addWidget(self.train_model_btn)
        
        self.stop_train_btn = QPushButton("停止训练")
        self.stop_train_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.stop_train_btn.clicked.connect(self.stop_training)
        self.stop_train_btn.setEnabled(False)
        button_layout.addWidget(self.stop_train_btn)
        
        control_layout.addLayout(button_layout)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # 连接信号
        self.training_classification_radio.toggled.connect(self.toggle_training_task)
        # 不需要为detection_radio再次连接，因为它们是互斥的，一个切换会自动触发另一个
        
        # 初始化显示
        self.toggle_training_task(self.training_classification_radio.isChecked())

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
        
        # 创建标注数据目录
        self.annotation_folder = os.path.join(self.output_folder, 'annotations')
        
        os.makedirs(self.preprocessed_folder, exist_ok=True)
        os.makedirs(self.dataset_folder, exist_ok=True)
        os.makedirs(self.annotation_folder, exist_ok=True)
            
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
        self.processed_info_label.setText(f'预处理完成。\n预处理图片保存在: {self.preprocessed_folder}\n数据集保存在: {self.dataset_folder}\n标注数据将保存在: {self.annotation_folder}')
        self.goto_annotation_btn.setEnabled(True)
        
        # 更新标注界面的文件夹路径
        self.processed_folder_label.setText(self.preprocessed_folder)
        self.annotation_folder_label.setText(self.annotation_folder)

    def select_processed_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择待标注的图片文件夹")
        if folder:
            self.processed_folder = folder
            self.processed_folder_label.setText(folder)
            self.update_status('已选择待标注的图片文件夹')
            self.check_annotation_ready()
            
            # 如果选择了内置标注工具，则设置图像文件夹
            if self.labelimg_radio.isChecked():
                self.annotation_widget.set_image_folder(folder)

    def check_annotation_ready(self):
        """检查是否可以开始标注"""
        # 确保ready是布尔类型
        ready = bool(hasattr(self, 'processed_folder') and self.processed_folder and hasattr(self, 'annotation_folder') and self.annotation_folder)
        
        # 打印调试信息
        print(f"check_annotation_ready: ready = {ready}, type = {type(ready)}")
        
        if self.training_classification_radio.isChecked():
            # 分类任务
            if ready:
                self.create_folders_btn.setEnabled(True)
                self.open_folders_btn.setEnabled(True)
        else:
            # 目标检测任务
            # 更新外部标注工具按钮状态
            self.start_annotation_btn.setEnabled(ready)
            
            # 如果选择了内置标注工具，则更新内置标注工具的设置
            if self.labelimg_radio.isChecked() and ready:
                # 设置类别
                class_names = [self.class_list.item(i).text() for i in range(self.class_list.count())]
                if class_names:
                    self.annotation_widget.set_class_names(class_names)
                
                # 设置图像文件夹
                if hasattr(self, 'processed_folder') and self.processed_folder and os.path.exists(self.processed_folder):
                    print(f"check_annotation_ready: 设置图像文件夹 {self.processed_folder}")
                    self.annotation_widget.set_image_folder(self.processed_folder)
                
                # 设置输出文件夹
                if hasattr(self, 'annotation_folder') and self.annotation_folder:
                    print(f"check_annotation_ready: 设置输出文件夹 {self.annotation_folder}")
                    self.annotation_widget.set_output_folder(self.annotation_folder)

    def add_defect_class(self):
        """添加缺陷类别"""
        class_name, ok = QInputDialog.getText(self, '添加类别', '请输入缺陷类别名称:')
        if ok and class_name:
            self.class_list.addItem(class_name)
            
            # 更新内置标注工具的类别
            if self.labelimg_radio.isChecked():
                class_names = [self.class_list.item(i).text() for i in range(self.class_list.count())]
                self.annotation_widget.set_class_names(class_names)

    def remove_defect_class(self):
        """删除缺陷类别"""
        selected_items = self.class_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, '警告', '请先选择要删除的类别')
            return
            
        for item in selected_items:
            self.class_list.takeItem(self.class_list.row(item))
            
        # 更新内置标注工具的类别
        if self.labelimg_radio.isChecked():
            class_names = [self.class_list.item(i).text() for i in range(self.class_list.count())]
            self.annotation_widget.set_class_names(class_names)

    def start_annotation(self):
        if not self.processed_folder:
            QMessageBox.warning(self, '警告', '请先选择待标注的图片文件夹')
            return
            
        # 检查标注输出目录
        if not hasattr(self, 'annotation_folder') or not self.annotation_folder:
            QMessageBox.warning(self, '警告', '请先选择标注结果保存目录')
            return
            
        # 确保标注输出目录存在
        os.makedirs(self.annotation_folder, exist_ok=True)
            
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
        # 将图片文件夹路径传递给标注工具
        self.annotation_started.emit(self.processed_folder)

    def select_annotation_folder(self):
        """选择标注数据文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择标注数据文件夹")
        if folder:
            # 标准化路径格式，确保使用正斜杠
            folder = os.path.normpath(folder).replace('\\', '/')
            self.annotation_folder = folder
            
            # 更新所有相关标签
            if hasattr(self, 'annotation_folder_label'):
                self.annotation_folder_label.setText(folder)
            if hasattr(self, 'training_annotation_folder_label'):
                self.training_annotation_folder_label.setText(folder)
                
            # 启用训练按钮
            if hasattr(self, 'train_model_btn'):
                self.train_model_btn.setEnabled(True)
            
            # 根据当前任务类型启用相应的按钮
            if hasattr(self, 'training_classification_radio') and self.training_classification_radio.isChecked():
                # 分类任务
                if hasattr(self, 'create_folders_btn'):
                    self.create_folders_btn.setEnabled(True)
                if hasattr(self, 'open_folders_btn'):
                    self.open_folders_btn.setEnabled(True)
            else:
                # 目标检测任务
                if hasattr(self, 'start_annotation_btn'):
                    self.start_annotation_btn.setEnabled(True)
                    
            print(f"已选择标注数据文件夹: {folder}")
            return folder
        return None

    def process_data(self):
        if not self.processed_folder:
            QMessageBox.warning(self, '警告', '请先完成图片预处理')
            return
        self.update_status('正在处理数据...')
        self.data_processing_started.emit()

    def train_model(self):
        """开始模型训练"""
        # 检查是否选择了标注数据文件夹
        if not hasattr(self, 'annotation_folder') or not self.annotation_folder:
            QMessageBox.warning(self, '警告', '请先选择标注数据文件夹')
            return
            
        print(f"开始训练模型，使用标注数据文件夹: {self.annotation_folder}")
            
        # 获取当前任务类型
        is_classification = hasattr(self, 'training_classification_radio') and self.training_classification_radio.isChecked()
        
        # 检查数据集目录结构
        train_dir = os.path.join(self.annotation_folder, 'train')
        val_dir = os.path.join(self.annotation_folder, 'val')
        
        if not os.path.exists(train_dir):
            QMessageBox.warning(self, '警告', f'训练数据集目录不存在: {train_dir}\n请确保标注数据文件夹下有train和val子目录')
            return
            
        if not os.path.exists(val_dir):
            QMessageBox.warning(self, '警告', f'验证数据集目录不存在: {val_dir}\n请确保标注数据文件夹下有train和val子目录')
            return
            
        # 获取训练参数
        if is_classification:
            # 分类任务
            selected_model = self.classification_model_combo.currentText()
            task_type = 'classification'
            
            # 基本训练参数
            batch_size = self.batch_size_spin.value()
            learning_rate = self.lr_spin.value()
            epochs = self.epochs_spin.value()
            optimizer = self.optimizer_combo.currentText()
            lr_scheduler = self.lr_scheduler_combo.currentText()
            weight_decay = self.weight_decay_spin.value()
            
            # 高级训练参数
            use_pretrained = self.pretrained_checkbox.isChecked()
            use_mixed_precision = self.mixed_precision_checkbox.isChecked()
            use_early_stopping = self.early_stopping_checkbox.isChecked()
            patience = self.patience_spin.value()
            class_weight = self.class_weight_combo.currentText()
            use_tensorboard = self.tensorboard_checkbox.isChecked()
            
            # 评估指标
            metrics = []
            if self.accuracy_checkbox.isChecked():
                metrics.append('accuracy')
            if self.precision_checkbox.isChecked():
                metrics.append('precision')
            if self.recall_checkbox.isChecked():
                metrics.append('recall')
            if self.f1_checkbox.isChecked():
                metrics.append('f1')
            if self.confusion_matrix_checkbox.isChecked():
                metrics.append('confusion_matrix')
            
            # 检查数据集是否符合分类任务的要求
            class_names = [self.class_list.item(i).text() for i in range(self.class_list.count())]
            if not class_names:
                QMessageBox.warning(self, '警告', '请先添加缺陷类别')
                return
                
            # 检查每个类别目录是否存在
            missing_dirs = []
            for class_name in class_names:
                train_class_dir = os.path.join(train_dir, class_name)
                val_class_dir = os.path.join(val_dir, class_name)
                
                if not os.path.exists(train_class_dir):
                    missing_dirs.append(f"训练集缺少类别目录: {train_class_dir}")
                    
                if not os.path.exists(val_class_dir):
                    missing_dirs.append(f"验证集缺少类别目录: {val_class_dir}")
                    
            if missing_dirs:
                error_msg = "\n".join(missing_dirs)
                QMessageBox.warning(self, '警告', f'数据集目录结构不完整:\n{error_msg}\n\n请先创建完整的分类数据集目录结构')
                return
                
            # 创建训练配置字典
            training_config = {
                'model': selected_model,
                'task_type': task_type,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
                'weight_decay': weight_decay,
                'use_pretrained': use_pretrained,
                'use_mixed_precision': use_mixed_precision,
                'use_early_stopping': use_early_stopping,
                'patience': patience,
                'class_weight': class_weight,
                'metrics': metrics,
                'use_tensorboard': use_tensorboard
            }
            
        else:
            # 目标检测任务
            selected_model = self.detection_model_combo.currentText()
            task_type = 'detection'
            
            # 基本训练参数
            batch_size = self.detection_batch_size_spin.value()
            learning_rate = self.detection_lr_spin.value()
            epochs = self.detection_epochs_spin.value()
            optimizer = self.detection_optimizer_combo.currentText()
            lr_scheduler = self.detection_lr_scheduler_combo.currentText()
            weight_decay = self.detection_weight_decay_spin.value()
            
            # 检测特有参数
            iou_threshold = self.iou_threshold_spin.value()
            nms_threshold = self.nms_threshold_spin.value()
            conf_threshold = self.conf_threshold_spin.value()
            anchor_size = self.anchor_size_combo.currentText()
            use_fpn = self.use_fpn_checkbox.isChecked()
            
            # 评估指标
            metrics = []
            if self.map_checkbox.isChecked():
                metrics.append('mAP')
            if self.map50_checkbox.isChecked():
                metrics.append('mAP_50')
            if self.map75_checkbox.isChecked():
                metrics.append('mAP_75')
            if self.precision_curve_checkbox.isChecked():
                metrics.append('precision_recall_curve')
            if self.detection_speed_checkbox.isChecked():
                metrics.append('fps')
            
            # 检查数据集是否符合目标检测任务的要求
            # 这里简化处理，只检查目录是否存在
            # 实际应用中可能需要检查XML文件或其他标注文件
            xml_files = []
            for root, _, files in os.walk(train_dir):
                for file in files:
                    if file.lower().endswith('.xml'):
                        xml_files.append(os.path.join(root, file))
                        
            if not xml_files:
                QMessageBox.warning(self, '警告', f'训练数据集中未找到XML标注文件\n请确保已完成目标检测标注')
                return
                
            # 创建训练配置字典
            training_config = {
                'model': selected_model,
                'task_type': task_type,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
                'weight_decay': weight_decay,
                'iou_threshold': iou_threshold,
                'nms_threshold': nms_threshold,
                'conf_threshold': conf_threshold,
                'anchor_size': anchor_size,
                'use_fpn': use_fpn,
                'metrics': metrics,
                'use_tensorboard': use_tensorboard
            }
            
        # 更新状态并启动训练
        self.update_status(f'正在训练{selected_model}模型...')
        self.train_model_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)
        
        # 打印训练配置（调试用）
        print("训练配置:", training_config)
        
        # 发射训练信号，传递任务类型和模型名称
        self.training_started.emit()

    def stop_training(self):
        """停止模型训练"""
        # 实现停止训练的逻辑
        self.training_thread.stop()
        self.stop_train_btn.setEnabled(False)
        self.train_model_btn.setEnabled(True)
        self.update_training_status("训练已停止")

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
        """更新状态栏消息"""
        try:
            if hasattr(self, 'status_label') and self.status_label is not None:
                self.status_label.setText(message)
            print(f"状态更新: {message}")
        except Exception as e:
            print(f"更新状态时出错: {str(e)}")

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
        
        # 如果有预处理后的文件夹，则自动设置标注界面的文件夹路径
        if hasattr(self, 'preprocessed_folder') and self.preprocessed_folder:
            self.processed_folder = self.preprocessed_folder
            self.processed_folder_label.setText(self.preprocessed_folder)
            
            # 如果有标注输出目录，则自动设置
            if hasattr(self, 'annotation_folder') and self.annotation_folder:
                self.annotation_folder_label.setText(self.annotation_folder)
            
            # 启用标注按钮
            self.start_annotation_btn.setEnabled(True)
            
        # 从设置中加载缺陷类别
        try:
            from config_loader import ConfigLoader
            config_loader = ConfigLoader()
            config = config_loader.get_config()
            
            # 如果类别列表为空，则从设置中加载
            if self.class_list.count() == 0:
                defect_classes = config.get('defect_classes', [])
                for class_name in defect_classes:
                    self.class_list.addItem(class_name)
        except Exception as e:
            print(f"加载缺陷类别时出错: {str(e)}")

    def create_settings_tab(self, parent):
        """创建设置选项卡"""
        layout = QVBoxLayout(parent)
        
        # 路径设置组
        paths_group = QGroupBox("路径设置")
        paths_layout = QVBoxLayout()
        
        # 默认源文件夹
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("默认源文件夹:"))
        self.default_source_path = QLineEdit()
        self.default_source_path.setReadOnly(True)
        self.select_default_source_btn = QPushButton("选择")
        source_layout.addWidget(self.default_source_path)
        source_layout.addWidget(self.select_default_source_btn)
        paths_layout.addLayout(source_layout)
        source_tip = QLabel("提示: 存放原始图像的文件夹")
        source_tip.setStyleSheet("color: gray; font-size: 10px;")
        paths_layout.addWidget(source_tip)
        
        # 默认输出根目录
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("默认输出根目录:"))
        self.default_output_path = QLineEdit()
        self.default_output_path.setReadOnly(True)
        self.select_default_output_btn = QPushButton("选择")
        output_layout.addWidget(self.default_output_path)
        output_layout.addWidget(self.select_default_output_btn)
        paths_layout.addLayout(output_layout)
        output_tip = QLabel("提示: 存放预处理后的图像和训练结果的根目录")
        output_tip.setStyleSheet("color: gray; font-size: 10px;")
        paths_layout.addWidget(output_tip)
        
        # 默认待标注图片文件夹
        processed_layout = QHBoxLayout()
        processed_layout.addWidget(QLabel("默认待标注图片文件夹:"))
        self.default_processed_path = QLineEdit()
        self.default_processed_path.setReadOnly(True)
        self.select_default_processed_btn = QPushButton("选择")
        processed_layout.addWidget(self.default_processed_path)
        processed_layout.addWidget(self.select_default_processed_btn)
        paths_layout.addLayout(processed_layout)
        processed_tip = QLabel("提示: 存放待标注的图片文件夹，可以是预处理后的图片文件夹或其他文件夹")
        processed_tip.setStyleSheet("color: gray; font-size: 10px;")
        paths_layout.addWidget(processed_tip)
        
        # 默认标注数据文件夹
        annotation_layout = QHBoxLayout()
        annotation_layout.addWidget(QLabel("标注结果保存目录:"))
        self.default_annotation_path = QLineEdit()
        self.default_annotation_path.setReadOnly(True)
        self.select_default_annotation_btn = QPushButton("选择")
        annotation_layout.addWidget(self.default_annotation_path)
        annotation_layout.addWidget(self.select_default_annotation_btn)
        paths_layout.addLayout(annotation_layout)
        annotation_tip = QLabel("提示: 存放标注工具生成的标注文件的目录，与标注界面的标注结果保存目录相同")
        annotation_tip.setStyleSheet("color: gray; font-size: 10px;")
        paths_layout.addWidget(annotation_tip)
        
        # 自动设置标注文件夹为输出目录的子文件夹
        auto_annotation_layout = QHBoxLayout()
        self.auto_annotation_checkbox = QCheckBox("自动设置标注文件夹为输出目录的子文件夹")
        self.auto_annotation_checkbox.setChecked(True)
        self.auto_annotation_checkbox.stateChanged.connect(self.toggle_auto_annotation)
        auto_annotation_layout.addWidget(self.auto_annotation_checkbox)
        paths_layout.addLayout(auto_annotation_layout)
        
        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)
        
        # 缺陷类别设置组
        classes_group = QGroupBox("缺陷类别设置")
        classes_layout = QVBoxLayout()
        
        # 类别列表
        self.settings_class_list = QListWidget()
        classes_layout.addWidget(self.settings_class_list)
        
        # 添加/删除类别按钮
        class_btn_layout = QHBoxLayout()
        self.settings_add_class_btn = QPushButton("添加类别")
        self.settings_remove_class_btn = QPushButton("删除类别")
        class_btn_layout.addWidget(self.settings_add_class_btn)
        class_btn_layout.addWidget(self.settings_remove_class_btn)
        classes_layout.addLayout(class_btn_layout)
        
        classes_group.setLayout(classes_layout)
        layout.addWidget(classes_group)
        
        # 保存设置按钮
        self.save_settings_btn = QPushButton("保存设置")
        self.save_settings_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        layout.addWidget(self.save_settings_btn)
        
        # 绑定事件
        self.select_default_source_btn.clicked.connect(self.select_default_source_folder)
        self.select_default_output_btn.clicked.connect(self.select_default_output_folder)
        self.select_default_processed_btn.clicked.connect(self.select_default_processed_folder)
        self.select_default_annotation_btn.clicked.connect(self.select_default_annotation_folder)
        self.settings_add_class_btn.clicked.connect(self.settings_add_defect_class)
        self.settings_remove_class_btn.clicked.connect(self.settings_remove_defect_class)
        self.save_settings_btn.clicked.connect(self.save_settings)
        
        # 加载当前设置
        self.load_current_settings()
        
    def toggle_auto_annotation(self, state):
        """切换自动设置标注文件夹模式"""
        if state == Qt.Checked:
            # 如果已有输出目录，则自动设置标注文件夹
            if self.default_output_path.text():
                annotation_dir = os.path.join(self.default_output_path.text(), 'annotations')
                self.default_annotation_path.setText(annotation_dir)
            # 禁用手动选择按钮
            self.select_default_annotation_btn.setEnabled(False)
        else:
            # 启用手动选择按钮
            self.select_default_annotation_btn.setEnabled(True)
            
    def select_default_source_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择默认源文件夹")
        if folder:
            self.default_source_path.setText(folder)
            
    def select_default_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择默认输出根目录")
        if folder:
            self.default_output_path.setText(folder)
            # 如果启用了自动设置标注文件夹，则更新标注文件夹路径
            if self.auto_annotation_checkbox.isChecked():
                annotation_dir = os.path.join(folder, 'annotations')
                self.default_annotation_path.setText(annotation_dir)
            
    def select_default_processed_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择默认待标注图片文件夹")
        if folder:
            self.default_processed_path.setText(folder)
            
    def select_default_annotation_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择标注文件存储目录")
        if folder:
            self.default_annotation_path.setText(folder)
            
    def settings_add_defect_class(self):
        class_name, ok = QInputDialog.getText(self, "添加缺陷类别", "请输入缺陷类别名称:")
        if ok and class_name:
            self.settings_class_list.addItem(class_name)
            
    def settings_remove_defect_class(self):
        selected_items = self.settings_class_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择要删除的类别")
            return
            
        for item in selected_items:
            self.settings_class_list.takeItem(self.settings_class_list.row(item))
            
    def load_current_settings(self):
        """加载当前设置"""
        try:
            from config_loader import ConfigLoader
            config_loader = ConfigLoader()
            config = config_loader.get_config()
            
            # 加载路径设置
            paths = config.get('paths', {})
            self.default_source_path.setText(paths.get('default_source_dir', ''))
            self.default_output_path.setText(paths.get('default_output_dir', ''))
            self.default_processed_path.setText(paths.get('default_processed_dir', ''))
            
            # 处理自动标注文件夹模式
            auto_annotation = paths.get('auto_annotation', True)
            self.auto_annotation_checkbox.setChecked(auto_annotation)
            
            if auto_annotation and paths.get('default_output_dir', ''):
                # 自动设置标注文件夹为输出目录的子文件夹
                annotation_dir = os.path.join(paths.get('default_output_dir', ''), 'annotations')
                self.default_annotation_path.setText(annotation_dir)
                self.select_default_annotation_btn.setEnabled(False)
            else:
                # 使用配置中的标注文件夹或空
                self.default_annotation_path.setText(paths.get('default_annotation_dir', ''))
                self.select_default_annotation_btn.setEnabled(True)
            
            # 加载缺陷类别
            defect_classes = config.get('defect_classes', [])
            self.settings_class_list.clear()
            for class_name in defect_classes:
                self.settings_class_list.addItem(class_name)
                
            # 应用路径配置
            paths_config = config.get('paths', {})
            default_source_dir = paths_config.get('default_source_dir', '')
            default_output_dir = paths_config.get('default_output_dir', '')
            default_processed_dir = paths_config.get('default_processed_dir', '')
            default_annotation_dir = paths_config.get('default_annotation_dir', '')
            
            # 标准化路径格式
            if default_source_dir:
                default_source_dir = os.path.normpath(default_source_dir).replace('\\', '/')
            if default_output_dir:
                default_output_dir = os.path.normpath(default_output_dir).replace('\\', '/')
            if default_processed_dir:
                default_processed_dir = os.path.normpath(default_processed_dir).replace('\\', '/')
            if default_annotation_dir:
                default_annotation_dir = os.path.normpath(default_annotation_dir).replace('\\', '/')
            
            # 设置初始文件夹
            if default_source_dir and os.path.exists(default_source_dir):
                self.data_folder = default_source_dir
                if hasattr(self, 'source_path_label'):
                    self.source_path_label.setText(default_source_dir)
                
            if default_output_dir and os.path.exists(default_output_dir):
                self.output_folder = default_output_dir
                if hasattr(self, 'output_path_label'):
                    self.output_path_label.setText(default_output_dir)
                
            if default_processed_dir and os.path.exists(default_processed_dir):
                self.processed_folder = default_processed_dir
                if hasattr(self, 'processed_folder_label'):
                    self.processed_folder_label.setText(default_processed_dir)
                
            if default_annotation_dir and os.path.exists(default_annotation_dir):
                self.annotation_folder = default_annotation_dir
                if hasattr(self, 'annotation_folder_label'):
                    self.annotation_folder_label.setText(default_annotation_dir)
                if hasattr(self, 'training_annotation_folder_label'):
                    self.training_annotation_folder_label.setText(default_annotation_dir)
                    # 启用训练按钮
                    if hasattr(self, 'train_model_btn'):
                        self.train_model_btn.setEnabled(True)
            
            # 检查是否可以启用开始处理按钮
            self.check_preprocess_ready()
            
            # 检查是否可以启用标注按钮
            if hasattr(self, 'check_annotation_ready'):
                self.check_annotation_ready()
                
        except Exception as e:
            print(f"加载设置时出错: {str(e)}")
            
    def save_settings(self):
        """保存设置"""
        try:
            # 创建配置对象
            config = {
                'paths': {
                    'default_source_dir': self.default_source_path.text(),
                    'default_output_dir': self.default_output_path.text(),
                    'default_processed_dir': self.default_processed_path.text(),
                    'default_annotation_dir': self.default_annotation_path.text(),
                    'auto_annotation': self.auto_annotation_checkbox.isChecked()
                },
                'defect_classes': [self.settings_class_list.item(i).text() for i in range(self.settings_class_list.count())],
                'preprocessing': {
                    'default_size': self.size_combo.currentText(),
                    'default_format': self.format_combo.currentText().lower(),
                    'default_train_ratio': self.train_ratio_spin.value(),
                    'default_augmentation': self.augmentation_combo.currentText()
                },
                'training': {
                    'default_model': self.classification_model_combo.currentText(),
                    'default_batch_size': self.batch_size_spin.value(),
                    'default_learning_rate': self.lr_spin.value(),
                    'default_epochs': self.epochs_spin.value()
                }
            }
            
            # 保存配置
            from config_loader import ConfigLoader
            config_loader = ConfigLoader()
            if config_loader.save_config(config):
                QMessageBox.information(self, '成功', '设置已保存')
                
                # 更新当前路径
                if self.default_source_path.text():
                    self.data_folder = self.default_source_path.text()
                    self.source_path_label.setText(self.data_folder)
                
                if self.default_output_path.text():
                    self.output_folder = self.default_output_path.text()
                    self.output_path_label.setText(self.output_folder)
                
                if self.default_processed_path.text():
                    self.processed_folder = self.default_processed_path.text()
                    if hasattr(self, 'processed_folder_label'):
                        self.processed_folder_label.setText(self.processed_folder)
                    # 检查是否可以启用标注按钮
                    if hasattr(self, 'check_annotation_ready'):
                        self.check_annotation_ready()
                
                if self.default_annotation_path.text():
                    self.annotation_folder = self.default_annotation_path.text()
                    if hasattr(self, 'annotation_folder_label'):
                        self.annotation_folder_label.setText(self.annotation_folder)
                    if hasattr(self, 'training_annotation_folder_label'):
                        self.training_annotation_folder_label.setText(self.annotation_folder)
                
                # 更新类别列表
                self.class_list.clear()
                for i in range(self.settings_class_list.count()):
                    self.class_list.addItem(self.settings_class_list.item(i).text())
                    
                # 检查是否可以启用相关按钮
                self.check_preprocess_ready()
                if self.annotation_folder:
                    self.train_model_btn.setEnabled(True)
                
                self.update_status('设置已保存并应用')
            else:
                QMessageBox.warning(self, '错误', '保存设置失败')
        except Exception as e:
            QMessageBox.critical(self, '错误', f'保存设置时出错: {str(e)}')

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
            
            index = self.classification_model_combo.findText(default_model)
            if index >= 0:
                self.classification_model_combo.setCurrentIndex(index)
            
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
            default_source_dir = paths_config.get('default_source_dir', '')
            default_output_dir = paths_config.get('default_output_dir', '')
            default_processed_dir = paths_config.get('default_processed_dir', '')
            default_annotation_dir = paths_config.get('default_annotation_dir', '')
            
            # 标准化路径格式
            if default_source_dir:
                default_source_dir = os.path.normpath(default_source_dir).replace('\\', '/')
            if default_output_dir:
                default_output_dir = os.path.normpath(default_output_dir).replace('\\', '/')
            if default_processed_dir:
                default_processed_dir = os.path.normpath(default_processed_dir).replace('\\', '/')
            if default_annotation_dir:
                default_annotation_dir = os.path.normpath(default_annotation_dir).replace('\\', '/')
            
            # 设置初始文件夹
            if default_source_dir and os.path.exists(default_source_dir):
                self.data_folder = default_source_dir
                if hasattr(self, 'source_path_label'):
                    self.source_path_label.setText(default_source_dir)
                
            if default_output_dir and os.path.exists(default_output_dir):
                self.output_folder = default_output_dir
                if hasattr(self, 'output_path_label'):
                    self.output_path_label.setText(default_output_dir)
                
            if default_processed_dir and os.path.exists(default_processed_dir):
                self.processed_folder = default_processed_dir
                if hasattr(self, 'processed_folder_label'):
                    self.processed_folder_label.setText(default_processed_dir)
                
            if default_annotation_dir and os.path.exists(default_annotation_dir):
                self.annotation_folder = default_annotation_dir
                if hasattr(self, 'annotation_folder_label'):
                    self.annotation_folder_label.setText(default_annotation_dir)
                if hasattr(self, 'training_annotation_folder_label'):
                    self.training_annotation_folder_label.setText(default_annotation_dir)
                    # 启用训练按钮
                    if hasattr(self, 'train_model_btn'):
                        self.train_model_btn.setEnabled(True)
            
            # 检查是否可以启用开始处理按钮
            self.check_preprocess_ready()
            
            # 检查是否可以启用标注按钮
            if hasattr(self, 'check_annotation_ready'):
                self.check_annotation_ready()
                
            # 初始化内置标注工具
            if hasattr(self, 'annotation_widget'):
                # 设置类别
                class_names = [self.class_list.item(i).text() for i in range(self.class_list.count())]
                if class_names:
                    self.annotation_widget.set_class_names(class_names)
                
                # 设置图像文件夹
                if hasattr(self, 'processed_folder') and self.processed_folder and os.path.exists(self.processed_folder):
                    print(f"初始化标注工具：设置图像文件夹 {self.processed_folder}")
                    self.annotation_widget.set_image_folder(self.processed_folder)
                
                # 设置输出文件夹
                if hasattr(self, 'annotation_folder') and self.annotation_folder:
                    print(f"初始化标注工具：设置输出文件夹 {self.annotation_folder}")
                    self.annotation_widget.set_output_folder(self.annotation_folder)
                
        except Exception as e:
            print(f"应用配置时出错: {str(e)}")

    def toggle_annotation_task(self, checked):
        """切换任务类型"""
        if checked:  # 选择了分类任务
            self.annotation_task_stack.setCurrentIndex(0)
        else:  # 选择了目标检测任务
            self.annotation_task_stack.setCurrentIndex(1)

    def toggle_detection_tool(self, checked):
        """切换标注工具显示"""
        if checked:  # 选择了内置标注工具
            self.detection_tool_stack.setCurrentIndex(0)
            
            # 设置内置标注工具的路径和类别
            if hasattr(self, 'processed_folder') and self.processed_folder and os.path.exists(self.processed_folder):
                print(f"toggle_detection_tool: 设置图像文件夹 {self.processed_folder}")
                self.annotation_widget.set_image_folder(self.processed_folder)
                
            if hasattr(self, 'annotation_folder') and self.annotation_folder:
                print(f"toggle_detection_tool: 设置输出文件夹 {self.annotation_folder}")
                self.annotation_widget.set_output_folder(self.annotation_folder)
                
            # 设置类别
            class_names = [self.class_list.item(i).text() for i in range(self.class_list.count())]
            if class_names:
                self.annotation_widget.set_class_names(class_names)
        else:  # 选择了外部标注工具
            self.detection_tool_stack.setCurrentIndex(1)

    def create_classification_folders(self):
        """创建分类数据集目录结构"""
        if not self.annotation_folder:
            QMessageBox.warning(self, '警告', '请先选择标注结果保存目录')
            return
            
        try:
            # 创建train和val目录
            train_dir = os.path.join(self.annotation_folder, 'train')
            val_dir = os.path.join(self.annotation_folder, 'val')
            
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            
            # 为每个类别创建子目录
            class_names = [self.class_list.item(i).text() for i in range(self.class_list.count())]
            if not class_names:
                QMessageBox.warning(self, '警告', '请先添加缺陷类别')
                return
                
            for class_name in class_names:
                os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
                os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
                
            QMessageBox.information(self, '成功', '分类数据集目录结构创建成功')
            self.update_status('分类数据集目录结构创建成功')
            
            # 启用打开文件夹按钮
            self.open_folders_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, '错误', f'创建目录结构时出错: {str(e)}')
            
    def open_annotation_folder(self):
        """打开标注结果目录"""
        if not self.annotation_folder:
            QMessageBox.warning(self, '警告', '请先选择标注结果保存目录')
            return
            
        try:
            # 使用系统默认程序打开文件夹
            if sys.platform == 'win32':
                os.startfile(self.annotation_folder)
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', self.annotation_folder])
            else:  # Linux
                subprocess.run(['xdg-open', self.annotation_folder])
                
        except Exception as e:
            QMessageBox.critical(self, '错误', f'打开文件夹时出错: {str(e)}')

    def toggle_training_task(self, checked):
        """切换训练任务类型"""
        try:
            print(f"切换训练任务类型: {'分类' if checked else '检测'}")
            
            if not hasattr(self, 'model_stack') or not hasattr(self, 'params_stack'):
                print("错误: model_stack 或 params_stack 未初始化")
                return
                
            if checked:  # 选择了分类任务
                if self.model_stack.count() > 0:
                    self.model_stack.setCurrentIndex(0)  # 显示分类模型选择
                if self.params_stack.count() > 0:
                    self.params_stack.setCurrentIndex(0)  # 显示分类参数设置
                print("已切换到分类任务界面")
            else:  # 选择了目标检测任务
                if self.model_stack.count() > 1:
                    self.model_stack.setCurrentIndex(1)  # 显示检测模型选择
                if self.params_stack.count() > 1:
                    self.params_stack.setCurrentIndex(1)  # 显示检测参数设置
                print("已切换到检测任务界面")
                
            # 强制更新界面
            if hasattr(self, 'model_stack'):
                self.model_stack.update()
            if hasattr(self, 'params_stack'):
                self.params_stack.update()
                
            # 打印当前状态
            print(f"model_stack 当前索引: {self.model_stack.currentIndex()}")
            print(f"params_stack 当前索引: {self.params_stack.currentIndex()}")
            
        except Exception as e:
            print(f"切换任务类型时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def show_training_help(self):
        """显示训练参数和评估指标的详细说明对话框"""
        help_text = """
        <h2>训练参数说明</h2>
        
        <h3>基本训练参数</h3>
        <ul>
            <li><b>批次大小</b>：模型一次处理的图像数量。较大的批次可以加速训练，但需要更多内存。较小的批次可能提高模型泛化能力，但训练速度较慢。</li>
            <li><b>学习率</b>：控制模型参数更新的步长。较大的学习率可能导致训练不稳定。较小的学习率可能导致训练过慢或陷入局部最优。</li>
            <li><b>训练轮数</b>：模型遍历整个训练数据集的次数。轮数过少可能导致欠拟合，轮数过多可能导致过拟合。</li>
            <li><b>学习率调度</b>：控制训练过程中学习率的变化。固定学习率整个训练过程使用相同的学习率；阶梯下降在特定轮数后降低学习率；余弦退火学习率按余弦函数逐渐降低。</li>
            <li><b>优化器</b>：决定如何更新模型参数。Adam是自适应优化器，适用于大多数情况；SGD是随机梯度下降，可能需要更多调参但有时效果更好；RMSprop是自适应学习率优化器。</li>
            <li><b>权重衰减</b>：一种正则化技术，用于防止过拟合。较大的值会使模型更简单但可能欠拟合。较小的值允许模型更复杂但可能过拟合。</li>
        </ul>
        
        <h3>高级训练参数</h3>
        <ul>
            <li><b>使用预训练权重</b>：使用在大型数据集上预训练的权重。这通常可以加速训练并提高模型性能，特别是在训练数据较少的情况下。</li>
            <li><b>混合精度训练</b>：使用FP16和FP32混合计算，可以加速训练并减少内存使用，在支持的GPU上可提高训练速度2-3倍。</li>
            <li><b>使用早停</b>：早停机制会监控验证集性能，当性能不再提升时自动停止训练，有助于防止过拟合并节省训练时间。</li>
            <li><b>早停耐心值</b>：指定在停止训练前，模型可以连续多少轮没有改进。较大的值使训练更有耐心，较小的值使训练更早停止。</li>
            <li><b>类别权重</b>：用于处理不平衡的数据集。均衡为所有类别分配相同权重；自动计算根据类别频率自动计算权重；不使用则不应用类别权重。</li>
        </ul>
        
        <h3>检测特有参数</h3>
        <ul>
            <li><b>IoU阈值</b>：交并比阈值，用于确定预测框和真实框的匹配程度。较高的阈值要求更精确的定位，较低的阈值更宽松。</li>
            <li><b>NMS阈值</b>：非极大值抑制阈值，用于过滤重叠的检测框。较高的阈值保留更多可能重叠的框，较低的阈值更严格地过滤重叠框。</li>
            <li><b>置信度阈值</b>：决定检测结果的最低可信度。较高的阈值减少误报但可能增加漏报，较低的阈值减少漏报但可能增加误报。</li>
            <li><b>锚框尺寸</b>：检测算法预设的边界框。自动根据数据集统计自动生成；小目标优化适合检测小物体；大目标优化适合检测大物体；混合目标平衡不同尺寸的目标。</li>
            <li><b>使用特征金字塔</b>：特征金字塔网络通过结合不同尺度的特征，提高检测不同大小目标的能力，特别是小目标。</li>
        </ul>
        
        <h2>评估指标说明</h2>
        
        <h3>分类评估指标</h3>
        <ul>
            <li><b>准确率</b>：正确预测的样本比例。计算公式：正确预测数 / 总样本数。适用于类别平衡的数据集，但在类别不平衡时可能产生误导。</li>
            <li><b>精确率</b>：预测为正类的样本中真正属于正类的比例。计算公式：真正例 / (真正例 + 假正例)。高精确率意味着模型预测为正类的结果很可靠，误报率低。</li>
            <li><b>召回率</b>：真实正类样本中被正确预测为正类的比例。计算公式：真正例 / (真正例 + 假负例)。高召回率意味着模型能够捕获大部分正类样本，漏报率低。</li>
            <li><b>F1分数</b>：精确率和召回率的调和平均。计算公式：2 * (精确率 * 召回率) / (精确率 + 召回率)。当需要在精确率和召回率之间取得平衡时，F1分数是很好的选择。</li>
            <li><b>混淆矩阵</b>：详细展示各类别之间的预测情况，包含真正例、假正例、真负例和假负例的数量。帮助理解模型在哪些类别上表现好，哪些类别容易混淆。</li>
        </ul>
        
        <h3>检测评估指标</h3>
        <ul>
            <li><b>平均精度均值(mAP)</b>：目标检测的主要评估指标，计算不同IoU阈值和类别下的平均精度。值越高表示检测性能越好，同时考虑了定位和分类准确性。</li>
            <li><b>mAP@0.5</b>：IoU阈值为0.5时的平均精度，是COCO数据集的标准评估指标之一。这是一个相对宽松的评估标准，适合一般应用。</li>
            <li><b>mAP@0.75</b>：IoU阈值为0.75时的平均精度，这是一个更严格的评估标准，要求更精确的目标定位。在需要高精度定位的应用中很有用。</li>
            <li><b>精确率-召回率曲线</b>：展示不同置信度阈值下，精确率和召回率的变化关系。帮助选择最佳置信度阈值，平衡误报和漏报。</li>
            <li><b>检测速度(FPS)</b>：以每秒帧数衡量，反映模型在实际应用中的实时性能。对于需要实时处理的应用尤为重要。</li>
        </ul>
        """
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("训练参数和评估指标说明")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(help_text)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def create_tensorboard_tab(self, parent):
        layout = QVBoxLayout(parent)
        
        # 创建TensorBoard组件
        self.tensorboard_widget = TensorBoardWidget()
        layout.addWidget(self.tensorboard_widget)

    def train_model(self):
        """开始模型训练"""
        # 检查是否选择了标注数据文件夹
        if not hasattr(self, 'annotation_folder') or not self.annotation_folder:
            QMessageBox.warning(self, '警告', '请先选择标注数据文件夹')
            return
            
        print(f"开始训练模型，使用标注数据文件夹: {self.annotation_folder}")
            
        # 获取当前任务类型
        is_classification = hasattr(self, 'training_classification_radio') and self.training_classification_radio.isChecked()
        
        # 检查数据集目录结构
        train_dir = os.path.join(self.annotation_folder, 'train')
        val_dir = os.path.join(self.annotation_folder, 'val')
        
        if not os.path.exists(train_dir):
            QMessageBox.warning(self, '警告', f'训练数据集目录不存在: {train_dir}\n请确保标注数据文件夹下有train和val子目录')
            return
            
        if not os.path.exists(val_dir):
            QMessageBox.warning(self, '警告', f'验证数据集目录不存在: {val_dir}\n请确保标注数据文件夹下有train和val子目录')
            return
            
        # 获取训练参数
        if is_classification:
            # 分类任务
            selected_model = self.classification_model_combo.currentText()
            task_type = 'classification'
            
            # 基本训练参数
            batch_size = self.batch_size_spin.value()
            learning_rate = self.lr_spin.value()
            epochs = self.epochs_spin.value()
            optimizer = self.optimizer_combo.currentText()
            lr_scheduler = self.lr_scheduler_combo.currentText()
            weight_decay = self.weight_decay_spin.value()
            
            # 高级训练参数
            use_pretrained = self.pretrained_checkbox.isChecked()
            use_mixed_precision = self.mixed_precision_checkbox.isChecked()
            use_early_stopping = self.early_stopping_checkbox.isChecked()
            patience = self.patience_spin.value()
            class_weight = self.class_weight_combo.currentText()
            use_tensorboard = self.tensorboard_checkbox.isChecked()
            
            # 评估指标
            metrics = []
            if self.accuracy_checkbox.isChecked():
                metrics.append('accuracy')
            if self.precision_checkbox.isChecked():
                metrics.append('precision')
            if self.recall_checkbox.isChecked():
                metrics.append('recall')
            if self.f1_checkbox.isChecked():
                metrics.append('f1')
            if self.confusion_matrix_checkbox.isChecked():
                metrics.append('confusion_matrix')
            
            # 检查数据集是否符合分类任务的要求
            class_names = [self.class_list.item(i).text() for i in range(self.class_list.count())]
            if not class_names:
                QMessageBox.warning(self, '警告', '请先添加缺陷类别')
                return
                
            # 检查每个类别目录是否存在
            missing_dirs = []
            for class_name in class_names:
                train_class_dir = os.path.join(train_dir, class_name)
                val_class_dir = os.path.join(val_dir, class_name)
                
                if not os.path.exists(train_class_dir):
                    missing_dirs.append(f"训练集缺少类别目录: {train_class_dir}")
                    
                if not os.path.exists(val_class_dir):
                    missing_dirs.append(f"验证集缺少类别目录: {val_class_dir}")
                    
            if missing_dirs:
                error_msg = "\n".join(missing_dirs)
                QMessageBox.warning(self, '警告', f'数据集目录结构不完整:\n{error_msg}\n\n请先创建完整的分类数据集目录结构')
                return
                
            # 创建训练配置字典
            training_config = {
                'model': selected_model,
                'task_type': task_type,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
                'weight_decay': weight_decay,
                'use_pretrained': use_pretrained,
                'use_mixed_precision': use_mixed_precision,
                'use_early_stopping': use_early_stopping,
                'patience': patience,
                'class_weight': class_weight,
                'metrics': metrics,
                'use_tensorboard': use_tensorboard
            }
            
        else:
            # 目标检测任务
            selected_model = self.detection_model_combo.currentText()
            task_type = 'detection'
            
            # 基本训练参数
            batch_size = self.detection_batch_size_spin.value()
            learning_rate = self.detection_lr_spin.value()
            epochs = self.detection_epochs_spin.value()
            optimizer = self.detection_optimizer_combo.currentText()
            lr_scheduler = self.detection_lr_scheduler_combo.currentText()
            weight_decay = self.detection_weight_decay_spin.value()
            
            # 检测特有参数
            iou_threshold = self.iou_threshold_spin.value()
            nms_threshold = self.nms_threshold_spin.value()
            conf_threshold = self.conf_threshold_spin.value()
            anchor_size = self.anchor_size_combo.currentText()
            use_fpn = self.use_fpn_checkbox.isChecked()
            
            # 评估指标
            metrics = []
            if self.map_checkbox.isChecked():
                metrics.append('mAP')
            if self.map50_checkbox.isChecked():
                metrics.append('mAP_50')
            if self.map75_checkbox.isChecked():
                metrics.append('mAP_75')
            if self.precision_curve_checkbox.isChecked():
                metrics.append('precision_recall_curve')
            if self.detection_speed_checkbox.isChecked():
                metrics.append('fps')
            
            # 检查数据集是否符合目标检测任务的要求
            # 这里简化处理，只检查目录是否存在
            # 实际应用中可能需要检查XML文件或其他标注文件
            xml_files = []
            for root, _, files in os.walk(train_dir):
                for file in files:
                    if file.lower().endswith('.xml'):
                        xml_files.append(os.path.join(root, file))
                        
            if not xml_files:
                QMessageBox.warning(self, '警告', f'训练数据集中未找到XML标注文件\n请确保已完成目标检测标注')
                return
                
            # 创建训练配置字典
            training_config = {
                'model': selected_model,
                'task_type': task_type,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
                'weight_decay': weight_decay,
                'iou_threshold': iou_threshold,
                'nms_threshold': nms_threshold,
                'conf_threshold': conf_threshold,
                'anchor_size': anchor_size,
                'use_fpn': use_fpn,
                'metrics': metrics,
                'use_tensorboard': use_tensorboard
            }
            
        # 更新状态并启动训练
        self.update_status(f'正在训练{selected_model}模型...')
        self.train_model_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)
        
        # 打印训练配置（调试用）
        print("训练配置:", training_config)
        
        # 发射训练信号，传递任务类型和模型名称
        self.training_started.emit()

        # 创建模型保存目录
        model_save_dir = os.path.join(self.annotation_folder, 'models', selected_model)
        os.makedirs(model_save_dir, exist_ok=True)
        
        # 如果启用了TensorBoard，设置TensorBoard日志目录
        if use_tensorboard:
            tensorboard_dir = os.path.join(model_save_dir, 'tensorboard_logs')
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.tensorboard_widget.set_tensorboard_dir(tensorboard_dir)
        
        # 连接信号
        self.worker.model_trainer.progress_updated.connect(self.update_training_progress)
        self.worker.model_trainer.status_updated.connect(self.update_training_status)
        self.worker.model_trainer.training_finished.connect(self.on_training_finished)
        self.worker.model_trainer.training_error.connect(self.on_training_error)
        self.worker.model_trainer.epoch_finished.connect(self.training_visualization.update_plots)

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

    def update_training_progress(self, value):
        self.training_progress_bar.setValue(value)

    def update_training_status(self, message):
        self.training_status_label.setText(message)

    def on_training_finished(self):
        self.update_status('训练完成')
        self.train_model_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)

    def on_training_error(self, error):
        QMessageBox.critical(self, '错误', f'训练过程中出错: {str(error)}')
        self.train_model_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)

    def create_evaluation_tab(self, parent):
        """创建模型评估标签页"""
        layout = QVBoxLayout(parent)
        
        # 添加标题和说明
        title_label = QLabel("模型评估与可视化")
        title_label.setFont(QFont('Arial', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        description_label = QLabel("在此页面可以查看模型训练过程的实时指标和评估结果，以及使用TensorBoard进行深入分析。")
        description_label.setWordWrap(True)
        description_label.setStyleSheet("color: #555; font-style: italic;")
        layout.addWidget(description_label)
        
        # 创建分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # 创建选项卡，分为训练可视化和TensorBoard两个子标签页
        eval_tab_widget = QTabWidget()
        layout.addWidget(eval_tab_widget)
        
        # 训练可视化子标签页
        training_viz_tab = QWidget()
        training_viz_layout = QVBoxLayout(training_viz_tab)
        
        # 训练可视化组件
        visualization_group = QGroupBox("训练过程可视化")
        visualization_layout = QVBoxLayout()
        self.training_visualization = TrainingVisualizationWidget()
        visualization_layout.addWidget(self.training_visualization)
        visualization_group.setLayout(visualization_layout)
        training_viz_layout.addWidget(visualization_group)
        
        # 添加训练可视化子标签页
        eval_tab_widget.addTab(training_viz_tab, "训练曲线")
        
        # TensorBoard子标签页
        tensorboard_tab = QWidget()
        tensorboard_layout = QVBoxLayout(tensorboard_tab)
        
        # 创建TensorBoard组件
        self.tensorboard_widget = TensorBoardWidget()
        tensorboard_layout.addWidget(self.tensorboard_widget)
        
        # 添加TensorBoard子标签页
        eval_tab_widget.addTab(tensorboard_tab, "TensorBoard")
        
        # 添加模型比较部分
        comparison_group = QGroupBox("模型比较")
        comparison_layout = QVBoxLayout()
        
        # 模型比较说明
        comparison_description = QLabel("选择多个训练好的模型进行性能比较，查看不同模型在各项指标上的表现。")
        comparison_description.setWordWrap(True)
        comparison_layout.addWidget(comparison_description)
        
        # 模型选择区域
        model_selection_layout = QHBoxLayout()
        model_selection_layout.addWidget(QLabel("选择模型:"))
        self.model_comparison_list = QListWidget()
        self.model_comparison_list.setSelectionMode(QListWidget.MultiSelection)
        model_selection_layout.addWidget(self.model_comparison_list)
        
        # 添加和刷新按钮
        buttons_layout = QVBoxLayout()
        refresh_models_btn = QPushButton("刷新模型列表")
        refresh_models_btn.clicked.connect(self.refresh_model_list)
        compare_models_btn = QPushButton("比较所选模型")
        compare_models_btn.clicked.connect(self.compare_models)
        buttons_layout.addWidget(refresh_models_btn)
        buttons_layout.addWidget(compare_models_btn)
        buttons_layout.addStretch()
        model_selection_layout.addLayout(buttons_layout)
        
        comparison_layout.addLayout(model_selection_layout)
        comparison_group.setLayout(comparison_layout)
        layout.addWidget(comparison_group)
        
    def refresh_model_list(self):
        """刷新模型列表"""
        self.model_comparison_list.clear()
        
        if not hasattr(self, 'annotation_folder') or not self.annotation_folder:
            QMessageBox.warning(self, '警告', '请先在模型训练标签页选择标注数据文件夹')
            return
            
        models_dir = os.path.join(self.annotation_folder, 'models')
        if not os.path.exists(models_dir):
            QMessageBox.information(self, '提示', '未找到已训练的模型')
            return
            
        # 查找所有模型目录
        model_dirs = []
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'best_model.pth')):
                model_dirs.append(item)
                
        if not model_dirs:
            QMessageBox.information(self, '提示', '未找到已训练的模型')
            return
            
        # 添加到列表
        for model_name in model_dirs:
            self.model_comparison_list.addItem(model_name)
            
        QMessageBox.information(self, '提示', f'找到 {len(model_dirs)} 个已训练的模型')
        
    def compare_models(self):
        """比较所选模型"""
        selected_items = self.model_comparison_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, '警告', '请先选择要比较的模型')
            return
            
        selected_models = [item.text() for item in selected_items]
        QMessageBox.information(self, '提示', f'已选择 {len(selected_models)} 个模型进行比较: {", ".join(selected_models)}\n\n此功能正在开发中...')

    def save_training_params(self):
        """保存训练参数"""
        try:
            # 获取当前任务类型
            is_classification = self.training_classification_radio.isChecked()
            
            # 创建参数字典
            params = {
                'task_type': 'classification' if is_classification else 'detection',
                'model': self.classification_model_combo.currentText() if is_classification else self.detection_model_combo.currentText(),
                'basic_params': {
                    'batch_size': self.batch_size_spin.value() if is_classification else self.detection_batch_size_spin.value(),
                    'learning_rate': self.lr_spin.value() if is_classification else self.detection_lr_spin.value(),
                    'epochs': self.epochs_spin.value() if is_classification else self.detection_epochs_spin.value(),
                    'optimizer': self.optimizer_combo.currentText() if is_classification else self.detection_optimizer_combo.currentText(),
                    'lr_scheduler': self.lr_scheduler_combo.currentText() if is_classification else self.detection_lr_scheduler_combo.currentText(),
                    'weight_decay': self.weight_decay_spin.value() if is_classification else self.detection_weight_decay_spin.value()
                }
            }
            
            if is_classification:
                # 分类任务特有参数
                params.update({
                    'advanced_params': {
                        'use_pretrained': self.pretrained_checkbox.isChecked(),
                        'use_mixed_precision': self.mixed_precision_checkbox.isChecked(),
                        'use_early_stopping': self.early_stopping_checkbox.isChecked(),
                        'patience': self.patience_spin.value(),
                        'class_weight': self.class_weight_combo.currentText()
                    },
                    'metrics': {
                        'accuracy': self.accuracy_checkbox.isChecked(),
                        'precision': self.precision_checkbox.isChecked(),
                        'recall': self.recall_checkbox.isChecked(),
                        'f1': self.f1_checkbox.isChecked(),
                        'confusion_matrix': self.confusion_matrix_checkbox.isChecked()
                    }
                })
            else:
                # 目标检测任务特有参数
                params.update({
                    'detection_params': {
                        'iou_threshold': self.iou_threshold_spin.value(),
                        'nms_threshold': self.nms_threshold_spin.value(),
                        'conf_threshold': self.conf_threshold_spin.value(),
                        'anchor_size': self.anchor_size_combo.currentText(),
                        'use_fpn': self.use_fpn_checkbox.isChecked()
                    },
                    'metrics': {
                        'map': self.map_checkbox.isChecked(),
                        'map50': self.map50_checkbox.isChecked(),
                        'map75': self.map75_checkbox.isChecked(),
                        'precision_curve': self.precision_curve_checkbox.isChecked(),
                        'detection_speed': self.detection_speed_checkbox.isChecked()
                    }
                })
            
            # 保存到配置文件
            from config_loader import ConfigLoader
            config_loader = ConfigLoader()
            config = config_loader.get_config()
            config['training_params'] = params
            
            # 保存标注文件夹路径
            if hasattr(self, 'annotation_folder') and self.annotation_folder:
                config['annotation_folder'] = self.annotation_folder
                print(f"已保存标注文件夹路径: {self.annotation_folder}")
            
            config_loader.save_config(config)
            
            return True
        except Exception as e:
            print(f"保存训练参数时出错: {str(e)}")
            return False
            
    def load_training_params(self):
        """加载训练参数"""
        try:
            from config_loader import ConfigLoader
            config_loader = ConfigLoader()
            config = config_loader.get_config()
            
            if 'training_params' not in config:
                return
                
            params = config['training_params']
            
            # 加载标注文件夹路径
            if 'annotation_folder' in config:
                self.annotation_folder = config['annotation_folder']
                if hasattr(self, 'training_annotation_folder_label'):
                    self.training_annotation_folder_label.setText(self.annotation_folder)
                if hasattr(self, 'train_model_btn'):
                    self.train_model_btn.setEnabled(True)
                print(f"已加载标注文件夹路径: {self.annotation_folder}")
            
            # 设置任务类型
            is_classification = params.get('task_type') == 'classification'
            self.training_classification_radio.setChecked(is_classification)
            self.training_detection_radio.setChecked(not is_classification)
            
            # 设置基本参数
            basic_params = params.get('basic_params', {})
            if is_classification:
                self.batch_size_spin.setValue(basic_params.get('batch_size', 32))
                self.lr_spin.setValue(basic_params.get('learning_rate', 0.001))
                self.epochs_spin.setValue(basic_params.get('epochs', 20))
                
                optimizer = basic_params.get('optimizer', 'Adam')
                index = self.optimizer_combo.findText(optimizer)
                if index >= 0:
                    self.optimizer_combo.setCurrentIndex(index)
                    
                lr_scheduler = basic_params.get('lr_scheduler', '固定')
                index = self.lr_scheduler_combo.findText(lr_scheduler)
                if index >= 0:
                    self.lr_scheduler_combo.setCurrentIndex(index)
                    
                self.weight_decay_spin.setValue(basic_params.get('weight_decay', 0.0001))
                
                # 设置高级参数
                advanced_params = params.get('advanced_params', {})
                self.pretrained_checkbox.setChecked(advanced_params.get('use_pretrained', True))
                self.mixed_precision_checkbox.setChecked(advanced_params.get('use_mixed_precision', True))
                self.early_stopping_checkbox.setChecked(advanced_params.get('use_early_stopping', True))
                self.patience_spin.setValue(advanced_params.get('patience', 5))
                
                class_weight = advanced_params.get('class_weight', '均衡')
                index = self.class_weight_combo.findText(class_weight)
                if index >= 0:
                    self.class_weight_combo.setCurrentIndex(index)
                    
                # 设置评估指标
                metrics = params.get('metrics', {})
                self.accuracy_checkbox.setChecked(metrics.get('accuracy', True))
                self.precision_checkbox.setChecked(metrics.get('precision', True))
                self.recall_checkbox.setChecked(metrics.get('recall', True))
                self.f1_checkbox.setChecked(metrics.get('f1', True))
                self.confusion_matrix_checkbox.setChecked(metrics.get('confusion_matrix', True))
            else:
                self.detection_batch_size_spin.setValue(basic_params.get('batch_size', 16))
                self.detection_lr_spin.setValue(basic_params.get('learning_rate', 0.001))
                self.detection_epochs_spin.setValue(basic_params.get('epochs', 50))
                
                optimizer = basic_params.get('optimizer', 'Adam')
                index = self.detection_optimizer_combo.findText(optimizer)
                if index >= 0:
                    self.detection_optimizer_combo.setCurrentIndex(index)
                    
                lr_scheduler = basic_params.get('lr_scheduler', '固定')
                index = self.detection_lr_scheduler_combo.findText(lr_scheduler)
                if index >= 0:
                    self.detection_lr_scheduler_combo.setCurrentIndex(index)
                    
                self.detection_weight_decay_spin.setValue(basic_params.get('weight_decay', 0.0001))
                
                # 设置检测特有参数
                detection_params = params.get('detection_params', {})
                self.iou_threshold_spin.setValue(detection_params.get('iou_threshold', 0.5))
                self.nms_threshold_spin.setValue(detection_params.get('nms_threshold', 0.45))
                self.conf_threshold_spin.setValue(detection_params.get('conf_threshold', 0.25))
                
                anchor_size = detection_params.get('anchor_size', '自动')
                index = self.anchor_size_combo.findText(anchor_size)
                if index >= 0:
                    self.anchor_size_combo.setCurrentIndex(index)
                    
                self.use_fpn_checkbox.setChecked(detection_params.get('use_fpn', True))
                
                # 设置评估指标
                metrics = params.get('metrics', {})
                self.map_checkbox.setChecked(metrics.get('map', True))
                self.map50_checkbox.setChecked(metrics.get('map50', True))
                self.map75_checkbox.setChecked(metrics.get('map75', True))
                self.precision_curve_checkbox.setChecked(metrics.get('precision_curve', True))
                self.detection_speed_checkbox.setChecked(metrics.get('detection_speed', True))
                
        except Exception as e:
            print(f"加载训练参数时出错: {str(e)}")
