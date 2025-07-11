from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QGroupBox, QGridLayout, QSizePolicy, QLineEdit, 
                           QMessageBox, QFrame, QStackedWidget, QRadioButton, QButtonGroup,
                           QProgressBar, QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage
import os
from .base_tab import BaseTab
from .components.prediction.auto_review_widget import AutoReviewWidget
from .components.prediction.accuracy_calculator_widget import AccuracyCalculatorWidget

class PredictionTab(BaseTab):
    """预测标签页，负责模型预测功能，包括单张预测和批量预测"""
    
    # 定义信号
    prediction_started = pyqtSignal(dict)
    batch_prediction_started = pyqtSignal(dict)
    batch_prediction_stopped = pyqtSignal()
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent, main_window)
        self.main_window = main_window
        self.model_file = ""
        self.class_info_file = ""
        self.image_file = ""
        self.input_folder = ""
        self.output_folder = ""
        self.top_k = 3  # 默认显示前3个类别
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
        title_label = QLabel("模型预测")
        title_label.setFont(QFont('微软雅黑', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 创建预测模式选择
        mode_group = QGroupBox("预测模式")
        mode_layout = QHBoxLayout()
        
        self.mode_group = QButtonGroup(self)
        self.single_mode_radio = QRadioButton("单张预测")
        self.batch_mode_radio = QRadioButton("批量预测")
        self.auto_review_mode_radio = QRadioButton("自动Review")
        self.single_mode_radio.setChecked(True)
        
        self.mode_group.addButton(self.single_mode_radio, 0)
        self.mode_group.addButton(self.batch_mode_radio, 1)
        self.mode_group.addButton(self.auto_review_mode_radio, 2)
        
        self.mode_group.buttonClicked.connect(self.switch_prediction_mode)
        
        mode_layout.addWidget(self.single_mode_radio)
        mode_layout.addWidget(self.batch_mode_radio)
        mode_layout.addWidget(self.auto_review_mode_radio)
        mode_layout.addStretch()
        
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)
        
        # 创建模型文件选择组
        model_group = QGroupBox("模型文件")
        model_layout = QGridLayout()
        
        # 添加模型类型选择
        model_layout.addWidget(QLabel("模型类型:"), 0, 0)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["分类模型", "检测模型"])
        self.model_type_combo.currentIndexChanged.connect(self.switch_model_type)
        model_layout.addWidget(self.model_type_combo, 0, 1, 1, 1)
        
        # 添加模型架构选择
        model_layout.addWidget(QLabel("模型架构:"), 0, 2)
        self.model_arch_combo = QComboBox()
        self.model_arch_combo.addItems([
            "MobileNetV2", "MobileNetV3", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
            "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4",
            "VGG16", "VGG19", "DenseNet121", "DenseNet169", "DenseNet201", "InceptionV3", "Xception"
        ])
        model_layout.addWidget(self.model_arch_combo, 0, 3, 1, 1)
        
        # 模型文件选择
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText("请选择训练好的模型文件")
        
        model_btn = QPushButton("浏览...")
        model_btn.clicked.connect(self.select_model_file)
        
        model_layout.addWidget(QLabel("模型文件:"), 1, 0)
        model_layout.addWidget(self.model_path_edit, 1, 1, 1, 2)
        model_layout.addWidget(model_btn, 1, 3)
        
        # 类别信息文件
        self.class_info_path_edit = QLineEdit()
        self.class_info_path_edit.setReadOnly(True)
        self.class_info_path_edit.setPlaceholderText("请选择类别信息文件")
        
        class_info_btn = QPushButton("浏览...")
        class_info_btn.clicked.connect(self.select_class_info_file)
        
        model_layout.addWidget(QLabel("类别信息:"), 2, 0)
        model_layout.addWidget(self.class_info_path_edit, 2, 1, 1, 2)
        model_layout.addWidget(class_info_btn, 2, 3)
        
        # 加载模型按钮
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.load_prediction_model)
        self.load_model_btn.setEnabled(False)
        model_layout.addWidget(self.load_model_btn, 3, 0, 1, 4)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # 创建堆叠部件用于切换不同预测模式界面
        self.stacked_widget = QStackedWidget()
        
        # 创建单张预测界面
        self.single_prediction_widget = QWidget()
        self.init_single_prediction_ui()
        
        # 创建批量预测界面
        self.batch_prediction_widget = QWidget()
        self.init_batch_prediction_ui()
        
        # 创建自动Review界面
        self.auto_review_widget = AutoReviewWidget(parent=self, main_window=self.main_window)
        
        # 添加到堆叠部件
        self.stacked_widget.addWidget(self.single_prediction_widget)
        self.stacked_widget.addWidget(self.batch_prediction_widget)
        self.stacked_widget.addWidget(self.auto_review_widget)
        
        main_layout.addWidget(self.stacked_widget)
        
        # 添加弹性空间
        main_layout.addStretch()
    
    def _do_apply_config(self, config):
        """实现具体的配置应用逻辑 - 智能配置系统"""
        print(f"PredictionTab: 智能应用配置，包含 {len(config)} 个配置项")
        
        # 应用默认模型文件
        if 'default_model_file' in config and config['default_model_file']:
            print(f"PredictionTab: 应用模型文件配置: {config['default_model_file']}")
            self.model_file = config['default_model_file']
            if hasattr(self, 'model_path_edit'):
                self.model_path_edit.setText(config['default_model_file'])
        
        # 应用默认类别信息文件
        if 'default_class_info_file' in config and config['default_class_info_file']:
            print(f"PredictionTab: 应用类别信息文件配置: {config['default_class_info_file']}")
            self.class_info_file = config['default_class_info_file']
            if hasattr(self, 'class_info_path_edit'):
                self.class_info_path_edit.setText(config['default_class_info_file'])
                
            # 如果类别信息文件有效，加载类别信息
            if os.path.exists(config['default_class_info_file']):
                try:
                    import json
                    with open(config['default_class_info_file'], 'r', encoding='utf-8') as f:
                        class_info = json.load(f)
                        if hasattr(self, 'class_info'):
                            self.class_info = class_info
                            print("PredictionTab: 成功加载类别信息")
                except Exception as e:
                    print(f"PredictionTab: 加载类别信息文件失败: {str(e)}")
        
        # 检查是否可以加载模型
        if hasattr(self, 'check_model_ready'):
            self.check_model_ready()
            
        print("PredictionTab: 智能配置应用完成")
    
    def init_single_prediction_ui(self):
        """初始化单张预测UI"""
        layout = QVBoxLayout(self.single_prediction_widget)
        
        # 创建图像选择组
        image_group = QGroupBox("图像选择")
        image_layout = QVBoxLayout()
        
        # 添加显示类别数量设置
        top_k_layout = QHBoxLayout()
        top_k_layout.addWidget(QLabel("显示类别数量:"))
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 100)  # 设置范围从1到100
        self.top_k_spin.setValue(self.top_k)  # 设置默认值
        self.top_k_spin.valueChanged.connect(self.update_top_k)
        top_k_layout.addWidget(self.top_k_spin)
        top_k_layout.addStretch()
        image_layout.addLayout(top_k_layout)
        
        # 图像选择按钮
        image_btn_layout = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setReadOnly(True)
        self.image_path_edit.setPlaceholderText("请选择要预测的图像")
        
        select_image_btn = QPushButton("浏览...")
        select_image_btn.clicked.connect(self.select_image)
        
        image_btn_layout.addWidget(QLabel("图像文件:"))
        image_btn_layout.addWidget(self.image_path_edit)
        image_btn_layout.addWidget(select_image_btn)
        
        image_layout.addLayout(image_btn_layout)
        
        # 图像预览
        preview_layout = QHBoxLayout()
        
        # 原始图像
        original_layout = QVBoxLayout()
        original_layout.addWidget(QLabel("原始图像:"))
        
        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(224, 224)
        self.original_image_label.setFrameShape(QFrame.Box)
        original_layout.addWidget(self.original_image_label)
        
        preview_layout.addLayout(original_layout)
        
        # 预测结果
        result_layout = QVBoxLayout()
        result_layout.addWidget(QLabel("预测结果:"))
        
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(224, 224)
        self.result_label.setFrameShape(QFrame.Box)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        result_layout.addWidget(self.result_label)
        
        preview_layout.addLayout(result_layout)
        
        image_layout.addLayout(preview_layout)
        
        # 预测按钮
        self.predict_btn = QPushButton("开始预测")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setEnabled(False)
        self.predict_btn.setMinimumHeight(40)
        image_layout.addWidget(self.predict_btn)
        
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
    
    def init_batch_prediction_ui(self):
        """初始化批量预测UI"""
        layout = QVBoxLayout(self.batch_prediction_widget)
        
        # 输入文件夹选择
        input_group = QGroupBox("输入/输出文件夹")
        input_layout = QGridLayout()
        
        # 输入文件夹
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setReadOnly(True)
        self.input_path_edit.setPlaceholderText("请选择输入文件夹")
        
        input_btn = QPushButton("浏览...")
        input_btn.clicked.connect(self.browse_input_folder)
        
        input_layout.addWidget(QLabel("输入文件夹:"), 0, 0)
        input_layout.addWidget(self.input_path_edit, 0, 1)
        input_layout.addWidget(input_btn, 0, 2)
        
        # 输出文件夹
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setReadOnly(True)
        self.output_path_edit.setPlaceholderText("请选择输出文件夹")
        
        output_btn = QPushButton("浏览...")
        output_btn.clicked.connect(self.browse_output_folder)
        
        input_layout.addWidget(QLabel("输出文件夹:"), 1, 0)
        input_layout.addWidget(self.output_path_edit, 1, 1)
        input_layout.addWidget(output_btn, 1, 2)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # 预测选项
        options_group = QGroupBox("预测选项")
        options_layout = QGridLayout()
        
        # 置信度阈值
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.5)
        
        options_layout.addWidget(QLabel("置信度阈值:"), 0, 0)
        options_layout.addWidget(self.threshold_spin, 0, 1)
        
        # 批处理大小
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(32)
        
        options_layout.addWidget(QLabel("批处理大小:"), 0, 2)
        options_layout.addWidget(self.batch_size_spin, 0, 3)
        
        # 文件操作模式
        options_layout.addWidget(QLabel("文件操作:"), 1, 0)
        self.copy_mode_combo = QComboBox()
        self.copy_mode_combo.addItems(["复制", "移动"])
        self.copy_mode_combo.setCurrentIndex(0)  # 默认复制
        options_layout.addWidget(self.copy_mode_combo, 1, 1)
        
        # 创建子文件夹选项
        self.create_subfolders_check = QCheckBox("为每个类别创建子文件夹")
        self.create_subfolders_check.setChecked(True)
        options_layout.addWidget(self.create_subfolders_check, 1, 2, 1, 2)
        
        # 保存选项
        self.save_images_check = QCheckBox("保存预测图像")
        self.save_images_check.setChecked(True)
        
        self.save_csv_check = QCheckBox("保存CSV结果")
        self.save_csv_check.setChecked(True)
        
        options_layout.addWidget(self.save_images_check, 2, 0, 1, 2)
        options_layout.addWidget(self.save_csv_check, 2, 2, 1, 2)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # 进度条
        progress_group = QGroupBox("预测进度")
        progress_layout = QVBoxLayout()
        
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)
        
        progress_layout.addWidget(self.batch_progress_bar)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        self.start_batch_btn = QPushButton("开始批量预测")
        self.start_batch_btn.clicked.connect(self.start_batch_prediction)
        self.start_batch_btn.setEnabled(False)
        self.start_batch_btn.setMinimumHeight(40)
        
        self.stop_batch_btn = QPushButton("停止")
        self.stop_batch_btn.clicked.connect(self.stop_batch_prediction)
        self.stop_batch_btn.setEnabled(False)
        
        self.open_output_btn = QPushButton("打开输出文件夹")
        self.open_output_btn.clicked.connect(self.open_output_folder)
        self.open_output_btn.setEnabled(False)  # 初始状态禁用
        
        button_layout.addWidget(self.start_batch_btn)
        button_layout.addWidget(self.stop_batch_btn)
        button_layout.addWidget(self.open_output_btn)
        
        layout.addLayout(button_layout)
        
        # 添加准确率计算组件
        accuracy_group = QGroupBox("预测准确率分析")
        accuracy_layout = QVBoxLayout()
        
        self.accuracy_calculator = AccuracyCalculatorWidget(parent=self)
        accuracy_layout.addWidget(self.accuracy_calculator)
        
        accuracy_group.setLayout(accuracy_layout)
        layout.addWidget(accuracy_group)
    
    def switch_prediction_mode(self, button):
        """切换预测模式"""
        if button == self.single_mode_radio:
            self.stacked_widget.setCurrentIndex(0)
        elif button == self.batch_mode_radio:
            self.stacked_widget.setCurrentIndex(1)
        elif button == self.auto_review_mode_radio:
            self.stacked_widget.setCurrentIndex(2)
    
    def select_model_file(self):
        """选择模型文件"""
        file, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "模型文件 (*.h5 *.pb *.tflite *.pt *.pth);;所有文件 (*)")
        if file:
            self.model_file = file
            self.model_path_edit.setText(file)
            self.check_model_ready()
    
    def select_class_info_file(self):
        """选择类别信息文件"""
        file, _ = QFileDialog.getOpenFileName(self, "选择类别信息文件", "", "JSON文件 (*.json);;所有文件 (*)")
        if file:
            self.class_info_file = file
            self.class_info_path_edit.setText(file)
            self.check_model_ready()
    
    def check_model_ready(self):
        """检查是否可以加载模型"""
        is_ready = bool(self.model_file and self.class_info_file)
        self.load_model_btn.setEnabled(is_ready)
        return is_ready
    
    def switch_model_type(self, index):
        """切换模型类型"""
        if index == 0:  # 分类模型
            self.model_arch_combo.clear()
            self.model_arch_combo.addItems([
                "MobileNetV2", "MobileNetV3", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
                "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4",
                "VGG16", "VGG19", "DenseNet121", "DenseNet169", "DenseNet201", "InceptionV3", "Xception"
            ])
        else:  # 检测模型
            self.model_arch_combo.clear()
            self.model_arch_combo.addItems([
                "YOLOv5", "YOLOv8", "YOLOv7", "YOLOv6", "YOLOv4", "YOLOv3",
                "SSD", "SSD512", "SSD300", "Faster R-CNN", "Mask R-CNN",
                "RetinaNet", "DETR"
            ])
            
    def load_prediction_model(self):
        """加载预测模型"""
        if not self.model_file or not self.class_info_file:
            QMessageBox.warning(self, "警告", "请先选择模型文件和类别信息文件!")
            return
            
        self.update_status("正在加载模型...")
        
        # 获取模型类型和架构
        model_type = self.model_type_combo.currentText()
        model_arch = self.model_arch_combo.currentText()
        
        # 发送加载模型的信号
        try:
            # 创建模型信息字典
            model_info = {
                "model_path": self.model_file,
                "class_info_path": self.class_info_file,
                "model_type": model_type,
                "model_arch": model_arch
            }
            
            # 调用predictor的load_model方法，传入模型信息
            self.main_window.worker.predictor.load_model_with_info(model_info)
            
            # 模型加载成功
            self.update_status("模型加载成功")
            
            # 显示成功提示弹窗
            QMessageBox.information(self, "成功", f"模型 {model_arch} 加载成功！\n现在可以进行图像预测了。")
            
            # 禁用加载模型按钮，表示模型已加载
            self.load_model_btn.setEnabled(False)
            
            # 如果已经选择了图像，则启用预测按钮
            if self.image_file:
                self.predict_btn.setEnabled(True)
                self.update_status("模型已加载，可以开始预测")
            else:
                self.update_status("模型已加载，请选择要预测的图像")
        except Exception as e:
            self.update_status(f"模型加载失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            return
    
    def select_image(self):
        """选择图像文件"""
        file, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "图像文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)")
        if file:
            self.image_file = file
            self.image_path_edit.setText(file)
            
            # 显示图像预览
            pixmap = QPixmap(file)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.original_image_label.setPixmap(pixmap)
            else:
                self.original_image_label.setText("无法加载图像")
            
            # 如果模型已加载，则启用预测按钮
            if not self.load_model_btn.isEnabled() and self.model_file and self.class_info_file:
                self.predict_btn.setEnabled(True)
                self.update_status("已选择图像，可以开始预测")
    
    def predict(self):
        """开始单张预测"""
        if not self.model_file or not self.class_info_file:
            QMessageBox.warning(self, "警告", "请先加载模型!")
            return
            
        if not self.image_file:
            QMessageBox.warning(self, "警告", "请先选择图像!")
            return
            
        # 创建预测参数字典
        predict_params = {
            'image_path': self.image_file,
            'top_k': self.top_k
        }
        
        # 更新状态并发送预测参数
        self.update_status("开始预测...")
        self.prediction_started.emit(predict_params)
    
    def update_prediction_result(self, result):
        """更新预测结果"""
        # 保存最后的预测结果
        self.last_prediction_result = result
        
        if isinstance(result, dict):
            # 格式化结果显示
            result_text = "<h3>预测结果:</h3>"
            
            # 检查结果格式
            if 'predictions' in result:
                # 新格式：包含预测结果列表
                predictions = result['predictions'][:self.top_k]  # 只取前top_k个结果
                for pred in predictions:
                    class_name = pred.get('class_name', '未知')
                    probability = pred.get('probability', 0)
                    result_text += f"<p>{class_name}: {probability:.2f}%</p>"
            else:
                # 兼容旧格式
                # 将字典转换为列表并按概率排序
                items = []
                for class_name, prob in result.items():
                    if isinstance(prob, (int, float)):
                        items.append((class_name, prob))
                    elif isinstance(prob, (list, tuple)) and len(prob) > 0:
                        items.append((class_name, prob[0]))
                    else:
                        items.append((class_name, 0))
                
                # 按概率降序排序并只取前top_k个
                items.sort(key=lambda x: x[1], reverse=True)
                for class_name, prob in items[:self.top_k]:
                    if isinstance(prob, (int, float)):
                        result_text += f"<p>{class_name}: {prob:.2%}</p>"
                    else:
                        result_text += f"<p>{class_name}: {prob:.2f}%</p>"
            
            self.result_label.setText(result_text)
        else:
            self.result_label.setText(f"<h3>预测结果:</h3><p>{result}</p>")
    
    # 批量预测相关方法
    def browse_input_folder(self):
        """浏览输入文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if folder_path:
            self.input_folder = folder_path
            self.input_path_edit.setText(folder_path)
            self.check_batch_ready()
    
    def browse_output_folder(self):
        """浏览输出文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder_path:
            self.output_folder = folder_path
            self.output_path_edit.setText(folder_path)
            self.check_batch_ready()
    
    def check_batch_ready(self):
        """检查批量预测是否准备就绪"""
        # 检查模型是否已加载（通过检查加载模型按钮是否被禁用来判断）
        model_loaded = not self.load_model_btn.isEnabled()
        
        is_ready = bool(self.model_file and os.path.exists(self.model_file) and
                   self.class_info_file and os.path.exists(self.class_info_file) and
                   self.input_folder and os.path.exists(self.input_folder) and
                   self.output_folder and model_loaded)
        self.start_batch_btn.setEnabled(is_ready)
        
        # 打开输出文件夹按钮只需要输出文件夹存在即可
        output_folder_exists = bool(self.output_folder and os.path.exists(self.output_folder))
        self.open_output_btn.setEnabled(output_folder_exists)
        
        return is_ready
    
    def start_batch_prediction(self):
        """开始批量预测"""
        if not self.check_batch_ready():
            QMessageBox.warning(self, "警告", "请确保所有必要的文件和文件夹都已选择。")
            return
        
        # 准备参数，匹配predictor.batch_predict的参数格式
        params = {
            'source_folder': self.input_folder,
            'target_folder': self.output_folder,
            'confidence_threshold': self.threshold_spin.value() * 100,  # 转换为百分比
            'copy_mode': 'copy' if self.copy_mode_combo.currentText() == '复制' else 'move',
            'create_subfolders': self.create_subfolders_check.isChecked()
        }
        
        # 添加调试信息
        print(f"批量预测参数:")
        print(f"  source_folder: {params['source_folder']}")
        print(f"  target_folder: {params['target_folder']}")
        print(f"  confidence_threshold: {params['confidence_threshold']}")
        print(f"  copy_mode: {params['copy_mode']}")
        print(f"  create_subfolders: {params['create_subfolders']}")
        
        # 更新UI状态
        self.start_batch_btn.setEnabled(False)
        self.stop_batch_btn.setEnabled(True)
        self.batch_progress_bar.setValue(0)
        self.update_status("批量预测开始...")
        
        # 发射信号
        self.batch_prediction_started.emit(params)
    
    def stop_batch_prediction(self):
        """停止批量预测"""
        self.update_status("正在停止批量预测...")
        self.batch_prediction_stopped.emit()
    
    def update_batch_progress(self, value):
        """更新批量预测进度"""
        self.batch_progress_bar.setValue(value)
        self.update_progress(value)
    
    def batch_prediction_finished(self, results=None):
        """批量预测完成"""
        self.start_batch_btn.setEnabled(True)
        self.stop_batch_btn.setEnabled(False)
        self.batch_progress_bar.setValue(100)
        self.update_status("批量预测完成")
        self.update_progress(100)
        
        # 自动设置准确率计算组件的文件夹路径
        if hasattr(self, 'accuracy_calculator') and self.input_folder and self.output_folder:
            self.accuracy_calculator.set_folders_from_parent(self.input_folder, self.output_folder)
        
        # 显示预测结果统计
        if results and isinstance(results, dict):
            total = results.get('total', 0)
            processed = results.get('processed', 0)
            classified = results.get('classified', 0)
            unclassified = results.get('unclassified', 0)
            class_counts = results.get('class_counts', {})
            
            # 构建结果消息
            result_msg = f"批量预测完成！\n\n"
            result_msg += f"总图片数: {total}\n"
            result_msg += f"已处理: {processed}\n"
            result_msg += f"已分类: {classified}\n"
            result_msg += f"未分类: {unclassified}\n\n"
            
            if class_counts:
                result_msg += "各类别统计:\n"
                for class_name, count in class_counts.items():
                    if count > 0:
                        result_msg += f"  {class_name}: {count} 张\n"
            
            # 添加准确率计算提示
            result_msg += "\n💡 提示: 您可以使用下方的准确率分析工具来计算预测准确率。"
            
            QMessageBox.information(self, "批量预测完成", result_msg)
        else:
            QMessageBox.information(self, "批量预测完成", "批量预测已完成！")
    
    def update_top_k(self, value):
        """更新要显示的类别数量"""
        self.top_k = value
        # 如果已经有预测结果，重新显示结果
        if hasattr(self, 'last_prediction_result'):
            self.update_prediction_result(self.last_prediction_result)

    def open_output_folder(self):
        """打开输出文件夹"""
        if not self.output_folder:
            QMessageBox.warning(self, "警告", "请先选择输出文件夹")
            return
            
        if not os.path.exists(self.output_folder):
            QMessageBox.warning(self, "警告", f"输出文件夹不存在: {self.output_folder}")
            return
            
        try:
            import platform
            import subprocess
            
            system = platform.system()
            if system == "Windows":
                os.startfile(self.output_folder)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", self.output_folder])
            else:  # Linux
                subprocess.run(["xdg-open", self.output_folder])
                
            self.update_status(f"已打开输出文件夹: {self.output_folder}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开文件夹: {str(e)}")
            self.update_status(f"打开文件夹失败: {str(e)}") 