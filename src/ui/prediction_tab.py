from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QGroupBox, QGridLayout, QSizePolicy, QLineEdit, 
                           QMessageBox, QFrame, QStackedWidget, QRadioButton, QButtonGroup,
                           QProgressBar, QCheckBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage
import os
from .base_tab import BaseTab

class PredictionTab(BaseTab):
    """预测标签页，负责模型预测功能，包括单张预测和批量预测"""
    
    # 定义信号
    prediction_started = pyqtSignal()
    batch_prediction_started = pyqtSignal(dict)
    batch_prediction_stopped = pyqtSignal()
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent, main_window)
        self.model_file = ""
        self.class_info_file = ""
        self.image_file = ""
        self.input_folder = ""
        self.output_folder = ""
        self.init_ui()
        
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
        self.single_mode_radio.setChecked(True)
        
        self.mode_group.addButton(self.single_mode_radio, 0)
        self.mode_group.addButton(self.batch_mode_radio, 1)
        
        self.mode_group.buttonClicked.connect(self.switch_prediction_mode)
        
        mode_layout.addWidget(self.single_mode_radio)
        mode_layout.addWidget(self.batch_mode_radio)
        mode_layout.addStretch()
        
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)
        
        # 创建模型文件选择组
        model_group = QGroupBox("模型文件")
        model_layout = QGridLayout()
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText("请选择训练好的模型文件")
        
        model_btn = QPushButton("浏览...")
        model_btn.clicked.connect(self.select_model_file)
        
        model_layout.addWidget(QLabel("模型文件:"), 0, 0)
        model_layout.addWidget(self.model_path_edit, 0, 1)
        model_layout.addWidget(model_btn, 0, 2)
        
        # 类别信息文件
        self.class_info_path_edit = QLineEdit()
        self.class_info_path_edit.setReadOnly(True)
        self.class_info_path_edit.setPlaceholderText("请选择类别信息文件")
        
        class_info_btn = QPushButton("浏览...")
        class_info_btn.clicked.connect(self.select_class_info_file)
        
        model_layout.addWidget(QLabel("类别信息:"), 1, 0)
        model_layout.addWidget(self.class_info_path_edit, 1, 1)
        model_layout.addWidget(class_info_btn, 1, 2)
        
        # 加载模型按钮
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.load_prediction_model)
        self.load_model_btn.setEnabled(False)
        model_layout.addWidget(self.load_model_btn, 2, 0, 1, 3)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # 创建堆叠部件用于切换单张预测和批量预测界面
        self.stacked_widget = QStackedWidget()
        
        # 创建单张预测界面
        self.single_prediction_widget = QWidget()
        self.init_single_prediction_ui()
        
        # 创建批量预测界面
        self.batch_prediction_widget = QWidget()
        self.init_batch_prediction_ui()
        
        # 添加到堆叠部件
        self.stacked_widget.addWidget(self.single_prediction_widget)
        self.stacked_widget.addWidget(self.batch_prediction_widget)
        
        main_layout.addWidget(self.stacked_widget)
        
        # 添加弹性空间
        main_layout.addStretch()
    
    def init_single_prediction_ui(self):
        """初始化单张预测UI"""
        layout = QVBoxLayout(self.single_prediction_widget)
        
        # 创建图像选择组
        image_group = QGroupBox("图像选择")
        image_layout = QVBoxLayout()
        
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
        
        # 保存选项
        self.save_images_check = QCheckBox("保存预测图像")
        self.save_images_check.setChecked(True)
        
        self.save_csv_check = QCheckBox("保存CSV结果")
        self.save_csv_check.setChecked(True)
        
        options_layout.addWidget(self.save_images_check, 1, 0, 1, 2)
        options_layout.addWidget(self.save_csv_check, 1, 2, 1, 2)
        
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
        
        button_layout.addWidget(self.start_batch_btn)
        button_layout.addWidget(self.stop_batch_btn)
        
        layout.addLayout(button_layout)
    
    def switch_prediction_mode(self, button):
        """切换预测模式"""
        if button == self.single_mode_radio:
            self.stacked_widget.setCurrentIndex(0)
        else:
            self.stacked_widget.setCurrentIndex(1)
    
    def select_model_file(self):
        """选择模型文件"""
        file, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "模型文件 (*.h5 *.pb *.tflite *.pt);;所有文件 (*)")
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
    
    def load_prediction_model(self):
        """加载预测模型"""
        if not self.model_file or not self.class_info_file:
            QMessageBox.warning(self, "警告", "请先选择模型文件和类别信息文件!")
            return
            
        self.update_status("正在加载模型...")
        # 这里需要发送加载模型的信号
        # 在主窗口中处理
        
        # 模拟加载成功
        self.update_status("模型加载成功")
        
        # 更新单张预测和批量预测按钮状态
        self.predict_btn.setEnabled(self.image_file != "")
        self.check_batch_ready()
    
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
            if self.load_model_btn.isEnabled() == False and self.model_file and self.class_info_file:
                self.predict_btn.setEnabled(True)
    
    def predict(self):
        """开始单张预测"""
        if not self.model_file or not self.class_info_file:
            QMessageBox.warning(self, "警告", "请先加载模型!")
            return
            
        if not self.image_file:
            QMessageBox.warning(self, "警告", "请先选择图像!")
            return
            
        # 发出预测开始信号
        self.prediction_started.emit()
        self.update_status("开始预测...")
        
        # 这里需要发送预测的信号
        # 在主窗口中处理
    
    def update_prediction_result(self, result):
        """更新预测结果"""
        if isinstance(result, dict):
            # 格式化结果显示
            result_text = "<h3>预测结果:</h3>"
            for class_name, prob in result.items():
                result_text += f"<p>{class_name}: {prob:.2%}</p>"
            
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
        is_ready = (self.model_file and os.path.exists(self.model_file) and
                   self.class_info_file and os.path.exists(self.class_info_file) and
                   self.input_folder and os.path.exists(self.input_folder) and
                   self.output_folder)
        self.start_batch_btn.setEnabled(is_ready)
        return is_ready
    
    def start_batch_prediction(self):
        """开始批量预测"""
        if not self.check_batch_ready():
            QMessageBox.warning(self, "警告", "请确保所有必要的文件和文件夹都已选择。")
            return
        
        # 准备参数
        params = {
            'model_file': self.model_file,
            'class_info_file': self.class_info_file,
            'input_folder': self.input_folder,
            'output_folder': self.output_folder,
            'threshold': self.threshold_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'save_images': self.save_images_check.isChecked(),
            'save_csv': self.save_csv_check.isChecked()
        }
        
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
    
    def batch_prediction_finished(self):
        """批量预测完成"""
        self.start_batch_btn.setEnabled(True)
        self.stop_batch_btn.setEnabled(False)
        self.batch_progress_bar.setValue(100)
        self.update_status("批量预测完成")
        self.update_progress(100) 