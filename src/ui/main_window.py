from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QVBoxLayout, QWidget,
                           QStatusBar, QProgressBar, QLabel, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import os
import sys

from .data_processing_tab import DataProcessingTab
from .annotation_tab import AnnotationTab
from .training_tab import TrainingTab
from .prediction_tab import PredictionTab
from .settings_tab import SettingsTab
from .evaluation_tab import EvaluationTab
from .about_tab import AboutTab

class MainWindow(QMainWindow):
    """主窗口类，负责组织和管理所有标签页"""
    
    # 定义信号
    data_processing_started = pyqtSignal()
    training_started = pyqtSignal()
    prediction_started = pyqtSignal()
    image_preprocessing_started = pyqtSignal(dict)
    annotation_started = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        
        # 设置工具提示字体和样式
        QApplication.setFont(QFont('微软雅黑', 9))
        
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
        
        # 初始化UI
        self.init_ui()
        
        # 加载配置
        self.load_config()

    def init_ui(self):
        """初始化UI"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)  # 减少边距
        main_layout.setSpacing(0)  # 减少间距
        
        # 创建标签页控件
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)  # 使标签页看起来更现代
        self.tabs.setTabsClosable(False)  # 不显示关闭按钮
        
        # 创建各个标签页
        self.data_processing_tab = DataProcessingTab(self.tabs, self)
        self.annotation_tab = AnnotationTab(self.tabs, self)
        self.training_tab = TrainingTab(self.tabs, self)
        self.prediction_tab = PredictionTab(self.tabs, self)
        self.evaluation_tab = EvaluationTab(self.tabs, self)
        self.settings_tab = SettingsTab(self.tabs, self)
        self.about_tab = AboutTab(self.tabs, self)
        
        # 添加标签页（调整顺序，将模型预测放在模型评估与可视化后面）
        self.tabs.addTab(self.data_processing_tab, "图像预处理")
        self.tabs.addTab(self.annotation_tab, "图像标注")
        self.tabs.addTab(self.training_tab, "模型训练")
        self.tabs.addTab(self.evaluation_tab, "模型评估与可视化")
        self.tabs.addTab(self.prediction_tab, "模型预测")
        self.tabs.addTab(self.settings_tab, "设置")
        self.tabs.addTab(self.about_tab, "关于")
        
        # 添加标签页控件到主布局
        main_layout.addWidget(self.tabs)
        
        # 创建状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # 添加状态标签
        self.status_label = QLabel("就绪")
        self.statusBar.addWidget(self.status_label)
        
        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        # 删除最大宽度限制
        # self.progress_bar.setMaximumWidth(200)
        # 使进度条铺满状态栏剩余空间
        self.statusBar.addPermanentWidget(self.progress_bar, 1)
        
        # 连接信号
        self.connect_signals()
        
    def connect_signals(self):
        """连接信号和槽"""
        # 连接各个标签页的状态更新信号
        self.data_processing_tab.status_updated.connect(self.update_status)
        self.data_processing_tab.progress_updated.connect(self.update_progress)
        self.data_processing_tab.image_preprocessing_started.connect(self.on_image_preprocessing_started)
        
        self.annotation_tab.status_updated.connect(self.update_status)
        self.annotation_tab.progress_updated.connect(self.update_progress)
        self.annotation_tab.annotation_started.connect(self.on_annotation_started)
        
        self.training_tab.status_updated.connect(self.update_status)
        self.training_tab.progress_updated.connect(self.update_progress)
        self.training_tab.training_started.connect(self.on_training_started)
        # 连接训练进度更新信号到评估标签页的实时训练曲线更新函数
        self.training_tab.training_progress_updated.connect(self.evaluation_tab.update_training_visualization)
        
        self.prediction_tab.status_updated.connect(self.update_status)
        self.prediction_tab.progress_updated.connect(self.update_progress)
        self.prediction_tab.prediction_started.connect(self.on_prediction_started)
        
        # 连接批量预测信号（现在在prediction_tab中）
        if hasattr(self.prediction_tab, 'batch_prediction_started'):
            self.prediction_tab.batch_prediction_started.connect(self.on_batch_prediction_started)
        if hasattr(self.prediction_tab, 'batch_prediction_stopped'):
            self.prediction_tab.batch_prediction_stopped.connect(self.on_batch_prediction_stopped)
        
        self.evaluation_tab.status_updated.connect(self.update_status)
        self.evaluation_tab.progress_updated.connect(self.update_progress)
        
        self.settings_tab.status_updated.connect(self.update_status)
        self.settings_tab.progress_updated.connect(self.update_progress)
        self.settings_tab.settings_saved.connect(self.apply_config)

    def update_status(self, message):
        """更新状态栏信息"""
        self.status_label.setText(message)

    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        
    def on_image_preprocessing_started(self, params):
        """图像预处理开始时调用"""
        # 这里可以添加实际的图像预处理逻辑
        # 例如，创建一个工作线程来处理图像
        self.image_preprocessing_started.emit(params)
    
    def on_annotation_started(self, folder):
        """标注开始时调用"""
        # 这里可以添加实际的标注逻辑
        self.annotation_started.emit(folder)
    
    def on_training_started(self):
        """训练开始时调用"""
        # 重置评估标签页的实时训练曲线
        self.evaluation_tab.reset_training_visualization()
        # 这里可以添加实际的训练逻辑
        self.training_started.emit()

    def on_prediction_started(self):
        """预测开始时调用"""
        # 这里可以添加实际的预测逻辑
        self.prediction_started.emit()

    def on_batch_prediction_started(self, params):
        """批量预测开始时调用"""
        # 这里可以添加实际的批量预测逻辑
        pass
    
    def on_batch_prediction_stopped(self):
        """批量预测停止时调用"""
        # 这里可以添加实际的批量预测停止逻辑
        pass
    
    def load_config(self):
        """加载配置"""
        # 这里可以添加加载配置的逻辑
        # 例如，从配置文件加载设置
        pass

    def apply_config(self, config):
        """应用配置"""
        # 将配置应用到各个标签页
        
        # 数据处理标签页
        if 'default_source_folder' in config and config['default_source_folder']:
            self.data_processing_tab.source_folder = config['default_source_folder']
            self.data_processing_tab.source_path_edit.setText(config['default_source_folder'])
        
        if 'default_output_folder' in config and config['default_output_folder']:
            self.data_processing_tab.output_folder = config['default_output_folder']
            self.data_processing_tab.output_path_edit.setText(config['default_output_folder'])
        
        self.data_processing_tab.check_preprocess_ready()
        
        # 标注标签页
        if 'default_processed_folder' in config and config['default_processed_folder']:
            self.annotation_tab.processed_folder = config['default_processed_folder']
            self.annotation_tab.processed_path_edit.setText(config['default_processed_folder'])
        
        if 'default_classes' in config and config['default_classes']:
            self.annotation_tab.defect_classes = config['default_classes'].copy()
            self.annotation_tab.class_list.clear()
            for class_name in config['default_classes']:
                self.annotation_tab.class_list.addItem(class_name)
        
        self.annotation_tab.check_annotation_ready()
        
        # 训练标签页
        if 'default_annotation_folder' in config and config['default_annotation_folder']:
            self.training_tab.annotation_folder = config['default_annotation_folder']
            self.training_tab.annotation_path_edit.setText(config['default_annotation_folder'])
        
        self.training_tab.check_training_ready()
        
        # 预测标签页
        if 'default_model_file' in config and config['default_model_file']:
            self.prediction_tab.model_file = config['default_model_file']
            self.prediction_tab.model_path_edit.setText(config['default_model_file'])
        
        if 'default_class_info_file' in config and config['default_class_info_file']:
            self.prediction_tab.class_info_file = config['default_class_info_file']
            self.prediction_tab.class_info_path_edit.setText(config['default_class_info_file'])
        
        self.prediction_tab.check_model_ready()
    
    def goto_annotation_tab(self):
        """切换到标注选项卡"""
        self.tabs.setCurrentWidget(self.annotation_tab)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 确保在关闭窗口时停止TensorBoard进程
        if hasattr(self.evaluation_tab, 'stop_tensorboard'):
            self.evaluation_tab.stop_tensorboard()
        super().closeEvent(event)
        
    def update_prediction_result(self, result):
        """更新预测结果"""
        if hasattr(self, 'prediction_tab'):
            self.prediction_tab.update_prediction_result(result)
        self.update_status("预测完成") 