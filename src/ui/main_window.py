from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QVBoxLayout, QWidget,
                           QStatusBar, QProgressBar, QLabel, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import os
import sys
import json

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
    prediction_started = pyqtSignal(dict)  # 修改为接收字典参数
    image_preprocessing_started = pyqtSignal(dict)
    annotation_started = pyqtSignal(str)
    create_class_folders_signal = pyqtSignal(str, list)  # 添加创建类别文件夹信号

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
        # 移除最大宽度限制，使进度条铺满窗口
        # self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()  # 默认隐藏进度条
        # 使用addWidget而不是addPermanentWidget，并添加stretch参数使进度条填满状态栏
        self.statusBar.addWidget(self.progress_bar, 1)  # stretch=1使进度条占据所有可用空间
        
        # 连接信号
        self.connect_signals()
        
    def connect_signals(self):
        """连接信号和槽"""
        # 连接各个标签页的状态更新信号
        self.data_processing_tab.status_updated.connect(self.update_status)
        self.data_processing_tab.progress_updated.connect(self.update_progress)
        self.data_processing_tab.image_preprocessing_started.connect(self.on_image_preprocessing_started)
        # 连接创建类别文件夹信号
        self.data_processing_tab.create_class_folders_signal.connect(self.on_create_class_folders)
        
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
        """更新状态栏消息"""
        if hasattr(self, 'status_label'):
            self.status_label.setText(message)
            
    def update_progress(self, value):
        """更新进度条"""
        if hasattr(self, 'progress_bar'):
            if value > 0:
                self.progress_bar.show()
                self.progress_bar.setValue(value)
            else:
                self.progress_bar.hide()

    def preprocessing_finished(self):
        """处理图像预处理完成后的操作"""
        # 重新启用预处理按钮
        if hasattr(self, 'data_processing_tab') and self.data_processing_tab:
            self.data_processing_tab.enable_preprocess_button()
        self.update_status("预处理完成！")
        
    def on_image_preprocessing_started(self, params):
        """当图像预处理开始时的处理函数"""
        self.update_status("开始图像预处理...")
        self.image_preprocessing_started.emit(params)
    
    def on_create_class_folders(self, base_folder, class_names):
        """创建类别文件夹的处理函数"""
        self.update_status(f"正在创建类别文件夹: {len(class_names)} 个")
        # 将信号传递给ImagePreprocessor
        self.create_class_folders_signal.emit(base_folder, class_names)
    
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

    def on_prediction_started(self, predict_params):
        """预测开始时调用"""
        try:
            # 使用传入的参数进行预测
            self.worker.predictor.predict(
                image_path=predict_params['image_path'],
                top_k=predict_params['top_k']
            )
        except Exception as e:
            self.update_status(f"预测出错: {str(e)}")
            import traceback
            traceback.print_exc()

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
        try:
            # 获取配置文件路径
            config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
            
            # 如果配置文件存在，则加载配置
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 应用配置
                self.apply_config(config)
                print(f"MainWindow成功加载配置文件: {config_file}")
                print(f"配置内容: {config}")
                
        except Exception as e:
            print(f"MainWindow加载配置失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def apply_config(self, config):
        """应用配置"""
        print(f"MainWindow.apply_config被调用，配置内容: {config}")
        # 保存配置到实例变量中，以便其他标签页可以访问
        self.config = config
        
        # 将配置应用到各个标签页
        
        # 数据处理标签页
        if hasattr(self, 'data_processing_tab'):
            if 'default_source_folder' in config and config['default_source_folder']:
                if hasattr(self.data_processing_tab, 'source_path_edit'):
                    self.data_processing_tab.source_path_edit.setText(config['default_source_folder'])
                if hasattr(self.data_processing_tab, 'source_folder'):
                    self.data_processing_tab.source_folder = config['default_source_folder']
            
            if 'default_output_folder' in config and config['default_output_folder']:
                if hasattr(self.data_processing_tab, 'output_path_edit'):
                    self.data_processing_tab.output_path_edit.setText(config['default_output_folder'])
                if hasattr(self.data_processing_tab, 'output_folder'):
                    self.data_processing_tab.output_folder = config['default_output_folder']
            
            # 在设置了source_folder和output_folder后，检查是否可以开始预处理
            if hasattr(self.data_processing_tab, 'check_preprocess_ready'):
                self.data_processing_tab.check_preprocess_ready()
                print("MainWindow: 已调用data_processing_tab.check_preprocess_ready()方法")
            
            # 添加默认类别设置应用
            if 'default_classes' in config and config['default_classes']:
                print(f"MainWindow: 向数据处理标签页应用默认类别: {config['default_classes']}")
                if hasattr(self.data_processing_tab, 'defect_classes'):
                    self.data_processing_tab.defect_classes = config['default_classes'].copy()
                # 更新类别列表
                if hasattr(self.data_processing_tab, 'class_list'):
                    self.data_processing_tab.class_list.clear()
                    for class_name in config['default_classes']:
                        self.data_processing_tab.class_list.addItem(class_name)
                    print(f"MainWindow: 数据处理标签页的类别列表已更新，现在有 {self.data_processing_tab.class_list.count()} 个类别")
                elif hasattr(self.data_processing_tab, 'defect_class_list'):
                    self.data_processing_tab.defect_class_list.clear()
                    for class_name in config['default_classes']:
                        self.data_processing_tab.defect_class_list.addItem(class_name)
                    print(f"MainWindow: 数据处理标签页的类别列表已更新，现在有 {self.data_processing_tab.defect_class_list.count()} 个类别")
                
                # 如果标签页有apply_config方法，也调用它
                if hasattr(self.data_processing_tab, 'apply_config'):
                    self.data_processing_tab.apply_config(config)
                    print("MainWindow: 已调用data_processing_tab.apply_config方法")
        
        # 标注标签页
        if hasattr(self, 'annotation_tab'):
            # 将配置应用到标注界面
            if hasattr(self.annotation_tab, 'apply_config'):
                self.annotation_tab.apply_config(config)
        
        # 训练标签页
        if hasattr(self, 'training_tab'):
            # 将配置应用到训练界面
            if hasattr(self.training_tab, 'apply_config'):
                self.training_tab.apply_config(config)
                
        # 评估标签页
        if hasattr(self, 'evaluation_tab'):
            # 将配置应用到评估界面
            if hasattr(self.evaluation_tab, 'apply_config'):
                self.evaluation_tab.apply_config(config)
        
        # 预测标签页
        if hasattr(self, 'prediction_tab'):
            # 应用默认模型文件
            if 'default_model_file' in config and config['default_model_file']:
                if hasattr(self.prediction_tab, 'model_path_edit'):
                    self.prediction_tab.model_path_edit.setText(config['default_model_file'])
                if hasattr(self.prediction_tab, 'model_file'):
                    self.prediction_tab.model_file = config['default_model_file']
            
            # 应用默认类别信息文件
            if 'default_class_info_file' in config and config['default_class_info_file']:
                if hasattr(self.prediction_tab, 'class_info_path_edit'):
                    self.prediction_tab.class_info_path_edit.setText(config['default_class_info_file'])
                if hasattr(self.prediction_tab, 'class_info_file'):
                    self.prediction_tab.class_info_file = config['default_class_info_file']
                    
                    # 如果类别信息文件有效，加载类别信息
                    if os.path.exists(config['default_class_info_file']):
                        try:
                            with open(config['default_class_info_file'], 'r', encoding='utf-8') as f:
                                class_info = json.load(f)
                                if hasattr(self.prediction_tab, 'class_info'):
                                    self.prediction_tab.class_info = class_info
                        except Exception as e:
                            print(f"加载类别信息文件失败: {str(e)}")
                            
        print("MainWindow.apply_config应用完成")

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