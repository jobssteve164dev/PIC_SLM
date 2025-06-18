from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QVBoxLayout, QWidget,
                           QStatusBar, QProgressBar, QLabel, QApplication, QMessageBox, QPushButton)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
import os
import sys
import json
from typing import Dict

# 导入统一的配置路径工具
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))
from utils.config_path import get_config_file_path

from .data_processing_tab import DataProcessingTab
from .annotation_tab import AnnotationTab
from .training_tab import TrainingTab
from .prediction_tab import PredictionTab
from .settings_tab import SettingsTab
from .evaluation_tab import EvaluationTab
from .about_tab import AboutTab
from .dataset_evaluation_tab import DatasetEvaluationTab  # 导入新的数据集评估标签页
from .model_analysis_tab import ModelAnalysisTab  # 导入新的模型分析标签页
from .base_tab import BaseTab

# 导入预处理线程
from ..image_processing.preprocessing_thread import PreprocessingThread

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
        self.setWindowTitle('图片模型训练系统 - AGPL-3.0许可')
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化配置
        self.config = {}
        
        # 初始化预处理线程
        self.preprocessing_thread = None
        
        # 初始化UI
        self.init_ui()
        
        # 加载配置
        self.load_config()
        
        # 确保所有tab都已收到配置信息，使用定时器延迟一点再次强制应用配置
        QTimer.singleShot(100, self._ensure_all_tabs_configured)

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
        
        # 连接标签页切换信号，确保每次切换都刷新布局
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        # 创建各个标签页
        self.data_processing_tab = DataProcessingTab(self.tabs, self)
        print(f"数据处理标签页已创建: {self.data_processing_tab}")
        print(f"数据处理标签页预处理按钮: {hasattr(self.data_processing_tab, 'preprocess_btn')}")
        self.annotation_tab = AnnotationTab(self.tabs, self)
        self.training_tab = TrainingTab(self.tabs, self)
        self.prediction_tab = PredictionTab(self.tabs, self)
        self.evaluation_tab = EvaluationTab(self.tabs, self)
        self.settings_tab = SettingsTab(self.tabs, self)
        self.about_tab = AboutTab(self.tabs, self)
        self.dataset_evaluation_tab = DatasetEvaluationTab(self.tabs, self)  # 创建数据集评估标签页
        self.model_analysis_tab = ModelAnalysisTab(self.tabs, self)  # 创建模型分析标签页
        
        # 添加标签页（调整顺序，将模型预测放在模型评估与可视化后面）
        self.tabs.addTab(self.data_processing_tab, "图像预处理")
        self.tabs.addTab(self.annotation_tab, "图像标注")
        self.tabs.addTab(self.dataset_evaluation_tab, "数据集评估")  # 移动到图像标注后面
        self.tabs.addTab(self.training_tab, "模型训练")
        self.tabs.addTab(self.evaluation_tab, "模型评估与可视化")
        self.tabs.addTab(self.model_analysis_tab, "模型分析")  # 添加新的模型分析标签页
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
        
        self.dataset_evaluation_tab.status_updated.connect(self.update_status)  # 连接数据集评估标签页的信号
        self.dataset_evaluation_tab.progress_updated.connect(self.update_progress)
        
        self.model_analysis_tab.status_updated.connect(self.update_status)  # 连接模型分析标签页的信号
        self.model_analysis_tab.progress_updated.connect(self.update_progress)

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
        """预处理完成时的处理"""
        print("MainWindow.preprocessing_finished被调用")
        
        # 重新启用预处理按钮
        if hasattr(self.data_processing_tab, 'enable_preprocess_button'):
            print("调用data_processing_tab.enable_preprocess_button")
            self.data_processing_tab.enable_preprocess_button()
        
        # 强制设置UI状态
        if hasattr(self.data_processing_tab, 'preprocess_btn'):
            # 确保源文件夹和输出文件夹已设置
            if (hasattr(self.data_processing_tab, 'source_folder') and 
                hasattr(self.data_processing_tab, 'output_folder') and
                self.data_processing_tab.source_folder and 
                self.data_processing_tab.output_folder):
                # 重新启用按钮
                self.data_processing_tab.preprocess_btn.setEnabled(True)
                # 强制更新UI
                self.data_processing_tab.update()
                self.data_processing_tab.repaint()
                print("MainWindow: 预处理按钮已重新启用")
            else:
                print("MainWindow: 源文件夹或输出文件夹未设置，不启用预处理按钮")
        
        # 设置一个延迟定时器，确保按钮状态得到更新
        QTimer.singleShot(500, self._try_enable_button_again)
        
        # 清理线程资源
        if self.preprocessing_thread:
            self.preprocessing_thread.quit()
            self.preprocessing_thread.wait()
            self.preprocessing_thread = None
            print("MainWindow: 预处理线程已清理")
        
        # 显示完成提示
        self.update_status("图片预处理完成！")
        QMessageBox.information(self, "完成", "图片预处理已完成！")

    def _try_enable_button_again(self):
        """尝试再次启用按钮的定时器方法"""
        try:
            print("MainWindow._try_enable_button_again被调用")
            if hasattr(self.data_processing_tab, 'preprocess_btn'):
                if (hasattr(self.data_processing_tab, 'source_folder') and 
                    hasattr(self.data_processing_tab, 'output_folder') and
                    self.data_processing_tab.source_folder and 
                    self.data_processing_tab.output_folder):
                    
                    # 多次尝试启用按钮，确保成功
                    for i in range(3):
                        self.data_processing_tab.preprocess_btn.setEnabled(True)
                        self.data_processing_tab.preprocess_btn.update()
                        self.data_processing_tab.preprocess_btn.repaint()
                        QApplication.processEvents()
                        print(f"MainWindow: 第{i+1}次尝试启用预处理按钮")
                    
                    # 检查预处理按钮是否存在并且是可用的
                    if hasattr(self.data_processing_tab, 'preprocess_btn'):
                        current_state = self.data_processing_tab.preprocess_btn.isEnabled()
                        print(f"MainWindow: 预处理按钮当前状态: {current_state}")
                        
                        # 如果按钮仍然不可用，尝试重建按钮
                        if not current_state:
                            print("MainWindow: 按钮状态异常，尝试重建按钮")
                            try:
                                # 获取按钮的父布局
                                parent_layout = self.data_processing_tab.preprocess_btn.parent().layout()
                                if parent_layout:
                                    # 移除原按钮
                                    parent_layout.removeWidget(self.data_processing_tab.preprocess_btn)
                                    self.data_processing_tab.preprocess_btn.deleteLater()
                                    
                                    # 创建新按钮
                                    new_btn = QPushButton("开始预处理")
                                    new_btn.clicked.connect(self.data_processing_tab.preprocess_images)
                                    new_btn.setEnabled(True)
                                    new_btn.setMinimumWidth(200)
                                    new_btn.setMinimumHeight(40)
                                    
                                    # 添加新按钮到布局
                                    parent_layout.addWidget(new_btn)
                                    
                                    # 更新引用
                                    self.data_processing_tab.preprocess_btn = new_btn
                                    print("已重建预处理按钮")
                            except Exception as e:
                                print(f"尝试重建按钮时出错: {str(e)}")
                        
                    # 全局刷新UI
                    self.update()
                    QApplication.processEvents()

        except Exception as e:
            print(f"MainWindow._try_enable_button_again发生错误: {str(e)}")

    def on_image_preprocessing_started(self, params):
        """当图像预处理开始时的处理函数"""
        print("MainWindow: 收到图像预处理开始信号")
        self.update_status("正在初始化图像预处理线程...")
        
        # 创建新的预处理线程
        self.preprocessing_thread = PreprocessingThread(self)
        
        # 连接线程信号
        self.preprocessing_thread.progress_updated.connect(self.update_progress)
        self.preprocessing_thread.status_updated.connect(self.update_status)
        self.preprocessing_thread.preprocessing_finished.connect(self.preprocessing_finished)
        self.preprocessing_thread.preprocessing_error.connect(self.on_preprocessing_error)
        
        # 设置预处理参数并启动线程
        self.preprocessing_thread.setup_preprocessing(params)
        self.preprocessing_thread.start()
        
        print("MainWindow: 图像预处理线程已启动")
    
    def on_preprocessing_error(self, error_msg):
        """处理预处理错误"""
        print(f"MainWindow: 预处理错误 - {error_msg}")
        self.update_status(f"预处理错误: {error_msg}")
        QMessageBox.critical(self, "预处理错误", error_msg)
        
        # 重新启用预处理按钮
        self.preprocessing_finished()
    
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
        try:
            # 准备批量预测参数，匹配predictor.batch_predict的参数格式
            batch_params = {
                'source_folder': params.get('input_folder'),
                'target_folder': params.get('output_folder'),
                'confidence_threshold': params.get('threshold', 0.5) * 100,  # 转换为百分比
                'copy_mode': 'copy',  # 默认复制模式
                'create_subfolders': True  # 创建子文件夹
            }
            
            self.update_status("开始批量预测...")
            # 调用predictor的批量预测方法
            self.worker.predictor.batch_predict(batch_params)
        except Exception as e:
            self.update_status(f"批量预测启动失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def on_batch_prediction_stopped(self):
        """批量预测停止时调用"""
        try:
            # 调用predictor的停止批量处理方法
            self.worker.predictor.stop_batch_processing()
            self.update_status("批量预测已停止")
        except Exception as e:
            self.update_status(f"停止批量预测失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_config(self):
        """加载配置"""
        try:
            # 获取配置文件路径（使用统一工具）
            config_file = get_config_file_path()
            
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
        print(f"MainWindow.apply_config: 源文件夹配置 = {config.get('default_source_folder', 'NOT_SET')}")
        print(f"MainWindow.apply_config: 输出文件夹配置 = {config.get('default_output_folder', 'NOT_SET')}")
        # 保存配置到实例变量中，以便其他标签页可以访问
        self.config = config
        
        # 将配置应用到各个标签页
        
        # 数据处理标签页
        if hasattr(self, 'data_processing_tab'):
            if 'default_source_folder' in config and config['default_source_folder']:
                # 设置UI输入框
                if hasattr(self.data_processing_tab, 'source_path_edit'):
                    self.data_processing_tab.source_path_edit.setText(config['default_source_folder'])
                # 设置实例变量 - 这是关键的修复点
                self.data_processing_tab.source_folder = config['default_source_folder']
                print(f"MainWindow: 设置源文件夹路径: {config['default_source_folder']}")
            
            if 'default_output_folder' in config and config['default_output_folder']:
                # 设置UI输入框
                if hasattr(self.data_processing_tab, 'output_path_edit'):
                    self.data_processing_tab.output_path_edit.setText(config['default_output_folder'])
                # 设置实例变量 - 这是关键的修复点
                self.data_processing_tab.output_folder = config['default_output_folder']
                print(f"MainWindow: 设置输出文件夹路径: {config['default_output_folder']}")
            
            # 在设置了source_folder和output_folder后，检查是否可以开始预处理
            if hasattr(self.data_processing_tab, 'check_preprocess_ready'):
                self.data_processing_tab.check_preprocess_ready()
                print("MainWindow: 已调用data_processing_tab.check_preprocess_ready()方法")
            
            # 如果标签页有apply_config方法，调用它来处理所有配置
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
        
        # 数据集评估标签页
        if hasattr(self, 'dataset_evaluation_tab'):
            # 将配置应用到数据集评估界面
            if hasattr(self.dataset_evaluation_tab, 'apply_config'):
                self.dataset_evaluation_tab.apply_config(config)
                print("MainWindow: 已调用dataset_evaluation_tab.apply_config方法")
        
        # 模型分析标签页
        if hasattr(self, 'model_analysis_tab'):
            # 将配置应用到模型分析界面
            if hasattr(self.model_analysis_tab, 'apply_config'):
                self.model_analysis_tab.apply_config(config)
                print("MainWindow: 已调用model_analysis_tab.apply_config方法")
        
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
            
            # 在设置了模型文件和类别信息文件后，检查是否可以加载模型
            if hasattr(self.prediction_tab, 'check_model_ready'):
                self.prediction_tab.check_model_ready()
                print("MainWindow: 已调用prediction_tab.check_model_ready()方法")
                    
        print("MainWindow.apply_config应用完成")

    def goto_annotation_tab(self):
        """切换到标注选项卡"""
        self.tabs.setCurrentWidget(self.annotation_tab)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 确保在关闭窗口时停止TensorBoard进程
        if hasattr(self, 'evaluation_tab') and hasattr(self.evaluation_tab, 'stop_tensorboard'):
            try:
                print("MainWindow: 正在停止TensorBoard进程...")
                self.evaluation_tab.stop_tensorboard()
            except Exception as e:
                print(f"MainWindow: 停止TensorBoard失败: {str(e)}")
                
        # 额外确保通过操作系统命令终止所有TensorBoard进程
        try:
            print("MainWindow: 正在确保所有TensorBoard进程终止...")
            import subprocess, os
            if os.name == 'nt':  # Windows
                subprocess.call(['taskkill', '/F', '/IM', 'tensorboard.exe'], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:  # Linux/Mac
                subprocess.call("pkill -f tensorboard", shell=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"MainWindow: 终止所有TensorBoard进程时发生错误: {str(e)}")
        
        super().closeEvent(event)
        
    def update_prediction_result(self, result):
        """更新预测结果"""
        if hasattr(self, 'prediction_tab'):
            self.prediction_tab.update_prediction_result(result)
        self.update_status("预测完成")

    # 添加处理标签页切换的方法
    def on_tab_changed(self, index):
        """处理标签页切换事件"""
        current_tab = self.tabs.widget(index)
        if current_tab:
            # 立即应用一次快速布局更新
            current_tab.update()
            self.update()
            
            # 强制处理所有待处理的事件
            QApplication.processEvents()
            
            # 如果当前标签页有refresh_layout方法，延迟调用以确保布局完全刷新
            if hasattr(current_tab, 'refresh_layout'):
                # 使用多个定时器，在不同的时间点尝试刷新布局，以确保最终布局正确
                QTimer.singleShot(10, current_tab.refresh_layout)
                QTimer.singleShot(100, current_tab.refresh_layout)
                
                # 如果是设置标签页，额外再延迟一次刷新以处理可能的问题
                if current_tab.__class__.__name__ == 'SettingsTab':
                    QTimer.singleShot(200, current_tab.refresh_layout)
                    # 调整窗口大小，触发布局重新计算
                    QTimer.singleShot(300, lambda: self._force_resize())
    
    def _force_resize(self):
        """强制重新调整窗口大小，以触发布局重新计算"""
        # 保存当前大小
        current_size = self.size()
        # 稍微改变大小
        self.resize(current_size.width() + 1, current_size.height())
        # 恢复原来的大小
        QTimer.singleShot(50, lambda: self.resize(current_size))
        
    def _ensure_all_tabs_configured(self):
        """确保所有tab都正确配置"""
        print("MainWindow._ensure_all_tabs_configured: 开始强制确保所有tab配置正确...")
        
        if hasattr(self, 'config') and self.config:
            print(f"MainWindow._ensure_all_tabs_configured: 使用已加载的配置")
            
            # 强制重新应用配置到数据处理tab
            if hasattr(self, 'data_processing_tab'):
                print("MainWindow._ensure_all_tabs_configured: 强制配置数据处理tab...")
                if hasattr(self.data_processing_tab, 'apply_config'):
                    self.data_processing_tab.apply_config(self.config)
                else:
                    # 手动设置关键配置
                    if 'default_source_folder' in self.config and self.config['default_source_folder']:
                        self.data_processing_tab.source_folder = self.config['default_source_folder']
                        if hasattr(self.data_processing_tab, 'source_path_edit'):
                            self.data_processing_tab.source_path_edit.setText(self.config['default_source_folder'])
                    
                    if 'default_output_folder' in self.config and self.config['default_output_folder']:
                        self.data_processing_tab.output_folder = self.config['default_output_folder']
                        if hasattr(self.data_processing_tab, 'output_path_edit'):
                            self.data_processing_tab.output_path_edit.setText(self.config['default_output_folder'])
                    
                    # 检查预处理准备状态
                    if hasattr(self.data_processing_tab, 'check_preprocess_ready'):
                        self.data_processing_tab.check_preprocess_ready()
                
                print(f"MainWindow._ensure_all_tabs_configured: 数据处理tab配置完成")
                print(f"  源文件夹: '{getattr(self.data_processing_tab, 'source_folder', 'NOT_SET')}'")
                print(f"  输出文件夹: '{getattr(self.data_processing_tab, 'output_folder', 'NOT_SET')}'")
                print(f"  预处理按钮状态: {getattr(self.data_processing_tab, 'preprocess_btn', None) and self.data_processing_tab.preprocess_btn.isEnabled()}")
            
            # 同样确保其他tab的配置
            for tab_name in ['annotation_tab', 'training_tab', 'prediction_tab', 'evaluation_tab', 'dataset_evaluation_tab', 'model_analysis_tab']:
                if hasattr(self, tab_name):
                    tab = getattr(self, tab_name)
                    if hasattr(tab, 'apply_config'):
                        print(f"MainWindow._ensure_all_tabs_configured: 强制配置 {tab_name}...")
                        tab.apply_config(self.config)
        else:
            print("MainWindow._ensure_all_tabs_configured: 警告 - 没有可用的配置")

    def showEvent(self, event):
        """窗口显示事件，确保所有标签页布局正确"""
        super().showEvent(event)
        
        # 当窗口首次显示时，强制刷新当前标签页的布局
        current_index = self.tabs.currentIndex()
        current_tab = self.tabs.widget(current_index)
        
        # 先尝试立即更新
        self.update()
        QApplication.processEvents()
        
        # 然后延迟执行多次布局刷新
        if current_tab and hasattr(current_tab, 'refresh_layout'):
            QTimer.singleShot(100, current_tab.refresh_layout)
            QTimer.singleShot(300, current_tab.refresh_layout)
            QTimer.singleShot(500, lambda: self._force_resize()) 