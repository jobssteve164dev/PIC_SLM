from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QVBoxLayout, QWidget,
                           QStatusBar, QProgressBar, QLabel, QApplication, QMessageBox, QPushButton,
                           QSystemTrayIcon, QMenu, QAction, QCheckBox, QHBoxLayout)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon
import os
import sys
import json
from typing import Dict

# 导入统一的配置路径工具
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))
from utils.config_path import get_config_file_path
from utils.config_manager import config_manager

from .data_processing_tab import DataProcessingTab
from .annotation_tab import AnnotationTab
from .training_tab import TrainingTab
from .prediction_tab import PredictionTab
from .settings_tab import SettingsTab
from .evaluation_tab import EvaluationTab
from .about_tab import AboutTab
from .dataset_evaluation_tab import DatasetEvaluationTab  # 导入新的数据集评估标签页
from .model_analysis_tab import ModelAnalysisTab  # 导入新的模型分析标签页
from .model_factory_tab import ModelFactoryTab  # 导入新的模型工厂标签页
from .base_tab import BaseTab

# 导入预处理线程
from ..image_processing.preprocessing_thread import PreprocessingThread
from ..training_components.intelligent_training_manager import IntelligentTrainingManager

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
        
        # 初始化配置标志，避免重复初始化
        self._config_initialized = False
        self.config = {}
        
        # 初始化预处理线程
        self.preprocessing_thread = None
        # 智能训练管理器
        self.intelligent_manager = IntelligentTrainingManager()
        
        # 初始化系统托盘
        self.init_system_tray()
        
        # 初始化UI
        self.init_ui()
        
        # 加载配置 - 只在初始化时加载一次
        self.load_config()
        
        # 标记配置已初始化
        self._config_initialized = True
        
        print("MainWindow: 初始化完成，跳过重复配置检查")

    def init_system_tray(self):
        """初始化系统托盘"""
        # 检查系统是否支持托盘图标
        if not QSystemTrayIcon.isSystemTrayAvailable():
            QMessageBox.critical(None, "系统托盘", "系统不支持托盘图标功能")
            return
        
        # 创建托盘图标
        self.tray_icon = QSystemTrayIcon(self)
        
        # 设置托盘图标
        icon_path = os.path.join(os.path.dirname(__file__), 'icons', 'app.png')
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
        else:
            # 如果图标文件不存在，尝试创建它
            try:
                from .create_icon import create_app_icon
                created_icon_path = create_app_icon()
                if os.path.exists(created_icon_path):
                    icon = QIcon(created_icon_path)
                else:
                    icon = self.style().standardIcon(self.style().SP_ComputerIcon)
            except Exception as e:
                print(f"创建图标失败: {e}")
                # 使用默认图标
                icon = self.style().standardIcon(self.style().SP_ComputerIcon)
        self.tray_icon.setIcon(icon)
        
        # 同时设置窗口图标
        self.setWindowIcon(icon)
        
        # 创建托盘菜单
        self.tray_menu = QMenu()
        
        # 显示/隐藏窗口动作
        self.show_action = QAction("显示主窗口", self)
        self.show_action.triggered.connect(self.show_window)
        self.tray_menu.addAction(self.show_action)
        
        self.hide_action = QAction("隐藏到托盘", self)
        self.hide_action.triggered.connect(self.hide_to_tray)
        self.tray_menu.addAction(self.hide_action)
        
        self.tray_menu.addSeparator()
        
        # 退出动作
        self.quit_action = QAction("退出程序", self)
        self.quit_action.triggered.connect(self.quit_application)
        self.tray_menu.addAction(self.quit_action)
        
        # 设置托盘菜单
        self.tray_icon.setContextMenu(self.tray_menu)
        
        # 设置托盘图标提示
        self.tray_icon.setToolTip("图片模型训练系统")
        
        # 连接托盘图标双击事件
        self.tray_icon.activated.connect(self.tray_icon_activated)
        
        # 显示托盘图标
        self.tray_icon.show()
        
        # 初始化最小化到托盘的选项状态
        self.minimize_to_tray_enabled = True

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
        self.model_factory_tab = ModelFactoryTab(self.tabs, self)  # 创建模型工厂标签页
        
        # 添加标签页（调整顺序，将模型预测放在模型评估与可视化后面）
        self.tabs.addTab(self.data_processing_tab, "图像预处理")
        self.tabs.addTab(self.annotation_tab, "图像标注")
        self.tabs.addTab(self.dataset_evaluation_tab, "数据集评估")  # 移动到图像标注后面
        self.tabs.addTab(self.training_tab, "模型训练")
        self.tabs.addTab(self.evaluation_tab, "模型评估与可视化")
        self.tabs.addTab(self.model_analysis_tab, "模型分析")  # 添加新的模型分析标签页
        self.tabs.addTab(self.model_factory_tab, "🏭 AI模型工厂")  # 添加新的模型工厂标签页
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
        # 连接智能训练管理器信号
        try:
            self.intelligent_manager.status_updated.connect(self.update_status)
            self.intelligent_manager.error_occurred.connect(lambda e: QMessageBox.warning(self, "智能训练错误", e))
            # 智能训练事件桥接到评估曲线
            self.intelligent_manager.analysis_completed.connect(lambda d: self.update_status("AI分析完成"))
        except Exception:
            pass
        
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
        # 连接训练进度到智能训练管理器
        self.training_tab.training_progress_updated.connect(self.intelligent_manager.update_training_progress)
        
        # 智能训练控件信号桥接
        if hasattr(self.training_tab, 'intelligent_widget'):
            self.training_tab.intelligent_widget.start_monitoring_requested.connect(self.on_start_intelligent_monitoring)
            self.training_tab.intelligent_widget.stop_monitoring_requested.connect(self.on_stop_intelligent_monitoring)
            self.training_tab.intelligent_widget.restart_training_requested.connect(self.on_restart_training_with_params)
        
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
        # 不直接连接settings_saved信号到apply_config，避免重复应用配置
        # self.settings_tab.settings_saved.connect(self.apply_config)
        
        self.dataset_evaluation_tab.status_updated.connect(self.update_status)  # 连接数据集评估标签页的信号
        self.dataset_evaluation_tab.progress_updated.connect(self.update_progress)
        
        self.model_analysis_tab.status_updated.connect(self.update_status)  # 连接模型分析标签页的信号
        self.model_analysis_tab.progress_updated.connect(self.update_progress)
        
        # 连接模型工厂标签页的信号
        self.model_factory_tab.status_updated.connect(self.update_status)
        self.model_factory_tab.progress_updated.connect(self.update_progress)

    # 智能训练：开始监控
    def on_start_intelligent_monitoring(self, it_config: Dict):
        try:
            # 从主配置中加载智能训练配置
            self._load_intelligent_training_config_from_main_config()
            
            # 将UI配置合并到管理器（UI配置优先级更高）
            self.intelligent_manager.config.update(it_config)
            
            # 绑定模型训练器
            if hasattr(self, 'worker') and hasattr(self.worker, 'model_trainer'):
                self.intelligent_manager.set_model_trainer(self.worker.model_trainer)
            # 统一构建完整训练配置（与普通开始训练一致）
            training_config = self._build_training_config_from_ui()
            self.intelligent_manager.start_intelligent_training(training_config)
            # 同步按钮UI
            if hasattr(self.training_tab, 'intelligent_widget'):
                self.training_tab.intelligent_widget.start_monitoring(training_config)
            self.update_status('智能训练监控已启动')
        except Exception as e:
            self.update_status(f'启动智能训练失败: {e}')
    
    def _load_intelligent_training_config_from_main_config(self):
        """从主配置中加载智能训练配置"""
        try:
            from src.config_loader import ConfigLoader
            cfg = ConfigLoader().get_config()
            intelligent_config = cfg.get('intelligent_training', {})
            if intelligent_config:
                self.intelligent_manager.config.update(intelligent_config)
                print(f"从主配置加载智能训练配置: {intelligent_config}")
        except Exception as e:
            print(f"从主配置加载智能训练配置失败: {e}")

    def _build_training_config_from_ui(self) -> Dict:
        """从训练页与集中配置构建完整训练配置，保证两条启动路径一致。"""
        try:
            params = self.training_tab.get_training_params() if hasattr(self, 'training_tab') else {}
            task_type = params.get('task_type', 'classification')
            from src.config_loader import ConfigLoader
            cfg = ConfigLoader().get_config()
            model_save_dir = cfg.get('default_model_save_dir', '') or os.path.join('models', 'saved_models')
            param_save_dir = cfg.get('default_param_save_dir', '') or model_save_dir
            tensorboard_log_dir = cfg.get('default_tensorboard_log_dir', '') or os.path.join('runs', 'tensorboard')
            data_dir = self.training_tab.annotation_folder if hasattr(self, 'training_tab') else ''

            if task_type == 'classification':
                model_name = params.get('model', 'ResNet50')
                batch_size = params.get('batch_size', 32)
                epochs = params.get('epochs', 20)
                learning_rate = params.get('learning_rate', 0.001)
                optimizer = params.get('optimizer', 'Adam')
                use_pretrained = params.get('use_pretrained', True)
                pretrained_path = params.get('pretrained_path', '')
                metrics = params.get('metrics', ['accuracy'])
                weight_decay = params.get('weight_decay', 0.0001)
                lr_scheduler = params.get('lr_scheduler', 'StepLR')
                use_augmentation = params.get('use_augmentation', True)
                early_stopping = params.get('early_stopping', True)
                early_stopping_patience = params.get('early_stopping_patience', 10)
                gradient_clipping = params.get('gradient_clipping', False)
                gradient_clipping_value = params.get('gradient_clipping_value', 1.0)
                mixed_precision = params.get('mixed_precision', True)
                dropout_rate = params.get('dropout_rate', 0.0)
            else:
                model_name = params.get('model', 'YOLOv5')
                batch_size = params.get('batch_size', 16)
                epochs = params.get('epochs', 50)
                learning_rate = params.get('learning_rate', 0.0005)
                optimizer = params.get('optimizer', 'Adam')
                use_pretrained = params.get('use_pretrained', True)
                pretrained_path = params.get('pretrained_path', '')
                metrics = params.get('metrics', ['mAP'])
                weight_decay = params.get('weight_decay', 0.0005)
                lr_scheduler = params.get('lr_scheduler', 'StepLR')
                use_augmentation = params.get('use_augmentation', True)
                early_stopping = params.get('early_stopping', True)
                early_stopping_patience = params.get('early_stopping_patience', 10)
                gradient_clipping = params.get('gradient_clipping', False)
                gradient_clipping_value = params.get('gradient_clipping_value', 1.0)
                mixed_precision = params.get('mixed_precision', True)
                dropout_rate = params.get('dropout_rate', 0.0)
                iou_threshold = params.get('iou_threshold', 0.5)
                conf_threshold = params.get('conf_threshold', 0.25)
                use_mosaic = params.get('use_mosaic', True)
                use_multiscale = params.get('use_multiscale', True)
                use_ema = params.get('use_ema', True)

            use_class_weights = cfg.get('use_class_weights', True)
            weight_strategy = cfg.get('weight_strategy', 'balanced')
            weight_config = {}
            if 'class_weights' in cfg and cfg['class_weights']:
                weight_config['class_weights'] = cfg['class_weights']
            if 'custom_class_weights' in cfg and cfg['custom_class_weights']:
                weight_config['custom_class_weights'] = cfg['custom_class_weights']
            if 'weight_config_file' in cfg and cfg['weight_config_file']:
                weight_config['weight_config_file'] = cfg['weight_config_file']
            if 'all_strategies' in cfg and cfg['all_strategies']:
                weight_config['all_strategies'] = cfg['all_strategies']

            training_config = {
                'data_dir': data_dir,
                'model_name': model_name,
                'num_epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'model_save_dir': os.path.normpath(model_save_dir),
                'default_param_save_dir': os.path.normpath(param_save_dir),
                'tensorboard_log_dir': os.path.normpath(tensorboard_log_dir),
                'task_type': task_type,
                'optimizer': optimizer,
                'use_pretrained': use_pretrained,
                'pretrained_path': pretrained_path,
                'metrics': metrics,
                'use_tensorboard': True,
                'model_note': params.get('model_note', ''),
                'use_class_weights': use_class_weights,
                'weight_strategy': weight_strategy,
                'weight_decay': weight_decay,
                'lr_scheduler': lr_scheduler,
                'use_augmentation': use_augmentation,
                'early_stopping': early_stopping,
                'early_stopping_patience': early_stopping_patience,
                'gradient_clipping': gradient_clipping,
                'gradient_clipping_value': gradient_clipping_value,
                'mixed_precision': mixed_precision,
                'dropout_rate': dropout_rate,
                'activation_function': params.get('activation_function', 'ReLU'),
                'beta1': params.get('beta1', 0.9),
                'beta2': params.get('beta2', 0.999),
                'momentum': params.get('momentum', 0.9),
                'nesterov': params.get('nesterov', False),
                'warmup_enabled': params.get('warmup_enabled', False),
                'warmup_steps': params.get('warmup_steps', 0),
                'warmup_ratio': params.get('warmup_ratio', 0.0),
                'warmup_method': params.get('warmup_method', 'linear'),
                'min_lr_enabled': params.get('min_lr_enabled', False),
                'min_lr': params.get('min_lr', 1e-6),
                'label_smoothing_enabled': params.get('label_smoothing_enabled', False),
                'label_smoothing': params.get('label_smoothing', 0.0),
                'model_ema': params.get('model_ema', False),
                'model_ema_decay': params.get('model_ema_decay', 0.9999),
                'gradient_accumulation_enabled': params.get('gradient_accumulation_enabled', False),
                'gradient_accumulation_steps': params.get('gradient_accumulation_steps', 1),
                'advanced_augmentation_enabled': params.get('advanced_augmentation_enabled', False),
                'cutmix_prob': params.get('cutmix_prob', 0.0),
                'mixup_alpha': params.get('mixup_alpha', 0.0),
                'loss_scaling_enabled': params.get('loss_scaling_enabled', False),
                'loss_scale': params.get('loss_scale', 'dynamic'),
                'static_loss_scale': params.get('static_loss_scale', 128.0),
            }

            training_config.update(weight_config)
            if task_type == 'detection':
                training_config.update({
                    'iou_threshold': iou_threshold,
                    'conf_threshold': conf_threshold,
                    'resolution': params.get('resolution', '640x640'),
                    'nms_threshold': 0.45,
                    'use_fpn': True,
                    'use_mosaic': use_mosaic,
                    'use_multiscale': use_multiscale,
                    'use_ema': use_ema
                })
            return training_config
        except Exception:
            return {
                'data_dir': self.training_tab.annotation_folder if hasattr(self, 'training_tab') else '',
                'model_name': 'ResNet50',
                'num_epochs': 20,
                'batch_size': 32,
                'learning_rate': 0.001,
                'model_save_dir': os.path.join('models', 'saved_models'),
                'tensorboard_log_dir': os.path.join('runs', 'tensorboard'),
                'task_type': 'classification'
            }

    # 智能训练：停止监控
    def on_stop_intelligent_monitoring(self):
        try:
            self.intelligent_manager.stop_intelligent_training()
            try:
                if hasattr(self.training_tab, 'intelligent_widget'):
                    self.training_tab.intelligent_widget.stop_monitoring()
            except Exception:
                pass
            self.update_status('智能训练监控已停止')
        except Exception as e:
            self.update_status(f'停止智能训练失败: {e}')

    # 智能训练：根据建议参数重启
    def on_restart_training_with_params(self, data: Dict):
        try:
            suggested = data.get('suggested_params', {}) if isinstance(data, dict) else {}
            if not suggested:
                self.update_status('无可用的优化参数，忽略重启请求')
                return
            # 将建议交给管理器处理（它内部会更新配置并重启）
            # 直接调用公开接口：合并建议并触发重启
            updated_config = self.intelligent_manager._update_training_config(suggested)
            # 模拟控制器触发的延时重启流程
            self.intelligent_manager._on_training_restart_requested({'new_params': suggested})
        except Exception as e:
            self.update_status(f'处理智能重启失败: {e}')

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
        
        # 修复：调用prepare_training_config方法来正确传递参数
        # 检查是否有worker和model_trainer
        if hasattr(self, 'worker') and hasattr(self.worker, 'model_trainer'):
            # 调用prepare_training_config方法
            if hasattr(self, 'prepare_training_config'):
                self.prepare_training_config(self.worker.model_trainer)
            else:
                print("错误：prepare_training_config方法未找到")
        else:
            print("错误：worker或model_trainer未找到")
        
        # 发射训练开始信号
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
            # 直接使用prediction_tab传递的参数，无需重新构建
            print(f"接收到的批量预测参数: {params}")
            
            self.update_status("开始批量预测...")
            # 调用predictor的批量预测方法
            self.worker.predictor.batch_predict(params)
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
        """加载配置 - 使用集中化配置管理器"""
        try:
            config_file = get_config_file_path()
            print(f"MainWindow: 使用集中化配置管理器加载配置: {config_file}")
            
            # 设置配置管理器路径
            config_manager.set_config_path(config_file)
            
            # 获取配置
            config = config_manager.get_config()
            
            if config:
                print(f"MainWindow: 成功通过配置管理器加载配置，包含 {len(config)} 个配置项")
                self.apply_config(config)
            else:
                print("MainWindow: 配置为空或加载失败")
                
        except Exception as e:
            print(f"MainWindow: 加载配置文件失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def apply_config(self, config):
        """应用配置"""
        # 检查是否已经初始化完成，避免重复应用配置
        if self._config_initialized and self.config == config:
            print("MainWindow.apply_config: 配置未变化，跳过重复应用")
            return
            
        print(f"MainWindow.apply_config被调用，配置内容: {len(config)} 个配置项")
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
        
        # 应用系统托盘配置
        if 'minimize_to_tray' in config:
            minimize_to_tray = config['minimize_to_tray']
            self.set_minimize_to_tray_enabled(minimize_to_tray)
            print(f"MainWindow: 已应用系统托盘配置: {minimize_to_tray}")
                    
        print("MainWindow.apply_config应用完成")

    def goto_annotation_tab(self):
        """切换到标注选项卡"""
        self.tabs.setCurrentWidget(self.annotation_tab)
    
    def set_minimize_to_tray_enabled(self, enabled):
        """设置最小化到托盘功能的启用状态"""
        self.minimize_to_tray_enabled = enabled
        if enabled:
            self.tray_icon.showMessage(
                "系统托盘",
                "程序将在最小化或关闭时隐藏到系统托盘。双击托盘图标可以重新显示窗口。",
                QSystemTrayIcon.Information,
                3000
            )
        else:
            self.tray_icon.showMessage(
                "系统托盘",
                "已关闭最小化到托盘功能。程序将按传统方式最小化和关闭。",
                QSystemTrayIcon.Information,
                2000
            )
    
    def tray_icon_activated(self, reason):
        """处理托盘图标激活事件"""
        if reason == QSystemTrayIcon.DoubleClick:
            self.show_window()
    
    def show_window(self):
        """显示主窗口"""
        self.show()
        self.raise_()
        self.activateWindow()
        # 如果窗口被最小化，恢复正常状态
        if self.isMinimized():
            self.showNormal()
    
    def hide_to_tray(self):
        """隐藏到系统托盘"""
        self.hide()
        self.tray_icon.showMessage(
            "系统托盘",
            "程序已最小化到系统托盘。双击托盘图标可以重新显示窗口。",
            QSystemTrayIcon.Information,
            2000
        )
    
    def quit_application(self):
        """退出应用程序"""
        # 显示确认对话框
        reply = QMessageBox.question(
            self, 
            '确认退出', 
            '确定要退出程序吗？',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 隐藏托盘图标
            if hasattr(self, 'tray_icon'):
                self.tray_icon.hide()
            # 退出应用程序
            QApplication.quit()
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 如果启用了最小化到托盘，则隐藏到托盘而不是关闭
        if self.minimize_to_tray_enabled and hasattr(self, 'tray_icon') and self.tray_icon.isVisible():
            event.ignore()
            self.hide_to_tray()
            return
        
        # 否则执行正常的关闭流程
        print("MainWindow: 开始清理资源...")
        
        # 停止批量预测线程
        if hasattr(self, 'worker') and hasattr(self.worker, 'predictor'):
            try:
                print("MainWindow: 正在停止批量预测线程...")
                if self.worker.predictor.is_batch_prediction_running():
                    self.worker.predictor.stop_batch_processing()
                    # 等待线程结束，最多等待3秒
                    if not self.worker.predictor.wait_for_batch_prediction_to_finish(3000):
                        print("MainWindow: 批量预测线程停止超时，强制清理")
                    self.worker.predictor.cleanup_batch_prediction_thread()
                    print("MainWindow: 批量预测线程已清理")
            except Exception as e:
                print(f"MainWindow: 停止批量预测线程失败: {str(e)}")
        
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
        
        # 隐藏托盘图标
        if hasattr(self, 'tray_icon'):
            self.tray_icon.hide()
        
        print("MainWindow: 资源清理完成")
        super().closeEvent(event)
    
    def changeEvent(self, event):
        """窗口状态改变事件"""
        # 如果窗口被最小化且启用了最小化到托盘
        if (event.type() == event.WindowStateChange and 
            self.isMinimized() and 
            self.minimize_to_tray_enabled and 
            hasattr(self, 'tray_icon') and 
            self.tray_icon.isVisible()):
            
            # 延迟隐藏到托盘，避免闪烁
            QTimer.singleShot(100, self.hide_to_tray)
            event.ignore()
            return
        
        super().changeEvent(event)
        
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
        """确保所有tab都正确配置 - 简化版本，减少重复操作"""
        if not self._config_initialized:
            print("MainWindow._ensure_all_tabs_configured: 初始化尚未完成，跳过")
            return
            
        if not hasattr(self, 'config') or not self.config:
            print("MainWindow._ensure_all_tabs_configured: 没有可用的配置，跳过")
            return
            
        print("MainWindow._ensure_all_tabs_configured: 执行最小必要的配置检查...")
        
        # 只检查关键配置是否正确应用，不重复应用整个配置
        if hasattr(self, 'data_processing_tab'):
            source_set = getattr(self.data_processing_tab, 'source_folder', None)
            output_set = getattr(self.data_processing_tab, 'output_folder', None)
            
            if not source_set or not output_set:
                print("MainWindow._ensure_all_tabs_configured: 数据处理tab缺少关键配置，执行一次修复")
                if hasattr(self.data_processing_tab, 'apply_config'):
                    self.data_processing_tab.apply_config(self.config)
                    
        print("MainWindow._ensure_all_tabs_configured: 配置检查完成")

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