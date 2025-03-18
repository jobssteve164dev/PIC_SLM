from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout,
                           QSizePolicy, QLineEdit, QCheckBox, QMessageBox, QRadioButton, QButtonGroup,
                           QStackedWidget, QScrollArea, QListWidget, QDialog, QTextBrowser)
from PyQt5.QtCore import Qt, pyqtSignal, QUrl
from PyQt5.QtGui import QFont, QDesktopServices
import os
import sys
import platform
from .base_tab import BaseTab
from .training_help_dialog import TrainingHelpDialog
import json

class ModelDownloadDialog(QDialog):
    """模型下载失败时显示的对话框，提供下载链接和文件夹打开按钮"""
    
    def __init__(self, model_name, model_link, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.model_link = model_link
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("预训练模型下载失败")
        self.setMinimumWidth(600)
        self.setMinimumHeight(450)
        
        layout = QVBoxLayout(self)
        
        # 添加说明标签
        title_label = QLabel(f"预训练模型 <b>{self.model_name}</b> 下载失败")
        title_label.setStyleSheet("font-size: 16px; color: #C62828;")
        layout.addWidget(title_label)
        
        # 添加详细说明
        info_text = QTextBrowser()
        info_text.setOpenExternalLinks(True)
        info_text.setStyleSheet("background-color: #F5F5F5; border: 1px solid #E0E0E0;")
        info_text.setHtml(f"""
        <h3>预训练模型下载失败</h3>
        <p>PyTorch无法自动下载预训练模型，可能是由于以下原因：</p>
        <ul>
            <li>网络连接问题（如防火墙限制、代理设置等）</li>
            <li>服务器暂时不可用</li>
            <li>下载超时</li>
        </ul>
        
        <h3>预训练模型的优势</h3>
        <p>使用预训练模型可以显著提高训练效果和速度：</p>
        <ul>
            <li>缩短训练时间，通常减少50%-90%的训练轮数</li>
            <li>提高模型精度，特别是在训练数据量较少的情况下</li>
            <li>更好的泛化能力和特征提取能力</li>
        </ul>
        
        <h3>解决方法</h3>
        <p>您可以：</p>
        <ol>
            <li>手动下载模型文件: <a href="{self.model_link}"><b>{self.model_name}</b> 模型下载链接</a></li>
            <li>将下载的文件放入PyTorch的模型缓存目录</li>
        </ol>
        
        <h4>模型缓存目录通常位于:</h4>
        <ul>
            <li>Windows: <code>%USERPROFILE%\\.cache\\torch\\hub\\checkpoints</code></li>
            <li>Linux: <code>~/.cache/torch/hub/checkpoints</code></li>
            <li>macOS: <code>~/Library/Caches/torch/hub/checkpoints</code> 或 <code>~/.cache/torch/hub/checkpoints</code></li>
        </ul>
        
        <p>放置模型文件后，<b>不需要重命名文件</b>，直接将下载的文件放入目录即可。</p>
        <p>下载完成后，重新开始训练即可使用预训练模型。</p>
        
        <p><i>注意：如果不使用预训练模型，当前训练会继续进行，但可能需要更长时间才能达到相同精度。</i></p>
        """)
        layout.addWidget(info_text)
        
        # 添加按钮
        button_layout = QHBoxLayout()
        
        # 打开下载链接按钮
        open_link_btn = QPushButton("打开下载链接")
        open_link_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        open_link_btn.setMinimumWidth(120)
        open_link_btn.clicked.connect(self.open_download_link)
        
        # 打开缓存目录按钮
        open_cache_btn = QPushButton("打开模型缓存目录")
        open_cache_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        open_cache_btn.setMinimumWidth(150)
        open_cache_btn.clicked.connect(self.open_cache_directory)
        
        # 关闭按钮
        close_btn = QPushButton("继续训练(不使用预训练)")
        close_btn.setStyleSheet("padding: 8px;")
        close_btn.clicked.connect(self.accept)
        
        button_layout.addWidget(open_link_btn)
        button_layout.addWidget(open_cache_btn)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def open_download_link(self):
        """打开模型下载链接"""
        QDesktopServices.openUrl(QUrl(self.model_link))
    
    def open_cache_directory(self):
        """打开PyTorch模型缓存目录"""
        try:
            # 根据不同操作系统找到缓存目录
            cache_dir = self.get_torch_cache_dir()
            
            # 确保目录存在
            os.makedirs(cache_dir, exist_ok=True)
            
            # 打开目录
            if platform.system() == "Windows":
                os.startfile(cache_dir)
            elif platform.system() == "Darwin":  # macOS
                import subprocess
                subprocess.run(["open", cache_dir])
            else:  # Linux
                import subprocess
                subprocess.run(["xdg-open", cache_dir])
            
        except Exception as e:
            QMessageBox.warning(self, "无法打开目录", f"无法打开PyTorch缓存目录: {str(e)}")
    
    def get_torch_cache_dir(self):
        """获取PyTorch缓存目录路径"""
        # 获取用户主目录
        home_dir = os.path.expanduser("~")
        
        # 根据操作系统确定缓存目录
        if platform.system() == "Windows":
            cache_dir = os.path.join(os.environ.get("USERPROFILE", home_dir), ".cache", "torch", "hub", "checkpoints")
        elif platform.system() == "Darwin":  # macOS
            # 先尝试macOS特有的缓存目录，如果不存在则使用通用的.cache目录
            macos_cache = os.path.join(home_dir, "Library", "Caches", "torch", "hub", "checkpoints")
            if os.path.exists(os.path.dirname(macos_cache)):
                cache_dir = macos_cache
            else:
                cache_dir = os.path.join(home_dir, ".cache", "torch", "hub", "checkpoints")
        else:  # Linux
            cache_dir = os.path.join(home_dir, ".cache", "torch", "hub", "checkpoints")
        
        return cache_dir

class TrainingTab(BaseTab):
    """训练标签页，负责模型训练功能"""
    
    # 定义信号
    training_started = pyqtSignal()
    training_progress_updated = pyqtSignal(int, dict)  # 添加训练进度更新信号
    training_stopped = pyqtSignal()  # 添加训练停止信号
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent, main_window)
        self.annotation_folder = ""
        self.task_type = "classification"  # 默认为图片分类任务
        self.init_ui()
        
        # 尝试直接从配置文件加载默认标注文件夹
        try:
            config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
            print(f"训练标签页直接加载配置文件: {config_file}")
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if 'default_annotation_folder' in config and config['default_annotation_folder']:
                        self.annotation_folder = config['default_annotation_folder']
                        print(f"训练标签页直接设置标注文件夹: {self.annotation_folder}")
                        
                        # 设置相应的路径输入框
                        if hasattr(self, 'classification_path_edit'):
                            self.classification_path_edit.setText(self.annotation_folder)
                            print(f"直接设置分类路径输入框: {self.annotation_folder}")
                        
                        if hasattr(self, 'detection_path_edit'):
                            self.detection_path_edit.setText(self.annotation_folder)
                            print(f"直接设置检测路径输入框: {self.annotation_folder}")
                        
                        # 检查是否可以开始训练
                        self.check_training_ready()
        except Exception as e:
            print(f"训练标签页直接加载配置文件出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def apply_config(self, config):
        """应用配置设置到训练标签页"""
        try:
            print(f"训练标签页apply_config被调用，配置内容：{config}")
            
            # 加载默认标注文件夹路径
            if 'default_annotation_folder' in config and config['default_annotation_folder']:
                print(f"发现默认标注文件夹配置: {config['default_annotation_folder']}")
                self.annotation_folder = config['default_annotation_folder']
                
                # 根据当前任务类型设置相应的路径输入框
                if hasattr(self, 'classification_path_edit'):
                    print(f"设置classification_path_edit: {self.annotation_folder}")
                    self.classification_path_edit.setText(self.annotation_folder)
                else:
                    print("classification_path_edit尚未创建")
                    
                if hasattr(self, 'detection_path_edit'):
                    print(f"设置detection_path_edit: {self.annotation_folder}")
                    self.detection_path_edit.setText(self.annotation_folder)
                else:
                    print("detection_path_edit尚未创建")
                
                # 检查是否满足开始训练的条件
                self.check_training_ready()
            else:
                print("配置中没有找到default_annotation_folder或为空值")
                
            print(f"已应用训练标签页配置，标注文件夹: {self.annotation_folder}")
        except Exception as e:
            print(f"应用训练标签页配置时出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def init_ui(self):
        """初始化UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self.scroll_content)
        
        # 添加标题
        title_label = QLabel("模型训练")
        title_label.setFont(QFont('微软雅黑', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 添加训练任务类型选择
        task_group = QGroupBox("训练任务")
        task_layout = QHBoxLayout()
        
        # 创建单选按钮
        self.classification_radio = QRadioButton("图片分类")
        self.detection_radio = QRadioButton("目标检测")
        
        # 创建按钮组
        self.task_button_group = QButtonGroup()
        self.task_button_group.addButton(self.classification_radio, 0)
        self.task_button_group.addButton(self.detection_radio, 1)
        self.classification_radio.setChecked(True)  # 默认选择图片分类
        
        # 添加到布局
        task_layout.addWidget(self.classification_radio)
        task_layout.addWidget(self.detection_radio)
        task_layout.addStretch()
        task_group.setLayout(task_layout)
        main_layout.addWidget(task_group)
        
        # 创建堆叠部件用于切换不同的训练界面
        self.stacked_widget = QStackedWidget()
        
        # 创建图片分类训练界面
        self.classification_widget = QWidget()
        self.init_classification_ui()
        
        # 创建目标检测训练界面
        self.detection_widget = QWidget()
        self.init_detection_ui()
        
        # 添加到堆叠部件
        self.stacked_widget.addWidget(self.classification_widget)
        self.stacked_widget.addWidget(self.detection_widget)
        
        # 添加到主布局
        main_layout.addWidget(self.stacked_widget)
        
        # 创建底部控制区域
        bottom_widget = QWidget()
        bottom_widget.setMaximumHeight(100)  # 增加底部区域高度
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setSpacing(10)
        bottom_layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建训练按钮布局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)  # 增加按钮之间的间距
        
        # 设置按钮样式
        button_style = """
            QPushButton {
                font-size: 14px;
                font-weight: bold;
                padding: 4px;
                min-height: 30px;
                border-radius: 4px;
                border: 1px solid #BDBDBD;
            }
            QPushButton:enabled {
                color: #333333;
            }
            QPushButton:disabled {
                color: #757575;
            }
            QPushButton:hover:enabled {
                background-color: #F5F5F5;
            }
        """
        
        self.train_btn = QPushButton("开始训练")
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self.train_model)
        
        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        
        help_btn = QPushButton("训练帮助")
        help_btn.clicked.connect(self.show_training_help)
        
        # 设置按钮样式
        self.train_btn.setStyleSheet(button_style)
        self.stop_btn.setStyleSheet(button_style)
        help_btn.setStyleSheet(button_style)
        
        # 添加按钮到布局，设置拉伸因子使其平均分配空间
        button_layout.addWidget(self.train_btn, 1)
        button_layout.addWidget(self.stop_btn, 1)
        button_layout.addWidget(help_btn, 1)
        
        # 添加训练状态标签
        self.training_status_label = QLabel("等待训练开始...")
        self.training_status_label.setAlignment(Qt.AlignCenter)
        self.training_status_label.setStyleSheet("font-size: 12px; color: #666;")
        
        # 将按钮和状态标签添加到底部布局
        bottom_layout.addLayout(button_layout)
        bottom_layout.addWidget(self.training_status_label)
        
        # 将底部控制区域添加到滚动布局
        main_layout.addWidget(bottom_widget)
        
        # 连接信号
        self.task_button_group.buttonClicked.connect(self.on_task_changed)

    def init_classification_ui(self):
        """初始化图片分类训练界面"""
        # 创建主布局
        main_layout = QVBoxLayout(self.classification_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建标注文件夹选择组
        folder_group = QGroupBox("标注文件夹")
        folder_group.setMaximumHeight(70)  # 限制文件夹选择组的高度
        folder_layout = QHBoxLayout()
        folder_layout.setContentsMargins(10, 5, 10, 5)
        
        self.classification_path_edit = QLineEdit()
        self.classification_path_edit.setReadOnly(True)
        self.classification_path_edit.setPlaceholderText("请选择包含已标注图片的分类文件夹")
        
        # 如果有设置默认标注文件夹，在创建完控件后立即设置
        if hasattr(self, 'annotation_folder') and self.annotation_folder:
            self.classification_path_edit.setText(self.annotation_folder)
            print(f"初始化分类界面时设置路径: {self.annotation_folder}")
        
        folder_btn = QPushButton("浏览...")
        folder_btn.setFixedWidth(60)
        folder_btn.clicked.connect(self.select_classification_folder)
        
        folder_layout.addWidget(self.classification_path_edit)
        folder_layout.addWidget(folder_btn)
        folder_group.setLayout(folder_layout)
        main_layout.addWidget(folder_group)
        
        # 添加预训练模型选择组
        pretrained_group = QGroupBox("预训练模型")
        pretrained_group.setMaximumHeight(70)
        pretrained_layout = QHBoxLayout()
        pretrained_layout.setContentsMargins(10, 5, 10, 5)
        
        self.classification_use_local_pretrained_checkbox = QCheckBox("使用本地预训练模型")
        self.classification_use_local_pretrained_checkbox.setChecked(False)
        self.classification_use_local_pretrained_checkbox.stateChanged.connect(
            lambda state: self.toggle_pretrained_controls(state == Qt.Checked, is_classification=True)
        )
        pretrained_layout.addWidget(self.classification_use_local_pretrained_checkbox)
        pretrained_layout.addWidget(QLabel("预训练模型:"))
        self.classification_pretrained_path_edit = QLineEdit()
        self.classification_pretrained_path_edit.setReadOnly(True)
        self.classification_pretrained_path_edit.setEnabled(False)
        self.classification_pretrained_path_edit.setPlaceholderText("选择本地预训练模型文件")
        pretrained_btn = QPushButton("浏览...")
        pretrained_btn.setFixedWidth(60)
        pretrained_btn.setEnabled(False)
        pretrained_btn.clicked.connect(self.select_pretrained_model)
        pretrained_layout.addWidget(self.classification_pretrained_path_edit)
        pretrained_layout.addWidget(pretrained_btn)
        
        pretrained_group.setLayout(pretrained_layout)
        main_layout.addWidget(pretrained_group)
        
        # 创建基础训练参数组
        basic_group = QGroupBox("基础训练参数")
        basic_layout = QGridLayout()
        basic_layout.setContentsMargins(10, 15, 10, 15)
        basic_layout.setSpacing(10)
        
        # 模型选择
        basic_layout.addWidget(QLabel("模型:"), 0, 0)
        self.classification_model_combo = QComboBox()
        self.classification_model_combo.addItems([
            "MobileNetV2", "MobileNetV3", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
            "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4",
            "VGG16", "VGG19", "DenseNet121", "DenseNet169", "DenseNet201", "InceptionV3", "Xception"
        ])
        basic_layout.addWidget(self.classification_model_combo, 0, 1)
        
        # 批次大小
        basic_layout.addWidget(QLabel("批次大小:"), 1, 0)
        self.classification_batch_size_spin = QSpinBox()
        self.classification_batch_size_spin.setRange(1, 256)
        self.classification_batch_size_spin.setValue(32)
        basic_layout.addWidget(self.classification_batch_size_spin, 1, 1)
        
        # 训练轮数
        basic_layout.addWidget(QLabel("训练轮数:"), 2, 0)
        self.classification_epochs_spin = QSpinBox()
        self.classification_epochs_spin.setRange(1, 1000)
        self.classification_epochs_spin.setValue(20)
        basic_layout.addWidget(self.classification_epochs_spin, 2, 1)
        
        # 学习率
        basic_layout.addWidget(QLabel("学习率:"), 3, 0)
        self.classification_lr_spin = QDoubleSpinBox()
        self.classification_lr_spin.setRange(0.00001, 0.1)
        self.classification_lr_spin.setSingleStep(0.0001)
        self.classification_lr_spin.setDecimals(5)
        self.classification_lr_spin.setValue(0.001)
        basic_layout.addWidget(self.classification_lr_spin, 3, 1)
        
        # 学习率调度器
        basic_layout.addWidget(QLabel("学习率调度:"), 4, 0)
        self.classification_lr_scheduler_combo = QComboBox()
        self.classification_lr_scheduler_combo.addItems([
            "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau", "OneCycleLR", "CyclicLR"
        ])
        basic_layout.addWidget(self.classification_lr_scheduler_combo, 4, 1)
        
        # 优化器
        basic_layout.addWidget(QLabel("优化器:"), 5, 0)
        self.classification_optimizer_combo = QComboBox()
        self.classification_optimizer_combo.addItems([
            "Adam", "SGD", "RMSprop", "Adagrad", "AdamW", "RAdam", "AdaBelief"
        ])
        basic_layout.addWidget(self.classification_optimizer_combo, 5, 1)
        
        # 权重衰减
        basic_layout.addWidget(QLabel("权重衰减:"), 6, 0)
        self.classification_weight_decay_spin = QDoubleSpinBox()
        self.classification_weight_decay_spin.setRange(0, 0.1)
        self.classification_weight_decay_spin.setSingleStep(0.0001)
        self.classification_weight_decay_spin.setDecimals(5)
        self.classification_weight_decay_spin.setValue(0.0001)
        basic_layout.addWidget(self.classification_weight_decay_spin, 6, 1)
        
        # 评估指标
        basic_layout.addWidget(QLabel("评估指标:"), 7, 0)
        self.classification_metrics_list = QListWidget()
        self.classification_metrics_list.setSelectionMode(QListWidget.MultiSelection)
        self.classification_metrics_list.addItems([
            "accuracy", "precision", "recall", "f1_score", "confusion_matrix",
            "roc_auc", "average_precision", "top_k_accuracy", "balanced_accuracy"
        ])
        # 默认选中accuracy
        self.classification_metrics_list.setCurrentRow(0)
        basic_layout.addWidget(self.classification_metrics_list, 7, 1)
        
        # 使用预训练权重
        self.classification_pretrained_checkbox = QCheckBox("使用预训练权重")
        self.classification_pretrained_checkbox.setChecked(True)
        basic_layout.addWidget(self.classification_pretrained_checkbox, 8, 0, 1, 2)
        
        # 数据增强
        self.classification_augmentation_checkbox = QCheckBox("使用数据增强")
        self.classification_augmentation_checkbox.setChecked(True)
        basic_layout.addWidget(self.classification_augmentation_checkbox, 9, 0, 1, 2)
        
        basic_group.setLayout(basic_layout)
        main_layout.addWidget(basic_group)
        
        # 创建高级训练参数组
        advanced_group = QGroupBox("高级训练参数")
        advanced_layout = QGridLayout()
        advanced_layout.setContentsMargins(10, 15, 10, 15)
        advanced_layout.setSpacing(10)
        
        # 早停
        advanced_layout.addWidget(QLabel("启用早停:"), 0, 0)
        self.classification_early_stopping_checkbox = QCheckBox("启用早停")
        self.classification_early_stopping_checkbox.setChecked(True)
        advanced_layout.addWidget(self.classification_early_stopping_checkbox, 0, 1)
        
        # 早停耐心值
        advanced_layout.addWidget(QLabel("早停耐心值:"), 0, 2)
        self.classification_early_stopping_patience_spin = QSpinBox()
        self.classification_early_stopping_patience_spin.setRange(1, 50)
        self.classification_early_stopping_patience_spin.setValue(10)
        advanced_layout.addWidget(self.classification_early_stopping_patience_spin, 0, 3)
        
        # 梯度裁剪
        advanced_layout.addWidget(QLabel("启用梯度裁剪:"), 1, 0)
        self.classification_gradient_clipping_checkbox = QCheckBox("启用梯度裁剪")
        self.classification_gradient_clipping_checkbox.setChecked(False)
        advanced_layout.addWidget(self.classification_gradient_clipping_checkbox, 1, 1)
        
        # 梯度裁剪阈值
        advanced_layout.addWidget(QLabel("梯度裁剪阈值:"), 1, 2)
        self.classification_gradient_clipping_value_spin = QDoubleSpinBox()
        self.classification_gradient_clipping_value_spin.setRange(0.1, 10.0)
        self.classification_gradient_clipping_value_spin.setSingleStep(0.1)
        self.classification_gradient_clipping_value_spin.setValue(1.0)
        advanced_layout.addWidget(self.classification_gradient_clipping_value_spin, 1, 3)
        
        # 混合精度训练
        advanced_layout.addWidget(QLabel("启用混合精度训练:"), 2, 0)
        self.classification_mixed_precision_checkbox = QCheckBox("启用混合精度训练")
        self.classification_mixed_precision_checkbox.setChecked(True)
        advanced_layout.addWidget(self.classification_mixed_precision_checkbox, 2, 1)
        
        advanced_group.setLayout(advanced_layout)
        main_layout.addWidget(advanced_group)

    def init_detection_ui(self):
        """初始化目标检测训练界面"""
        # 创建主布局
        main_layout = QVBoxLayout(self.detection_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建标注文件夹选择组
        folder_group = QGroupBox("检测标注文件夹")
        folder_group.setMaximumHeight(70)  # 限制文件夹选择组的高度
        folder_layout = QHBoxLayout()
        folder_layout.setContentsMargins(10, 5, 10, 5)
        
        self.detection_path_edit = QLineEdit()
        self.detection_path_edit.setReadOnly(True)
        self.detection_path_edit.setPlaceholderText("请选择包含目标检测标注的文件夹")
        
        # 如果有设置默认标注文件夹，在创建完控件后立即设置
        if hasattr(self, 'annotation_folder') and self.annotation_folder:
            self.detection_path_edit.setText(self.annotation_folder)
            print(f"初始化检测界面时设置路径: {self.annotation_folder}")
        
        folder_btn = QPushButton("浏览...")
        folder_btn.setFixedWidth(60)
        folder_btn.clicked.connect(self.select_detection_folder)
        
        folder_layout.addWidget(self.detection_path_edit)
        folder_layout.addWidget(folder_btn)
        folder_group.setLayout(folder_layout)
        main_layout.addWidget(folder_group)
        
        # 添加预训练模型选择组
        pretrained_group = QGroupBox("预训练模型")
        pretrained_group.setMaximumHeight(70)
        pretrained_layout = QHBoxLayout()
        pretrained_layout.setContentsMargins(10, 5, 10, 5)
        
        self.detection_use_local_pretrained_checkbox = QCheckBox("使用本地预训练模型")
        self.detection_use_local_pretrained_checkbox.setChecked(False)
        self.detection_use_local_pretrained_checkbox.stateChanged.connect(
            lambda state: self.toggle_pretrained_controls(state == Qt.Checked, is_classification=False)
        )
        pretrained_layout.addWidget(self.detection_use_local_pretrained_checkbox)
        pretrained_layout.addWidget(QLabel("预训练模型:"))
        self.detection_pretrained_path_edit = QLineEdit()
        self.detection_pretrained_path_edit.setReadOnly(True)
        self.detection_pretrained_path_edit.setEnabled(False)
        self.detection_pretrained_path_edit.setPlaceholderText("选择本地预训练模型文件")
        pretrained_btn = QPushButton("浏览...")
        pretrained_btn.setFixedWidth(60)
        pretrained_btn.setEnabled(False)
        pretrained_btn.clicked.connect(self.select_pretrained_model)
        pretrained_layout.addWidget(self.detection_pretrained_path_edit)
        pretrained_layout.addWidget(pretrained_btn)
        
        pretrained_group.setLayout(pretrained_layout)
        main_layout.addWidget(pretrained_group)
        
        # 创建基础训练参数组
        basic_group = QGroupBox("基础训练参数")
        basic_layout = QGridLayout()
        basic_layout.setContentsMargins(10, 15, 10, 15)
        basic_layout.setSpacing(10)
        
        # 检测模型选择
        basic_layout.addWidget(QLabel("检测模型:"), 0, 0)
        self.detection_model_combo = QComboBox()
        self.detection_model_combo.addItems([
            "YOLOv5", "YOLOv8", "YOLOv7", "YOLOv6", "YOLOv4", "YOLOv3",
            "SSD", "SSD512", "SSD300", "Faster R-CNN", "Mask R-CNN",
            "RetinaNet", "DETR", "Swin Transformer", "DINO", "Deformable DETR"
        ])
        basic_layout.addWidget(self.detection_model_combo, 0, 1)
        
        # 批次大小
        basic_layout.addWidget(QLabel("批次大小:"), 1, 0)
        self.detection_batch_size_spin = QSpinBox()
        self.detection_batch_size_spin.setRange(1, 128)
        self.detection_batch_size_spin.setValue(16)
        basic_layout.addWidget(self.detection_batch_size_spin, 1, 1)
        
        # 训练轮数
        basic_layout.addWidget(QLabel("训练轮数:"), 2, 0)
        self.detection_epochs_spin = QSpinBox()
        self.detection_epochs_spin.setRange(1, 1000)
        self.detection_epochs_spin.setValue(50)
        basic_layout.addWidget(self.detection_epochs_spin, 2, 1)
        
        # 学习率
        basic_layout.addWidget(QLabel("学习率:"), 3, 0)
        self.detection_lr_spin = QDoubleSpinBox()
        self.detection_lr_spin.setRange(0.00001, 0.01)
        self.detection_lr_spin.setSingleStep(0.0001)
        self.detection_lr_spin.setDecimals(5)
        self.detection_lr_spin.setValue(0.0005)
        basic_layout.addWidget(self.detection_lr_spin, 3, 1)
        
        # 学习率调度器
        basic_layout.addWidget(QLabel("学习率调度:"), 4, 0)
        self.detection_lr_scheduler_combo = QComboBox()
        self.detection_lr_scheduler_combo.addItems([
            "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau", "OneCycleLR", "CyclicLR",
            "WarmupCosineLR", "LinearWarmup"
        ])
        basic_layout.addWidget(self.detection_lr_scheduler_combo, 4, 1)
        
        # 优化器
        basic_layout.addWidget(QLabel("优化器:"), 5, 0)
        self.detection_optimizer_combo = QComboBox()
        self.detection_optimizer_combo.addItems([
            "Adam", "SGD", "AdamW", "RAdam", "AdaBelief", "Lion"
        ])
        basic_layout.addWidget(self.detection_optimizer_combo, 5, 1)
        
        # 权重衰减
        basic_layout.addWidget(QLabel("权重衰减:"), 6, 0)
        self.detection_weight_decay_spin = QDoubleSpinBox()
        self.detection_weight_decay_spin.setRange(0, 0.1)
        self.detection_weight_decay_spin.setSingleStep(0.0001)
        self.detection_weight_decay_spin.setDecimals(5)
        self.detection_weight_decay_spin.setValue(0.0005)
        basic_layout.addWidget(self.detection_weight_decay_spin, 6, 1)
        
        # 评估指标
        basic_layout.addWidget(QLabel("评估指标:"), 7, 0)
        self.detection_metrics_list = QListWidget()
        self.detection_metrics_list.setSelectionMode(QListWidget.MultiSelection)
        self.detection_metrics_list.addItems([
            "mAP", "mAP50", "mAP75", "mAP50-95", "precision", "recall", "f1_score",
            "confusion_matrix", "per_class_metrics", "coco_metrics"
        ])
        # 默认选中mAP
        self.detection_metrics_list.setCurrentRow(0)
        basic_layout.addWidget(self.detection_metrics_list, 7, 1)
        
        # 输入分辨率
        basic_layout.addWidget(QLabel("输入分辨率:"), 8, 0)
        self.detection_resolution_combo = QComboBox()
        self.detection_resolution_combo.addItems([
            "416x416", "640x640", "512x512", "800x800", "1024x1024",
            "1280x1280", "1536x1536", "1920x1920"
        ])
        basic_layout.addWidget(self.detection_resolution_combo, 8, 1)
        
        # IOU阈值
        basic_layout.addWidget(QLabel("IOU阈值:"), 9, 0)
        self.detection_iou_spin = QDoubleSpinBox()
        self.detection_iou_spin.setRange(0.1, 0.9)
        self.detection_iou_spin.setSingleStep(0.05)
        self.detection_iou_spin.setDecimals(2)
        self.detection_iou_spin.setValue(0.5)
        basic_layout.addWidget(self.detection_iou_spin, 9, 1)
        
        # 置信度阈值
        basic_layout.addWidget(QLabel("置信度阈值:"), 10, 0)
        self.detection_conf_spin = QDoubleSpinBox()
        self.detection_conf_spin.setRange(0.05, 0.95)
        self.detection_conf_spin.setSingleStep(0.05)
        self.detection_conf_spin.setDecimals(2)
        self.detection_conf_spin.setValue(0.25)
        basic_layout.addWidget(self.detection_conf_spin, 10, 1)
        
        # 使用预训练权重
        basic_layout.addWidget(QLabel("使用预训练权重:"), 11, 0, 1, 2)
        self.detection_pretrained_checkbox = QCheckBox("使用预训练权重")
        self.detection_pretrained_checkbox.setChecked(True)
        basic_layout.addWidget(self.detection_pretrained_checkbox, 11, 2)
        
        # 数据增强
        basic_layout.addWidget(QLabel("使用数据增强:"), 12, 0, 1, 2)
        self.detection_augmentation_checkbox = QCheckBox("使用数据增强")
        self.detection_augmentation_checkbox.setChecked(True)
        basic_layout.addWidget(self.detection_augmentation_checkbox, 12, 2)
        
        basic_group.setLayout(basic_layout)
        main_layout.addWidget(basic_group)
        
        # 创建高级训练参数组
        advanced_group = QGroupBox("高级训练参数")
        advanced_layout = QGridLayout()
        advanced_layout.setContentsMargins(10, 15, 10, 15)
        advanced_layout.setSpacing(10)
        
        # 早停
        advanced_layout.addWidget(QLabel("启用早停:"), 0, 0)
        self.detection_early_stopping_checkbox = QCheckBox("启用早停")
        self.detection_early_stopping_checkbox.setChecked(True)
        advanced_layout.addWidget(self.detection_early_stopping_checkbox, 0, 1)
        
        # 早停耐心值
        advanced_layout.addWidget(QLabel("早停耐心值:"), 0, 2)
        self.detection_early_stopping_patience_spin = QSpinBox()
        self.detection_early_stopping_patience_spin.setRange(1, 50)
        self.detection_early_stopping_patience_spin.setValue(10)
        advanced_layout.addWidget(self.detection_early_stopping_patience_spin, 0, 3)
        
        # 梯度裁剪
        advanced_layout.addWidget(QLabel("启用梯度裁剪:"), 1, 0)
        self.detection_gradient_clipping_checkbox = QCheckBox("启用梯度裁剪")
        self.detection_gradient_clipping_checkbox.setChecked(False)
        advanced_layout.addWidget(self.detection_gradient_clipping_checkbox, 1, 1)
        
        # 梯度裁剪阈值
        advanced_layout.addWidget(QLabel("梯度裁剪阈值:"), 1, 2)
        self.detection_gradient_clipping_value_spin = QDoubleSpinBox()
        self.detection_gradient_clipping_value_spin.setRange(0.1, 10.0)
        self.detection_gradient_clipping_value_spin.setSingleStep(0.1)
        self.detection_gradient_clipping_value_spin.setValue(1.0)
        advanced_layout.addWidget(self.detection_gradient_clipping_value_spin, 1, 3)
        
        # 混合精度训练
        advanced_layout.addWidget(QLabel("启用混合精度训练:"), 2, 0)
        self.detection_mixed_precision_checkbox = QCheckBox("启用混合精度训练")
        self.detection_mixed_precision_checkbox.setChecked(True)
        advanced_layout.addWidget(self.detection_mixed_precision_checkbox, 2, 1)
        
        # 多GPU训练
        advanced_layout.addWidget(QLabel("启用多GPU训练:"), 2, 2)
        self.detection_multi_gpu_checkbox = QCheckBox("启用多GPU训练")
        self.detection_multi_gpu_checkbox.setChecked(False)
        advanced_layout.addWidget(self.detection_multi_gpu_checkbox, 2, 3)
        
        # 分布式训练
        advanced_layout.addWidget(QLabel("启用分布式训练:"), 2, 4)
        self.detection_distributed_checkbox = QCheckBox("启用分布式训练")
        self.detection_distributed_checkbox.setChecked(False)
        advanced_layout.addWidget(self.detection_distributed_checkbox, 2, 5)
        
        advanced_group.setLayout(advanced_layout)
        main_layout.addWidget(advanced_group)

    def on_task_changed(self, button):
        """训练任务改变时调用"""
        if button == self.classification_radio:
            self.stacked_widget.setCurrentIndex(0)
            self.task_type = "classification"
        else:
            self.stacked_widget.setCurrentIndex(1)
            self.task_type = "detection"
        self.check_training_ready()
        
    def select_classification_folder(self):
        """选择分类标注文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择分类标注文件夹")
        if folder:
            self.annotation_folder = folder
            self.classification_path_edit.setText(folder)
            self.check_training_ready()
    
    def select_detection_folder(self):
        """选择检测标注文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择检测标注文件夹")
        if folder:
            self.annotation_folder = folder
            self.detection_path_edit.setText(folder)
            self.check_training_ready()
    
    def check_training_ready(self):
        """检查是否可以开始训练"""
        if self.task_type == "classification":
            path_edit = self.classification_path_edit
        else:
            path_edit = self.detection_path_edit
            
        if path_edit.text() and os.path.exists(path_edit.text()):
            self.train_btn.setEnabled(True)
            return True
        else:
            self.train_btn.setEnabled(False)
            return False
    
    def train_model(self):
        """开始训练模型"""
        if not self.check_training_ready():
            QMessageBox.warning(self, "警告", "请先选择标注文件夹")
            return
        
        # 更新UI状态
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.update_status("正在准备训练...")
        self.update_progress(0)
        
        # 更新训练状态标签
        self.training_status_label.setText("正在准备训练...")
        
        # 发射训练开始信号
        self.training_started.emit()
    
    def stop_training(self):
        """停止训练"""
        reply = QMessageBox.question(
            self, 
            "确认停止", 
            "确定要停止训练吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.update_status("正在停止训练...")
            self.training_status_label.setText("正在停止训练...")
            
            # 直接发射停止训练信号
            if hasattr(self, 'main_window') and hasattr(self.main_window, 'worker') and hasattr(self.main_window.worker, 'model_trainer'):
                self.main_window.worker.model_trainer.stop()
                self.training_stopped.emit()  # 发射训练停止信号
            else:
                print("无法找到model_trainer对象，停止训练失败")
                QMessageBox.warning(self, "警告", "无法停止训练，请尝试关闭程序")
    
    def show_training_help(self):
        """显示训练帮助"""
        dialog = TrainingHelpDialog(self)
        dialog.exec_()
    
    def update_training_progress(self, epoch, logs):
        """更新训练进度"""
        # 根据当前任务类型获取训练轮数
        if self.task_type == "classification":
            epochs = self.classification_epochs_spin.value()
        else:
            epochs = self.detection_epochs_spin.value()
            
        # 更新进度条
        progress = int((epoch + 1) / epochs * 100)
        self.update_progress(progress)
        
        # 更新状态信息
        status = f"训练中... 轮次: {epoch + 1}/{epochs}"
        if logs:
            status += f" - 损失: {logs.get('loss', 0):.4f}"
            if 'accuracy' in logs:
                status += f" - 准确率: {logs.get('accuracy', 0):.4f}"
            elif 'acc' in logs:
                status += f" - 准确率: {logs.get('acc', 0):.4f}"
            elif 'mAP' in logs:
                status += f" - mAP: {logs.get('mAP', 0):.4f}"
        
        self.update_status(status)
        self.training_status_label.setText(status)
        
        # 发射训练进度更新信号，用于更新评估标签页中的实时训练曲线
        self.training_progress_updated.emit(epoch, logs)
    
    def on_training_finished(self):
        """训练完成时调用"""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.update_status("训练完成")
        self.update_progress(100)
        self.training_status_label.setText("训练完成")
    
    def on_training_error(self, error):
        """训练出错时调用"""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.update_status(f"训练出错: {error}")
        self.training_status_label.setText(f"训练出错: {error}")
        QMessageBox.critical(self, "训练错误", str(error))
    
    def select_pretrained_model(self):
        """选择本地预训练模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择预训练模型文件",
            "",
            "模型文件 (*.pth *.pt *.h5 *.hdf5 *.pkl);;所有文件 (*.*)"
        )
        if file_path:
            if self.task_type == "classification":
                self.classification_pretrained_path_edit.setText(file_path)
            else:
                self.detection_pretrained_path_edit.setText(file_path)
    
    def get_training_params(self):
        """获取当前训练参数"""
        params = {"task_type": self.task_type}
        
        if self.task_type == "classification":
            # 获取所有选中的评估指标
            selected_metrics = [item.text() for item in self.classification_metrics_list.selectedItems()]
            if not selected_metrics:
                selected_metrics = ["accuracy"]  # 默认使用accuracy
                
            # 获取预训练模型信息
            use_local_pretrained = self.classification_use_local_pretrained_checkbox.isChecked()
            pretrained_path = self.classification_pretrained_path_edit.text() if use_local_pretrained else ""
            pretrained_model = "" if use_local_pretrained else self.classification_model_combo.currentText()
                
            params.update({
                "model": self.classification_model_combo.currentText(),
                "batch_size": self.classification_batch_size_spin.value(),
                "epochs": self.classification_epochs_spin.value(),
                "learning_rate": self.classification_lr_spin.value(),
                "optimizer": self.classification_optimizer_combo.currentText(),
                "metrics": selected_metrics,
                "use_pretrained": self.classification_pretrained_checkbox.isChecked(),
                "use_local_pretrained": use_local_pretrained,
                "pretrained_path": pretrained_path,
                "pretrained_model": pretrained_model,  # 添加预训练模型名称
                "use_augmentation": self.classification_augmentation_checkbox.isChecked()
            })
        else:
            # 获取所有选中的评估指标
            selected_metrics = [item.text() for item in self.detection_metrics_list.selectedItems()]
            if not selected_metrics:
                selected_metrics = ["mAP"]  # 默认使用mAP
                
            # 获取预训练模型信息
            use_local_pretrained = self.detection_use_local_pretrained_checkbox.isChecked()
            pretrained_path = self.detection_pretrained_path_edit.text() if use_local_pretrained else ""
            pretrained_model = "" if use_local_pretrained else self.detection_model_combo.currentText()
                
            params.update({
                "model": self.detection_model_combo.currentText(),
                "batch_size": self.detection_batch_size_spin.value(),
                "epochs": self.detection_epochs_spin.value(),
                "learning_rate": self.detection_lr_spin.value(),
                "optimizer": self.detection_optimizer_combo.currentText(),
                "metrics": selected_metrics,
                "resolution": self.detection_resolution_combo.currentText(),
                "iou_threshold": self.detection_iou_spin.value(),
                "conf_threshold": self.detection_conf_spin.value(),
                "use_pretrained": self.detection_pretrained_checkbox.isChecked(),
                "use_local_pretrained": use_local_pretrained,
                "pretrained_path": pretrained_path,
                "pretrained_model": pretrained_model,  # 添加预训练模型名称
                "use_augmentation": self.detection_augmentation_checkbox.isChecked()
            })
            
        return params 

    def toggle_pretrained_controls(self, enabled, is_classification=True):
        """切换预训练模型控件的启用状态"""
        if is_classification:
            self.classification_pretrained_path_edit.setEnabled(enabled)
            # 找到对应的浏览按钮并设置状态
            for widget in self.classification_widget.findChildren(QPushButton):
                if widget.text() == "浏览..." and widget.parent() == self.classification_pretrained_path_edit.parent():
                    widget.setEnabled(enabled)
                    break
        else:
            self.detection_pretrained_path_edit.setEnabled(enabled)
            # 找到对应的浏览按钮并设置状态
            for widget in self.detection_widget.findChildren(QPushButton):
                if widget.text() == "浏览..." and widget.parent() == self.detection_pretrained_path_edit.parent():
                    widget.setEnabled(enabled)
                    break 

    def on_model_download_failed(self, model_name, model_link):
        """处理模型下载失败事件"""
        dialog = ModelDownloadDialog(model_name, model_link, self)
        dialog.exec_()
        
    def on_training_stopped(self):
        """训练停止完成时调用"""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.update_status("训练已停止")
        self.training_status_label.setText("训练已停止")
        QMessageBox.information(self, "训练状态", "训练已成功停止")

    def connect_model_trainer_signals(self, model_trainer):
        """连接模型训练器的信号"""
        if hasattr(model_trainer, 'model_download_failed'):
            model_trainer.model_download_failed.connect(self.on_model_download_failed)
        if hasattr(model_trainer, 'training_finished'):
            model_trainer.training_finished.connect(self.on_training_finished)
        if hasattr(model_trainer, 'training_error'):
            model_trainer.training_error.connect(self.on_training_error)
        if hasattr(model_trainer, 'epoch_finished'):
            model_trainer.epoch_finished.connect(self.update_training_progress)
        if hasattr(model_trainer, 'training_stopped'):
            model_trainer.training_stopped.connect(self.on_training_stopped) 