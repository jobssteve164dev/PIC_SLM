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
    training_progress_updated = pyqtSignal(dict)  # 修改为只接收dict参数
    training_stopped = pyqtSignal()  # 添加训练停止信号
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent, main_window)
        self.annotation_folder = ""
        self.task_type = "classification"  # 默认为图片分类任务
        self.init_ui()
        
        # 尝试直接从配置文件加载默认输出文件夹
        try:
            config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
            print(f"训练标签页直接加载配置文件: {config_file}")
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if 'default_output_folder' in config and config['default_output_folder']:
                        self.annotation_folder = config['default_output_folder']
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
            
            # 加载默认输出文件夹路径作为标注文件夹
            if 'default_output_folder' in config and config['default_output_folder']:
                print(f"发现默认输出文件夹配置: {config['default_output_folder']}")
                self.annotation_folder = config['default_output_folder']
                
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
                print("配置中没有找到default_output_folder或为空值")
                
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
        
        # 添加打开模型保存文件夹按钮
        self.open_model_folder_btn = QPushButton("打开模型文件夹")
        self.open_model_folder_btn.setEnabled(False)  # 默认禁用
        self.open_model_folder_btn.clicked.connect(self.open_model_folder)
        
        # 检查模型文件夹是否存在，如果存在则启用按钮
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_dir = os.path.join(root_dir, "models", "saved_models")
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            self.open_model_folder_btn.setEnabled(True)
        
        # 设置按钮样式
        self.train_btn.setStyleSheet(button_style)
        self.stop_btn.setStyleSheet(button_style)
        help_btn.setStyleSheet(button_style)
        self.open_model_folder_btn.setStyleSheet(button_style)
        
        # 添加按钮到布局，设置拉伸因子使其平均分配空间
        button_layout.addWidget(self.train_btn, 1)
        button_layout.addWidget(self.stop_btn, 1)
        button_layout.addWidget(help_btn, 1)
        button_layout.addWidget(self.open_model_folder_btn, 1)
        
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
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # 创建分类数据文件夹选择组
        folder_group = QGroupBox("训练数据目录")
        folder_layout = QHBoxLayout()
        folder_layout.setContentsMargins(10, 15, 10, 15)
        
        folder_layout.addWidget(QLabel("数据集路径:"))
        self.classification_path_edit = QLineEdit()
        self.classification_path_edit.setReadOnly(True)
        self.classification_path_edit.setPlaceholderText("请选择包含分类训练数据的文件夹")
        self.classification_path_edit.setToolTip("选择包含已分类图像的文件夹，文件夹结构应为每个分类在单独的子文件夹中")
        
        folder_layout.addWidget(self.classification_path_edit)
        
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.select_classification_folder)
        browse_btn.setToolTip("选择包含训练数据的根目录，每个子文件夹代表一个类别")
        folder_layout.addWidget(browse_btn)
        
        folder_group.setLayout(folder_layout)
        main_layout.addWidget(folder_group)
        
        # 创建预训练模型组
        pretrained_group = QGroupBox("预训练模型")
        pretrained_layout = QHBoxLayout()
        pretrained_layout.setContentsMargins(10, 15, 10, 15)
        
        # 使用本地预训练模型复选框
        self.classification_use_local_pretrained_checkbox = QCheckBox("使用本地预训练模型")
        self.classification_use_local_pretrained_checkbox.setToolTip("选择是否使用本地已有的预训练模型文件，而非从网络下载")
        self.classification_use_local_pretrained_checkbox.setChecked(False)
        self.classification_use_local_pretrained_checkbox.stateChanged.connect(
            lambda state: self.toggle_pretrained_controls(state == Qt.Checked, True)
        )
        pretrained_layout.addWidget(self.classification_use_local_pretrained_checkbox)
        
        pretrained_layout.addWidget(QLabel("预训练模型:"))
        self.classification_pretrained_path_edit = QLineEdit()
        self.classification_pretrained_path_edit.setReadOnly(True)
        self.classification_pretrained_path_edit.setEnabled(False)
        self.classification_pretrained_path_edit.setPlaceholderText("选择本地预训练模型文件")
        self.classification_pretrained_path_edit.setToolTip("选择本地已有的预训练模型文件（.pth/.h5/.pb格式）")
        pretrained_btn = QPushButton("浏览...")
        pretrained_btn.setFixedWidth(60)
        pretrained_btn.setEnabled(False)
        pretrained_btn.clicked.connect(self.select_pretrained_model)
        pretrained_btn.setToolTip("浏览选择本地预训练模型文件")
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
        model_label = QLabel("模型:")
        model_label.setToolTip("选择用于图像分类的深度学习模型架构")
        basic_layout.addWidget(model_label, 0, 0)
        self.classification_model_combo = QComboBox()
        self.classification_model_combo.addItems([
            "MobileNetV2", "MobileNetV3", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
            "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4",
            "VGG16", "VGG19", "DenseNet121", "DenseNet169", "DenseNet201", "InceptionV3", "Xception"
        ])
        self.classification_model_combo.setToolTip("选择不同的模型架构：\n- MobileNet系列：轻量级模型，适合移动设备\n- ResNet系列：残差网络，深度较大但训练稳定\n- EfficientNet系列：效率较高的模型\n- VGG系列：经典但参数较多的模型\n- DenseNet系列：密集连接的网络，参数利用率高\n- Inception/Xception：适合复杂特征提取")
        basic_layout.addWidget(self.classification_model_combo, 0, 1)
        
        # 批次大小
        batch_label = QLabel("批次大小:")
        batch_label.setToolTip("每次模型权重更新时处理的样本数量")
        basic_layout.addWidget(batch_label, 1, 0)
        self.classification_batch_size_spin = QSpinBox()
        self.classification_batch_size_spin.setRange(1, 256)
        self.classification_batch_size_spin.setValue(32)
        self.classification_batch_size_spin.setToolTip("批次大小影响训练速度和内存占用：\n- 较大批次：训练更稳定，但需要更多内存\n- 较小批次：内存占用少，但训练可能不稳定\n- 根据GPU内存大小调整，内存不足时请减小该值")
        basic_layout.addWidget(self.classification_batch_size_spin, 1, 1)
        
        # 训练轮数
        epochs_label = QLabel("训练轮数:")
        epochs_label.setToolTip("模型训练的完整周期数")
        basic_layout.addWidget(epochs_label, 2, 0)
        self.classification_epochs_spin = QSpinBox()
        self.classification_epochs_spin.setRange(1, 1000)
        self.classification_epochs_spin.setValue(20)
        self.classification_epochs_spin.setToolTip("训练轮数决定训练时长：\n- 轮数过少：模型可能欠拟合\n- 轮数过多：可能过拟合，浪费计算资源\n- 搭配早停策略使用效果更佳\n- 使用预训练模型时可适当减少轮数")
        basic_layout.addWidget(self.classification_epochs_spin, 2, 1)
        
        # 学习率
        lr_label = QLabel("学习率:")
        lr_label.setToolTip("模型权重更新的步长大小")
        basic_layout.addWidget(lr_label, 3, 0)
        self.classification_lr_spin = QDoubleSpinBox()
        self.classification_lr_spin.setRange(0.00001, 0.1)
        self.classification_lr_spin.setSingleStep(0.0001)
        self.classification_lr_spin.setDecimals(5)
        self.classification_lr_spin.setValue(0.001)
        self.classification_lr_spin.setToolTip("学习率是最重要的超参数之一：\n- 太大：训练不稳定，可能无法收敛\n- 太小：训练缓慢，可能陷入局部最优\n- 典型值：0.1 (SGD), 0.001 (Adam)\n- 微调预训练模型时使用较小学习率(0.0001)")
        basic_layout.addWidget(self.classification_lr_spin, 3, 1)
        
        # 学习率调度器
        lr_sched_label = QLabel("学习率调度:")
        lr_sched_label.setToolTip("学习率随训练进程自动调整的策略")
        basic_layout.addWidget(lr_sched_label, 4, 0)
        self.classification_lr_scheduler_combo = QComboBox()
        self.classification_lr_scheduler_combo.addItems([
            "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau", "OneCycleLR", "CyclicLR"
        ])
        self.classification_lr_scheduler_combo.setToolTip("学习率调度策略：\n- StepLR：按固定间隔降低学习率\n- CosineAnnealingLR：余弦周期性调整学习率\n- ReduceLROnPlateau：当指标不再改善时降低学习率\n- OneCycleLR：先增大再减小学习率，适合较短训练\n- CyclicLR：在两个界限间循环调整学习率")
        basic_layout.addWidget(self.classification_lr_scheduler_combo, 4, 1)
        
        # 优化器
        optimizer_label = QLabel("优化器:")
        optimizer_label.setToolTip("控制模型权重如何根据梯度更新")
        basic_layout.addWidget(optimizer_label, 5, 0)
        self.classification_optimizer_combo = QComboBox()
        self.classification_optimizer_combo.addItems([
            "Adam", "SGD", "RMSprop", "Adagrad", "AdamW", "RAdam", "AdaBelief"
        ])
        self.classification_optimizer_combo.setToolTip("不同的优化算法：\n- Adam：自适应算法，适用大多数情况\n- SGD：经典算法，配合动量可获得良好结果\n- RMSprop：类似于带衰减的AdaGrad\n- AdamW：修正Adam的权重衰减\n- RAdam：带修正的Adam，收敛更稳定\n- AdaBelief：最新优化器，通常更稳定")
        basic_layout.addWidget(self.classification_optimizer_combo, 5, 1)
        
        # 权重衰减
        wd_label = QLabel("权重衰减:")
        wd_label.setToolTip("L2正则化参数，控制模型复杂度")
        basic_layout.addWidget(wd_label, 6, 0)
        self.classification_weight_decay_spin = QDoubleSpinBox()
        self.classification_weight_decay_spin.setRange(0, 0.1)
        self.classification_weight_decay_spin.setSingleStep(0.0001)
        self.classification_weight_decay_spin.setDecimals(5)
        self.classification_weight_decay_spin.setValue(0.0001)
        self.classification_weight_decay_spin.setToolTip("权重衰减可防止过拟合：\n- 较大值：模型更简单，泛化能力可能更强\n- 较小值：允许模型更复杂，拟合能力更强\n- 典型值：0.0001-0.001\n- 数据较少时可适当增大")
        basic_layout.addWidget(self.classification_weight_decay_spin, 6, 1)
        
        # 评估指标
        metrics_label = QLabel("评估指标:")
        metrics_label.setToolTip("用于评估模型性能的指标")
        basic_layout.addWidget(metrics_label, 7, 0)
        self.classification_metrics_list = QListWidget()
        self.classification_metrics_list.setSelectionMode(QListWidget.MultiSelection)
        self.classification_metrics_list.addItems([
            "accuracy", "precision", "recall", "f1_score", "confusion_matrix",
            "roc_auc", "average_precision", "top_k_accuracy", "balanced_accuracy"
        ])
        self.classification_metrics_list.setToolTip("选择用于评估模型的指标：\n- accuracy：准确率，适用于平衡数据集\n- precision：精确率，关注减少假阳性\n- recall：召回率，关注减少假阴性\n- f1_score：精确率和召回率的调和平均\n- confusion_matrix：混淆矩阵\n- roc_auc：ROC曲线下面积\n- balanced_accuracy：平衡准确率，适用于不平衡数据集")
        # 默认选中accuracy
        self.classification_metrics_list.setCurrentRow(0)
        basic_layout.addWidget(self.classification_metrics_list, 7, 1)
        
        # 使用预训练权重
        self.classification_pretrained_checkbox = QCheckBox("使用预训练权重")
        self.classification_pretrained_checkbox.setChecked(True)
        self.classification_pretrained_checkbox.setToolTip("使用在大型数据集(如ImageNet)上预训练的模型权重：\n- 加快训练速度\n- 提高模型性能\n- 尤其在训练数据较少时效果显著\n- 需要网络连接下载权重文件")
        basic_layout.addWidget(self.classification_pretrained_checkbox, 8, 0, 1, 2)
        
        # 数据增强
        self.classification_augmentation_checkbox = QCheckBox("使用数据增强")
        self.classification_augmentation_checkbox.setChecked(True)
        self.classification_augmentation_checkbox.setToolTip("通过随机变换（旋转、裁剪、翻转等）增加训练数据多样性：\n- 减少过拟合\n- 提高模型泛化能力\n- 尤其在训练数据较少时非常有效")
        basic_layout.addWidget(self.classification_augmentation_checkbox, 9, 0, 1, 2)
        
        basic_group.setLayout(basic_layout)
        main_layout.addWidget(basic_group)
        
        # 创建高级训练参数组
        advanced_group = QGroupBox("高级训练参数")
        advanced_layout = QGridLayout()
        advanced_layout.setContentsMargins(10, 15, 10, 15)
        advanced_layout.setSpacing(10)
        
        # 早停
        early_stop_label = QLabel("启用早停:")
        early_stop_label.setToolTip("当验证指标不再改善时自动停止训练")
        advanced_layout.addWidget(early_stop_label, 0, 0)
        self.classification_early_stopping_checkbox = QCheckBox("启用早停")
        self.classification_early_stopping_checkbox.setChecked(True)
        self.classification_early_stopping_checkbox.setToolTip("当验证指标在一定轮数内不再改善时停止训练：\n- 减少过拟合风险\n- 节省训练时间\n- 自动选择最佳模型")
        advanced_layout.addWidget(self.classification_early_stopping_checkbox, 0, 1)
        
        # 早停耐心值
        patience_label = QLabel("早停耐心值:")
        patience_label.setToolTip("早停前允许的不改善轮数")
        advanced_layout.addWidget(patience_label, 0, 2)
        self.classification_early_stopping_patience_spin = QSpinBox()
        self.classification_early_stopping_patience_spin.setRange(1, 50)
        self.classification_early_stopping_patience_spin.setValue(10)
        self.classification_early_stopping_patience_spin.setToolTip("在停止训练前容忍的无改善轮数：\n- 较大值：更有耐心，可能训练更久\n- 较小值：更积极停止，可能过早停止\n- 典型值：5-15轮")
        advanced_layout.addWidget(self.classification_early_stopping_patience_spin, 0, 3)
        
        # 梯度裁剪
        grad_clip_label = QLabel("启用梯度裁剪:")
        grad_clip_label.setToolTip("限制梯度大小以稳定训练")
        advanced_layout.addWidget(grad_clip_label, 1, 0)
        self.classification_gradient_clipping_checkbox = QCheckBox("启用梯度裁剪")
        self.classification_gradient_clipping_checkbox.setChecked(False)
        self.classification_gradient_clipping_checkbox.setToolTip("限制梯度的最大范数，防止梯度爆炸：\n- 稳定训练过程\n- 尤其适用于循环神经网络\n- 可以使用更大的学习率")
        advanced_layout.addWidget(self.classification_gradient_clipping_checkbox, 1, 1)
        
        # 梯度裁剪阈值
        clip_value_label = QLabel("梯度裁剪阈值:")
        clip_value_label.setToolTip("梯度裁剪的最大范数值")
        advanced_layout.addWidget(clip_value_label, 1, 2)
        self.classification_gradient_clipping_value_spin = QDoubleSpinBox()
        self.classification_gradient_clipping_value_spin.setRange(0.1, 10.0)
        self.classification_gradient_clipping_value_spin.setSingleStep(0.1)
        self.classification_gradient_clipping_value_spin.setValue(1.0)
        self.classification_gradient_clipping_value_spin.setToolTip("梯度范数的最大允许值：\n- 较小值：裁剪更积极，训练更稳定但可能较慢\n- 较大值：裁剪更宽松\n- 典型值：1.0-5.0")
        advanced_layout.addWidget(self.classification_gradient_clipping_value_spin, 1, 3)
        
        # 混合精度训练
        mixed_precision_label = QLabel("启用混合精度训练:")
        mixed_precision_label.setToolTip("使用FP16和FP32混合精度加速训练")
        advanced_layout.addWidget(mixed_precision_label, 2, 0)
        self.classification_mixed_precision_checkbox = QCheckBox("启用混合精度训练")
        self.classification_mixed_precision_checkbox.setChecked(True)
        self.classification_mixed_precision_checkbox.setToolTip("使用FP16和FP32混合精度：\n- 加速训练(最高2倍)\n- 减少内存占用\n- 几乎不影响精度\n- 需要支持FP16的GPU")
        advanced_layout.addWidget(self.classification_mixed_precision_checkbox, 2, 1)
        
        # 模型命名备注
        model_note_label = QLabel("模型命名备注:")
        model_note_label.setToolTip("添加到训练输出模型文件名中的备注")
        advanced_layout.addWidget(model_note_label, 3, 0)
        self.classification_model_note_edit = QLineEdit()
        self.classification_model_note_edit.setPlaceholderText("可选: 添加模型命名备注")
        self.classification_model_note_edit.setToolTip("这个备注将添加到输出模型文件名中，方便识别不同训练的模型")
        advanced_layout.addWidget(self.classification_model_note_edit, 3, 1, 1, 3)
        
        advanced_group.setLayout(advanced_layout)
        main_layout.addWidget(advanced_group)

    def init_detection_ui(self):
        """初始化目标检测训练界面"""
        # 创建主布局
        main_layout = QVBoxLayout(self.detection_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # 创建检测数据文件夹选择组
        folder_group = QGroupBox("训练数据目录")
        folder_layout = QHBoxLayout()
        folder_layout.setContentsMargins(10, 15, 10, 15)
        
        folder_layout.addWidget(QLabel("数据集路径:"))
        self.detection_path_edit = QLineEdit()
        self.detection_path_edit.setReadOnly(True)
        self.detection_path_edit.setPlaceholderText("请选择包含目标检测训练数据的文件夹")
        self.detection_path_edit.setToolTip("选择包含目标检测数据的文件夹，需要标注文件（YOLO或COCO格式）和图像文件")
        
        folder_layout.addWidget(self.detection_path_edit)
        
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.select_detection_folder)
        browse_btn.setToolTip("选择包含已标注目标检测数据的文件夹，包括图像和标注文件")
        folder_layout.addWidget(browse_btn)
        
        folder_group.setLayout(folder_layout)
        main_layout.addWidget(folder_group)
        
        # 创建预训练模型组
        pretrained_group = QGroupBox("预训练模型")
        pretrained_layout = QHBoxLayout()
        pretrained_layout.setContentsMargins(10, 15, 10, 15)
        
        # 使用本地预训练模型复选框
        self.detection_use_local_pretrained_checkbox = QCheckBox("使用本地预训练模型")
        self.detection_use_local_pretrained_checkbox.setToolTip("选择是否使用本地已有的预训练模型文件，而非从网络下载")
        self.detection_use_local_pretrained_checkbox.setChecked(False)
        self.detection_use_local_pretrained_checkbox.stateChanged.connect(
            lambda state: self.toggle_pretrained_controls(state == Qt.Checked, False)
        )
        pretrained_layout.addWidget(self.detection_use_local_pretrained_checkbox)
        
        pretrained_layout.addWidget(QLabel("预训练模型:"))
        self.detection_pretrained_path_edit = QLineEdit()
        self.detection_pretrained_path_edit.setReadOnly(True)
        self.detection_pretrained_path_edit.setEnabled(False)
        self.detection_pretrained_path_edit.setPlaceholderText("选择本地预训练模型文件")
        self.detection_pretrained_path_edit.setToolTip("选择本地已有的预训练目标检测模型文件（.pth/.weights/.pt格式）")
        pretrained_btn = QPushButton("浏览...")
        pretrained_btn.setFixedWidth(60)
        pretrained_btn.setEnabled(False)
        pretrained_btn.clicked.connect(self.select_pretrained_model)
        pretrained_btn.setToolTip("浏览选择本地预训练目标检测模型文件")
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
        model_label = QLabel("检测模型:")
        model_label.setToolTip("选择用于目标检测的深度学习模型架构")
        basic_layout.addWidget(model_label, 0, 0)
        self.detection_model_combo = QComboBox()
        self.detection_model_combo.addItems([
            "YOLOv5", "YOLOv8", "YOLOv7", "YOLOv6", "YOLOv4", "YOLOv3",
            "SSD", "SSD512", "SSD300", "Faster R-CNN", "Mask R-CNN",
            "RetinaNet", "DETR", "Swin Transformer", "DINO", "Deformable DETR"
        ])
        self.detection_model_combo.setToolTip("选择不同的目标检测模型：\n- YOLO系列：单阶段检测器，速度快精度适中\n- SSD系列：单阶段多尺度检测器\n- Faster/Mask R-CNN：两阶段检测器，精度高\n- DETR系列：基于Transformer的端到端检测器\n- 不同模型在速度和精度上有权衡")
        basic_layout.addWidget(self.detection_model_combo, 0, 1)
        
        # 批次大小
        batch_label = QLabel("批次大小:")
        batch_label.setToolTip("每次模型权重更新时处理的样本数量")
        basic_layout.addWidget(batch_label, 1, 0)
        self.detection_batch_size_spin = QSpinBox()
        self.detection_batch_size_spin.setRange(1, 128)
        self.detection_batch_size_spin.setValue(16)
        self.detection_batch_size_spin.setToolTip("目标检测训练的批次大小：\n- 检测模型通常需要更大内存\n- 典型值：8-16（普通GPU）\n- 内存不足时请减小该值\n- 较小值也可能提高精度但训练更慢")
        basic_layout.addWidget(self.detection_batch_size_spin, 1, 1)
        
        # 训练轮数
        epochs_label = QLabel("训练轮数:")
        epochs_label.setToolTip("模型训练的完整周期数")
        basic_layout.addWidget(epochs_label, 2, 0)
        self.detection_epochs_spin = QSpinBox()
        self.detection_epochs_spin.setRange(1, 1000)
        self.detection_epochs_spin.setValue(50)
        self.detection_epochs_spin.setToolTip("检测模型训练轮数：\n- 检测模型通常需要更多轮数收敛\n- 典型值：50-100轮\n- 使用预训练时可减少到30-50轮\n- 搭配早停策略效果更佳")
        basic_layout.addWidget(self.detection_epochs_spin, 2, 1)
        
        # 学习率
        lr_label = QLabel("学习率:")
        lr_label.setToolTip("模型权重更新的步长大小")
        basic_layout.addWidget(lr_label, 3, 0)
        self.detection_lr_spin = QDoubleSpinBox()
        self.detection_lr_spin.setRange(0.00001, 0.01)
        self.detection_lr_spin.setSingleStep(0.0001)
        self.detection_lr_spin.setDecimals(5)
        self.detection_lr_spin.setValue(0.0005)
        self.detection_lr_spin.setToolTip("检测模型学习率：\n- 通常比分类模型小一个数量级\n- 典型值：0.0005-0.001\n- 训练不稳定时可减小\n- 微调预训练模型时使用更小值(0.00005)")
        basic_layout.addWidget(self.detection_lr_spin, 3, 1)
        
        # 学习率调度器
        lr_sched_label = QLabel("学习率调度:")
        lr_sched_label.setToolTip("学习率随训练进程自动调整的策略")
        basic_layout.addWidget(lr_sched_label, 4, 0)
        self.detection_lr_scheduler_combo = QComboBox()
        self.detection_lr_scheduler_combo.addItems([
            "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau", "OneCycleLR", "CyclicLR",
            "WarmupCosineLR", "LinearWarmup"
        ])
        self.detection_lr_scheduler_combo.setToolTip("目标检测特有的学习率调度：\n- WarmupCosineLR：先预热再余弦衰减，YOLO常用\n- LinearWarmup：线性预热，目标检测常用\n- ReduceLROnPlateau：性能平台时降低学习率\n- CosineAnnealingLR：余弦周期调整")
        basic_layout.addWidget(self.detection_lr_scheduler_combo, 4, 1)
        
        # 优化器
        optimizer_label = QLabel("优化器:")
        optimizer_label.setToolTip("控制模型权重如何根据梯度更新")
        basic_layout.addWidget(optimizer_label, 5, 0)
        self.detection_optimizer_combo = QComboBox()
        self.detection_optimizer_combo.addItems([
            "Adam", "SGD", "AdamW", "RAdam", "AdaBelief", "Lion"
        ])
        self.detection_optimizer_combo.setToolTip("检测模型优化器：\n- SGD：YOLO推荐使用，配合动量和预热\n- Adam：稳定但可能精度略低\n- AdamW：带修正权重衰减的Adam\n- Lion：新型高效优化器，可节省显存")
        basic_layout.addWidget(self.detection_optimizer_combo, 5, 1)
        
        # 权重衰减
        wd_label = QLabel("权重衰减:")
        wd_label.setToolTip("L2正则化参数，控制模型复杂度")
        basic_layout.addWidget(wd_label, 6, 0)
        self.detection_weight_decay_spin = QDoubleSpinBox()
        self.detection_weight_decay_spin.setRange(0, 0.1)
        self.detection_weight_decay_spin.setSingleStep(0.0001)
        self.detection_weight_decay_spin.setDecimals(5)
        self.detection_weight_decay_spin.setValue(0.0005)
        self.detection_weight_decay_spin.setToolTip("检测模型权重衰减：\n- YOLO系列通常使用0.0005\n- Faster R-CNN使用0.0001\n- 数据量小时可适当增大\n- 过大可能导致欠拟合")
        basic_layout.addWidget(self.detection_weight_decay_spin, 6, 1)
        
        # 评估指标
        metrics_label = QLabel("评估指标:")
        metrics_label.setToolTip("用于评估检测模型性能的指标")
        basic_layout.addWidget(metrics_label, 7, 0)
        self.detection_metrics_list = QListWidget()
        self.detection_metrics_list.setSelectionMode(QListWidget.MultiSelection)
        self.detection_metrics_list.addItems([
            "mAP", "mAP50", "mAP75", "precision", "recall", "f1_score", 
            "box_loss", "class_loss", "obj_loss"
        ])
        self.detection_metrics_list.setToolTip("检测模型评估指标：\n- mAP：不同IOU阈值下的平均精度，主要指标\n- mAP50：IOU阈值为0.5的平均精度\n- mAP75：IOU阈值为0.75的平均精度，更严格\n- precision/recall：精确率/召回率\n- f1_score：精确率和召回率的调和平均值\n- 各种loss：用于诊断训练问题")
        # 默认选中mAP
        self.detection_metrics_list.setCurrentRow(0)
        basic_layout.addWidget(self.detection_metrics_list, 7, 1)
        
        # 输入分辨率
        resolution_label = QLabel("输入分辨率:")
        resolution_label.setToolTip("模型训练和推理的图像尺寸")
        basic_layout.addWidget(resolution_label, 8, 0)
        self.detection_resolution_combo = QComboBox()
        self.detection_resolution_combo.addItems([
            "416x416", "512x512", "640x640", "768x768", "1024x1024", "1280x1280"
        ])
        self.detection_resolution_combo.setToolTip("输入分辨率影响速度和精度：\n- 更大分辨率：更高精度，尤其对小物体\n- 更小分辨率：更快速度，适合实时应用\n- 640x640是YOLO常用分辨率\n- 根据目标大小和GPU内存选择")
        self.detection_resolution_combo.setCurrentText("640x640")
        basic_layout.addWidget(self.detection_resolution_combo, 8, 1)
        
        # IOU阈值
        iou_label = QLabel("IOU阈值:")
        iou_label.setToolTip("交并比阈值，用于训练和非极大值抑制")
        basic_layout.addWidget(iou_label, 9, 0)
        self.detection_iou_spin = QDoubleSpinBox()
        self.detection_iou_spin.setRange(0.1, 0.9)
        self.detection_iou_spin.setSingleStep(0.05)
        self.detection_iou_spin.setDecimals(2)
        self.detection_iou_spin.setValue(0.5)
        self.detection_iou_spin.setToolTip("IOU(交并比)阈值：\n- 训练中用于正负样本分配\n- 推理中用于NMS筛选\n- 较高值：更严格的重叠判定\n- 较低值：更宽松的重叠判定\n- 典型值：0.5或0.45")
        basic_layout.addWidget(self.detection_iou_spin, 9, 1)
        
        # 置信度阈值
        conf_label = QLabel("置信度阈值:")
        conf_label.setToolTip("检测结果的最小置信度分数")
        basic_layout.addWidget(conf_label, 10, 0)
        self.detection_conf_spin = QDoubleSpinBox()
        self.detection_conf_spin.setRange(0.01, 0.99)
        self.detection_conf_spin.setSingleStep(0.05)
        self.detection_conf_spin.setDecimals(2)
        self.detection_conf_spin.setValue(0.25)
        self.detection_conf_spin.setToolTip("置信度阈值：\n- 推理时保留的最小目标置信度\n- 较高值：减少假阳性，但可能漏检\n- 较低值：提高召回率，但增加假阳性\n- 典型值：0.25-0.45\n- 可根据应用场景调整")
        basic_layout.addWidget(self.detection_conf_spin, 10, 1)
        
        # Multi-scale 训练
        ms_label = QLabel("多尺度训练:")
        ms_label.setToolTip("在训练中随机调整输入图像尺寸")
        basic_layout.addWidget(ms_label, 11, 0)
        self.detection_multiscale_checkbox = QCheckBox("启用多尺度训练")
        self.detection_multiscale_checkbox.setChecked(True)
        self.detection_multiscale_checkbox.setToolTip("多尺度训练增强模型泛化能力：\n- 在训练中随机调整输入图像大小\n- 提高模型对不同尺寸目标的适应性\n- 可能增加训练时间\n- YOLO模型常用技术")
        basic_layout.addWidget(self.detection_multiscale_checkbox, 11, 1)
        
        # Mosaic 数据增强
        mosaic_label = QLabel("马赛克增强:")
        mosaic_label.setToolTip("YOLO系列特有的数据增强方法")
        basic_layout.addWidget(mosaic_label, 12, 0)
        self.detection_mosaic_checkbox = QCheckBox("启用马赛克增强")
        self.detection_mosaic_checkbox.setChecked(True)
        self.detection_mosaic_checkbox.setToolTip("马赛克数据增强：\n- 将4张图像拼接成1张\n- 大幅增加目标数量和上下文变化\n- 显著提高小目标检测性能\n- YOLOv5之后广泛使用的增强方法")
        basic_layout.addWidget(self.detection_mosaic_checkbox, 12, 1)
        
        # 使用预训练权重
        self.detection_pretrained_checkbox = QCheckBox("使用预训练权重")
        self.detection_pretrained_checkbox.setChecked(True)
        self.detection_pretrained_checkbox.setToolTip("使用在COCO等大型数据集上预训练的模型：\n- 加快检测模型收敛速度\n- 显著提高最终精度\n- 尤其在训练数据较少时更有效\n- YOLO和大多数检测模型都支持")
        basic_layout.addWidget(self.detection_pretrained_checkbox, 11, 2)
        
        # 常规数据增强
        self.detection_augmentation_checkbox = QCheckBox("使用数据增强")
        self.detection_augmentation_checkbox.setChecked(True)
        self.detection_augmentation_checkbox.setToolTip("常规数据增强：\n- 翻转、旋转、色彩变换等\n- 减少过拟合，提高泛化能力\n- 检测任务中一般都需要启用\n- 需要bbox坐标同步转换")
        basic_layout.addWidget(self.detection_augmentation_checkbox, 12, 2)
        
        basic_group.setLayout(basic_layout)
        main_layout.addWidget(basic_group)
        
        # 创建高级训练参数组
        advanced_group = QGroupBox("高级训练参数")
        advanced_layout = QGridLayout()
        advanced_layout.setContentsMargins(10, 15, 10, 15)
        advanced_layout.setSpacing(10)
        
        # 早停
        early_stop_label = QLabel("启用早停:")
        early_stop_label.setToolTip("当验证指标不再改善时自动停止训练")
        advanced_layout.addWidget(early_stop_label, 0, 0)
        self.detection_early_stopping_checkbox = QCheckBox("启用早停")
        self.detection_early_stopping_checkbox.setChecked(True)
        self.detection_early_stopping_checkbox.setToolTip("当mAP在一定轮数内不再提高时停止训练：\n- 避免不必要的训练时间\n- 减少过拟合风险\n- 自动保存最佳模型")
        advanced_layout.addWidget(self.detection_early_stopping_checkbox, 0, 1)
        
        # 早停耐心值
        patience_label = QLabel("早停耐心值:")
        patience_label.setToolTip("早停前允许的不改善轮数")
        advanced_layout.addWidget(patience_label, 0, 2)
        self.detection_early_stopping_patience_spin = QSpinBox()
        self.detection_early_stopping_patience_spin.setRange(1, 50)
        self.detection_early_stopping_patience_spin.setValue(10)
        self.detection_early_stopping_patience_spin.setToolTip("目标检测早停耐心值：\n- 检测模型可能需要更大耐心值\n- 典型值：10-15轮\n- 过小可能过早停止\n- 过大则失去早停意义")
        advanced_layout.addWidget(self.detection_early_stopping_patience_spin, 0, 3)
        
        # 梯度裁剪
        grad_clip_label = QLabel("启用梯度裁剪:")
        grad_clip_label.setToolTip("限制梯度大小以稳定训练")
        advanced_layout.addWidget(grad_clip_label, 1, 0)
        self.detection_gradient_clipping_checkbox = QCheckBox("启用梯度裁剪")
        self.detection_gradient_clipping_checkbox.setChecked(False)
        self.detection_gradient_clipping_checkbox.setToolTip("目标检测中的梯度裁剪：\n- 大型检测模型更容易出现梯度不稳定\n- 预防梯度爆炸和训练不稳定\n- 尤其在高学习率时有用\n- Faster R-CNN等常用技术")
        advanced_layout.addWidget(self.detection_gradient_clipping_checkbox, 1, 1)
        
        # 梯度裁剪阈值
        clip_value_label = QLabel("梯度裁剪阈值:")
        clip_value_label.setToolTip("梯度裁剪的最大范数值")
        advanced_layout.addWidget(clip_value_label, 1, 2)
        self.detection_gradient_clipping_value_spin = QDoubleSpinBox()
        self.detection_gradient_clipping_value_spin.setRange(0.1, 10.0)
        self.detection_gradient_clipping_value_spin.setSingleStep(0.1)
        self.detection_gradient_clipping_value_spin.setValue(1.0)
        self.detection_gradient_clipping_value_spin.setToolTip("检测模型梯度裁剪阈值：\n- 两阶段检测器常用值：10.0\n- YOLO常用值：4.0\n- 训练不稳定时可降低该值\n- 精调时通常使用较小值")
        advanced_layout.addWidget(self.detection_gradient_clipping_value_spin, 1, 3)
        
        # 混合精度训练
        mixed_precision_label = QLabel("启用混合精度训练:")
        mixed_precision_label.setToolTip("使用FP16和FP32混合精度加速训练")
        advanced_layout.addWidget(mixed_precision_label, 2, 0)
        self.detection_mixed_precision_checkbox = QCheckBox("启用混合精度训练")
        self.detection_mixed_precision_checkbox.setChecked(True)
        self.detection_mixed_precision_checkbox.setToolTip("目标检测模型混合精度训练：\n- 检测模型受益更大，加速可达2倍\n- 减少50%GPU内存使用\n- 几乎不影响最终精度\n- 建议所有支持FP16的GPU都启用")
        advanced_layout.addWidget(self.detection_mixed_precision_checkbox, 2, 1)
        
        # EMA - 指数移动平均
        ema_label = QLabel("启用EMA:")
        ema_label.setToolTip("使用权重的指数移动平均提高稳定性")
        advanced_layout.addWidget(ema_label, 2, 2)
        self.detection_ema_checkbox = QCheckBox("启用指数移动平均")
        self.detection_ema_checkbox.setChecked(True)
        self.detection_ema_checkbox.setToolTip("模型权重的指数移动平均：\n- 产生更平滑和稳定的模型\n- 提高测试精度和泛化能力\n- YOLO默认开启此功能\n- 几乎不增加计算负担")
        advanced_layout.addWidget(self.detection_ema_checkbox, 2, 3)
        
        # 模型命名备注
        model_note_label = QLabel("模型命名备注:")
        model_note_label.setToolTip("添加到训练输出模型文件名中的备注")
        advanced_layout.addWidget(model_note_label, 3, 0)
        self.detection_model_note_edit = QLineEdit()
        self.detection_model_note_edit.setPlaceholderText("可选: 添加模型命名备注")
        self.detection_model_note_edit.setToolTip("这个备注将添加到输出模型文件名中，方便识别不同训练的模型")
        advanced_layout.addWidget(self.detection_model_note_edit, 3, 1, 1, 3)
        
        advanced_group.setLayout(advanced_layout)
        main_layout.addWidget(advanced_group)

    def on_task_changed(self, button):
        """训练任务改变时调用"""
        if button == self.classification_radio:
            self.stacked_widget.setCurrentIndex(0)
            self.task_type = "classification"
            # 自动设置分类数据集路径
            self.select_classification_folder()
        else:
            self.stacked_widget.setCurrentIndex(1)
            self.task_type = "detection"
            # 自动设置目标检测数据集路径
            self.select_detection_folder()
            
        self.check_training_ready()
        
    def select_classification_folder(self):
        """选择分类标注文件夹"""
        try:
            # 尝试从配置文件直接加载
            config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    default_output_folder = config.get('default_output_folder', '')
            else:
                # 如果配置文件不存在，尝试从main_window获取
                if hasattr(self, 'main_window') and hasattr(self.main_window, 'config'):
                    default_output_folder = self.main_window.config.get('default_output_folder', '')
                else:
                    default_output_folder = ''
            
            if not default_output_folder:
                QMessageBox.warning(self, "错误", "请先在设置中配置默认输出文件夹")
                return
                
            dataset_folder = os.path.join(default_output_folder, 'dataset')
            if not os.path.exists(dataset_folder):
                QMessageBox.warning(self, "错误", "未找到数据集文件夹，请先完成图像标注")
                return
                
            self.annotation_folder = dataset_folder
            self.classification_path_edit.setText(dataset_folder)
            self.check_training_ready()
            
            # 显示成功信息
            self.update_status("成功检测到分类数据集文件夹结构")
            # 更新训练状态标签
            self.training_status_label.setText("数据集文件夹已刷新")
            
        except Exception as e:
            print(f"选择分类文件夹时出错: {str(e)}")
            QMessageBox.warning(self, "错误", f"无法获取默认输出文件夹设置: {str(e)}")
    
    def select_detection_folder(self):
        """选择检测标注文件夹"""
        try:
            # 尝试从配置文件直接加载
            config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    default_output_folder = config.get('default_output_folder', '')
            else:
                # 如果配置文件不存在，尝试从main_window获取
                if hasattr(self, 'main_window') and hasattr(self.main_window, 'config'):
                    default_output_folder = self.main_window.config.get('default_output_folder', '')
                else:
                    default_output_folder = ''
            
            if not default_output_folder:
                QMessageBox.warning(self, "错误", "请先在设置中配置默认输出文件夹")
                return
                
            detection_data_folder = os.path.join(default_output_folder, 'detection_data')
            if not os.path.exists(detection_data_folder):
                QMessageBox.warning(self, "错误", "未找到目标检测数据集文件夹，请先完成目标检测标注")
                return
                
            self.annotation_folder = detection_data_folder
            self.detection_path_edit.setText(detection_data_folder)
            self.check_training_ready()
            
            # 显示成功信息
            self.update_status("成功检测到目标检测数据集文件夹结构")
            # 更新训练状态标签
            self.training_status_label.setText("数据集文件夹已刷新")
            
        except Exception as e:
            print(f"选择检测文件夹时出错: {str(e)}")
            QMessageBox.warning(self, "错误", f"无法获取默认输出文件夹设置: {str(e)}")
    
    def check_training_ready(self):
        """检查是否可以开始训练"""
        try:
            # 尝试从配置文件直接加载
            config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    default_output_folder = config.get('default_output_folder', '')
            else:
                # 如果配置文件不存在，尝试从main_window获取
                if hasattr(self, 'main_window') and hasattr(self.main_window, 'config'):
                    default_output_folder = self.main_window.config.get('default_output_folder', '')
                else:
                    default_output_folder = ''
            
            if not default_output_folder:
                self.train_btn.setEnabled(False)
                self.update_status("请先在设置中配置默认输出文件夹")
                return False
                
            # 根据任务类型检查相应的数据集文件夹
            if self.task_type == "classification":
                dataset_folder = os.path.join(default_output_folder, 'dataset')
                train_folder = os.path.join(dataset_folder, 'train')
                val_folder = os.path.join(dataset_folder, 'val')
                
                # 检查训练集和验证集文件夹是否存在
                if os.path.exists(train_folder) and os.path.exists(val_folder):
                    self.train_btn.setEnabled(True)
                    self.update_status("分类数据集结构检查完成，可以开始训练")
                    return True
                else:
                    missing_folders = []
                    if not os.path.exists(dataset_folder):
                        missing_folders.append("dataset")
                    else:
                        if not os.path.exists(train_folder):
                            missing_folders.append("train")
                        if not os.path.exists(val_folder):
                            missing_folders.append("val")
                    
                    error_msg = f"缺少必要的文件夹: {', '.join(missing_folders)}"
                    self.update_status(error_msg)
                    self.train_btn.setEnabled(False)
                    return False
            else:
                detection_data_folder = os.path.join(default_output_folder, 'detection_data')
                train_images = os.path.join(detection_data_folder, 'images', 'train')
                val_images = os.path.join(detection_data_folder, 'images', 'val')
                train_labels = os.path.join(detection_data_folder, 'labels', 'train')
                val_labels = os.path.join(detection_data_folder, 'labels', 'val')
                
                # 检查目标检测数据集的完整性
                if (os.path.exists(train_images) and os.path.exists(val_images) and
                    os.path.exists(train_labels) and os.path.exists(val_labels)):
                    self.train_btn.setEnabled(True)
                    self.update_status("目标检测数据集结构检查完成，可以开始训练")
                    return True
                else:
                    missing_folders = []
                    if not os.path.exists(detection_data_folder):
                        missing_folders.append("detection_data")
                    else:
                        if not os.path.exists(os.path.join(detection_data_folder, 'images')):
                            missing_folders.append("images")
                        else:
                            if not os.path.exists(train_images):
                                missing_folders.append("images/train")
                            if not os.path.exists(val_images):
                                missing_folders.append("images/val")
                            
                        if not os.path.exists(os.path.join(detection_data_folder, 'labels')):
                            missing_folders.append("labels")
                        else:
                            if not os.path.exists(train_labels):
                                missing_folders.append("labels/train")
                            if not os.path.exists(val_labels):
                                missing_folders.append("labels/val")
                    
                    error_msg = f"缺少必要的文件夹: {', '.join(missing_folders)}"
                    self.update_status(error_msg)
                    self.train_btn.setEnabled(False)
                    return False
            
        except Exception as e:
            print(f"检查训练准备状态时出错: {str(e)}")
            self.train_btn.setEnabled(False)
            self.update_status(f"检查数据集结构时出错: {str(e)}")
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
    
    def update_training_progress(self, data_dict):
        """更新训练进度"""
        # 从字典中提取epoch和logs信息
        epoch = data_dict.get('epoch', 0) - 1  # 减1是因为我们的epoch从1开始，但索引从0开始
        logs = data_dict  # 直接使用整个字典作为logs
        
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
        
        # 直接传递整个data_dict到训练进度更新信号
        # 这个信号被连接到evaluation_tab.update_training_visualization
        self.training_progress_updated.emit(data_dict)
    
    def on_training_finished(self):
        """训练完成时调用"""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.open_model_folder_btn.setEnabled(True)  # 训练完成后启用打开模型文件夹按钮
        self.update_status("训练完成")
        self.update_progress(100)
        self.training_status_label.setText("训练完成")
        
        # 检查并创建模型保存目录
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_dir = os.path.join(root_dir, "models", "saved_models")
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            self.open_model_folder_btn.setEnabled(True)
    
    def on_training_error(self, error):
        """训练出错时调用"""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.open_model_folder_btn.setEnabled(False)  # 训练出错时禁用打开模型文件夹按钮
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
            
            # 获取模型命名备注
            model_note = self.classification_model_note_edit.text().strip()
                
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
                "use_augmentation": self.classification_augmentation_checkbox.isChecked(),
                "model_note": model_note  # 添加模型命名备注
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
            
            # 获取模型命名备注
            model_note = self.detection_model_note_edit.text().strip()
                
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
                "use_augmentation": self.detection_augmentation_checkbox.isChecked(),
                "model_note": model_note  # 添加模型命名备注
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
        # 检查防止重复调用的标志
        if hasattr(self, '_stopping_in_progress') and self._stopping_in_progress:
            return
            
        # 设置标志防止重复调用
        self._stopping_in_progress = True
        
        try:
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.update_status("训练已停止")
            self.training_status_label.setText("训练已停止")
            QMessageBox.information(self, "训练状态", "训练已成功停止")
        finally:
            # 重置标志
            self._stopping_in_progress = False

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

    def open_model_folder(self):
        """打开模型保存文件夹"""
        try:
            # 获取程序根目录
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            # 模型保存目录在models/saved_models目录下
            model_dir = os.path.join(root_dir, "models", "saved_models")
            
            if not os.path.exists(model_dir):
                QMessageBox.warning(self, "警告", "模型保存文件夹不存在，请先完成训练。")
                return
                
            # 打开文件夹
            if platform.system() == "Windows":
                os.startfile(model_dir)
            elif platform.system() == "Darwin":  # macOS
                import subprocess
                subprocess.run(["open", model_dir])
            else:  # Linux
                import subprocess
                subprocess.run(["xdg-open", model_dir])
                
        except Exception as e:
            QMessageBox.warning(self, "错误", f"无法打开模型保存文件夹: {str(e)}")

    def goto_annotation_tab(self):
        """切换到标注标签页"""
        if self.main_window and hasattr(self.main_window, 'tabs'):
            annotation_tab_index = -1
            for i in range(self.main_window.tabs.count()):
                if self.main_window.tabs.tabText(i) == "图像标注":
                    annotation_tab_index = i
                    break
            
            if annotation_tab_index >= 0:
                self.main_window.tabs.setCurrentIndex(annotation_tab_index)
                # 如果有标注标签页的引用，尝试切换到目标检测模式
                if hasattr(self.main_window, 'annotation_tab'):
                    # 检查是否有mode_radio_group
                    if hasattr(self.main_window.annotation_tab, 'mode_radio_group'):
                        # 获取按钮列表
                        buttons = self.main_window.annotation_tab.mode_radio_group.buttons()
                        # 选中目标检测按钮(通常是第二个按钮)
                        if len(buttons) >= 2:
                            buttons[1].setChecked(True)

    def update_status(self, message):
        """更新状态信息"""
        if hasattr(self, 'main_window') and hasattr(self.main_window, 'update_status'):
            self.main_window.update_status(message)
        # 更新训练状态标签
        if hasattr(self, 'training_status_label'):
            self.training_status_label.setText(message) 