from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, 
                           QGridLayout, QCheckBox, QListWidget, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
import os
import json
from .layer_config_widget import LayerConfigWidget
from .weight_config_manager import WeightConfigDisplayWidget


class ClassificationTrainingWidget(QWidget):
    """图片分类训练界面组件"""
    
    # 定义信号
    folder_changed = pyqtSignal(str)
    params_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.annotation_folder = ""
        self.layer_config = None
        self.init_ui()
    
    def init_ui(self):
        """初始化图片分类训练界面"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # 创建分类数据文件夹选择组
        folder_group = QGroupBox("训练数据目录")
        folder_layout = QHBoxLayout()
        folder_layout.setContentsMargins(10, 15, 10, 15)
        
        folder_layout.addWidget(QLabel("数据集路径:"))
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("请选择包含分类训练数据的文件夹")
        self.path_edit.setToolTip("选择包含已分类图像的文件夹，文件夹结构应为每个分类在单独的子文件夹中")
        
        folder_layout.addWidget(self.path_edit)
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_folder)
        refresh_btn.setToolTip("刷新训练数据集文件夹的路径和状态")
        folder_layout.addWidget(refresh_btn)
        
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_folder)
        browse_btn.setToolTip("浏览选择包含训练数据的根目录，每个子文件夹代表一个类别")
        folder_layout.addWidget(browse_btn)
        
        folder_group.setLayout(folder_layout)
        main_layout.addWidget(folder_group)
        
        # 创建类别权重信息显示组
        weight_info_group = QGroupBox("类别权重配置")
        self.weight_config_widget = WeightConfigDisplayWidget(self)
        weight_info_group.setLayout(QVBoxLayout())
        weight_info_group.layout().addWidget(self.weight_config_widget)
        main_layout.addWidget(weight_info_group)
        
        # 创建预训练模型组
        pretrained_group = QGroupBox("预训练模型")
        pretrained_layout = QHBoxLayout()
        pretrained_layout.setContentsMargins(10, 15, 10, 15)
        
        # 使用本地预训练模型复选框
        self.use_local_pretrained_checkbox = QCheckBox("使用本地预训练模型")
        self.use_local_pretrained_checkbox.setToolTip("选择是否使用本地已有的预训练模型文件，而非从网络下载")
        self.use_local_pretrained_checkbox.setChecked(False)
        self.use_local_pretrained_checkbox.stateChanged.connect(self.toggle_pretrained_controls)
        pretrained_layout.addWidget(self.use_local_pretrained_checkbox)
        
        pretrained_layout.addWidget(QLabel("预训练模型:"))
        self.pretrained_path_edit = QLineEdit()
        self.pretrained_path_edit.setReadOnly(True)
        self.pretrained_path_edit.setEnabled(False)
        self.pretrained_path_edit.setPlaceholderText("选择本地预训练模型文件")
        self.pretrained_path_edit.setToolTip("选择本地已有的预训练模型文件（.pth/.h5/.pb格式）")
        self.pretrained_btn = QPushButton("浏览...")
        self.pretrained_btn.setFixedWidth(60)
        self.pretrained_btn.setEnabled(False)
        self.pretrained_btn.clicked.connect(self.select_pretrained_model)
        self.pretrained_btn.setToolTip("浏览选择本地预训练模型文件")
        pretrained_layout.addWidget(self.pretrained_path_edit)
        pretrained_layout.addWidget(self.pretrained_btn)
        
        pretrained_group.setLayout(pretrained_layout)
        main_layout.addWidget(pretrained_group)
        
        # 创建基础训练参数组
        self.create_basic_params_group(main_layout)
        
        # 创建高级训练参数组
        self.create_advanced_params_group(main_layout)
        
        # 添加层配置组件
        self.layer_config_widget = LayerConfigWidget(self)
        self.layer_config_widget.set_task_type("classification")
        self.layer_config_widget.config_changed.connect(self.on_layer_config_changed)
        main_layout.addWidget(self.layer_config_widget)
    
    def create_basic_params_group(self, main_layout):
        """创建基础训练参数组"""
        basic_group = QGroupBox("基础训练参数")
        basic_layout = QGridLayout()
        basic_layout.setContentsMargins(10, 15, 10, 15)
        basic_layout.setSpacing(10)
        
        # 模型选择
        model_label = QLabel("模型:")
        model_label.setToolTip("选择用于图像分类的深度学习模型架构")
        basic_layout.addWidget(model_label, 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "MobileNetV2", "MobileNetV3", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
            "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4",
            "VGG16", "VGG19", "DenseNet121", "DenseNet169", "DenseNet201", "InceptionV3", "Xception"
        ])
        self.model_combo.setToolTip("选择不同的模型架构：\n- MobileNet系列：轻量级模型，适合移动设备\n- ResNet系列：残差网络，深度较大但训练稳定\n- EfficientNet系列：效率较高的模型\n- VGG系列：经典但参数较多的模型\n- DenseNet系列：密集连接的网络，参数利用率高\n- Inception/Xception：适合复杂特征提取")
        basic_layout.addWidget(self.model_combo, 0, 1)
        
        # 批次大小
        batch_label = QLabel("批次大小:")
        batch_label.setToolTip("每次模型权重更新时处理的样本数量")
        basic_layout.addWidget(batch_label, 1, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 256)
        self.batch_size_spin.setValue(32)
        self.batch_size_spin.setToolTip("批次大小影响训练速度和内存占用：\n- 较大批次：训练更稳定，但需要更多内存\n- 较小批次：内存占用少，但训练可能不稳定\n- 根据GPU内存大小调整，内存不足时请减小该值")
        basic_layout.addWidget(self.batch_size_spin, 1, 1)
        
        # 训练轮数
        epochs_label = QLabel("训练轮数:")
        epochs_label.setToolTip("模型训练的完整周期数")
        basic_layout.addWidget(epochs_label, 2, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(20)
        self.epochs_spin.setToolTip("训练轮数决定训练时长：\n- 轮数过少：模型可能欠拟合\n- 轮数过多：可能过拟合，浪费计算资源\n- 搭配早停策略使用效果更佳\n- 使用预训练模型时可适当减少轮数")
        basic_layout.addWidget(self.epochs_spin, 2, 1)
        
        # 学习率
        lr_label = QLabel("学习率:")
        lr_label.setToolTip("模型权重更新的步长大小")
        basic_layout.addWidget(lr_label, 3, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setToolTip("学习率是最重要的超参数之一：\n- 太大：训练不稳定，可能无法收敛\n- 太小：训练缓慢，可能陷入局部最优\n- 典型值：0.1 (SGD), 0.001 (Adam)\n- 微调预训练模型时使用较小学习率(0.0001)")
        basic_layout.addWidget(self.lr_spin, 3, 1)
        
        # 学习率调度器
        lr_sched_label = QLabel("学习率调度:")
        lr_sched_label.setToolTip("学习率随训练进程自动调整的策略")
        basic_layout.addWidget(lr_sched_label, 4, 0)
        self.lr_scheduler_combo = QComboBox()
        self.lr_scheduler_combo.addItems([
            "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau", "OneCycleLR", "CyclicLR"
        ])
        self.lr_scheduler_combo.setToolTip("学习率调度策略：\n- StepLR：按固定间隔降低学习率\n- CosineAnnealingLR：余弦周期性调整学习率\n- ReduceLROnPlateau：当指标不再改善时降低学习率\n- OneCycleLR：先增大再减小学习率，适合较短训练\n- CyclicLR：在两个界限间循环调整学习率")
        basic_layout.addWidget(self.lr_scheduler_combo, 4, 1)
        
        # 优化器
        optimizer_label = QLabel("优化器:")
        optimizer_label.setToolTip("控制模型权重如何根据梯度更新")
        basic_layout.addWidget(optimizer_label, 5, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems([
            "Adam", "SGD", "RMSprop", "Adagrad", "AdamW", "RAdam", "AdaBelief"
        ])
        self.optimizer_combo.setToolTip("不同的优化算法：\n- Adam：自适应算法，适用大多数情况\n- SGD：经典算法，配合动量可获得良好结果\n- RMSprop：类似于带衰减的AdaGrad\n- AdamW：修正Adam的权重衰减\n- RAdam：带修正的Adam，收敛更稳定\n- AdaBelief：最新优化器，通常更稳定")
        basic_layout.addWidget(self.optimizer_combo, 5, 1)
        
        # 权重衰减
        wd_label = QLabel("权重衰减:")
        wd_label.setToolTip("L2正则化参数，控制模型复杂度")
        basic_layout.addWidget(wd_label, 6, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0, 0.1)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.weight_decay_spin.setDecimals(5)
        self.weight_decay_spin.setValue(0.0001)
        self.weight_decay_spin.setToolTip("权重衰减可防止过拟合：\n- 较大值：模型更简单，泛化能力可能更强\n- 较小值：允许模型更复杂，拟合能力更强\n- 典型值：0.0001-0.001\n- 数据较少时可适当增大")
        basic_layout.addWidget(self.weight_decay_spin, 6, 1)
        
        # 激活函数
        activation_label = QLabel("激活函数:")
        activation_label.setToolTip("模型中使用的非线性激活函数")
        basic_layout.addWidget(activation_label, 7, 0)
        self.activation_combo = QComboBox()
        self.activation_combo.addItems([
            "None", "ReLU", "LeakyReLU", "PReLU", "ELU", "SELU", "GELU", "Mish", "Swish"
        ])
        self.activation_combo.setToolTip("不同的激活函数对过拟合有不同影响：\n- None：不使用激活函数，对某些模型效果更好\n- ReLU：最常用，但可能出现神经元死亡\n- LeakyReLU：解决ReLU神经元死亡问题\n- PReLU：带可学习参数的LeakyReLU\n- ELU：指数线性单元，有更好的梯度\n- SELU：自归一化激活函数\n- GELU：高斯误差线性单元\n- Mish：平滑激活函数\n- Swish：SiLU，自门控激活函数")
        basic_layout.addWidget(self.activation_combo, 7, 1)
        
        # 添加Dropout参数控制
        dropout_label = QLabel("Dropout率:")
        dropout_label.setToolTip("随机丢弃神经元的概率，用于防止过拟合")
        basic_layout.addWidget(dropout_label, 8, 0)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setValue(0.0)
        self.dropout_spin.setToolTip("Dropout会在训练时随机丢弃部分神经元：\n- 0值：不使用Dropout\n- 0.1-0.3：轻度正则化\n- 0.4-0.6：中度正则化，常用范围\n- >0.6：强度正则化，小数据集可尝试\n- 太大的值会导致模型欠拟合")
        basic_layout.addWidget(self.dropout_spin, 8, 1)
        
        # 评估指标
        metrics_label = QLabel("评估指标:")
        metrics_label.setToolTip("用于评估模型性能的指标")
        basic_layout.addWidget(metrics_label, 9, 0)
        self.metrics_list = QListWidget()
        self.metrics_list.setSelectionMode(QListWidget.MultiSelection)
        self.metrics_list.addItems([
            "accuracy", "precision", "recall", "f1_score", "confusion_matrix",
            "roc_auc", "average_precision", "top_k_accuracy", "balanced_accuracy"
        ])
        self.metrics_list.setToolTip("选择用于评估模型的指标：\n- accuracy：准确率，适用于平衡数据集\n- precision：精确率，关注减少假阳性\n- recall：召回率，关注减少假阴性\n- f1_score：精确率和召回率的调和平均\n- confusion_matrix：混淆矩阵\n- roc_auc：ROC曲线下面积\n- balanced_accuracy：平衡准确率，适用于不平衡数据集")
        # 默认选中accuracy
        self.metrics_list.setCurrentRow(0)
        # 设置固定高度，避免与其他控件重叠
        self.metrics_list.setFixedHeight(150)
        basic_layout.addWidget(self.metrics_list, 9, 1)
        
        # 使用预训练权重
        self.pretrained_checkbox = QCheckBox("使用预训练权重")
        self.pretrained_checkbox.setChecked(True)
        self.pretrained_checkbox.setToolTip("使用在大型数据集(如ImageNet)上预训练的模型权重：\n- 加快训练速度\n- 提高模型性能\n- 尤其在训练数据较少时效果显著\n- 需要网络连接下载权重文件")
        basic_layout.addWidget(self.pretrained_checkbox, 10, 0, 1, 2)
        
        # 数据增强
        self.augmentation_checkbox = QCheckBox("使用数据增强")
        self.augmentation_checkbox.setChecked(True)
        self.augmentation_checkbox.setToolTip("通过随机变换（旋转、裁剪、翻转等）增加训练数据多样性：\n- 减少过拟合\n- 提高模型泛化能力\n- 尤其在训练数据较少时非常有效")
        basic_layout.addWidget(self.augmentation_checkbox, 11, 0, 1, 2)
        
        basic_group.setLayout(basic_layout)
        main_layout.addWidget(basic_group)
    
    def create_advanced_params_group(self, main_layout):
        """创建高级训练参数组"""
        advanced_group = QGroupBox("高级训练参数")
        advanced_layout = QGridLayout()
        advanced_layout.setContentsMargins(10, 15, 10, 15)
        advanced_layout.setSpacing(10)
        
        # 早停
        early_stop_label = QLabel("启用早停:")
        early_stop_label.setToolTip("当验证指标不再改善时自动停止训练")
        advanced_layout.addWidget(early_stop_label, 0, 0)
        self.early_stopping_checkbox = QCheckBox("启用早停")
        self.early_stopping_checkbox.setChecked(True)
        self.early_stopping_checkbox.setToolTip("当验证指标在一定轮数内不再改善时停止训练：\n- 减少过拟合风险\n- 节省训练时间\n- 自动选择最佳模型")
        advanced_layout.addWidget(self.early_stopping_checkbox, 0, 1)
        
        # 早停耐心值
        patience_label = QLabel("早停耐心值:")
        patience_label.setToolTip("早停前允许的不改善轮数")
        advanced_layout.addWidget(patience_label, 0, 2)
        self.early_stopping_patience_spin = QSpinBox()
        self.early_stopping_patience_spin.setRange(1, 50)
        self.early_stopping_patience_spin.setValue(10)
        self.early_stopping_patience_spin.setToolTip("在停止训练前容忍的无改善轮数：\n- 较大值：更有耐心，可能训练更久\n- 较小值：更积极停止，可能过早停止\n- 典型值：5-15轮")
        advanced_layout.addWidget(self.early_stopping_patience_spin, 0, 3)
        
        # 梯度裁剪
        grad_clip_label = QLabel("启用梯度裁剪:")
        grad_clip_label.setToolTip("限制梯度大小以稳定训练")
        advanced_layout.addWidget(grad_clip_label, 1, 0)
        self.gradient_clipping_checkbox = QCheckBox("启用梯度裁剪")
        self.gradient_clipping_checkbox.setChecked(False)
        self.gradient_clipping_checkbox.setToolTip("限制梯度的最大范数，防止梯度爆炸：\n- 稳定训练过程\n- 尤其适用于循环神经网络\n- 可以使用更大的学习率")
        advanced_layout.addWidget(self.gradient_clipping_checkbox, 1, 1)
        
        # 梯度裁剪阈值
        clip_value_label = QLabel("梯度裁剪阈值:")
        clip_value_label.setToolTip("梯度裁剪的最大范数值")
        advanced_layout.addWidget(clip_value_label, 1, 2)
        self.gradient_clipping_value_spin = QDoubleSpinBox()
        self.gradient_clipping_value_spin.setRange(0.1, 10.0)
        self.gradient_clipping_value_spin.setSingleStep(0.1)
        self.gradient_clipping_value_spin.setValue(1.0)
        self.gradient_clipping_value_spin.setToolTip("梯度范数的最大允许值：\n- 较小值：裁剪更积极，训练更稳定但可能较慢\n- 较大值：裁剪更宽松\n- 典型值：1.0-5.0")
        advanced_layout.addWidget(self.gradient_clipping_value_spin, 1, 3)
        
        # 混合精度训练
        mixed_precision_label = QLabel("启用混合精度训练:")
        mixed_precision_label.setToolTip("使用FP16和FP32混合精度加速训练")
        advanced_layout.addWidget(mixed_precision_label, 2, 0)
        self.mixed_precision_checkbox = QCheckBox("启用混合精度训练")
        self.mixed_precision_checkbox.setChecked(True)
        self.mixed_precision_checkbox.setToolTip("使用FP16和FP32混合精度：\n- 加速训练(最高2倍)\n- 减少内存占用\n- 几乎不影响精度\n- 需要支持FP16的GPU")
        advanced_layout.addWidget(self.mixed_precision_checkbox, 2, 1)
        
        # 模型命名备注
        model_note_label = QLabel("模型命名备注:")
        model_note_label.setToolTip("添加到训练输出模型文件名中的备注")
        advanced_layout.addWidget(model_note_label, 3, 0)
        self.model_note_edit = QLineEdit()
        self.model_note_edit.setPlaceholderText("可选: 添加模型命名备注")
        self.model_note_edit.setToolTip("这个备注将添加到输出模型文件名中，方便识别不同训练的模型")
        advanced_layout.addWidget(self.model_note_edit, 3, 1, 1, 3)
        
        advanced_group.setLayout(advanced_layout)
        main_layout.addWidget(advanced_group)
    
    def refresh_folder(self):
        """刷新文件夹"""
        try:
            # 尝试从配置文件直接加载
            config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    default_output_folder = config.get('default_output_folder', '')
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
            self.path_edit.setText(dataset_folder)
            self.folder_changed.emit(dataset_folder)
            self.weight_config_widget.refresh_weight_config()
            
        except Exception as e:
            print(f"选择分类文件夹时出错: {str(e)}")
            QMessageBox.warning(self, "错误", f"无法获取默认输出文件夹设置: {str(e)}")
    
    def browse_folder(self):
        """浏览选择文件夹"""
        try:
            # 获取上次使用的目录或默认目录
            last_dir = ""
            config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    last_dir = config.get('default_output_folder', '')
            
            # 打开文件对话框
            folder_path = QFileDialog.getExistingDirectory(
                self, 
                "选择图像分类数据集文件夹", 
                last_dir,
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )
            
            if folder_path:
                # 检查文件夹结构是否符合要求
                train_folder = os.path.join(folder_path, 'train')
                val_folder = os.path.join(folder_path, 'val')
                
                if not os.path.exists(train_folder):
                    QMessageBox.warning(self, "无效的数据集", "所选文件夹内未找到train子文件夹，请确保数据集包含train和val文件夹")
                    return
                    
                if not os.path.exists(val_folder):
                    QMessageBox.warning(self, "无效的数据集", "所选文件夹内未找到val子文件夹，请确保数据集包含train和val文件夹")
                    return
                
                # 检查是否存在类别子文件夹
                class_folders = [f for f in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, f))]
                if not class_folders:
                    QMessageBox.warning(self, "无效的数据集", "train文件夹内未找到任何类别子文件夹")
                    return
                
                # 设置路径
                self.annotation_folder = folder_path
                self.path_edit.setText(folder_path)
                self.folder_changed.emit(folder_path)
                self.weight_config_widget.refresh_weight_config()
                
        except Exception as e:
            print(f"浏览选择分类文件夹时出错: {str(e)}")
            QMessageBox.warning(self, "错误", f"选择文件夹时出错: {str(e)}")
    
    def toggle_pretrained_controls(self, state):
        """切换预训练模型控件的启用状态"""
        enabled = state == Qt.Checked
        self.pretrained_path_edit.setEnabled(enabled)
        self.pretrained_btn.setEnabled(enabled)
    
    def select_pretrained_model(self):
        """选择本地预训练模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择预训练模型文件",
            "",
            "模型文件 (*.pth *.pt *.h5 *.hdf5 *.pkl);;所有文件 (*.*)"
        )
        if file_path:
            self.pretrained_path_edit.setText(file_path)
    
    def on_layer_config_changed(self, config):
        """处理层配置变更"""
        self.layer_config = config
        self.params_changed.emit()
    
    def get_training_params(self):
        """获取训练参数"""
        # 获取所有选中的评估指标
        selected_metrics = [item.text() for item in self.metrics_list.selectedItems()]
        if not selected_metrics:
            selected_metrics = ["accuracy"]  # 默认使用accuracy
            
        # 获取预训练模型信息
        use_local_pretrained = self.use_local_pretrained_checkbox.isChecked()
        pretrained_path = self.pretrained_path_edit.text() if use_local_pretrained else ""
        pretrained_model = "" if use_local_pretrained else self.model_combo.currentText()
        
        # 获取模型命名备注
        model_note = self.model_note_edit.text().strip()
            
        params = {
            "task_type": "classification",
            "model": self.model_combo.currentText(),
            "batch_size": self.batch_size_spin.value(),
            "epochs": self.epochs_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "optimizer": self.optimizer_combo.currentText(),
            "metrics": selected_metrics,
            "use_pretrained": self.pretrained_checkbox.isChecked(),
            "use_local_pretrained": use_local_pretrained,
            "pretrained_path": pretrained_path,
            "pretrained_model": pretrained_model,
            "use_augmentation": self.augmentation_checkbox.isChecked(),
            "model_note": model_note,
            
            # 添加已有但之前未收集的参数
            "weight_decay": self.weight_decay_spin.value(),
            "lr_scheduler": self.lr_scheduler_combo.currentText(),
            "early_stopping": self.early_stopping_checkbox.isChecked(),
            "early_stopping_patience": self.early_stopping_patience_spin.value(),
            "gradient_clipping": self.gradient_clipping_checkbox.isChecked(),
            "gradient_clipping_value": self.gradient_clipping_value_spin.value(),
            "mixed_precision": self.mixed_precision_checkbox.isChecked(),
            "activation_function": self.activation_combo.currentText(),
            "dropout_rate": self.dropout_spin.value(),
        }
        
        # 添加层配置
        if self.layer_config:
            params['layer_config'] = self.layer_config
            
        return params
    
    def set_folder_path(self, path):
        """设置文件夹路径"""
        self.annotation_folder = path
        self.path_edit.setText(path)
        self.weight_config_widget.refresh_weight_config()
    
    def get_folder_path(self):
        """获取文件夹路径"""
        return self.annotation_folder 