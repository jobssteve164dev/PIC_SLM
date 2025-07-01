from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, 
                           QGridLayout, QCheckBox, QListWidget, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
import os
import json
from .layer_config_widget import LayerConfigWidget
from .weight_config_manager import WeightConfigDisplayWidget


class DetectionTrainingWidget(QWidget):
    """目标检测训练界面组件"""
    
    # 定义信号
    folder_changed = pyqtSignal(str)
    params_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.annotation_folder = ""
        self.layer_config = None
        self.init_ui()
    
    def init_ui(self):
        """初始化目标检测训练界面"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # 创建检测数据文件夹选择组
        folder_group = QGroupBox("训练数据目录")
        folder_layout = QHBoxLayout()
        folder_layout.setContentsMargins(10, 15, 10, 15)
        
        folder_layout.addWidget(QLabel("数据集路径:"))
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("请选择包含目标检测训练数据的文件夹")
        self.path_edit.setToolTip("选择包含目标检测数据的文件夹，需要标注文件（YOLO或COCO格式）和图像文件")
        
        folder_layout.addWidget(self.path_edit)
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_folder)
        refresh_btn.setToolTip("刷新目标检测数据集文件夹的路径和状态")
        folder_layout.addWidget(refresh_btn)
        
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_folder)
        browse_btn.setToolTip("浏览选择包含已标注目标检测数据的文件夹，包括图像和标注文件")
        folder_layout.addWidget(browse_btn)
        
        folder_group.setLayout(folder_layout)
        main_layout.addWidget(folder_group)
        
        # 创建类别权重信息显示组（目标检测）
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
        self.pretrained_path_edit.setToolTip("选择本地已有的预训练目标检测模型文件（.pth/.weights/.pt格式）")
        self.pretrained_btn = QPushButton("浏览...")
        self.pretrained_btn.setFixedWidth(60)
        self.pretrained_btn.setEnabled(False)
        self.pretrained_btn.clicked.connect(self.select_pretrained_model)
        self.pretrained_btn.setToolTip("浏览选择本地预训练目标检测模型文件")
        pretrained_layout.addWidget(self.pretrained_path_edit)
        pretrained_layout.addWidget(self.pretrained_btn)
        
        pretrained_group.setLayout(pretrained_layout)
        main_layout.addWidget(pretrained_group)
        
        # 创建基础训练参数组
        self.create_basic_params_group(main_layout)
        
        # 创建高级训练参数组
        self.create_advanced_params_group(main_layout)
        
        # 创建新增的高级超参数组件
        self.create_advanced_hyperparameters_group(main_layout)
        
        # 添加层配置组件
        self.layer_config_widget = LayerConfigWidget(self)
        self.layer_config_widget.set_task_type("detection")
        self.layer_config_widget.config_changed.connect(self.on_layer_config_changed)
        main_layout.addWidget(self.layer_config_widget)
    
    def create_basic_params_group(self, main_layout):
        """创建基础训练参数组"""
        basic_group = QGroupBox("基础训练参数")
        basic_layout = QGridLayout()
        basic_layout.setContentsMargins(10, 15, 10, 15)
        basic_layout.setSpacing(10)
        
        # 检测模型选择
        model_label = QLabel("检测模型:")
        model_label.setToolTip("选择用于目标检测的深度学习模型架构")
        basic_layout.addWidget(model_label, 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "YOLOv5", "YOLOv8", "YOLOv7", "YOLOv6", "YOLOv4", "YOLOv3",
            "SSD", "SSD512", "SSD300", "Faster R-CNN", "Mask R-CNN",
            "RetinaNet", "DETR", "Swin Transformer", "DINO", "Deformable DETR"
        ])
        self.model_combo.setToolTip("选择不同的目标检测模型：\n- YOLO系列：单阶段检测器，速度快精度适中\n- SSD系列：单阶段多尺度检测器\n- Faster/Mask R-CNN：两阶段检测器，精度高\n- DETR系列：基于Transformer的端到端检测器\n- 不同模型在速度和精度上有权衡")
        basic_layout.addWidget(self.model_combo, 0, 1)
        
        # 批次大小
        batch_label = QLabel("批次大小:")
        batch_label.setToolTip("每次模型权重更新时处理的样本数量")
        basic_layout.addWidget(batch_label, 1, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(16)
        self.batch_size_spin.setToolTip("目标检测训练的批次大小：\n- 检测模型通常需要更大内存\n- 典型值：8-16（普通GPU）\n- 内存不足时请减小该值\n- 较小值也可能提高精度但训练更慢")
        basic_layout.addWidget(self.batch_size_spin, 1, 1)
        
        # 训练轮数
        epochs_label = QLabel("训练轮数:")
        epochs_label.setToolTip("模型训练的完整周期数")
        basic_layout.addWidget(epochs_label, 2, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setToolTip("检测模型训练轮数：\n- 检测模型通常需要更多轮数收敛\n- 典型值：50-100轮\n- 使用预训练时可减少到30-50轮\n- 搭配早停策略效果更佳")
        basic_layout.addWidget(self.epochs_spin, 2, 1)
        
        # 学习率
        lr_label = QLabel("学习率:")
        lr_label.setToolTip("模型权重更新的步长大小")
        basic_layout.addWidget(lr_label, 3, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.01)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setValue(0.0005)
        self.lr_spin.setToolTip("检测模型学习率：\n- 通常比分类模型小一个数量级\n- 典型值：0.0005-0.001\n- 训练不稳定时可减小\n- 微调预训练模型时使用更小值(0.00005)")
        basic_layout.addWidget(self.lr_spin, 3, 1)
        
        # 学习率调度器
        lr_sched_label = QLabel("学习率调度:")
        lr_sched_label.setToolTip("学习率随训练进程自动调整的策略")
        basic_layout.addWidget(lr_sched_label, 4, 0)
        self.lr_scheduler_combo = QComboBox()
        self.lr_scheduler_combo.addItems([
            "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau", "OneCycleLR", "CyclicLR",
            "WarmupCosineLR", "LinearWarmup"
        ])
        self.lr_scheduler_combo.setToolTip("目标检测特有的学习率调度：\n- WarmupCosineLR：先预热再余弦衰减，YOLO常用\n- LinearWarmup：线性预热，目标检测常用\n- ReduceLROnPlateau：性能平台时降低学习率\n- CosineAnnealingLR：余弦周期调整")
        basic_layout.addWidget(self.lr_scheduler_combo, 4, 1)
        
        # 优化器
        optimizer_label = QLabel("优化器:")
        optimizer_label.setToolTip("控制模型权重如何根据梯度更新")
        basic_layout.addWidget(optimizer_label, 5, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems([
            "Adam", "SGD", "AdamW", "RAdam", "AdaBelief", "Lion"
        ])
        self.optimizer_combo.setToolTip("检测模型优化器：\n- SGD：YOLO推荐使用，配合动量和预热\n- Adam：稳定但可能精度略低\n- AdamW：带修正权重衰减的Adam\n- Lion：新型高效优化器，可节省显存")
        basic_layout.addWidget(self.optimizer_combo, 5, 1)
        
        # 权重衰减
        wd_label = QLabel("权重衰减:")
        wd_label.setToolTip("L2正则化参数，控制模型复杂度")
        basic_layout.addWidget(wd_label, 6, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0, 0.1)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.weight_decay_spin.setDecimals(5)
        self.weight_decay_spin.setValue(0.0005)
        self.weight_decay_spin.setToolTip("检测模型权重衰减：\n- YOLO系列通常使用0.0005\n- Faster R-CNN使用0.0001\n- 数据量小时可适当增大\n- 过大可能导致欠拟合")
        basic_layout.addWidget(self.weight_decay_spin, 6, 1)
        
        # 激活函数
        activation_label = QLabel("激活函数:")
        activation_label.setToolTip("模型中使用的非线性激活函数")
        basic_layout.addWidget(activation_label, 7, 0)
        self.activation_combo = QComboBox()
        self.activation_combo.addItems([
            "None", "ReLU", "LeakyReLU", "PReLU", "ELU", "SELU", "GELU", "Mish", "Swish", "SiLU"
        ])
        self.activation_combo.setCurrentText("SiLU")  # YOLO默认使用SiLU/Swish
        self.activation_combo.setToolTip("检测模型激活函数：\n- None：不使用激活函数，某些情况效果更好\n- LeakyReLU：YOLO早期版本默认激活函数\n- SiLU/Swish：YOLOv5/v8默认，性能更好\n- Mish：YOLOv4默认激活函数\n- 合适的激活函数可减轻过拟合\n- 不同架构有不同推荐激活函数")
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
            "mAP", "mAP50", "mAP75", "precision", "recall", "f1_score", 
            "box_loss", "class_loss", "obj_loss"
        ])
        self.metrics_list.setToolTip("检测模型评估指标：\n- mAP：不同IOU阈值下的平均精度，主要指标\n- mAP50：IOU阈值为0.5的平均精度\n- mAP75：IOU阈值为0.75的平均精度，更严格\n- precision/recall：精确率/召回率\n- f1_score：精确率和召回率的调和平均值\n- 各种loss：用于诊断训练问题")
        # 默认选中mAP
        self.metrics_list.setCurrentRow(0)
        # 设置固定高度，避免与其他控件重叠
        self.metrics_list.setFixedHeight(150)
        basic_layout.addWidget(self.metrics_list, 9, 1)
        
        # 输入分辨率
        resolution_label = QLabel("输入分辨率:")
        resolution_label.setToolTip("模型训练和推理的图像尺寸")
        basic_layout.addWidget(resolution_label, 10, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "416x416", "512x512", "640x640", "768x768", "1024x1024", "1280x1280"
        ])
        self.resolution_combo.setToolTip("输入分辨率影响速度和精度：\n- 更大分辨率：更高精度，尤其对小物体\n- 更小分辨率：更快速度，适合实时应用\n- 640x640是YOLO常用分辨率\n- 根据目标大小和GPU内存选择")
        self.resolution_combo.setCurrentText("640x640")
        basic_layout.addWidget(self.resolution_combo, 10, 1)
        
        # IOU阈值
        iou_label = QLabel("IOU阈值:")
        iou_label.setToolTip("交并比阈值，用于训练和非极大值抑制")
        basic_layout.addWidget(iou_label, 11, 0)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.1, 0.9)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setValue(0.5)
        self.iou_spin.setToolTip("IOU(交并比)阈值：\n- 训练中用于正负样本分配\n- 推理中用于NMS筛选\n- 较高值：更严格的重叠判定\n- 较低值：更宽松的重叠判定\n- 典型值：0.5或0.45")
        basic_layout.addWidget(self.iou_spin, 11, 1)
        
        # 置信度阈值
        conf_label = QLabel("置信度阈值:")
        conf_label.setToolTip("检测结果的最小置信度分数")
        basic_layout.addWidget(conf_label, 12, 0)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setToolTip("置信度阈值：\n- 推理时保留的最小目标置信度\n- 较高值：减少假阳性，但可能漏检\n- 较低值：提高召回率，但增加假阳性\n- 典型值：0.25-0.45\n- 可根据应用场景调整")
        basic_layout.addWidget(self.conf_spin, 12, 1)
        
        # Multi-scale 训练
        ms_label = QLabel("多尺度训练:")
        ms_label.setToolTip("在训练中随机调整输入图像尺寸")
        basic_layout.addWidget(ms_label, 13, 0)
        self.multiscale_checkbox = QCheckBox("启用多尺度训练")
        self.multiscale_checkbox.setChecked(True)
        self.multiscale_checkbox.setToolTip("多尺度训练增强模型泛化能力：\n- 在训练中随机调整输入图像大小\n- 提高模型对不同尺寸目标的适应性\n- 可能增加训练时间\n- YOLO模型常用技术")
        basic_layout.addWidget(self.multiscale_checkbox, 13, 1)
        
        # Mosaic 数据增强
        mosaic_label = QLabel("马赛克增强:")
        mosaic_label.setToolTip("YOLO系列特有的数据增强方法")
        basic_layout.addWidget(mosaic_label, 14, 0)
        self.mosaic_checkbox = QCheckBox("启用马赛克增强")
        self.mosaic_checkbox.setChecked(True)
        self.mosaic_checkbox.setToolTip("马赛克数据增强：\n- 将4张图像拼接成1张\n- 大幅增加目标数量和上下文变化\n- 显著提高小目标检测性能\n- YOLOv5之后广泛使用的增强方法")
        basic_layout.addWidget(self.mosaic_checkbox, 14, 1)
        
        # 使用预训练权重
        self.pretrained_checkbox = QCheckBox("使用预训练权重")
        self.pretrained_checkbox.setChecked(True)
        self.pretrained_checkbox.setToolTip("使用在COCO等大型数据集上预训练的模型：\n- 加快检测模型收敛速度\n- 显著提高最终精度\n- 尤其在训练数据较少时更有效\n- YOLO和大多数检测模型都支持")
        basic_layout.addWidget(self.pretrained_checkbox, 15, 0)
        
        # 常规数据增强
        self.augmentation_checkbox = QCheckBox("使用数据增强")
        self.augmentation_checkbox.setChecked(True)
        self.augmentation_checkbox.setToolTip("常规数据增强：\n- 翻转、旋转、色彩变换等\n- 减少过拟合，提高泛化能力\n- 检测任务中一般都需要启用\n- 需要bbox坐标同步转换")
        basic_layout.addWidget(self.augmentation_checkbox, 15, 1)
        
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
        self.early_stopping_checkbox.setToolTip("当mAP在一定轮数内不再提高时停止训练：\n- 避免不必要的训练时间\n- 减少过拟合风险\n- 自动保存最佳模型")
        advanced_layout.addWidget(self.early_stopping_checkbox, 0, 1)
        
        # 早停耐心值
        patience_label = QLabel("早停耐心值:")
        patience_label.setToolTip("早停前允许的不改善轮数")
        advanced_layout.addWidget(patience_label, 0, 2)
        self.early_stopping_patience_spin = QSpinBox()
        self.early_stopping_patience_spin.setRange(1, 50)
        self.early_stopping_patience_spin.setValue(10)
        self.early_stopping_patience_spin.setToolTip("目标检测早停耐心值：\n- 检测模型可能需要更大耐心值\n- 典型值：10-15轮\n- 过小可能过早停止\n- 过大则失去早停意义")
        advanced_layout.addWidget(self.early_stopping_patience_spin, 0, 3)
        
        # 梯度裁剪
        grad_clip_label = QLabel("启用梯度裁剪:")
        grad_clip_label.setToolTip("限制梯度大小以稳定训练")
        advanced_layout.addWidget(grad_clip_label, 1, 0)
        self.gradient_clipping_checkbox = QCheckBox("启用梯度裁剪")
        self.gradient_clipping_checkbox.setChecked(False)
        self.gradient_clipping_checkbox.setToolTip("目标检测中的梯度裁剪：\n- 大型检测模型更容易出现梯度不稳定\n- 预防梯度爆炸和训练不稳定\n- 尤其在高学习率时有用\n- Faster R-CNN等常用技术")
        advanced_layout.addWidget(self.gradient_clipping_checkbox, 1, 1)
        
        # 梯度裁剪阈值
        clip_value_label = QLabel("梯度裁剪阈值:")
        clip_value_label.setToolTip("梯度裁剪的最大范数值")
        advanced_layout.addWidget(clip_value_label, 1, 2)
        self.gradient_clipping_value_spin = QDoubleSpinBox()
        self.gradient_clipping_value_spin.setRange(0.1, 10.0)
        self.gradient_clipping_value_spin.setSingleStep(0.1)
        self.gradient_clipping_value_spin.setValue(1.0)
        self.gradient_clipping_value_spin.setToolTip("检测模型梯度裁剪阈值：\n- 两阶段检测器常用值：10.0\n- YOLO常用值：4.0\n- 训练不稳定时可降低该值\n- 精调时通常使用较小值")
        advanced_layout.addWidget(self.gradient_clipping_value_spin, 1, 3)
        
        # 混合精度训练
        mixed_precision_label = QLabel("启用混合精度训练:")
        mixed_precision_label.setToolTip("使用FP16和FP32混合精度加速训练")
        advanced_layout.addWidget(mixed_precision_label, 2, 0)
        self.mixed_precision_checkbox = QCheckBox("启用混合精度训练")
        self.mixed_precision_checkbox.setChecked(True)
        self.mixed_precision_checkbox.setToolTip("目标检测模型混合精度训练：\n- 检测模型受益更大，加速可达2倍\n- 减少50%GPU内存使用\n- 几乎不影响最终精度\n- 建议所有支持FP16的GPU都启用")
        advanced_layout.addWidget(self.mixed_precision_checkbox, 2, 1)
        
        # EMA - 指数移动平均
        ema_label = QLabel("启用EMA:")
        ema_label.setToolTip("使用权重的指数移动平均提高稳定性")
        advanced_layout.addWidget(ema_label, 2, 2)
        self.ema_checkbox = QCheckBox("启用指数移动平均")
        self.ema_checkbox.setChecked(True)
        self.ema_checkbox.setToolTip("模型权重的指数移动平均：\n- 产生更平滑和稳定的模型\n- 提高测试精度和泛化能力\n- YOLO默认开启此功能\n- 几乎不增加计算负担")
        advanced_layout.addWidget(self.ema_checkbox, 2, 3)
        
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
    
    def create_advanced_hyperparameters_group(self, main_layout):
        """创建高级超参数组（阶段一新增）"""
        from .advanced_hyperparameters_widget import AdvancedHyperparametersWidget
        
        advanced_hyperparams_group = QGroupBox("高级超参数 (专业)")
        advanced_hyperparams_group.setToolTip("专业用户使用的高级超参数配置，包括优化器高级参数、学习率预热、标签平滑等")
        advanced_hyperparams_layout = QVBoxLayout()
        
        # 创建高级超参数组件
        self.advanced_hyperparams_widget = AdvancedHyperparametersWidget(self)
        self.advanced_hyperparams_widget.params_changed.connect(self.params_changed)
        advanced_hyperparams_layout.addWidget(self.advanced_hyperparams_widget)
        
        advanced_hyperparams_group.setLayout(advanced_hyperparams_layout)
        main_layout.addWidget(advanced_hyperparams_group)
    
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
                
            detection_data_folder = os.path.join(default_output_folder, 'detection_data')
            if not os.path.exists(detection_data_folder):
                QMessageBox.warning(self, "错误", "未找到目标检测数据集文件夹，请先完成目标检测标注")
                return
                
            self.annotation_folder = detection_data_folder
            self.path_edit.setText(detection_data_folder)
            self.folder_changed.emit(detection_data_folder)
            self.weight_config_widget.refresh_weight_config()
            
        except Exception as e:
            print(f"选择检测文件夹时出错: {str(e)}")
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
                "选择目标检测数据集文件夹", 
                last_dir,
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )
            
            if folder_path:
                # 检查文件夹结构是否符合要求
                images_folder = os.path.join(folder_path, 'images')
                labels_folder = os.path.join(folder_path, 'labels')
                
                if not os.path.exists(images_folder):
                    QMessageBox.warning(self, "无效的数据集", "所选文件夹内未找到images子文件夹，请确保数据集符合YOLO格式")
                    return
                    
                if not os.path.exists(labels_folder):
                    QMessageBox.warning(self, "无效的数据集", "所选文件夹内未找到labels子文件夹，请确保数据集符合YOLO格式")
                    return
                
                # 检查是否存在train和val子文件夹
                train_images = os.path.join(images_folder, 'train')
                val_images = os.path.join(images_folder, 'val')
                
                if not (os.path.exists(train_images) and os.path.exists(val_images)):
                    QMessageBox.warning(self, "无效的数据集", "未找到images/train和images/val文件夹，请确保数据集符合YOLO格式")
                    return
                
                # 设置路径
                self.annotation_folder = folder_path
                self.path_edit.setText(folder_path)
                self.folder_changed.emit(folder_path)
                self.weight_config_widget.refresh_weight_config()
                
        except Exception as e:
            print(f"浏览选择检测文件夹时出错: {str(e)}")
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
            "模型文件 (*.pth *.pt *.weights *.h5 *.hdf5 *.pkl);;所有文件 (*.*)"
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
            selected_metrics = ["mAP"]  # 默认使用mAP
            
        # 获取预训练模型信息
        use_local_pretrained = self.use_local_pretrained_checkbox.isChecked()
        pretrained_path = self.pretrained_path_edit.text() if use_local_pretrained else ""
        pretrained_model = "" if use_local_pretrained else self.model_combo.currentText()
        
        # 获取模型命名备注
        model_note = self.model_note_edit.text().strip()
            
        params = {
            "task_type": "detection",
            "model": self.model_combo.currentText(),
            "batch_size": self.batch_size_spin.value(),
            "epochs": self.epochs_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "optimizer": self.optimizer_combo.currentText(),
            "metrics": selected_metrics,
            "resolution": self.resolution_combo.currentText(),
            "iou_threshold": self.iou_spin.value(),
            "conf_threshold": self.conf_spin.value(),
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
            "use_mosaic": self.mosaic_checkbox.isChecked(),
            "use_multiscale": self.multiscale_checkbox.isChecked(),
            "use_ema": self.ema_checkbox.isChecked(),
            "activation_function": self.activation_combo.currentText(),
            "dropout_rate": self.dropout_spin.value(),
        }
        
        # 添加新的高级超参数（阶段一新增）
        if hasattr(self, 'advanced_hyperparams_widget'):
            advanced_config = self.advanced_hyperparams_widget.get_config()
            params.update(advanced_config)
        
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