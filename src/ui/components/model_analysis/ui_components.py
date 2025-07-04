from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                           QFileDialog, QComboBox, QLineEdit, QGridLayout, QTabWidget,
                           QSpinBox, QDoubleSpinBox, QFormLayout, QCheckBox, QScrollArea,
                           QProgressBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from .image_viewer import ZoomableImageViewer


def create_model_section(parent):
    """创建模型加载区域"""
    model_group = QGroupBox("模型配置")
    model_layout = QGridLayout()
    
    # 模型类型选择
    model_layout.addWidget(QLabel("模型类型:"), 0, 0)
    model_type_combo = QComboBox()
    model_type_combo.addItems(["分类模型", "检测模型"])
    model_layout.addWidget(model_type_combo, 0, 1)
    
    # 模型架构选择
    model_layout.addWidget(QLabel("模型架构:"), 0, 2)
    model_arch_combo = QComboBox()
    model_arch_combo.addItems([
        "MobileNetV2", "MobileNetV3", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
        "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4",
        "VGG16", "VGG19", "DenseNet121", "DenseNet169", "DenseNet201", "InceptionV3", "Xception"
    ])
    model_layout.addWidget(model_arch_combo, 0, 3)
    
    # 模型文件选择
    model_path_edit = QLineEdit()
    model_path_edit.setReadOnly(True)
    model_path_edit.setPlaceholderText("请选择训练好的模型文件")
    
    model_btn = QPushButton("浏览...")
    
    model_layout.addWidget(QLabel("模型文件:"), 1, 0)
    model_layout.addWidget(model_path_edit, 1, 1, 1, 2)
    model_layout.addWidget(model_btn, 1, 3)
    
    # 类别信息文件
    class_info_path_edit = QLineEdit()
    class_info_path_edit.setReadOnly(True)
    class_info_path_edit.setPlaceholderText("请选择类别信息文件")
    
    class_info_btn = QPushButton("浏览...")
    
    model_layout.addWidget(QLabel("类别信息:"), 2, 0)
    model_layout.addWidget(class_info_path_edit, 2, 1, 1, 2)
    model_layout.addWidget(class_info_btn, 2, 3)
    
    # 加载模型按钮
    load_model_btn = QPushButton("加载模型")
    load_model_btn.setEnabled(False)
    model_layout.addWidget(load_model_btn, 3, 0, 1, 4)
    
    model_group.setLayout(model_layout)
    
    return {
        'group': model_group,
        'model_type_combo': model_type_combo,
        'model_arch_combo': model_arch_combo,
        'model_path_edit': model_path_edit,
        'model_btn': model_btn,
        'class_info_path_edit': class_info_path_edit,
        'class_info_btn': class_info_btn,
        'load_model_btn': load_model_btn
    }


def create_image_section(parent):
    """创建图片选择区域"""
    image_group = QGroupBox("图片配置")
    image_layout = QHBoxLayout()
    
    # 图片选择按钮
    select_image_btn = QPushButton("选择图片")
    image_layout.addWidget(select_image_btn)
    
    # 显示选中的图片路径
    image_path_edit = QLineEdit()
    image_path_edit.setReadOnly(True)
    image_path_edit.setPlaceholderText("请选择要分析的图片")
    image_layout.addWidget(image_path_edit)
    
    # 原始图片显示
    original_image_label = QLabel("原始图片")
    original_image_label.setAlignment(Qt.AlignCenter)
    original_image_label.setMinimumSize(200, 200)
    original_image_label.setStyleSheet("border: 1px solid gray;")
    image_layout.addWidget(original_image_label)
    
    image_group.setLayout(image_layout)
    
    return {
        'group': image_group,
        'select_image_btn': select_image_btn,
        'image_path_edit': image_path_edit,
        'original_image_label': original_image_label
    }


def create_analysis_section(parent):
    """创建分析选择区域"""
    analysis_group = QGroupBox("分析配置")
    analysis_layout = QVBoxLayout()
    
    # 分析方法选择
    methods_layout = QHBoxLayout()
    methods_layout.addWidget(QLabel("选择分析方法:"))
    
    feature_checkbox = QCheckBox("特征可视化")
    feature_checkbox.setToolTip("显示模型各层提取的特征图，帮助理解模型如何感知图像特征")
    
    gradcam_checkbox = QCheckBox("GradCAM")
    gradcam_checkbox.setToolTip("梯度加权类激活映射，通过热力图可视化模型关注的图像区域")
    
    lime_checkbox = QCheckBox("LIME解释")
    lime_checkbox.setToolTip("局部可解释性模型，通过扰动图像来分析哪些区域对预测结果影响最大")
    
    sensitivity_checkbox = QCheckBox("敏感性分析")
    sensitivity_checkbox.setToolTip("通过添加不同程度的噪声扰动，测试模型对输入变化的敏感程度")
    
    # 新增的第一阶段方法
    integrated_gradients_checkbox = QCheckBox("Integrated Gradients")
    integrated_gradients_checkbox.setToolTip("积分梯度方法，解决梯度饱和问题，提供更稳定的特征重要性解释")
    
    shap_checkbox = QCheckBox("SHAP解释")
    shap_checkbox.setToolTip("基于博弈论Shapley值的统一解释框架，提供全局和局部解释")
    
    smoothgrad_checkbox = QCheckBox("SmoothGrad")
    smoothgrad_checkbox.setToolTip("平滑梯度方法，通过添加噪声并平均梯度来减少噪声干扰")
    
    # 第一行布局
    methods_layout.addWidget(feature_checkbox)
    methods_layout.addWidget(gradcam_checkbox)
    methods_layout.addWidget(lime_checkbox)
    methods_layout.addWidget(sensitivity_checkbox)
    methods_layout.addStretch()
    
    analysis_layout.addLayout(methods_layout)
    
    # 第二行新方法
    new_methods_layout = QHBoxLayout()
    new_methods_layout.addWidget(integrated_gradients_checkbox)
    new_methods_layout.addWidget(shap_checkbox)
    new_methods_layout.addWidget(smoothgrad_checkbox)
    new_methods_layout.addStretch()
    
    analysis_layout.addLayout(new_methods_layout)
    
    # 公共参数
    common_layout = QHBoxLayout()
    
    # 目标类别选择
    common_layout.addWidget(QLabel("目标类别:"))
    class_combo = QComboBox()
    class_combo.setToolTip("选择要分析的目标类别，分析将针对该类别进行")
    class_combo.setMinimumWidth(200)  # 设置最小宽度为200像素
    class_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)  # 根据内容调整宽度
    common_layout.addWidget(class_combo)
    
    common_layout.addStretch()
    analysis_layout.addLayout(common_layout)
    
    # 分析参数设置
    params_layout = QFormLayout()
    
    # LIME参数
    num_superpixels = QSpinBox()
    num_superpixels.setRange(10, 1000)
    num_superpixels.setValue(100)
    num_superpixels.setToolTip("LIME分析时将图像分割的超像素数量，值越大分割越细致")
    params_layout.addRow("LIME超像素数量:", num_superpixels)
    
    num_samples = QSpinBox()
    num_samples.setRange(100, 10000)
    num_samples.setValue(1000)
    num_samples.setToolTip("LIME分析时生成的随机样本数量，值越大结果越准确但计算时间更长")
    params_layout.addRow("LIME样本数量:", num_samples)
    
    # 敏感性分析参数
    perturbation_range = QDoubleSpinBox()
    perturbation_range.setRange(0.01, 1.0)
    perturbation_range.setValue(0.1)
    perturbation_range.setSingleStep(0.01)
    perturbation_range.setToolTip("敏感性分析时添加的最大扰动范围，值越大扰动越明显")
    params_layout.addRow("敏感性扰动范围:", perturbation_range)
    
    num_steps = QSpinBox()
    num_steps.setRange(10, 100)
    num_steps.setValue(20)
    num_steps.setToolTip("敏感性分析时从0到最大扰动范围的步数，值越大曲线越平滑")
    params_layout.addRow("敏感性扰动步数:", num_steps)
    
    # Integrated Gradients参数
    ig_steps = QSpinBox()
    ig_steps.setRange(10, 200)
    ig_steps.setValue(50)
    ig_steps.setToolTip("Integrated Gradients的积分步数，值越大结果越精确但计算时间更长")
    params_layout.addRow("IG积分步数:", ig_steps)
    
    baseline_type = QComboBox()
    baseline_type.addItems(["zero", "gaussian", "mean"])
    baseline_type.setToolTip("基线图像类型：zero(全零)、gaussian(高斯噪声)、mean(平均值)")
    params_layout.addRow("IG基线类型:", baseline_type)
    
    # SHAP参数
    shap_samples = QSpinBox()
    shap_samples.setRange(50, 1000)
    shap_samples.setValue(100)
    shap_samples.setToolTip("SHAP分析的采样数量，值越大结果越准确但计算时间更长")
    params_layout.addRow("SHAP采样数量:", shap_samples)
    
    # SmoothGrad参数
    smoothgrad_samples = QSpinBox()
    smoothgrad_samples.setRange(20, 200)
    smoothgrad_samples.setValue(50)
    smoothgrad_samples.setToolTip("SmoothGrad的噪声采样次数，值越大平滑效果越好")
    params_layout.addRow("SmoothGrad采样次数:", smoothgrad_samples)
    
    noise_level = QDoubleSpinBox()
    noise_level.setRange(0.01, 0.5)
    noise_level.setValue(0.15)
    noise_level.setSingleStep(0.01)
    noise_level.setToolTip("SmoothGrad的噪声水平，值越大噪声越强")
    params_layout.addRow("SmoothGrad噪声水平:", noise_level)
    
    analysis_layout.addLayout(params_layout)
    
    # 开始分析按钮
    button_layout = QHBoxLayout()
    start_analysis_btn = QPushButton("开始分析")
    start_analysis_btn.setEnabled(False)
    start_analysis_btn.setToolTip("开始执行选定的分析方法")
    button_layout.addWidget(start_analysis_btn)
    
    # 添加停止分析按钮
    stop_analysis_btn = QPushButton("停止分析")
    stop_analysis_btn.setEnabled(False)
    stop_analysis_btn.setToolTip("停止当前正在执行的分析过程")
    button_layout.addWidget(stop_analysis_btn)
    
    # 进度条
    progress_bar = QProgressBar()
    progress_bar.setVisible(False)
    button_layout.addWidget(progress_bar)
    
    analysis_layout.addLayout(button_layout)
    
    analysis_group.setLayout(analysis_layout)
    
    return {
        'group': analysis_group,
        'feature_checkbox': feature_checkbox,
        'gradcam_checkbox': gradcam_checkbox,
        'lime_checkbox': lime_checkbox,
        'sensitivity_checkbox': sensitivity_checkbox,
        'integrated_gradients_checkbox': integrated_gradients_checkbox,
        'shap_checkbox': shap_checkbox,
        'smoothgrad_checkbox': smoothgrad_checkbox,
        'class_combo': class_combo,
        'num_superpixels': num_superpixels,
        'num_samples': num_samples,
        'perturbation_range': perturbation_range,
        'num_steps': num_steps,
        'ig_steps': ig_steps,
        'baseline_type': baseline_type,
        'shap_samples': shap_samples,
        'smoothgrad_samples': smoothgrad_samples,
        'noise_level': noise_level,
        'start_analysis_btn': start_analysis_btn,
        'stop_analysis_btn': stop_analysis_btn,
        'progress_bar': progress_bar
    }


def create_results_section(parent):
    """创建结果显示区域"""
    results_group = QGroupBox("分析结果")
    results_layout = QVBoxLayout()
    
    # 添加工具栏
    toolbar_layout = QHBoxLayout()
    
    # 复制当前分析结果图片按钮
    copy_image_btn = QPushButton("复制当前结果图片")
    copy_image_btn.setToolTip("将当前显示的分析结果图片复制到剪贴板")
    copy_image_btn.setEnabled(False)  # 初始状态禁用
    toolbar_layout.addWidget(copy_image_btn)
    
    # 保存当前分析结果图片按钮
    save_image_btn = QPushButton("保存当前结果图片")
    save_image_btn.setToolTip("将当前显示的分析结果图片保存到文件")
    save_image_btn.setEnabled(False)  # 初始状态禁用
    toolbar_layout.addWidget(save_image_btn)
    
    toolbar_layout.addStretch()  # 添加弹性空间
    results_layout.addLayout(toolbar_layout)
    
    # 创建标签页widget
    results_tabs = QTabWidget()
    results_tabs.setMinimumHeight(500)  # 设置最小高度为500像素
    
    # 特征可视化结果页
    feature_scroll = QScrollArea()
    feature_viewer = ZoomableImageViewer()
    feature_scroll.setWidget(feature_viewer)
    feature_scroll.setWidgetResizable(True)
    feature_scroll.setMinimumHeight(450)  # 设置滚动区域最小高度
    results_tabs.addTab(feature_scroll, "特征可视化")
    
    # GradCAM结果页
    gradcam_scroll = QScrollArea()
    gradcam_viewer = ZoomableImageViewer()
    gradcam_scroll.setWidget(gradcam_viewer)
    gradcam_scroll.setWidgetResizable(True)
    gradcam_scroll.setMinimumHeight(450)
    results_tabs.addTab(gradcam_scroll, "GradCAM")
    
    # LIME结果页
    lime_scroll = QScrollArea()
    lime_viewer = ZoomableImageViewer()
    lime_scroll.setWidget(lime_viewer)
    lime_scroll.setWidgetResizable(True)
    lime_scroll.setMinimumHeight(450)
    results_tabs.addTab(lime_scroll, "LIME解释")
    
    # 敏感性分析结果页
    sensitivity_scroll = QScrollArea()
    sensitivity_viewer = ZoomableImageViewer()
    sensitivity_scroll.setWidget(sensitivity_viewer)
    sensitivity_scroll.setWidgetResizable(True)
    sensitivity_scroll.setMinimumHeight(450)
    results_tabs.addTab(sensitivity_scroll, "敏感性分析")
    
    # Integrated Gradients结果页
    ig_scroll = QScrollArea()
    ig_viewer = ZoomableImageViewer()
    ig_scroll.setWidget(ig_viewer)
    ig_scroll.setWidgetResizable(True)
    ig_scroll.setMinimumHeight(450)
    results_tabs.addTab(ig_scroll, "Integrated Gradients")
    
    # SHAP结果页
    shap_scroll = QScrollArea()
    shap_viewer = ZoomableImageViewer()
    shap_scroll.setWidget(shap_viewer)
    shap_scroll.setWidgetResizable(True)
    shap_scroll.setMinimumHeight(450)
    results_tabs.addTab(shap_scroll, "SHAP解释")
    
    # SmoothGrad结果页
    smoothgrad_scroll = QScrollArea()
    smoothgrad_viewer = ZoomableImageViewer()
    smoothgrad_scroll.setWidget(smoothgrad_viewer)
    smoothgrad_scroll.setWidgetResizable(True)
    smoothgrad_scroll.setMinimumHeight(450)
    results_tabs.addTab(smoothgrad_scroll, "SmoothGrad")
    
    results_layout.addWidget(results_tabs)
    results_group.setLayout(results_layout)
    
    return {
        'group': results_group,
        'results_tabs': results_tabs,
        'feature_viewer': feature_viewer,
        'gradcam_viewer': gradcam_viewer,
        'lime_viewer': lime_viewer,
        'sensitivity_viewer': sensitivity_viewer,
        'ig_viewer': ig_viewer,
        'shap_viewer': shap_viewer,
        'smoothgrad_viewer': smoothgrad_viewer,
        'copy_image_btn': copy_image_btn,
        'save_image_btn': save_image_btn
    } 