from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                            QHBoxLayout, QGroupBox, QGridLayout, QTextEdit, QLineEdit,
                            QComboBox, QSplitter, QMessageBox, QSpinBox, QSlider, QCheckBox, QToolBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import torch
import torch.nn as nn
import torchvision.models as models
import os
import json
from torchsummary import summary
import io
from contextlib import redirect_stdout
import sys
import traceback
import torch.fx as fx
import networkx as nx
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

class ModelStructureViewer(QWidget):
    """模型结构可视化组件，使用torchsummary显示PyTorch模型结构"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.model_path = None
        self.class_names = []
        # 添加FX可视化相关变量
        self.graph = None
        self.positions = None
        self.layer_types = {}  # 保存层类型
        self.current_depth = 3 # 默认展示深度
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        
        # 模型选择组
        model_group = QGroupBox("模型选择")
        model_layout = QGridLayout()
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText("请选择模型文件")
        
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_model)
        
        self.input_size_label = QLabel("输入尺寸:")
        self.input_height = QSpinBox()
        self.input_height.setRange(1, 1024)
        self.input_height.setValue(224)
        self.input_width = QSpinBox()
        self.input_width.setRange(1, 1024)
        self.input_width.setValue(224)
        self.input_channels = QSpinBox()
        self.input_channels.setRange(1, 16)
        self.input_channels.setValue(3)
        
        size_layout = QHBoxLayout()
        size_layout.addWidget(self.input_height)
        size_layout.addWidget(QLabel("×"))
        size_layout.addWidget(self.input_width)
        size_layout.addWidget(QLabel("×"))
        size_layout.addWidget(self.input_channels)
        
        model_layout.addWidget(QLabel("模型文件:"), 0, 0)
        model_layout.addWidget(self.model_path_edit, 0, 1)
        model_layout.addWidget(browse_btn, 0, 2)
        model_layout.addWidget(self.input_size_label, 1, 0)
        model_layout.addLayout(size_layout, 1, 1)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # 控制按钮
        control_layout = QHBoxLayout()
        
        self.visualize_btn = QPushButton("可视化模型结构")
        self.visualize_btn.clicked.connect(self.visualize_model_structure)
        self.visualize_btn.setEnabled(False)
        
        self.fx_visualize_btn = QPushButton("FX可视化结构")
        self.fx_visualize_btn.clicked.connect(self.visualize_model_fx)
        self.fx_visualize_btn.setEnabled(False)
        if not HAS_MATPLOTLIB:
            self.fx_visualize_btn.setToolTip("需要安装matplotlib才能使用此功能")
        
        control_layout.addStretch()
        control_layout.addWidget(self.visualize_btn)
        control_layout.addWidget(self.fx_visualize_btn)
        control_layout.addStretch()
        
        main_layout.addLayout(control_layout)
        
        # 输出文本区域
        self.output_group = QGroupBox("模型结构")
        output_layout = QVBoxLayout()
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setLineWrapMode(QTextEdit.NoWrap)
        self.output_text.setFont(QFont("Courier New", 10))
        
        output_layout.addWidget(self.output_text)
        self.output_group.setLayout(output_layout)
        
        main_layout.addWidget(self.output_group)
        
        # 创建图表显示区域
        self.graph_container = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_container)
        
        # 添加FX可视化控制面板
        self.fx_control_panel = QGroupBox("可视化控制")
        fx_control_layout = QGridLayout()
        
        # 层次深度控制
        self.depth_label = QLabel("显示深度:")
        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setMinimum(1)
        self.depth_slider.setMaximum(10)
        self.depth_slider.setValue(3)
        self.depth_slider.setTickPosition(QSlider.TicksBelow)
        self.depth_slider.setTickInterval(1)
        self.depth_slider.valueChanged.connect(self.update_fx_visualization)
        self.depth_value = QLabel("3")
        self.depth_slider.valueChanged.connect(lambda v: self.depth_value.setText(str(v)))
        
        # 折叠/展开控制
        self.expand_btn = QPushButton("展开全部")
        self.expand_btn.clicked.connect(self.expand_all_layers)
        self.collapse_btn = QPushButton("折叠全部")
        self.collapse_btn.clicked.connect(self.collapse_all_layers)
        
        # 显示选项
        self.show_params_cb = QCheckBox("显示参数信息")
        self.show_params_cb.setChecked(True)
        self.show_params_cb.stateChanged.connect(self.update_fx_visualization)
        
        self.show_types_cb = QCheckBox("显示层类型")
        self.show_types_cb.setChecked(True)
        self.show_types_cb.stateChanged.connect(self.update_fx_visualization)
        
        # 布局选择
        self.layout_label = QLabel("布局方式:")
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["分层布局", "树形布局", "放射布局", "圆形布局", "随机布局"])
        self.layout_combo.setCurrentIndex(0)
        self.layout_combo.currentIndexChanged.connect(self.update_fx_visualization)
        
        # 应用布局
        fx_control_layout.addWidget(self.depth_label, 0, 0)
        fx_control_layout.addWidget(self.depth_slider, 0, 1)
        fx_control_layout.addWidget(self.depth_value, 0, 2)
        fx_control_layout.addWidget(self.expand_btn, 1, 0)
        fx_control_layout.addWidget(self.collapse_btn, 1, 1)
        fx_control_layout.addWidget(self.show_params_cb, 2, 0)
        fx_control_layout.addWidget(self.show_types_cb, 2, 1)
        fx_control_layout.addWidget(self.layout_label, 3, 0)
        fx_control_layout.addWidget(self.layout_combo, 3, 1, 1, 2)
        
        self.fx_control_panel.setLayout(fx_control_layout)
        self.graph_layout.addWidget(self.fx_control_panel)
        
        # 图表区域
        self.figure_canvas = None
        self.figure_container = QWidget()
        self.figure_layout = QVBoxLayout(self.figure_container)
        self.graph_layout.addWidget(self.figure_container)
        
        self.graph_container.setVisible(False)
        main_layout.addWidget(self.graph_container)
    
    def browse_model(self):
        """浏览并选择模型文件"""
        file_dialog = QFileDialog()
        model_path, _ = file_dialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch模型 (*.pth *.pt);;所有文件 (*.*)"
        )
        
        if model_path:
            self.model_path = model_path
            self.model_path_edit.setText(model_path)
            
            # 尝试寻找同目录下的类别信息文件
            model_dir = os.path.dirname(model_path)
            class_info_file = os.path.join(model_dir, "class_info.json")
            
            if os.path.exists(class_info_file):
                try:
                    with open(class_info_file, 'r', encoding='utf-8') as f:
                        class_info = json.load(f)
                        self.class_names = class_info.get('class_names', [])
                except Exception as e:
                    print(f"加载类别信息出错: {str(e)}")
            
            self.visualize_btn.setEnabled(True)
            self.fx_visualize_btn.setEnabled(HAS_MATPLOTLIB)
    
    def _create_densenet_model(self, model_type, num_classes):
        """创建DenseNet模型，兼容新旧PyTorch API"""
        try:
            # 尝试使用新API
            print(f"尝试使用新API加载{model_type}模型...")
            if model_type == "densenet121":
                model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            elif model_type == "densenet169":
                model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
            elif model_type == "densenet201":
                model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
            else:
                raise ValueError(f"不支持的DenseNet模型类型: {model_type}")
                
            # 修改分类器头
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            print(f"使用新API成功加载{model_type}模型")
            return model
        except Exception as e:
            # 新API失败，尝试使用旧API
            print(f"使用新API加载{model_type}模型失败: {str(e)}")
            try:
                print(f"尝试使用旧API加载{model_type}模型...")
                if model_type == "densenet121":
                    model = models.densenet121(pretrained=False)
                elif model_type == "densenet169":
                    model = models.densenet169(pretrained=False)
                elif model_type == "densenet201":
                    model = models.densenet201(pretrained=False)
                else:
                    raise ValueError(f"不支持的DenseNet模型类型: {model_type}")
                    
                # 修改分类器头
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                print(f"使用旧API成功加载{model_type}模型")
                return model
            except Exception as e2:
                # 再次失败，尝试直接从torchvision导入
                print(f"使用旧API加载{model_type}模型失败: {str(e2)}")
                try:
                    print(f"尝试直接从torchvision.models导入{model_type}...")
                    if model_type == "densenet121":
                        from torchvision.models import densenet121
                        model = densenet121(pretrained=False)
                    elif model_type == "densenet169":
                        from torchvision.models import densenet169
                        model = densenet169(pretrained=False)
                    elif model_type == "densenet201":
                        from torchvision.models import densenet201
                        model = densenet201(pretrained=False)
                    
                    # 使用安全的方式设置分类器
                    if hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
                        in_features = model.classifier.in_features
                        model.classifier = nn.Linear(in_features, num_classes)
                    else:
                        print(f"警告: 无法确定{model_type}的分类器结构，使用默认分类器")
                        
                    print(f"通过直接导入成功加载{model_type}模型")
                    return model
                except Exception as e3:
                    # 所有方法均失败
                    error_msg = f"所有方法加载{model_type}模型均失败:\n"
                    error_msg += f"新API错误: {str(e)}\n"
                    error_msg += f"旧API错误: {str(e2)}\n"
                    error_msg += f"直接导入错误: {str(e3)}"
                    print(error_msg)
                    raise ValueError(error_msg)
    
    def load_model(self):
        """加载模型"""
        try:
            if not os.path.exists(self.model_path):
                QMessageBox.warning(self, "错误", "模型文件不存在")
                return None
            
            # 获取模型类型
            model_name = os.path.basename(self.model_path).lower()
            
            # 创建模型实例 - 首先尝试从文件名猜测模型类型
            num_classes = len(self.class_names) if self.class_names else 1000
            
            if "resnet18" in model_name:
                model = models.resnet18(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif "resnet34" in model_name:
                model = models.resnet34(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif "resnet50" in model_name:
                model = models.resnet50(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif "resnet101" in model_name:
                model = models.resnet101(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif "resnet152" in model_name:
                model = models.resnet152(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif "mobilenetv2" in model_name or "mobilenet_v2" in model_name:
                model = models.mobilenet_v2(pretrained=False)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            elif "mobilenetv3" in model_name or "mobilenet_v3" in model_name:
                model = models.mobilenet_v3_large(pretrained=False)
                model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
            elif "vgg16" in model_name:
                model = models.vgg16(pretrained=False)
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            elif "vgg19" in model_name:
                model = models.vgg19(pretrained=False)
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            elif "densenet121" in model_name:
                model = self._create_densenet_model("densenet121", num_classes)
            elif "densenet169" in model_name:
                model = self._create_densenet_model("densenet169", num_classes)
            elif "densenet201" in model_name:
                model = self._create_densenet_model("densenet201", num_classes)
            elif "efficientnet" in model_name:
                # 尝试使用EfficientNet
                try:
                    from efficientnet_pytorch import EfficientNet
                    if "b0" in model_name:
                        model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
                    elif "b1" in model_name:
                        model = EfficientNet.from_name('efficientnet-b1', num_classes=num_classes)
                    elif "b2" in model_name:
                        model = EfficientNet.from_name('efficientnet-b2', num_classes=num_classes)
                    elif "b3" in model_name:
                        model = EfficientNet.from_name('efficientnet-b3', num_classes=num_classes)
                    elif "b4" in model_name:
                        model = EfficientNet.from_name('efficientnet-b4', num_classes=num_classes)
                    else:
                        model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
                except ImportError:
                    # 如果没有安装EfficientNet库，使用ResNet50替代
                    model = models.resnet50(pretrained=False)
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
            else:
                # 如果无法从文件名判断，默认使用ResNet50
                model = models.resnet50(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            # 加载模型权重
            if torch.cuda.is_available():
                state_dict = torch.load(self.model_path)
            else:
                state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
            
            # 尝试直接加载，可能会因为键名不匹配而失败
            try:
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"直接加载模型权重失败：{str(e)}")
                # 可能是DataParallel保存的模型，尝试移除 'module.' 前缀
                try:
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k.replace('module.', '')
                        new_state_dict[name] = v
                    model.load_state_dict(new_state_dict)
                except Exception as e2:
                    print(f"尝试调整模型权重键名后仍然失败：{str(e2)}")
                    # 加载失败可能是因为模型结构不匹配，直接返回创建的模型而不加载权重
                    QMessageBox.warning(self, "警告", f"无法加载模型权重，只显示模型结构: {str(e2)}")
            
            return model
        except Exception as e:
            error_msg = f"加载模型失败: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            QMessageBox.critical(self, "错误", error_msg)
            return None
    
    def visualize_model_structure(self):
        """可视化模型结构"""
        if not self.model_path:
            QMessageBox.warning(self, "警告", "请先选择模型文件")
            return
        
        # 加载模型
        model = self.load_model()
        if model is None:
            return
        
        # 隐藏图表区域，显示文本区域
        self.graph_container.setVisible(False)
        self.output_group.setVisible(True)
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 获取输入尺寸
        height = self.input_height.value()
        width = self.input_width.value()
        channels = self.input_channels.value()
        
        # 检查是否为DenseNet模型
        is_densenet = any(isinstance(model, getattr(models.densenet, name)) 
                          for name in dir(models.densenet) 
                          if 'DenseNet' in name and isinstance(getattr(models.densenet, name), type))
        
        # 对于DenseNet模型，使用替代方法显示结构
        if is_densenet:
            try:
                # 使用字符串形式的摘要替代torchsummary
                output = f"模型类型: {model.__class__.__name__}\n\n"
                output += f"特征提取层 (features):\n"
                for name, module in model.features.named_children():
                    output += f"  {name}: {module}\n"
                
                output += f"\n分类器 (classifier):\n  {model.classifier}\n"
                
                # 计算参数数量
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                output += f"\n\n总参数数量: {total_params:,}\n"
                output += f"可训练参数数量: {trainable_params:,}\n"
                
                # 显示结构图
                self.output_text.setText(output)
                
                # 添加DenseNet特有信息
                additional_info = (
                    f"\n\n注意: DenseNet模型结构复杂，无法使用标准方法可视化。\n"
                    f"DenseNet模型的主要组成部分:\n"
                    f"1. 卷积层 (Conv2d)\n"
                    f"2. BatchNorm层\n"
                    f"3. 密集连接块 (DenseBlock)\n"
                    f"4. 转换层 (Transition)\n"
                    f"5. 全局池化层\n"
                    f"6. 分类器 (Linear)\n\n"
                    f"输入尺寸: ({channels}, {height}, {width})"
                )
                self.output_text.append(additional_info)
                
            except Exception as e:
                error_msg = f"使用替代方法可视化DenseNet模型结构失败: {str(e)}"
                print(error_msg)
                print(traceback.format_exc())
                QMessageBox.critical(self, "错误", error_msg)
        else:
            # 对于非DenseNet模型，使用原始的torchsummary
            try:
                # 重定向stdout以捕获summary输出
                string_io = io.StringIO()
                with redirect_stdout(string_io):
                    summary(model, input_size=(channels, height, width), device=str(device))
                
                # 获取捕获的输出并显示在文本区域
                output = string_io.getvalue()
                self.output_text.setText(output)
                
                # 添加总参数数量和可训练参数数量信息
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                additional_info = f"\n\n总参数数量：{total_params:,}\n可训练参数数量：{trainable_params:,}"
                self.output_text.append(additional_info)
                
            except Exception as e:
                error_msg = f"可视化模型结构失败: {str(e)}"
                print(error_msg)
                print(traceback.format_exc())
                QMessageBox.critical(self, "错误", error_msg)
    
    def get_layer_color(self, layer_type):
        """根据层类型返回颜色"""
        color_map = {
            'Conv2d': '#FF9999',          # 红色
            'Linear': '#99CCFF',          # 蓝色
            'BatchNorm': '#FFCC99',       # 橙色
            'ReLU': '#99FF99',            # 绿色
            'Dropout': '#CCCCCC',         # 灰色
            'MaxPool': '#CC99FF',         # 紫色
            'AvgPool': '#FFFF99',         # 黄色
            'Flatten': '#FF99FF',         # 粉色
            'DenseBlock': '#66CCCC',      # 青色
            'Transition': '#CCFF99',      # 浅绿色
            'AdaptiveAvgPool': '#FFCC99', # 橙色
            'placeholder': '#F0F0F0',     # 占位符(输入/输出)为浅灰色
            'getattr': '#E0E0E0',         # getattr操作为灰色
            'call_method': '#D0D0D0',     # 调用方法为深灰色
            'output': '#B0E0E6'           # 输出为淡蓝色
        }
        
        # 检查是否包含已知类型的层
        for known_type, color in color_map.items():
            if known_type.lower() in layer_type.lower():
                return color
        
        # 默认颜色 - 浅灰色
        return '#F5F5F5'
    
    def get_node_size(self, node_type):
        """根据节点类型返回大小"""
        if 'conv' in node_type.lower() or 'linear' in node_type.lower():
            return 1500  # 重要层节点大一些
        elif 'input' in node_type.lower() or 'output' in node_type.lower():
            return 1800  # 输入输出节点最大
        else:
            return 1200  # 默认大小
    
    def custom_tree_layout(self, G, root=None):
        """自定义树形布局算法，不依赖pygraphviz"""
        if root is None:
            # 尝试找到根节点（入度为0的节点）
            roots = [n for n in G.nodes() if G.in_degree(n) == 0]
            if not roots:
                # 如果没有入度为0的节点，则选择第一个节点作为根
                root = list(G.nodes())[0]
            else:
                root = roots[0]
                
        pos = {}  # 节点位置字典
        visited = set([root])
        current_level = [root]
        level_count = 0
        nodes_per_level = {}  # 记录每层的节点数
        
        # 使用BFS遍历图以分配节点到层次
        while current_level:
            next_level = []
            for node in current_level:
                # 检查是否有子节点
                children = [n for n in G.successors(node) if n not in visited]
                for child in children:
                    visited.add(child)
                    next_level.append(child)
            
            # 更新每层的节点数
            nodes_per_level[level_count] = len(current_level)
            level_count += 1
            current_level = next_level
            
        # 计算总层数
        total_levels = level_count
        
        # 重置访问状态
        visited = set([root])
        current_level = [root]
        level_count = 0
        
        # 再次使用BFS遍历并分配坐标
        while current_level:
            width = 1.0
            y_coord = 1.0 - (level_count / (total_levels if total_levels > 0 else 1))
            
            for i, node in enumerate(current_level):
                # 计算x坐标
                if nodes_per_level[level_count] > 1:
                    x_coord = (i / (nodes_per_level[level_count] - 1)) * width
                else:
                    x_coord = 0.5 * width
                
                # 分配节点位置
                pos[node] = (x_coord, y_coord)
                
            next_level = []
            for node in current_level:
                # 检查是否有子节点
                children = [n for n in G.successors(node) if n not in visited]
                for child in children:
                    visited.add(child)
                    next_level.append(child)
            
            level_count += 1
            current_level = next_level
            
        # 检查是否有未访问的节点（未连接到主树）
        unvisited = set(G.nodes()) - visited
        if unvisited:
            # 将未访问的节点放在底部
            y_coord = -0.1
            for i, node in enumerate(unvisited):
                x_coord = (i / (len(unvisited) if len(unvisited) > 1 else 1)) * width
                pos[node] = (x_coord, y_coord)
                
        return pos

    def update_fx_visualization(self):
        """根据当前设置更新FX可视化"""
        if self.graph is None:
            return
            
        # 清除现有图表
        if self.figure_canvas:
            self.figure_layout.removeWidget(self.figure_canvas)
            if hasattr(self, 'toolbar') and self.toolbar:
                self.figure_layout.removeWidget(self.toolbar)
                self.toolbar = None
            self.figure_canvas.deleteLater()
            self.figure_canvas = None
        
        # 获取当前深度
        self.current_depth = self.depth_slider.value()
        
        try:
            # 创建新图表
            fig = Figure(figsize=(12, 10))
            ax = fig.add_subplot(111)
            
            # 根据当前布局选择布局算法
            layout_idx = self.layout_combo.currentIndex()
            G = nx.DiGraph()
            
            # 提取层类型和深度信息
            display_nodes = []
            for node, attrs in self.graph.nodes(data=True):
                # 检查节点深度是否小于或等于当前设置的显示深度
                if attrs.get('depth', 0) <= self.current_depth:
                    display_nodes.append(node)
                    
            # 创建子图
            subgraph = self.graph.subgraph(display_nodes)
            
            # 使用选定的布局算法
            if layout_idx == 0:  # 分层布局
                pos = nx.multipartite_layout(subgraph, subset_key='depth')
            elif layout_idx == 1:  # 树形布局
                pos = self.custom_tree_layout(subgraph)
            elif layout_idx == 2:  # 放射布局
                pos = nx.kamada_kawai_layout(subgraph)
            elif layout_idx == 3:  # 圆形布局
                pos = nx.circular_layout(subgraph)
            else:  # 随机布局
                pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
                
            # 获取节点大小和颜色
            node_colors = []
            node_sizes = []
            labels = {}
            
            for node in subgraph.nodes():
                node_type = subgraph.nodes[node].get('type', '')
                node_colors.append(self.get_layer_color(node_type))
                node_sizes.append(self.get_node_size(node_type))
                
                # 设置标签
                if self.show_types_cb.isChecked() and self.show_params_cb.isChecked():
                    # 同时显示类型和参数
                    params = subgraph.nodes[node].get('params', '')
                    if params:
                        labels[node] = f"{node}\n({node_type})\n{params}"
                    else:
                        labels[node] = f"{node}\n({node_type})"
                elif self.show_types_cb.isChecked():
                    # 仅显示类型
                    labels[node] = f"{node}\n({node_type})"
                elif self.show_params_cb.isChecked():
                    # 仅显示参数
                    params = subgraph.nodes[node].get('params', '')
                    if params:
                        labels[node] = f"{node}\n{params}"
                    else:
                        labels[node] = node
                else:
                    # 仅显示名称
                    labels[node] = node
            
            # 绘制图形
            nx.draw_networkx(
                subgraph, 
                pos=pos,
                with_labels=True,
                node_color=node_colors,
                node_size=node_sizes,
                labels=labels,
                font_size=9,
                font_weight='bold',
                arrows=True,
                arrowsize=15,
                ax=ax
            )
            
            # 添加图例
            legend_items = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9999', markersize=10, label='卷积层'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#99CCFF', markersize=10, label='全连接层'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFCC99', markersize=10, label='BatchNorm/归一化'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#99FF99', markersize=10, label='激活函数'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#CC99FF', markersize=10, label='池化层'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#66CCCC', markersize=10, label='DenseBlock'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F0F0F0', markersize=10, label='输入/占位符'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#B0E0E6', markersize=10, label='输出'),
            ]
            ax.legend(handles=legend_items, loc='upper right', bbox_to_anchor=(1.0, 1.0))
            
            # 设置标题
            if hasattr(self, 'model') and self.model:
                model_name = self.model.__class__.__name__
                ax.set_title(f"模型: {model_name} (深度: {self.current_depth})", fontsize=14)
            
            ax.set_axis_off()
            
            # 显示图表
            self.figure_canvas = FigureCanvas(fig)
            self.figure_layout.addWidget(self.figure_canvas)
            
            # 添加工具栏
            self.toolbar = NavigationToolbar(self.figure_canvas, self)
            self.figure_layout.addWidget(self.toolbar)
            
        except Exception as e:
            error_msg = f"更新FX可视化失败: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            QMessageBox.critical(self, "错误", error_msg)
    
    def expand_all_layers(self):
        """展开所有层"""
        max_depth = 0
        for _, attrs in self.graph.nodes(data=True):
            depth = attrs.get('depth', 0)
            if depth > max_depth:
                max_depth = depth
                
        self.depth_slider.setValue(max_depth)
    
    def collapse_all_layers(self):
        """折叠所有层"""
        self.depth_slider.setValue(1)
    
    def create_hierarchical_graph(self, model):
        """创建带层次结构的图"""
        G = nx.DiGraph()
        
        # 添加输入节点
        G.add_node("输入", type="placeholder", depth=0)
        
        # 获取模型名称
        model_name = model.__class__.__name__
        
        # 添加模型根节点
        G.add_node(model_name, type=model_name, depth=1)
        G.add_edge("输入", model_name)
        
        # 递归添加模块
        def add_module_to_graph(module, parent_name, depth):
            for name, layer in module.named_children():
                # 创建节点名称和类型
                node_name = f"{parent_name}.{name}" if parent_name else name
                layer_type = layer.__class__.__name__
                
                # 获取参数信息
                params_info = ""
                if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
                    params_info = f"{layer.in_features}→{layer.out_features}"
                elif hasattr(layer, "in_channels") and hasattr(layer, "out_channels"):
                    params_info = f"{layer.in_channels}→{layer.out_channels}"
                    if hasattr(layer, "kernel_size"):
                        kernel = layer.kernel_size
                        if isinstance(kernel, tuple):
                            kernel = 'x'.join(map(str, kernel))
                        params_info += f", k={kernel}"
                
                # 添加节点和边
                G.add_node(node_name, type=layer_type, depth=depth, params=params_info)
                G.add_edge(parent_name, node_name)
                
                # 记录层类型
                self.layer_types[node_name] = layer_type
                
                # 递归处理子模块
                if list(layer.named_children()):
                    add_module_to_graph(layer, node_name, depth+1)
                elif depth == 2:  # 如果是叶子节点且深度较浅，增加一个虚拟节点防止图太扁平
                    virtual_node = f"{node_name}.output"
                    G.add_node(virtual_node, type="output", depth=depth+1)
                    G.add_edge(node_name, virtual_node)
        
        # 从模型开始添加
        add_module_to_graph(model, model_name, 2)
        
        # 返回构建的图
        return G
    
    def visualize_model_fx(self):
        """使用FX可视化模型结构"""
        if not HAS_MATPLOTLIB:
            QMessageBox.warning(self, "警告", "需要安装matplotlib才能使用FX可视化功能")
            return
            
        if not self.model_path:
            QMessageBox.warning(self, "警告", "请先选择模型文件")
            return
        
        # 加载模型
        model = self.load_model()
        if model is None:
            return
        
        # 保存模型引用
        self.model = model
            
        try:
            # 清除之前的图表
            if self.figure_canvas:
                self.figure_layout.removeWidget(self.figure_canvas)
                if hasattr(self, 'toolbar') and self.toolbar:
                    self.figure_layout.removeWidget(self.toolbar)
                    self.toolbar = None
                self.figure_canvas.deleteLater()
                self.figure_canvas = None
                
            # 显示图表区域，隐藏文本区域
            self.output_group.setVisible(False)
            self.graph_container.setVisible(True)
            
            # 重置层类型字典
            self.layer_types = {}
            
            # 尝试使用FX符号跟踪
            try:
                model.eval()  # 设为评估模式
                traced = fx.symbolic_trace(model)
                
                # 输出FX跟踪结果到文本区域
                dot_graph = traced.graph.print_tabular()
                self.output_text.setText(f"FX符号跟踪结果:\n\n{dot_graph}")
                
                # 创建有向图
                G = nx.DiGraph()
                
                # 添加节点和边 - 带深度信息
                depth_map = {}  # 用于跟踪节点深度
                
                # 先添加所有节点
                for i, node in enumerate(traced.graph.nodes):
                    # 设置节点类型
                    node_type = node.op
                    if node.op == 'call_module':
                        target_type = str(type(traced.get_submodule(node.target)).__name__)
                        node_type = f"{node.op} - {target_type}"
                    elif node.op == 'call_function' or node.op == 'call_method':
                        node_type = f"{node.op} - {node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)}"
                        
                    # 计算参数信息
                    params_info = ""
                    if node.op == 'call_module':
                        module = traced.get_submodule(node.target)
                        if hasattr(module, "in_features") and hasattr(module, "out_features"):
                            params_info = f"{module.in_features}→{module.out_features}"
                        elif hasattr(module, "in_channels") and hasattr(module, "out_channels"):
                            params_info = f"{module.in_channels}→{module.out_channels}"
                            if hasattr(module, "kernel_size"):
                                kernel = module.kernel_size
                                if isinstance(kernel, tuple):
                                    kernel = 'x'.join(map(str, kernel))
                                params_info += f", k={kernel}"
                        
                    # 添加节点
                    G.add_node(node.name, type=node_type, depth=0, params=params_info)  # 初始深度为0
                    
                # 添加边并计算深度
                for node in traced.graph.nodes:
                    for input_node in node.all_input_nodes:
                        G.add_edge(input_node.name, node.name)
                
                # 计算深度 - 从源节点开始的最长路径
                for node in nx.topological_sort(G):
                    if not list(G.predecessors(node)):  # 如果是源节点
                        depth_map[node] = 1
                    else:
                        # 节点深度是所有前置节点的最大深度+1
                        max_pred_depth = max([depth_map.get(pred, 0) for pred in G.predecessors(node)])
                        depth_map[node] = max_pred_depth + 1
                
                # 更新节点深度
                for node, depth in depth_map.items():
                    G.nodes[node]['depth'] = depth
                    
                # 设置图的最大深度
                max_depth = max(depth_map.values())
                self.depth_slider.setMaximum(max_depth)
                
                # 保存图对象
                self.graph = G
                
                # 更新图形显示
                self.update_fx_visualization()
                
            except Exception as e:
                # 如果符号跟踪失败，使用分层结构代替
                print(f"FX符号跟踪失败: {str(e)}")
                self.output_text.setText(f"FX符号跟踪失败: {str(e)}\n\n使用模型层结构替代可视化:\n")
                
                # 创建分层结构图
                self.graph = self.create_hierarchical_graph(model)
                
                # 获取最大深度
                max_depth = max([attrs.get('depth', 0) for _, attrs in self.graph.nodes(data=True)])
                self.depth_slider.setMaximum(max_depth)
                
                # 更新图形
                self.update_fx_visualization()
                
                # 添加文本描述
                layers_text = "模型层次结构:\n"
                def format_module_info(module, prefix=""):
                    nonlocal layers_text
                    for name, layer in module.named_children():
                        layer_str = f"{prefix}└─ {name}: {layer.__class__.__name__}"
                        if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
                            layer_str += f" ({layer.in_features} → {layer.out_features})"
                        elif hasattr(layer, "in_channels") and hasattr(layer, "out_channels"):
                            layer_str += f" ({layer.in_channels} → {layer.out_channels})"
                        layers_text += layer_str + "\n"
                        
                        if list(layer.named_children()):
                            format_module_info(layer, prefix + "  ")
                
                format_module_info(model)
                self.output_text.append(layers_text)
                
        except Exception as e:
            error_msg = f"FX可视化过程中出错: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            QMessageBox.critical(self, "错误", error_msg)
            self.graph_container.setVisible(False)
    
    def set_model(self, model, class_names=None):
        """从外部设置模型"""
        if model is not None:
            self.model = model
            if class_names:
                self.class_names = class_names
            self.visualize_btn.setEnabled(True)
            self.fx_visualize_btn.setEnabled(HAS_MATPLOTLIB)
            
            # 隐藏图表区域，显示文本区域
            self.graph_container.setVisible(False)
            self.output_group.setVisible(True)
            
            # 自动触发可视化
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            # 获取输入尺寸
            height = self.input_height.value()
            width = self.input_width.value()
            channels = self.input_channels.value()
            
            # 检查是否为DenseNet模型
            is_densenet = any(isinstance(model, getattr(models.densenet, name)) 
                             for name in dir(models.densenet) 
                             if 'DenseNet' in name and isinstance(getattr(models.densenet, name), type))
            
            # 对于DenseNet模型，使用替代方法显示结构
            if is_densenet:
                try:
                    # 使用字符串形式的摘要替代torchsummary
                    output = f"模型类型: {model.__class__.__name__}\n\n"
                    output += f"特征提取层 (features):\n"
                    for name, module in model.features.named_children():
                        output += f"  {name}: {module}\n"
                    
                    output += f"\n分类器 (classifier):\n  {model.classifier}\n"
                    
                    # 计算参数数量
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    
                    output += f"\n\n总参数数量: {total_params:,}\n"
                    output += f"可训练参数数量: {trainable_params:,}\n"
                    
                    # 显示结构图
                    self.output_text.setText(output)
                    
                    # 添加DenseNet特有信息
                    additional_info = (
                        f"\n\n注意: DenseNet模型结构复杂，无法使用标准方法可视化。\n"
                        f"DenseNet模型的主要组成部分:\n"
                        f"1. 卷积层 (Conv2d)\n"
                        f"2. BatchNorm层\n"
                        f"3. 密集连接块 (DenseBlock)\n"
                        f"4. 转换层 (Transition)\n"
                        f"5. 全局池化层\n"
                        f"6. 分类器 (Linear)\n\n"
                        f"输入尺寸: ({channels}, {height}, {width})"
                    )
                    self.output_text.append(additional_info)
                    
                except Exception as e:
                    error_msg = f"使用替代方法可视化DenseNet模型结构失败: {str(e)}"
                    print(error_msg)
                    print(traceback.format_exc())
                    QMessageBox.critical(self, "错误", error_msg)
            else:
                # 对于非DenseNet模型，使用原始的torchsummary
                try:
                    # 重定向stdout以捕获summary输出
                    string_io = io.StringIO()
                    with redirect_stdout(string_io):
                        summary(model, input_size=(channels, height, width), device=str(device))
                    
                    # 获取捕获的输出并显示在文本区域
                    output = string_io.getvalue()
                    self.output_text.setText(output)
                    
                    # 添加总参数数量和可训练参数数量信息
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    
                    additional_info = f"\n\n总参数数量：{total_params:,}\n可训练参数数量：{trainable_params:,}"
                    self.output_text.append(additional_info)
                    
                except Exception as e:
                    error_msg = f"可视化模型结构失败: {str(e)}"
                    print(error_msg)
                    print(traceback.format_exc())
                    QMessageBox.critical(self, "错误", error_msg) 