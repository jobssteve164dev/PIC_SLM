from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                            QHBoxLayout, QGroupBox, QGridLayout, QTextEdit, QLineEdit,
                            QComboBox, QSplitter, QMessageBox, QSpinBox, QSlider, QCheckBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from .layout_algorithms import LayoutAlgorithms

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class UIComponents:
    """UI组件创建器，专门处理界面元素的创建和布局"""
    
    @staticmethod
    def create_model_selection_group():
        """创建模型选择组"""
        model_group = QGroupBox("模型选择")
        model_layout = QGridLayout()
        
        # 模型路径输入
        model_path_edit = QLineEdit()
        model_path_edit.setReadOnly(True)
        model_path_edit.setPlaceholderText("请选择模型文件")
        
        browse_btn = QPushButton("浏览...")
        
        # 输入尺寸控件
        input_size_label = QLabel("输入尺寸:")
        input_height = QSpinBox()
        input_height.setRange(1, 1024)
        input_height.setValue(224)
        input_width = QSpinBox()
        input_width.setRange(1, 1024)
        input_width.setValue(224)
        input_channels = QSpinBox()
        input_channels.setRange(1, 16)
        input_channels.setValue(3)
        
        size_layout = QHBoxLayout()
        size_layout.addWidget(input_height)
        size_layout.addWidget(QLabel("×"))
        size_layout.addWidget(input_width)
        size_layout.addWidget(QLabel("×"))
        size_layout.addWidget(input_channels)
        
        # 布局
        model_layout.addWidget(QLabel("模型文件:"), 0, 0)
        model_layout.addWidget(model_path_edit, 0, 1)
        model_layout.addWidget(browse_btn, 0, 2)
        model_layout.addWidget(input_size_label, 1, 0)
        model_layout.addLayout(size_layout, 1, 1)
        
        model_group.setLayout(model_layout)
        
        return model_group, model_path_edit, browse_btn, input_height, input_width, input_channels
    
    @staticmethod
    def create_control_buttons():
        """创建控制按钮"""
        control_layout = QHBoxLayout()
        
        visualize_btn = QPushButton("可视化模型结构")
        visualize_btn.setEnabled(False)
        
        fx_visualize_btn = QPushButton("FX可视化结构")
        fx_visualize_btn.setEnabled(False)
        if not HAS_MATPLOTLIB:
            fx_visualize_btn.setToolTip("需要安装matplotlib才能使用此功能")
        
        control_layout.addStretch()
        control_layout.addWidget(visualize_btn)
        control_layout.addWidget(fx_visualize_btn)
        control_layout.addStretch()
        
        return control_layout, visualize_btn, fx_visualize_btn
    
    @staticmethod
    def create_output_text_area():
        """创建输出文本区域"""
        output_group = QGroupBox("模型结构")
        output_layout = QVBoxLayout()
        
        output_text = QTextEdit()
        output_text.setReadOnly(True)
        output_text.setLineWrapMode(QTextEdit.NoWrap)
        output_text.setFont(QFont("Courier New", 10))
        
        output_layout.addWidget(output_text)
        output_group.setLayout(output_layout)
        
        return output_group, output_text
    
    @staticmethod
    def create_fx_control_panel():
        """创建FX可视化控制面板"""
        fx_control_panel = QGroupBox("可视化控制")
        fx_control_layout = QGridLayout()
        
        # 层次深度控制
        depth_label = QLabel("显示深度:")
        depth_slider = QSlider(Qt.Horizontal)
        depth_slider.setMinimum(1)
        depth_slider.setMaximum(10)
        depth_slider.setValue(3)
        depth_slider.setTickPosition(QSlider.TicksBelow)
        depth_slider.setTickInterval(1)
        depth_value = QLabel("3")
        depth_slider.valueChanged.connect(lambda v: depth_value.setText(str(v)))
        
        # 折叠/展开控制
        expand_btn = QPushButton("展开全部")
        collapse_btn = QPushButton("折叠全部")
        
        # 显示选项
        show_params_cb = QCheckBox("显示参数信息")
        show_params_cb.setChecked(True)
        
        show_types_cb = QCheckBox("显示层类型")
        show_types_cb.setChecked(True)
        
        # 布局选择
        layout_label = QLabel("布局方式:")
        layout_combo = QComboBox()
        layout_combo.addItems(LayoutAlgorithms.get_layout_names())
        layout_combo.setCurrentIndex(0)
        
        # 应用布局
        fx_control_layout.addWidget(depth_label, 0, 0)
        fx_control_layout.addWidget(depth_slider, 0, 1)
        fx_control_layout.addWidget(depth_value, 0, 2)
        fx_control_layout.addWidget(expand_btn, 1, 0)
        fx_control_layout.addWidget(collapse_btn, 1, 1)
        fx_control_layout.addWidget(show_params_cb, 2, 0)
        fx_control_layout.addWidget(show_types_cb, 2, 1)
        fx_control_layout.addWidget(layout_label, 3, 0)
        fx_control_layout.addWidget(layout_combo, 3, 1, 1, 2)
        
        fx_control_panel.setLayout(fx_control_layout)
        
        return (fx_control_panel, depth_slider, depth_value, expand_btn, collapse_btn,
                show_params_cb, show_types_cb, layout_combo)
    
    @staticmethod
    def create_graph_container():
        """创建图表显示容器"""
        graph_container = QWidget()
        graph_layout = QVBoxLayout(graph_container)
        
        # 图表区域
        figure_container = QWidget()
        figure_layout = QVBoxLayout(figure_container)
        
        graph_container.setVisible(False)
        
        return graph_container, graph_layout, figure_container, figure_layout
    
    @staticmethod
    def setup_main_layout(parent, model_group, control_layout, output_group, graph_container):
        """设置主布局"""
        main_layout = QVBoxLayout(parent)
        main_layout.addWidget(model_group)
        main_layout.addLayout(control_layout)
        main_layout.addWidget(output_group)
        main_layout.addWidget(graph_container)
        return main_layout 