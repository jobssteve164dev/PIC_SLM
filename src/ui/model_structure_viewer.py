from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt
import os
import traceback
from .components.model_structure_viewer.model_loader import ModelLoader
from .components.model_structure_viewer.visualization_controller import VisualizationController
from .components.model_structure_viewer.ui_components import UIComponents

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ModelStructureViewer(QWidget):
    """模型结构可视化组件，重构版本 - 提高可维护性"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 初始化各个子模块
        self.model_loader = ModelLoader()
        self.visualization_controller = VisualizationController()
        
        # UI状态
        self.figure_canvas = None
        self.toolbar = None
        self.current_depth = 3
        
        self.init_ui()
        self.connect_signals()
    
    def init_ui(self):
        """初始化UI"""
        # 创建UI组件
        (self.model_group, self.model_path_edit, self.browse_btn, 
         self.input_height, self.input_width, self.input_channels) = UIComponents.create_model_selection_group()
        
        (self.control_layout, self.visualize_btn, 
         self.fx_visualize_btn) = UIComponents.create_control_buttons()
        
        self.output_group, self.output_text = UIComponents.create_output_text_area()
        
        # 创建FX控制面板
        (self.fx_control_panel, self.depth_slider, self.depth_value, self.expand_btn,
         self.collapse_btn, self.show_params_cb, self.show_types_cb, 
         self.layout_combo) = UIComponents.create_fx_control_panel()
        
        # 创建图表容器
        (self.graph_container, self.graph_layout, self.figure_container, 
         self.figure_layout) = UIComponents.create_graph_container()
        
        # 将控制面板添加到图表布局
        self.graph_layout.addWidget(self.fx_control_panel)
        self.graph_layout.addWidget(self.figure_container)
        
        # 设置主布局
        UIComponents.setup_main_layout(
            self, self.model_group, self.control_layout, 
            self.output_group, self.graph_container)
    
    def connect_signals(self):
        """连接信号"""
        # 基本控制信号
        self.browse_btn.clicked.connect(self.browse_model)
        self.visualize_btn.clicked.connect(self.visualize_model_structure)
        self.fx_visualize_btn.clicked.connect(self.visualize_model_fx)
        
        # FX可视化控制信号
        self.depth_slider.valueChanged.connect(self.update_fx_visualization)
        self.expand_btn.clicked.connect(self.expand_all_layers)
        self.collapse_btn.clicked.connect(self.collapse_all_layers)
        self.show_params_cb.stateChanged.connect(self.update_fx_visualization)
        self.show_types_cb.stateChanged.connect(self.update_fx_visualization)
        self.layout_combo.currentIndexChanged.connect(self.update_fx_visualization)
    
    def browse_model(self):
        """浏览并选择模型文件"""
        file_dialog = QFileDialog()
        model_path, _ = file_dialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch模型 (*.pth *.pt);;所有文件 (*.*)"
        )
        
        if model_path:
            try:
                self.model_loader.set_model_path(model_path)
                self.model_path_edit.setText(model_path)
                
                # 启用按钮
                self.visualize_btn.setEnabled(True)
                self.fx_visualize_btn.setEnabled(HAS_MATPLOTLIB)
                
            except Exception as e:
                QMessageBox.warning(self, "警告", f"设置模型路径时出错: {str(e)}")
    
    def visualize_model_structure(self):
        """可视化模型结构"""
        if not self.model_loader.model_path:
            QMessageBox.warning(self, "警告", "请先选择模型文件")
            return
        
        try:
            # 加载模型
            model = self.model_loader.load_model()
            if model is None:
                return
            
            # 隐藏图表区域，显示文本区域
            self.graph_container.setVisible(False)
            self.output_group.setVisible(True)
            
            # 获取输入尺寸
            input_size = (
                self.input_channels.value(),
                self.input_height.value(),
                self.input_width.value()
            )
            
            # 创建文本可视化
            output = self.visualization_controller.create_text_visualization(model, input_size)
            self.output_text.setText(output)
            
        except Exception as e:
            error_msg = f"可视化模型结构失败: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            QMessageBox.critical(self, "错误", error_msg)
    
    def visualize_model_fx(self):
        """使用FX可视化模型结构"""
        if not HAS_MATPLOTLIB:
            QMessageBox.warning(self, "警告", "需要安装matplotlib才能使用FX可视化功能")
            return
            
        if not self.model_loader.model_path:
            QMessageBox.warning(self, "警告", "请先选择模型文件")
            return
        
        try:
            # 加载模型
            model = self.model_loader.load_model()
            if model is None:
                return
            
            # 清除之前的图表
            self._clear_previous_figure()
            
            # 显示图表区域，隐藏文本区域
            self.output_group.setVisible(False)
            self.graph_container.setVisible(True)
            
            # 创建FX可视化
            try:
                graph, max_depth, text_output = self.visualization_controller.create_fx_visualization(model)
                
                # 更新深度滑块范围
                self.depth_slider.setMaximum(max_depth)
                
                # 显示文本输出（FX跟踪结果或层次结构）
                self.output_text.setText(f"FX符号跟踪结果:\n\n{text_output}")
                
                # 更新图形显示
                self.update_fx_visualization()
                
            except Exception as e:
                error_msg = f"FX可视化过程中出错: {str(e)}"
                print(error_msg)
                print(traceback.format_exc())
                QMessageBox.critical(self, "错误", error_msg)
                self.graph_container.setVisible(False)
                
        except Exception as e:
            error_msg = f"加载模型失败: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            QMessageBox.critical(self, "错误", error_msg)
    
    def update_fx_visualization(self):
        """根据当前设置更新FX可视化"""
        if not self.visualization_controller.has_graph():
            return
        
        try:
            # 清除现有图表
            self._clear_previous_figure()
            
            # 获取当前设置
            current_depth = self.depth_slider.value()
            layout_idx = self.layout_combo.currentIndex()
            show_types = self.show_types_cb.isChecked()
            show_params = self.show_params_cb.isChecked()
            
            # 获取模型名称
            model_name = None
            if hasattr(self.model_loader, 'model') and self.model_loader.model:
                model_name = self.model_loader.model.__class__.__name__
            
            # 创建新图表
            fig = self.visualization_controller.create_graph_figure(
                current_depth, layout_idx, show_types, show_params, model_name)
            
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
        max_depth = self.visualization_controller.get_max_depth()
        self.depth_slider.setValue(max_depth)
    
    def collapse_all_layers(self):
        """折叠所有层"""
        self.depth_slider.setValue(1)
    
    def _clear_previous_figure(self):
        """清除之前的图表"""
        if self.figure_canvas:
            self.figure_layout.removeWidget(self.figure_canvas)
            if self.toolbar:
                self.figure_layout.removeWidget(self.toolbar)
                self.toolbar = None
            self.figure_canvas.deleteLater()
            self.figure_canvas = None
    
    def set_model(self, model, class_names=None):
        """从外部设置模型"""
        try:
            if model is not None:
                # 设置模型到加载器
                self.model_loader.set_external_model(model, class_names)
                
                # 启用按钮
                self.visualize_btn.setEnabled(True)
                self.fx_visualize_btn.setEnabled(HAS_MATPLOTLIB)
                
                # 隐藏图表区域，显示文本区域
                self.graph_container.setVisible(False)
                self.output_group.setVisible(True)
                
                # 获取输入尺寸
                input_size = (
                    self.input_channels.value(),
                    self.input_height.value(),
                    self.input_width.value()
                )
                
                # 自动触发文本可视化
                output = self.visualization_controller.create_text_visualization(model, input_size)
                self.output_text.setText(output)
                
        except Exception as e:
            error_msg = f"设置模型失败: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            QMessageBox.critical(self, "错误", error_msg)
    
    def get_model_info(self):
        """获取模型信息"""
        model_info = self.model_loader.get_model_info()
        graph_info = self.visualization_controller.get_graph_info()
        
        info = {}
        if model_info:
            info.update(model_info)
        if graph_info:
            info.update(graph_info)
            
        return info if info else None 