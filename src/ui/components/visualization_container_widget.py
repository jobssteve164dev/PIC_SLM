from PyQt5.QtWidgets import QWidget, QStackedWidget
from PyQt5.QtCore import pyqtSignal
import sys
import os

# 添加src目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 导入各种可视化组件
from ui.feature_visualization import FeatureVisualizationWidget
from ui.gradcam_visualization import GradCAMVisualizationWidget
from ui.sensitivity_analysis import SensitivityAnalysisWidget
from ui.lime_explanation import LIMEExplanationWidget
from ui.model_structure_viewer import ModelStructureViewer


class VisualizationContainerWidget(QWidget):
    """可视化组件容器，负责管理所有模型可视化组件"""
    
    status_updated = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 初始化可视化组件
        self.feature_viz_widget = None
        self.gradcam_widget = None
        self.sensitivity_widget = None
        self.lime_widget = None
        self.model_structure_widget = None
        
        self.init_components()
        
    def init_components(self):
        """初始化所有可视化组件"""
        try:
            # 创建特征可视化组件
            self.feature_viz_widget = FeatureVisualizationWidget()
            
            # 创建Grad-CAM可视化组件
            self.gradcam_widget = GradCAMVisualizationWidget()
            
            # 创建敏感性分析组件
            self.sensitivity_widget = SensitivityAnalysisWidget()
            
            # 创建LIME解释组件
            self.lime_widget = LIMEExplanationWidget()
            
            # 创建模型结构可视化组件
            self.model_structure_widget = ModelStructureViewer()
            
        except Exception as e:
            import traceback
            print(f"初始化可视化组件时出错: {str(e)}")
            print(traceback.format_exc())
            self.status_updated.emit(f"初始化可视化组件失败: {str(e)}")
    
    def get_feature_visualization_widget(self):
        """获取特征可视化组件"""
        return self.feature_viz_widget
    
    def get_gradcam_widget(self):
        """获取Grad-CAM组件"""
        return self.gradcam_widget
    
    def get_sensitivity_widget(self):
        """获取敏感性分析组件"""
        return self.sensitivity_widget
    
    def get_lime_widget(self):
        """获取LIME解释组件"""
        return self.lime_widget
    
    def get_model_structure_widget(self):
        """获取模型结构可视化组件"""
        return self.model_structure_widget
    
    def set_model(self, model, class_names=None):
        """为所有可视化组件设置模型"""
        try:
            if model is not None:
                if self.feature_viz_widget:
                    self.feature_viz_widget.set_model(model, class_names)
                
                if self.gradcam_widget:
                    self.gradcam_widget.set_model(model, class_names)
                    
                if self.sensitivity_widget:
                    self.sensitivity_widget.set_model(model, class_names)
                    
                if self.lime_widget:
                    self.lime_widget.set_model(model, class_names)
                    
                if self.model_structure_widget:
                    self.model_structure_widget.set_model(model, class_names)
                
                self.status_updated.emit("已为所有可视化组件设置模型")
        except Exception as e:
            import traceback
            print(f"设置模型时出错: {str(e)}")
            print(traceback.format_exc())
            self.status_updated.emit(f"设置模型失败: {str(e)}")
    
    def apply_config(self, config):
        """应用配置到所有组件"""
        try:
            if not config:
                return
                
            # 为每个组件应用配置（如果它们有apply_config方法）
            for widget in [self.feature_viz_widget, self.gradcam_widget, 
                          self.sensitivity_widget, self.lime_widget, 
                          self.model_structure_widget]:
                if widget and hasattr(widget, 'apply_config'):
                    widget.apply_config(config)
                    
        except Exception as e:
            import traceback
            print(f"应用配置时出错: {str(e)}")
            print(traceback.format_exc())
            self.status_updated.emit(f"应用配置失败: {str(e)}") 