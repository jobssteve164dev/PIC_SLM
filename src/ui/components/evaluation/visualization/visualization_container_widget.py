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
# 使用正确的相对导入路径
# 注意：只保留模型结构可视化组件，因为其他组件已经移除
from src.ui.model_structure_viewer import ModelStructureViewer


class VisualizationContainerWidget(QWidget):
    """可视化组件容器，负责管理所有模型可视化组件"""
    
    status_updated = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 初始化可视化组件
        self.model_structure_widget = None
        
        self.init_components()
        
    def init_components(self):
        """初始化所有可视化组件"""
        try:
            # 创建模型结构可视化组件
            self.model_structure_widget = ModelStructureViewer()
            
        except Exception as e:
            import traceback
            print(f"初始化可视化组件时出错: {str(e)}")
            print(traceback.format_exc())
            self.status_updated.emit(f"初始化可视化组件失败: {str(e)}")
    

    
    def get_model_structure_widget(self):
        """获取模型结构可视化组件"""
        return self.model_structure_widget
    
    def set_model(self, model, class_names=None):
        """为模型结构可视化组件设置模型"""
        try:
            if model is not None and self.model_structure_widget:
                self.model_structure_widget.set_model(model, class_names)
                self.status_updated.emit("已为模型结构可视化组件设置模型")
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
                
            # 为模型结构可视化组件应用配置
            if self.model_structure_widget and hasattr(self.model_structure_widget, 'apply_config'):
                self.model_structure_widget.apply_config(config)
                    
        except Exception as e:
            import traceback
            print(f"应用配置时出错: {str(e)}")
            print(traceback.format_exc())
            self.status_updated.emit(f"应用配置失败: {str(e)}") 