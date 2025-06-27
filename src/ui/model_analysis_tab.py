from PyQt5.QtWidgets import QVBoxLayout, QWidget
from PyQt5.QtCore import pyqtSignal
from .base_tab import BaseTab
from .model_analysis_widget import ModelAnalysisWidget


class ModelAnalysisTab(BaseTab):
    """模型分析标签页，集成了四种分析方法"""
    
    # 定义信号
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent, main_window)
        self.setWindowTitle("模型分析")
        self.init_ui()
        
        # 使用新的智能配置系统
        config = self.get_config_from_manager()
        if config:
            self.apply_config(config)
        
    def init_ui(self):
        """初始化UI"""
        try:
            # 使用BaseTab提供的滚动内容区域，而不是创建新的布局
            layout = QVBoxLayout(self.scroll_content)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(10)
            
            print("ModelAnalysisTab: 开始创建模型分析组件...")
            
            # 创建整合的模型分析组件
            self.model_analysis_widget = ModelAnalysisWidget(self.scroll_content)
            
            print("ModelAnalysisTab: 模型分析组件创建成功")
            
            # 连接信号
            self.model_analysis_widget.status_updated.connect(self.status_updated)
            
            # 添加到布局
            layout.addWidget(self.model_analysis_widget)
            
            print("ModelAnalysisTab: UI初始化完成")
            
        except ImportError as e:
            print(f"ModelAnalysisTab: 导入错误 - {str(e)}")
            # 创建一个简单的错误显示标签
            from PyQt5.QtWidgets import QLabel
            layout = QVBoxLayout(self.scroll_content)
            error_label = QLabel(f"模型分析组件加载失败：\n{str(e)}")
            layout.addWidget(error_label)
            
        except Exception as e:
            print(f"ModelAnalysisTab: 初始化错误 - {str(e)}")
            # 创建一个简单的错误显示标签
            from PyQt5.QtWidgets import QLabel
            layout = QVBoxLayout(self.scroll_content)
            error_label = QLabel(f"模型分析组件初始化失败：\n{str(e)}")
            layout.addWidget(error_label)
        
    def _do_apply_config(self, config):
        """实现具体的配置应用逻辑 - 智能配置系统"""
        print(f"ModelAnalysisTab: 智能应用配置，包含 {len(config)} 个配置项")
        
        # 如果需要应用配置到模型分析组件，可以在这里实现
        if hasattr(self, 'model_analysis_widget') and self.model_analysis_widget:
            # 可以将配置传递给模型分析组件
            if hasattr(self.model_analysis_widget, 'apply_config'):
                self.model_analysis_widget.apply_config(config)
                
        print("ModelAnalysisTab: 智能配置应用完成")
        
    def get_status(self):
        """获取当前状态"""
        return "模型分析标签页已就绪" 