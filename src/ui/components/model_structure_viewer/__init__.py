"""
模型结构可视化组件模块

这个模块包含了重构后的模型结构可视化组件，包括以下子模块：
- model_loader: 模型加载模块
- graph_builder: 图形构建模块  
- layout_algorithms: 布局算法模块
- visualization_controller: 可视化控制模块
- ui_components: UI组件模块
- model_structure_viewer: 主组件

主要用法:
    from src.ui.components.model_structure_viewer import ModelStructureViewer
    
    # 创建组件
    viewer = ModelStructureViewer()
    
    # 设置外部模型
    viewer.set_model(model, class_names)
"""

# 导入主组件
from .model_structure_viewer import ModelStructureViewer

# 导入子模块（可选，供高级用户使用）
from . import model_loader
from . import graph_builder
from . import layout_algorithms
from . import visualization_controller
from . import ui_components

# 公开的API
__all__ = [
    'ModelStructureViewer',
    'model_loader',
    'graph_builder', 
    'layout_algorithms',
    'visualization_controller',
    'ui_components'
]

# 版本信息
__version__ = '2.0.0'
__author__ = 'AI Assistant'
__description__ = '重构后的模型结构可视化组件' 