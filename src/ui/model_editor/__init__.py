"""
模型编辑器组件包

这是一个完整的模型结构编辑器，支持：
- 可视化模型结构编辑
- 预训练模型导入
- 模型结构导入/导出
- 交互式层参数编辑
- 缩放、平移、导航功能

主要类：
- ModelStructureEditor: 主编辑器对话框
- LayerParameterDialog: 层参数编辑对话框
- NetworkGraphicsScene: 图形场景
- NetworkGraphicsView: 图形视图
- LayerGraphicsItem: 层图形项
- ConnectionGraphicsItem: 连接图形项
"""

from .dialogs import ModelStructureEditor, LayerParameterDialog
from .graphics_scene import NetworkGraphicsScene, NetworkGraphicsView
from .graphics_items import LayerGraphicsItem, ConnectionGraphicsItem
from .utils import ModelExtractor

__all__ = [
    'ModelStructureEditor',
    'LayerParameterDialog',
    'NetworkGraphicsScene', 
    'NetworkGraphicsView',
    'LayerGraphicsItem',
    'ConnectionGraphicsItem',
    'ModelExtractor'
]

__version__ = '1.0.0' 