"""
模型结构编辑器 - 向后兼容性导入

此文件提供向后兼容性，将原有的类导入重定向到新的模块化结构。
建议直接从 model_editor 包导入类。

新的导入方式：
    from src.ui.model_editor import ModelStructureEditor
    
原有的导入方式（仍然支持）：
    from src.ui.model_structure_editor import ModelStructureEditor
"""

# 为了向后兼容，从新的模块化结构导入所有类
from .model_editor import (
    ModelStructureEditor,
    LayerParameterDialog,
    NetworkGraphicsScene,
    NetworkGraphicsView,
    LayerGraphicsItem,
    ConnectionGraphicsItem,
    ModelExtractor
)

# 保持原有的类名可用
__all__ = [
    'ModelStructureEditor',
    'LayerParameterDialog', 
    'NetworkGraphicsScene',
    'NetworkGraphicsView',
    'LayerGraphicsItem',
    'ConnectionGraphicsItem',
    'ModelExtractor'
]