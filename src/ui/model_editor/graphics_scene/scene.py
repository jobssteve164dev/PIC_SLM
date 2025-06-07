"""
模型网络图形场景
"""
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QBrush, QColor
from ..graphics_items import LayerGraphicsItem, ConnectionGraphicsItem


class NetworkGraphicsScene(QGraphicsScene):
    """模型网络图形场景"""
    
    def __init__(self, editor=None, parent=None):
        super().__init__(parent)
        self.editor = editor
        self.layer_items = {}  # 存储层名称到图形项的映射
        self.connection_items = []  # 存储连接图形项
        
        # 设置背景色
        self.setBackgroundBrush(QBrush(QColor("#ffffff")))
        
    def add_layer(self, layer_info, pos=None):
        """添加层图形项"""
        # 创建图形项
        layer_item = LayerGraphicsItem(layer_info, self.editor)
        
        # 设置位置
        if pos:
            layer_item.setPos(pos)
        else:
            # 计算默认位置 (网格布局)
            count = len(self.layer_items)
            row = count // 4
            col = count % 4
            layer_item.setPos(col * 140, row * 100)
            
        # 添加到场景
        self.addItem(layer_item)
        
        # 保存引用
        self.layer_items[layer_info['name']] = layer_item
        
        return layer_item
        
    def add_connection(self, from_name, to_name):
        """添加连接图形项"""
        # 获取源和目标图形项
        from_item = self.layer_items.get(from_name)
        to_item = self.layer_items.get(to_name)
        
        if not from_item or not to_item:
            return None
            
        # 创建连接图形项
        connection_item = ConnectionGraphicsItem(from_item, to_item)
        
        # 添加到场景
        self.addItem(connection_item)
        
        # 保存引用
        self.connection_items.append(connection_item)
        
        return connection_item
        
    def remove_layer(self, layer_name):
        """移除层图形项"""
        # 获取图形项
        layer_item = self.layer_items.get(layer_name)
        
        if not layer_item:
            return
            
        # 移除相关连接
        connections_to_remove = []
        for conn_item in self.connection_items:
            if (conn_item.from_item == layer_item or 
                conn_item.to_item == layer_item):
                connections_to_remove.append(conn_item)
                
        for conn_item in connections_to_remove:
            self.connection_items.remove(conn_item)
            self.removeItem(conn_item)
            
        # 移除层图形项
        self.removeItem(layer_item)
        del self.layer_items[layer_name]
        
    def update_connections(self):
        """更新所有连接的路径"""
        for conn_item in self.connection_items:
            conn_item.update_path()
            
    def clear_all(self):
        """清除所有图形项"""
        self.clear()
        self.layer_items.clear()
        self.connection_items.clear() 