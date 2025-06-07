"""
层图形项组件
"""
from PyQt5.QtWidgets import (QGraphicsRectItem, QGraphicsItem, QGraphicsTextItem,
                           QMenu, QGraphicsSceneMouseEvent)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QFont
from ..utils.constants import LAYER_COLORS, LAYER_ITEM_WIDTH, LAYER_ITEM_HEIGHT


class LayerGraphicsItem(QGraphicsRectItem):
    """表示网络层的图形项"""
    
    def __init__(self, layer_info, editor=None, parent=None):
        super().__init__(parent)
        self.layer_info = layer_info
        self.editor = editor
        self.is_selected = False
        
        # 设置图形项属性
        self.setRect(0, 0, LAYER_ITEM_WIDTH, LAYER_ITEM_HEIGHT)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        
        # 创建层名称文本项
        self.name_text = QGraphicsTextItem(layer_info['name'], self)
        self.name_text.setPos(10, 10)
        
        # 创建层类型文本项
        self.type_text = QGraphicsTextItem(layer_info['type'], self)
        self.type_text.setPos(10, 30)
        
        # 创建参数文本项
        self.param_text = QGraphicsTextItem(self.get_param_text(), self)
        self.param_text.setPos(10, 50)
        
        # 设置字体
        font = QFont()
        font.setPointSize(8)
        self.param_text.setFont(font)
        
        # 设置边框颜色
        self.update_style()
        
    def get_param_text(self):
        """根据层类型获取参数显示文本"""
        layer_type = self.layer_info['type']
        params = []
        
        if layer_type in ['Conv2d', 'ConvTranspose2d']:
            in_ch = self.layer_info.get('in_channels', 3)
            out_ch = self.layer_info.get('out_channels', 64)
            k_size = self.layer_info.get('kernel_size', 3)
            if isinstance(k_size, tuple):
                k_size = k_size[0]
            params.append(f"in:{in_ch}, out:{out_ch}")
            params.append(f"k:{k_size}×{k_size}")
        
        elif layer_type == 'Linear':
            in_feat = self.layer_info.get('in_features', 512)
            out_feat = self.layer_info.get('out_features', 10)
            params.append(f"in:{in_feat}")
            params.append(f"out:{out_feat}")
        
        elif layer_type in ['MaxPool2d', 'AvgPool2d']:
            k_size = self.layer_info.get('kernel_size', 2)
            if isinstance(k_size, tuple):
                k_size = k_size[0]
            params.append(f"size:{k_size}×{k_size}")
        
        elif layer_type == 'Dropout':
            p = self.layer_info.get('p', 0.5)
            params.append(f"prob:{p}")
        
        elif layer_type in ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', 'Flatten', 'BatchNorm2d']:
            # 这些层通常没有需要特别展示的参数
            if layer_type == 'BatchNorm2d' and 'num_features' in self.layer_info:
                params.append(f"features:{self.layer_info['num_features']}")
            elif layer_type == 'LeakyReLU' and 'negative_slope' in self.layer_info:
                params.append(f"slope:{self.layer_info['negative_slope']}")
        
        return "\n".join(params)
        
    def get_color_for_layer_type(self, layer_type):
        """根据层类型获取对应的颜色"""
        return LAYER_COLORS.get(layer_type, LAYER_COLORS['default'])
        
    def update_style(self):
        """更新图形项样式"""
        # 获取层类型对应的颜色
        color = self.get_color_for_layer_type(self.layer_info['type'])
        
        # 设置填充和边框
        self.setBrush(QBrush(QColor("#f0f0f0")))
        self.setPen(QPen(QColor(color), 2))
    
    def update_param_text(self):
        """更新参数文本"""
        if hasattr(self, 'param_text'):
            self.param_text.setPlainText(self.get_param_text())
        
    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        if event.button() == Qt.LeftButton:
            self.is_selected = True
            if self.editor:
                self.editor.on_layer_selected(self.layer_info)
        elif event.button() == Qt.RightButton:
            # 直接使用鼠标事件的屏幕坐标
            self.show_context_menu(event.screenPos())
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton:
            # 更新位置后通知编辑器
            if self.editor:
                self.editor.update_layer_position(self.layer_info['name'], self.pos())
            
    def show_context_menu(self, pos):
        """显示右键菜单"""
        if not self.editor:
            return
            
        menu = QMenu()
        
        # 编辑操作
        edit_action = menu.addAction("编辑参数")
        delete_action = menu.addAction("删除")
        
        # 直接使用屏幕坐标，无需调用toPoint()
        action = menu.exec_(pos)
        
        if action == edit_action:
            self.editor.edit_layer_parameters(self.layer_info)
        elif action == delete_action:
            self.editor.delete_layer(self.layer_info['name'])
            
    def paint(self, painter, option, widget):
        """自定义绘制"""
        # 绘制基本矩形
        super().paint(painter, option, widget)
        
        # 如果被选中，绘制高亮边框
        if self.is_selected:
            painter.setPen(QPen(QColor("#3399ff"), 3))
            painter.drawRect(self.rect().adjusted(2, 2, -2, -2))
            
    def itemChange(self, change, value):
        """处理项目状态变化"""
        if change == QGraphicsItem.ItemSelectedChange:
            self.is_selected = bool(value)
            
        return super().itemChange(change, value) 