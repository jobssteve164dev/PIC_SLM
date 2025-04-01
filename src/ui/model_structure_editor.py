import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                           QLabel, QPushButton, QComboBox, QSpinBox, 
                           QDoubleSpinBox, QLineEdit, QScrollArea, QWidget,
                           QGridLayout, QMenu, QMessageBox, QFileDialog,
                           QRubberBand, QGraphicsScene, QGraphicsView,
                           QGraphicsItem, QGraphicsRectItem, QGraphicsPathItem,
                           QGraphicsTextItem, QGraphicsSceneMouseEvent, QGraphicsSceneWheelEvent,
                           QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect, QSize, QRectF, QPointF
from PyQt5.QtGui import (QPainter, QPen, QColor, QBrush, QPainterPath, QTransform,
                       QFont, QPolygonF)
import json

class LayerGraphicsItem(QGraphicsRectItem):
    """表示网络层的图形项"""
    
    # 定义不同层类型的颜色
    LAYER_COLORS = {
        'Conv2d': "#4285F4",        # 蓝色
        'ConvTranspose2d': "#34A853", # 绿色
        'Linear': "#FBBC05",        # 黄色
        'MaxPool2d': "#EA4335",     # 红色
        'AvgPool2d': "#EA4335",     # 红色
        'BatchNorm2d': "#9C27B0",   # 紫色
        'Dropout': "#FF9800",       # 橙色
        'ReLU': "#03A9F4",          # 浅蓝色
        'LeakyReLU': "#03A9F4",     # 浅蓝色
        'Sigmoid': "#03A9F4",       # 浅蓝色
        'Tanh': "#03A9F4",          # 浅蓝色
        'Flatten': "#607D8B",       # 灰蓝色
        'default': "#757575"        # 默认灰色
    }
    
    def __init__(self, layer_info, editor=None, parent=None):
        super().__init__(parent)
        self.layer_info = layer_info
        self.editor = editor
        self.is_selected = False
        
        # 设置图形项属性
        self.setRect(0, 0, 120, 80)  # 设置大小
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        
        # 创建层名称文本项
        self.name_text = QGraphicsTextItem(layer_info['name'], self)
        self.name_text.setPos(10, 15)
        
        # 创建层类型文本项
        self.type_text = QGraphicsTextItem(layer_info['type'], self)
        self.type_text.setPos(10, 45)
        
        # 设置边框颜色
        self.update_style()
        
    def get_color_for_layer_type(self, layer_type):
        """根据层类型获取对应的颜色"""
        return self.LAYER_COLORS.get(layer_type, self.LAYER_COLORS['default'])
        
    def update_style(self):
        """更新图形项样式"""
        # 获取层类型对应的颜色
        color = self.get_color_for_layer_type(self.layer_info['type'])
        
        # 设置填充和边框
        self.setBrush(QBrush(QColor("#f0f0f0")))
        self.setPen(QPen(QColor(color), 2))
        
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


class ConnectionGraphicsItem(QGraphicsPathItem):
    """表示层之间连接的图形项"""
    
    def __init__(self, from_item, to_item, parent=None):
        super().__init__(parent)
        self.from_item = from_item
        self.to_item = to_item
        self.update_path()
        
        # 设置画笔
        self.setPen(QPen(QColor("#3399ff"), 2, Qt.SolidLine, Qt.RoundCap))
        
    def update_path(self):
        """更新连接路径"""
        if not self.from_item or not self.to_item:
            return
            
        # 计算源和目标位置
        from_rect = self.from_item.rect()
        to_rect = self.to_item.rect()
        
        from_pos = self.from_item.pos() + QPointF(from_rect.width() / 2, from_rect.height())
        to_pos = self.to_item.pos() + QPointF(to_rect.width() / 2, 0)
        
        # 创建路径
        path = QPainterPath()
        path.moveTo(from_pos)
        
        # 计算控制点
        control_dist = (to_pos.y() - from_pos.y()) * 0.5
        ctrl1 = QPointF(from_pos.x(), from_pos.y() + control_dist)
        ctrl2 = QPointF(to_pos.x(), to_pos.y() - control_dist)
        
        # 创建贝塞尔曲线
        path.cubicTo(ctrl1, ctrl2, to_pos)
        
        # 添加箭头
        self.arrow_head = self.create_arrow_head(to_pos, ctrl2)
        
        # 设置路径
        self.setPath(path)
        
    def create_arrow_head(self, tip_pos, control_point):
        """创建箭头"""
        # 计算方向向量
        dx = tip_pos.x() - control_point.x()
        dy = tip_pos.y() - control_point.y()
        dist = (dx**2 + dy**2)**0.5
        
        if dist < 0.1:  # 避免除以零
            return QPolygonF()
            
        # 归一化方向向量
        dx /= dist
        dy /= dist
        
        # 箭头大小和角度
        arrow_size = 10
        angle = 0.5
        
        # 计算箭头两侧的点
        p1 = QPointF(
            tip_pos.x() - arrow_size * dx - arrow_size * dy * angle,
            tip_pos.y() - arrow_size * dy + arrow_size * dx * angle
        )
        
        p2 = QPointF(
            tip_pos.x() - arrow_size * dx + arrow_size * dy * angle,
            tip_pos.y() - arrow_size * dy - arrow_size * dx * angle
        )
        
        # 创建多边形
        polygon = QPolygonF()
        polygon.append(tip_pos)
        polygon.append(p1)
        polygon.append(p2)
        
        return polygon
        
    def paint(self, painter, option, widget):
        """自定义绘制"""
        # 确保路径是最新的
        self.update_path()
        
        # 绘制路径
        super().paint(painter, option, widget)
        
        # 绘制箭头
        painter.setBrush(QBrush(QColor("#3399ff")))
        painter.drawPolygon(self.arrow_head)


class NetworkDesignArea(QWidget):
    """网络设计区域，包含层部件和连接线，支持缩放、平移和选择"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: white;")
        self.layers = []  # 存储层信息
        self.connections = []  # 存储连接信息
        self.layer_widgets = {}  # 存储层部件
        self.editor = None  # 存储编辑器引用
        
        # 缩放和平移相关
        self.scale_factor = 1.0  # 缩放因子
        self.translate_x = 0     # X轴平移量
        self.translate_y = 0     # Y轴平移量
        self.last_pan_point = None  # 上次平移点
        self.is_panning = False  # 是否正在平移
        
        # 橡皮筋选择相关
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self.rubber_band_origin = None
        self.is_selecting = False
        
        self.setMouseTracking(True)  # 启用鼠标跟踪
        self.setFocusPolicy(Qt.StrongFocus)  # 允许获取键盘焦点
        
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        self.layout = QGridLayout(self)
        self.layout.setSpacing(80)  # 设置很大的间距以便显示连接线
        self.setMinimumSize(600, 400)  # 设置最小大小
        
    def set_editor(self, editor):
        """设置编辑器引用"""
        self.editor = editor
        
    def add_layer(self, layer_info):
        """添加层"""
        layer_widget = LayerGraphicsItem(layer_info)
        
        # 计算位置 (每行最多4个)
        row = len(self.layers) // 4
        col = len(self.layers) % 4
        
        # 添加到布局
        self.layout.addWidget(layer_widget, row, col)
        
        # 保存引用
        self.layer_widgets[layer_info['name']] = layer_widget
        self.layers.append(layer_info)
        
        # 连接信号到编辑器
        if self.editor:
            layer_widget.layer_selected.connect(self.editor.on_layer_selected)
            layer_widget.layer_modified.connect(self.editor.on_layer_modified)
            layer_widget.layer_deleted.connect(self.editor.on_layer_deleted)
        
        # 更新视图
        self.update()
        return layer_widget
        
    def add_connection(self, from_layer, to_layer):
        """添加连接"""
        connection = {'from': from_layer, 'to': to_layer}
        self.connections.append(connection)
        self.update()  # 触发重绘
        
    def clear(self):
        """清除所有层和连接"""
        # 清除布局中的所有部件
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        # 清除数据
        self.layers.clear()
        self.connections.clear()
        self.layer_widgets.clear()
        
        # 更新视图
        self.update()
        
    def remove_layer(self, layer_name):
        """移除层"""
        # 移除层信息
        self.layers = [layer for layer in self.layers if layer['name'] != layer_name]
        
        # 移除相关连接
        self.connections = [conn for conn in self.connections 
                           if conn['from'] != layer_name and conn['to'] != layer_name]
        
        # 移除部件
        if layer_name in self.layer_widgets:
            self.layout.removeWidget(self.layer_widgets[layer_name])
            self.layer_widgets[layer_name].deleteLater()
            del self.layer_widgets[layer_name]
            
        # 重新排列部件
        self.rearrange_widgets()
        
        # 更新视图
        self.update()
        
    def rearrange_widgets(self):
        """重新排列部件"""
        # 先从布局中移除所有部件
        for widget in self.layer_widgets.values():
            self.layout.removeWidget(widget)
            
        # 重新添加
        for i, layer_info in enumerate(self.layers):
            widget = self.layer_widgets.get(layer_info['name'])
            if widget:
                row = i // 4
                col = i % 4
                self.layout.addWidget(widget, row, col)
                
    def wheelEvent(self, event):
        """处理鼠标滚轮事件，实现缩放功能"""
        # 获取鼠标位置作为缩放中心
        center_x = event.pos().x()
        center_y = event.pos().y()
        
        # 确定缩放方向和大小
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        
        # 应用缩放
        self.scale_factor *= zoom_factor
        
        # 限制缩放范围，避免过大或过小
        self.scale_factor = max(0.1, min(self.scale_factor, 5.0))
        
        # 更新视图
        self.update()
        
        # 接受事件
        event.accept()
        
    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        if event.button() == Qt.MiddleButton:
            # 中键按下，开始平移
            self.is_panning = True
            self.last_pan_point = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton and not self.childAt(event.pos()):
            # 左键按下且没有点击在子部件上，开始框选
            self.is_selecting = True
            self.rubber_band_origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.rubber_band_origin, QSize()))
            self.rubber_band.show()
        # 其他情况，调用默认处理
        else:
            super().mousePressEvent(event)
            
    def mouseMoveEvent(self, event):
        """处理鼠标移动事件"""
        if self.is_panning and self.last_pan_point:
            # 计算移动距离
            dx = event.pos().x() - self.last_pan_point.x()
            dy = event.pos().y() - self.last_pan_point.y()
            
            # 更新平移量
            self.translate_x += dx
            self.translate_y += dy
            
            # 更新上次平移点
            self.last_pan_point = event.pos()
            
            # 更新视图
            self.update()
        elif self.is_selecting:
            # 更新橡皮筋选框
            self.rubber_band.setGeometry(QRect(self.rubber_band_origin, event.pos()).normalized())
        else:
            super().mouseMoveEvent(event)
            
    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        if event.button() == Qt.MiddleButton and self.is_panning:
            # 结束平移
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
        elif event.button() == Qt.LeftButton and self.is_selecting:
            # 结束框选
            self.is_selecting = False
            selection_rect = self.rubber_band.geometry()
            self.rubber_band.hide()
            
            # 如果选择区域足够大，放大该区域
            if selection_rect.width() > 10 and selection_rect.height() > 10:
                # 计算选择区域的中心
                center_x = selection_rect.x() + selection_rect.width() / 2
                center_y = selection_rect.y() + selection_rect.height() / 2
                
                # 计算合适的缩放因子
                view_width = self.width()
                view_height = self.height()
                
                scale_x = view_width / selection_rect.width()
                scale_y = view_height / selection_rect.height()
                
                # 选择较小的缩放因子以确保整个选择区域都可见
                scale = min(scale_x, scale_y) * 0.8
                
                # 设置新的缩放和平移
                self.scale_factor = scale
                self.translate_x = view_width / 2 - center_x * scale
                self.translate_y = view_height / 2 - center_y * scale
                
                # 更新视图
                self.update()
        else:
            super().mouseReleaseEvent(event)
            
    def keyPressEvent(self, event):
        """处理键盘事件"""
        if event.key() == Qt.Key_Escape:
            # ESC键重置视图
            self.scale_factor = 1.0
            self.translate_x = 0
            self.translate_y = 0
            self.update()
        else:
            super().keyPressEvent(event)
            
    def paintEvent(self, event):
        """绘制事件，用于绘制连接线和应用变换"""
        # 创建画家
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 保存当前状态
        painter.save()
        
        # 应用变换（缩放和平移）
        transform = QTransform()
        transform.translate(self.translate_x, self.translate_y)
        transform.scale(self.scale_factor, self.scale_factor)
        painter.setTransform(transform)
        
        # 绘制背景
        painter.fillRect(self.rect(), QColor("white"))
        
        # 调用基类绘制
        super().paintEvent(event)
        
        # 如果没有连接，直接返回
        if not self.connections:
            painter.restore()
            return
            
        # 设置画笔
        pen = QPen(QColor("#3399ff"), 2)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        
        # 绘制每个连接
        for connection in self.connections:
            from_widget = self.layer_widgets.get(connection['from'])
            to_widget = self.layer_widgets.get(connection['to'])
            
            if from_widget and to_widget:
                # 计算连接点
                from_center = from_widget.mapTo(self, QPoint(from_widget.width() // 2, from_widget.height()))
                to_center = to_widget.mapTo(self, QPoint(to_widget.width() // 2, 0))
                
                # 创建路径
                path = QPainterPath()
                path.moveTo(from_center)
                
                # 计算控制点，使曲线平滑
                control_dist = (to_center.y() - from_center.y()) * 0.5
                ctrl1 = QPoint(from_center.x(), from_center.y() + control_dist)
                ctrl2 = QPoint(to_center.x(), to_center.y() - control_dist)
                
                # 绘制贝塞尔曲线
                path.cubicTo(ctrl1, ctrl2, to_center)
                painter.drawPath(path)
                
                # 绘制箭头
                self.draw_arrow(painter, path, to_center)
        
        # 恢复画家状态
        painter.restore()
    
    def draw_arrow(self, painter, path, point):
        """在路径端点绘制箭头"""
        # 箭头大小
        arrow_size = 10
        
        # 计算路径端点的切线方向
        angle = 0.5  # 箭头展开角度（弧度）
        
        # 假设路径的方向是向下的
        tangent = QPoint(0, -1)  # 默认向上的切线
        
        # 计算实际切线
        # 为简化，我们使用一个近似值
        if len(self.connections) > 0:
            # 找到当前连接
            for conn in self.connections:
                to_widget = self.layer_widgets.get(conn['to'])
                if to_widget and to_widget.mapTo(self, QPoint(to_widget.width() // 2, 0)) == point:
                    from_widget = self.layer_widgets.get(conn['from'])
                    if from_widget:
                        from_point = from_widget.mapTo(self, QPoint(from_widget.width() // 2, from_widget.height()))
                        dx = point.x() - from_point.x()
                        dy = point.y() - from_point.y()
                        length = (dx ** 2 + dy ** 2) ** 0.5
                        if length > 0:
                            tangent = QPoint(-dx / length, -dy / length)
                            break
        
        # 计算箭头的两个点
        left = QPoint(point.x() + tangent.x() * arrow_size - tangent.y() * arrow_size * angle,
                    point.y() + tangent.y() * arrow_size + tangent.x() * arrow_size * angle)
        right = QPoint(point.x() + tangent.x() * arrow_size + tangent.y() * arrow_size * angle,
                     point.y() + tangent.y() * arrow_size - tangent.x() * arrow_size * angle)
        
        # 创建箭头路径并填充
        arrow_path = QPainterPath()
        arrow_path.moveTo(point)
        arrow_path.lineTo(left)
        arrow_path.lineTo(right)
        arrow_path.closeSubpath()
        
        painter.fillPath(arrow_path, QBrush(QColor("#3399ff")))


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
        

class NetworkGraphicsView(QGraphicsView):
    """模型网络图形视图，支持缩放、平移和框选"""
    
    def __init__(self, scene=None, parent=None):
        super().__init__(scene, parent)
        
        # 设置渲染属性
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.TextAntialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # 设置拖拽模式
        self.setDragMode(QGraphicsView.RubberBandDrag)
        
        # 设置缩放属性
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        # 禁用滚动条，改用视图变换实现平移
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # 设置场景矩形为一个较大的区域，以支持自由平移
        self.setSceneRect(-10000, -10000, 20000, 20000)
        
        # 初始化变量
        self.zoom_factor = 1.0
        self.is_panning = False
        self.last_pan_pos = None
        
    def wheelEvent(self, event):
        """处理鼠标滚轮事件，实现缩放功能"""
        # 确定缩放方向和大小
        delta = event.angleDelta().y()
        zoom_in = delta > 0
        
        # 缩放因子
        factor = 1.1 if zoom_in else 0.9
        
        # 应用缩放
        self.scale(factor, factor)
        
        # 更新缩放因子
        self.zoom_factor *= factor
        
        # 限制缩放范围
        if self.zoom_factor < 0.1:
            # 恢复到最小缩放
            reset_factor = 0.1 / self.zoom_factor
            self.scale(reset_factor, reset_factor)
            self.zoom_factor = 0.1
        elif self.zoom_factor > 5.0:
            # 恢复到最大缩放
            reset_factor = 5.0 / self.zoom_factor
            self.scale(reset_factor, reset_factor)
            self.zoom_factor = 5.0
            
    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        if event.button() == Qt.MiddleButton:
            # 中键按下开始平移
            self.is_panning = True
            self.last_pan_pos = event.pos()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            # 其他按键使用默认处理
            super().mousePressEvent(event)
            
    def mouseMoveEvent(self, event):
        """处理鼠标移动事件"""
        if self.is_panning and self.last_pan_pos:
            # 计算视图坐标系中的移动距离
            delta = event.pos() - self.last_pan_pos
            self.last_pan_pos = event.pos()
            
            # 获取当前视图变换
            transform = self.transform()
            
            # 计算在当前缩放级别下的实际移动距离
            dx = delta.x()
            dy = delta.y()
            
            # 移动场景视图
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - dx)
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - dy)
            
            event.accept()
        else:
            super().mouseMoveEvent(event)
            
    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        if event.button() == Qt.MiddleButton and self.is_panning:
            # 结束平移
            self.is_panning = False
            self.last_pan_pos = None
            self.viewport().setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)
            
    def keyPressEvent(self, event):
        """处理键盘事件"""
        if event.key() == Qt.Key_Escape:
            # ESC键重置视图
            self.resetTransform()
            self.zoom_factor = 1.0
            # 重置视图位置到中心
            self.centerOn(0, 0)
            event.accept()
        else:
            super().keyPressEvent(event)


class LayerParameterDialog(QDialog):
    """层参数编辑对话框"""
    
    def __init__(self, layer_info, parent=None):
        super().__init__(parent)
        self.layer_info = layer_info.copy()
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("编辑层参数")
        layout = QVBoxLayout(self)
        
        # 创建参数编辑区域
        param_group = QGroupBox("参数设置")
        param_layout = QGridLayout()
        
        # 根据层类型添加不同的参数控件
        row = 0
        if self.layer_info['type'] in ['Conv2d', 'ConvTranspose2d']:
            # 输入通道
            param_layout.addWidget(QLabel("输入通道:"), row, 0)
            self.in_channels = QSpinBox()
            self.in_channels.setRange(1, 2048)
            self.in_channels.setValue(self.layer_info.get('in_channels', 3))
            param_layout.addWidget(self.in_channels, row, 1)
            row += 1
            
            # 输出通道
            param_layout.addWidget(QLabel("输出通道:"), row, 0)
            self.out_channels = QSpinBox()
            self.out_channels.setRange(1, 2048)
            self.out_channels.setValue(self.layer_info.get('out_channels', 64))
            param_layout.addWidget(self.out_channels, row, 1)
            row += 1
            
            # 卷积核大小
            param_layout.addWidget(QLabel("卷积核大小:"), row, 0)
            self.kernel_size = QComboBox()
            self.kernel_size.addItems(['1x1', '3x3', '5x5', '7x7'])
            kernel_size = self.layer_info.get('kernel_size', 3)
            # 处理元组或整数形式的kernel_size
            if isinstance(kernel_size, tuple):
                kernel_size = kernel_size[0]
            current_kernel = f"{kernel_size}x{kernel_size}"
            self.kernel_size.setCurrentText(current_kernel)
            param_layout.addWidget(self.kernel_size, row, 1)
            row += 1
            
        elif self.layer_info['type'] == 'Linear':
            # 输入特征
            param_layout.addWidget(QLabel("输入特征:"), row, 0)
            self.in_features = QSpinBox()
            self.in_features.setRange(1, 10000)
            self.in_features.setValue(self.layer_info.get('in_features', 512))
            param_layout.addWidget(self.in_features, row, 1)
            row += 1
            
            # 输出特征
            param_layout.addWidget(QLabel("输出特征:"), row, 0)
            self.out_features = QSpinBox()
            self.out_features.setRange(1, 10000)
            self.out_features.setValue(self.layer_info.get('out_features', 10))
            param_layout.addWidget(self.out_features, row, 1)
            row += 1
            
        elif self.layer_info['type'] in ['MaxPool2d', 'AvgPool2d']:
            # 池化核大小
            param_layout.addWidget(QLabel("池化核大小:"), row, 0)
            self.pool_size = QComboBox()
            self.pool_size.addItems(['2x2', '3x3', '4x4'])
            kernel_size = self.layer_info.get('kernel_size', 2)
            # 处理元组或整数形式的kernel_size
            if isinstance(kernel_size, tuple):
                kernel_size = kernel_size[0]
            current_pool = f"{kernel_size}x{kernel_size}"
            self.pool_size.setCurrentText(current_pool)
            param_layout.addWidget(self.pool_size, row, 1)
            row += 1
            
        elif self.layer_info['type'] == 'Dropout':
            # 丢弃率
            param_layout.addWidget(QLabel("丢弃率:"), row, 0)
            self.dropout_rate = QDoubleSpinBox()
            self.dropout_rate.setRange(0.0, 1.0)
            self.dropout_rate.setSingleStep(0.1)
            self.dropout_rate.setValue(self.layer_info.get('p', 0.5))
            param_layout.addWidget(self.dropout_rate, row, 1)
            row += 1
            
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
    def get_parameters(self):
        """获取修改后的参数"""
        params = self.layer_info.copy()
        
        if self.layer_info['type'] in ['Conv2d', 'ConvTranspose2d']:
            params['in_channels'] = self.in_channels.value()
            params['out_channels'] = self.out_channels.value()
            kernel = int(self.kernel_size.currentText().split('x')[0])
            params['kernel_size'] = (kernel, kernel)
            
        elif self.layer_info['type'] == 'Linear':
            params['in_features'] = self.in_features.value()
            params['out_features'] = self.out_features.value()
            
        elif self.layer_info['type'] in ['MaxPool2d', 'AvgPool2d']:
            pool = int(self.pool_size.currentText().split('x')[0])
            params['kernel_size'] = (pool, pool)
            
        elif self.layer_info['type'] == 'Dropout':
            params['p'] = self.dropout_rate.value()
            
        return params


class ModelStructureEditor(QDialog):
    """模型结构编辑器对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layers = []  # 存储所有层信息
        self.connections = []  # 存储层之间的连接
        self.selected_layer = None
        
        # 设置窗口标志，确保有最大化按钮
        self.setWindowFlags(
            Qt.Dialog | 
            Qt.WindowCloseButtonHint | 
            Qt.WindowMaximizeButtonHint | 
            Qt.WindowMinimizeButtonHint
        )
        
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("模型结构编辑器")
        self.resize(800, 600)
        
        main_layout = QHBoxLayout(self)
        
        # 左侧工具栏
        tools_group = QGroupBox("工具")
        tools_layout = QVBoxLayout()
        
        # 添加层按钮
        add_layer_button = QPushButton("添加层")
        add_layer_button.clicked.connect(self.add_layer)
        tools_layout.addWidget(add_layer_button)
        
        # 添加连接按钮
        add_connection_button = QPushButton("添加连接")
        add_connection_button.clicked.connect(self.add_connection)
        tools_layout.addWidget(add_connection_button)
        
        # 清除所有按钮
        clear_button = QPushButton("清除所有")
        clear_button.clicked.connect(self.clear_all)
        tools_layout.addWidget(clear_button)
        
        # 导入/导出按钮
        import_button = QPushButton("导入结构")
        import_button.clicked.connect(self.import_structure)
        tools_layout.addWidget(import_button)
        
        export_button = QPushButton("导出结构")
        export_button.clicked.connect(self.export_structure)
        tools_layout.addWidget(export_button)
        
        # 视图操作提示
        hint_group = QGroupBox("操作提示")
        hint_layout = QVBoxLayout()
        hint_layout.addWidget(QLabel("- 滚轮：缩放视图"))
        hint_layout.addWidget(QLabel("- 中键拖动：平移视图"))
        hint_layout.addWidget(QLabel("- 左键拖动：移动层"))
        hint_layout.addWidget(QLabel("- 右键点击：层选项"))
        hint_layout.addWidget(QLabel("- ESC键：重置视图"))
        hint_group.setLayout(hint_layout)
        tools_layout.addWidget(hint_group)
        
        # 层类型颜色说明
        color_group = QGroupBox("层类型颜色")
        color_layout = QVBoxLayout()
        
        # 添加颜色示例
        for layer_type, color in sorted(LayerGraphicsItem.LAYER_COLORS.items()):
            if layer_type != 'default':
                # 创建一个水平布局的容器
                color_container = QWidget()
                container_layout = QHBoxLayout(color_container)
                container_layout.setContentsMargins(5, 2, 5, 2)
                
                # 创建颜色示例框
                color_box = QFrame()
                color_box.setFixedSize(20, 20)
                color_box.setStyleSheet(f"""
                    QFrame {{
                        background-color: {color};
                        border: 1px solid #999999;
                        border-radius: 3px;
                    }}
                """)
                
                # 创建类型标签
                type_label = QLabel(layer_type)
                type_label.setStyleSheet("padding-left: 5px;")
                
                # 将颜色框和标签添加到容器中
                container_layout.addWidget(color_box)
                container_layout.addWidget(type_label)
                container_layout.addStretch()
                
                # 将容器添加到颜色组布局中
                color_layout.addWidget(color_container)
        
        color_layout.addStretch()
        color_group.setLayout(color_layout)
        tools_layout.addWidget(color_group)
        
        tools_layout.addStretch(1)
        tools_group.setLayout(tools_layout)
        main_layout.addWidget(tools_group)
        
        # 右侧编辑区域
        edit_group = QGroupBox("编辑区域")
        edit_layout = QVBoxLayout(edit_group)
        
        # 创建场景
        self.scene = NetworkGraphicsScene(self)
        
        # 创建视图
        self.view = NetworkGraphicsView(self.scene)
        edit_layout.addWidget(self.view)
        
        main_layout.addWidget(edit_group)
        
        # 设置布局比例
        main_layout.setStretch(0, 1)  # 工具栏占1
        main_layout.setStretch(1, 4)  # 编辑区域占4
        
    def add_layer(self):
        """添加新层"""
        layer_types = [
            'Conv2d', 'ConvTranspose2d', 'Linear', 'MaxPool2d', 
            'AvgPool2d', 'ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh',
            'BatchNorm2d', 'Dropout', 'Flatten'
        ]
        
        dialog = QDialog(self)
        dialog.setWindowTitle("添加层")
        layout = QVBoxLayout(dialog)
        
        # 层类型选择
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("层类型:"))
        type_combo = QComboBox()
        type_combo.addItems(layer_types)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)
        
        # 层名称输入
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("层名称:"))
        name_edit = QLineEdit()
        name_edit.setText(f"layer_{len(self.layers)}")
        name_layout.addWidget(name_edit)
        layout.addLayout(name_layout)
        
        # 按钮
        button_layout = QHBoxLayout()
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(dialog.accept)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        if dialog.exec_() == QDialog.Accepted:
            layer_info = {
                'name': name_edit.text(),
                'type': type_combo.currentText(),
                'position': {'x': 0, 'y': 0}  # 初始位置
            }
            
            # 添加到层列表
            self.layers.append(layer_info)
            
            # 添加到场景
            self.scene.add_layer(layer_info)
            
    def add_connection(self):
        """添加层之间的连接"""
        if len(self.layers) < 2:
            QMessageBox.warning(self, "警告", "需要至少两个层才能创建连接")
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle("添加连接")
        layout = QVBoxLayout(dialog)
        
        # 源层选择
        from_layout = QHBoxLayout()
        from_layout.addWidget(QLabel("从:"))
        from_combo = QComboBox()
        from_combo.addItems([layer['name'] for layer in self.layers])
        from_layout.addWidget(from_combo)
        layout.addLayout(from_layout)
        
        # 目标层选择
        to_layout = QHBoxLayout()
        to_layout.addWidget(QLabel("到:"))
        to_combo = QComboBox()
        to_combo.addItems([layer['name'] for layer in self.layers])
        to_layout.addWidget(to_combo)
        layout.addLayout(to_layout)
        
        # 按钮
        button_layout = QHBoxLayout()
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(dialog.accept)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        if dialog.exec_() == QDialog.Accepted:
            from_layer = from_combo.currentText()
            to_layer = to_combo.currentText()
            
            # 检查连接是否已存在
            for conn in self.connections:
                if conn['from'] == from_layer and conn['to'] == to_layer:
                    QMessageBox.warning(self, "警告", "该连接已存在")
                    return
            
            # 添加连接
            connection = {
                'from': from_layer,
                'to': to_layer
            }
            self.connections.append(connection)
            
            # 添加到场景
            self.scene.add_connection(from_layer, to_layer)
            
    def clear_all(self):
        """清除所有层和连接"""
        reply = QMessageBox.question(self, '确认清除', 
                                   '确定要清除所有层和连接吗？',
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            # 清除数据
            self.layers.clear()
            self.connections.clear()
            self.selected_layer = None
            
            # 清除场景
            self.scene.clear_all()
                    
    def import_structure(self):
        """导入模型结构"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "导入模型结构", "", "JSON文件 (*.json)")
            
        if file_name:
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 清除现有内容
                self.clear_all()
                
                # 加载层
                for layer_info in data.get('layers', []):
                    self.layers.append(layer_info)
                    
                    # 如果没有位置信息，添加默认位置
                    if 'position' not in layer_info:
                        layer_info['position'] = {'x': 0, 'y': 0}
                        
                    # 添加到场景
                    pos = QPointF(layer_info['position']['x'], layer_info['position']['y'])
                    self.scene.add_layer(layer_info, pos)
                    
                # 加载连接
                for connection in data.get('connections', []):
                    self.connections.append(connection)
                    
                    # 添加到场景
                    self.scene.add_connection(connection['from'], connection['to'])
                    
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导入失败: {str(e)}")
                
    def export_structure(self):
        """导出模型结构"""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "导出模型结构", "", "JSON文件 (*.json)")
            
        if file_name:
            try:
                # 更新层的位置信息
                for layer in self.layers:
                    layer_item = self.scene.layer_items.get(layer['name'])
                    if layer_item:
                        pos = layer_item.pos()
                        layer['position'] = {'x': pos.x(), 'y': pos.y()}
                
                data = {
                    'layers': self.layers,
                    'connections': self.connections
                }
                with open(file_name, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                QMessageBox.information(self, "成功", "模型结构已成功导出")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
                
    def on_layer_selected(self, layer_info):
        """处理层选中事件"""
        self.selected_layer = layer_info
        
    def update_layer_position(self, layer_name, pos):
        """更新层位置信息"""
        for layer in self.layers:
            if layer['name'] == layer_name:
                layer['position'] = {'x': pos.x(), 'y': pos.y()}
                break
                
    def edit_layer_parameters(self, layer_info):
        """编辑层参数"""
        dialog = LayerParameterDialog(layer_info, self)
        if dialog.exec_() == QDialog.Accepted:
            # 更新层参数
            updated_info = dialog.get_parameters()
            
            # 查找并更新层信息
            for i, layer in enumerate(self.layers):
                if layer['name'] == layer_info['name']:
                    self.layers[i].update(updated_info)
                    
                    # 更新图形项
                    layer_item = self.scene.layer_items.get(layer_info['name'])
                    if layer_item:
                        layer_item.layer_info.update(updated_info)
                        layer_item.update_style()
                        layer_item.update()
                    break
        
    def delete_layer(self, layer_name):
        """删除层"""
        reply = QMessageBox.question(self, '确认删除', 
                                   f'确定要删除层 {layer_name} 吗？',
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            # 从层列表删除
            self.layers = [layer for layer in self.layers if layer['name'] != layer_name]
            
            # 从连接列表删除相关连接
            self.connections = [conn for conn in self.connections 
                              if conn['from'] != layer_name and conn['to'] != layer_name]
            
            # 从场景删除
            self.scene.remove_layer(layer_name)
            
    def get_model_structure(self):
        """获取模型结构定义"""
        # 更新层的位置信息
        for layer in self.layers:
            layer_item = self.scene.layer_items.get(layer['name'])
            if layer_item:
                pos = layer_item.pos()
                layer['position'] = {'x': pos.x(), 'y': pos.y()}
                
        return {
            'layers': self.layers,
            'connections': self.connections
        } 