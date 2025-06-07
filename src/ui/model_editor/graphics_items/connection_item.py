"""
连接图形项组件
"""
from PyQt5.QtWidgets import QGraphicsPathItem
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QPolygonF


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
        
        # 连接线从底部中心到顶部中心
        from_pos = self.from_item.pos() + QPointF(from_rect.width() / 2, from_rect.height())
        to_pos = self.to_item.pos() + QPointF(to_rect.width() / 2, 0)
        
        # 创建路径
        path = QPainterPath()
        path.moveTo(from_pos)
        
        # 计算控制点 - 使曲线更平滑
        control_dist = min(80, (to_pos.y() - from_pos.y()) * 0.5)
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