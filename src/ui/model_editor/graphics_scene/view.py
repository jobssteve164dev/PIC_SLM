"""
模型网络图形视图
"""
from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter
from ..utils.constants import MIN_ZOOM_FACTOR, MAX_ZOOM_FACTOR, DEFAULT_ZOOM_FACTOR


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
        
        # 支持滚动条，并设置为按需显示
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 设置场景矩形为一个极大的区域，以支持超大型模型结构
        self.setSceneRect(-100000, -100000, 200000, 200000)
        
        # 初始化变量
        self.zoom_factor = DEFAULT_ZOOM_FACTOR
        self.is_panning = False
        self.last_pan_pos = None
    
    def resizeEvent(self, event):
        """窗口大小变化事件"""
        super().resizeEvent(event)
    
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
        
        # 限制缩放范围 - 扩大范围以适应大型模型
        if self.zoom_factor < MIN_ZOOM_FACTOR:
            # 恢复到最小缩放
            reset_factor = MIN_ZOOM_FACTOR / self.zoom_factor
            self.scale(reset_factor, reset_factor)
            self.zoom_factor = MIN_ZOOM_FACTOR
        elif self.zoom_factor > MAX_ZOOM_FACTOR:
            # 恢复到最大缩放
            reset_factor = MAX_ZOOM_FACTOR / self.zoom_factor
            self.scale(reset_factor, reset_factor)
            self.zoom_factor = MAX_ZOOM_FACTOR
    
    def zoom_in(self):
        """放大视图"""
        scale_factor = 1.2
        self.scale(scale_factor, scale_factor)
        self.zoom_factor *= scale_factor
        
        # 限制最大缩放
        if self.zoom_factor > MAX_ZOOM_FACTOR:
            reset_factor = MAX_ZOOM_FACTOR / self.zoom_factor
            self.scale(reset_factor, reset_factor)
            self.zoom_factor = MAX_ZOOM_FACTOR
    
    def zoom_out(self):
        """缩小视图"""
        scale_factor = 1.0 / 1.2
        self.scale(scale_factor, scale_factor)
        self.zoom_factor *= scale_factor
        
        # 限制最小缩放
        if self.zoom_factor < MIN_ZOOM_FACTOR:
            reset_factor = MIN_ZOOM_FACTOR / self.zoom_factor
            self.scale(reset_factor, reset_factor)
            self.zoom_factor = MIN_ZOOM_FACTOR
    
    def reset_view(self):
        """重置视图"""
        self.resetTransform()
        self.zoom_factor = DEFAULT_ZOOM_FACTOR
        self.centerOn(0, 0)
      
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
            
            # 修改平移处理，确保可以平移到场景边界
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            
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
            self.zoom_factor = DEFAULT_ZOOM_FACTOR
            # 重置视图位置到中心
            self.centerOn(0, 0)
            event.accept()
        else:
            super().keyPressEvent(event) 