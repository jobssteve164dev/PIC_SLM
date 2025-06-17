from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QScrollArea, QSizePolicy, QSlider)
from PyQt5.QtCore import Qt, QPoint, QRectF, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QCursor, QPainter, QPen, QColor, QBrush


class ZoomableImageViewer(QWidget):
    """可缩放的图像查看器组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.pixmap = None
        self.original_pixmap = None
        self.scale_factor = 1.0
        self.min_scale = 0.1
        self.max_scale = 10.0
        self.offset_x = 0  # 图像水平偏移量
        self.offset_y = 0  # 图像垂直偏移量
        
        # 平移相关
        self.panning = False
        self.pan_start_point = None
        
        # 初始化UI
        self.init_ui()
        
        # 设置鼠标追踪
        self.setMouseTracking(True)
        
        # 设置焦点策略
        self.setFocusPolicy(Qt.StrongFocus)
        
        # 设置最小尺寸
        self.setMinimumSize(300, 300)
        
        # 设置背景色
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.darkGray)
        self.setPalette(palette)
        
        # 设置属性，使其能够接收鼠标事件
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建工具栏
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(5, 5, 5, 5)
        
        # 缩放按钮
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setToolTip("放大")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_in_btn.setFixedSize(30, 30)
        
        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setToolTip("缩小")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.zoom_out_btn.setFixedSize(30, 30)
        
        self.reset_btn = QPushButton("重置")
        self.reset_btn.setToolTip("重置视图")
        self.reset_btn.clicked.connect(self.reset_view)
        self.reset_btn.setFixedSize(50, 30)
        
        # 移动模式按钮
        self.pan_btn = QPushButton("移动")
        self.pan_btn.setToolTip("切换移动模式")
        self.pan_btn.setCheckable(True)
        self.pan_btn.clicked.connect(self.toggle_pan_mode)
        self.pan_btn.setFixedSize(50, 30)
        
        # 缩放滑块
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(int(self.min_scale * 100), int(self.max_scale * 100))
        self.zoom_slider.setValue(int(self.scale_factor * 100))
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.setTickInterval(50)
        self.zoom_slider.valueChanged.connect(self.on_slider_changed)
        
        # 显示缩放比例的标签
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(50)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        
        # 添加到工具栏
        toolbar.addWidget(self.zoom_in_btn)
        toolbar.addWidget(self.zoom_out_btn)
        toolbar.addWidget(self.reset_btn)
        toolbar.addWidget(self.pan_btn)
        toolbar.addWidget(self.zoom_slider)
        toolbar.addWidget(self.zoom_label)
        toolbar.addStretch()
        
        layout.addLayout(toolbar)
        
        # 主内容区域占据大部分空间
        layout.addStretch(1)
        
    def set_image(self, pixmap):
        """设置图像"""
        if pixmap and not pixmap.isNull():
            self.pixmap = pixmap
            self.original_pixmap = pixmap
            
            # 重置状态
            self.reset_view()
            
            # 更新界面
            self.update()
        else:
            self.pixmap = None
            self.original_pixmap = None
            self.update()
    
    def reset_view(self):
        """重置视图"""
        if self.pixmap and not self.pixmap.isNull():
            # 计算初始缩放因子以适应窗口
            if self.width() > 0 and self.height() > 0:
                # 计算宽度和高度的缩放比例
                width_ratio = (self.width() - 20) / self.pixmap.width()
                height_ratio = (self.height() - 80) / self.pixmap.height()
                
                # 使用较小的缩放比例，确保图像完全显示在窗口中
                self.scale_factor = min(width_ratio, height_ratio, 1.0)  # 最大不超过原始大小
            else:
                self.scale_factor = 1.0
            
            # 更新滑块值
            self.zoom_slider.setValue(int(self.scale_factor * 100))
            self.zoom_label.setText(f"{int(self.scale_factor * 100)}%")
            
            # 计算居中位置
            self.center_image()
            
            # 更新界面
            self.update()
    
    def center_image(self):
        """居中显示图像"""
        if self.pixmap and not self.pixmap.isNull():
            scaled_width = int(self.pixmap.width() * self.scale_factor)
            scaled_height = int(self.pixmap.height() * self.scale_factor)
            
            self.offset_x = int((self.width() - scaled_width) / 2)
            self.offset_y = int((self.height() - scaled_height) / 2)
    
    def zoom_in(self):
        """放大图像"""
        if self.pixmap and not self.pixmap.isNull():
            if self.scale_factor < self.max_scale:
                self.scale_factor = min(self.scale_factor * 1.25, self.max_scale)
                self.zoom_slider.setValue(int(self.scale_factor * 100))
                self.zoom_label.setText(f"{int(self.scale_factor * 100)}%")
                self.update()
    
    def zoom_out(self):
        """缩小图像"""
        if self.pixmap and not self.pixmap.isNull():
            if self.scale_factor > self.min_scale:
                self.scale_factor = max(self.scale_factor / 1.25, self.min_scale)
                self.zoom_slider.setValue(int(self.scale_factor * 100))
                self.zoom_label.setText(f"{int(self.scale_factor * 100)}%")
                self.update()
    
    def on_slider_changed(self, value):
        """滑块值变化处理"""
        if self.pixmap and not self.pixmap.isNull():
            self.scale_factor = value / 100
            self.zoom_label.setText(f"{value}%")
            self.update()
    
    def toggle_pan_mode(self):
        """切换平移模式"""
        if self.pan_btn.isChecked():
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景
        painter.fillRect(self.rect(), Qt.darkGray)
        
        # 绘制图像
        if self.pixmap and not self.pixmap.isNull():
            # 计算缩放后的图像尺寸
            scaled_width = int(self.pixmap.width() * self.scale_factor)
            scaled_height = int(self.pixmap.height() * self.scale_factor)
            
            # 绘制图像
            painter.drawPixmap(
                self.offset_x,
                self.offset_y,
                scaled_width,
                scaled_height,
                self.pixmap
            )
            
            # 绘制图像边框
            pen = QPen(QColor(200, 200, 200), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(
                self.offset_x,
                self.offset_y,
                scaled_width,
                scaled_height
            )
        else:
            # 没有图像时显示提示
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "无图像")
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            if self.pan_btn.isChecked() or event.modifiers() & Qt.ControlModifier:
                self.panning = True
                self.pan_start_point = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.panning and self.pan_start_point:
            # 计算移动距离
            delta = event.pos() - self.pan_start_point
            
            # 更新偏移量
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            
            # 更新起始位置
            self.pan_start_point = event.pos()
            
            # 更新界面
            self.update()
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton and self.panning:
            self.panning = False
            if self.pan_btn.isChecked():
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)
    
    def wheelEvent(self, event):
        """鼠标滚轮事件"""
        if self.pixmap and not self.pixmap.isNull():
            # 获取当前鼠标位置
            pos = event.pos()
            
            # 计算鼠标位置相对于图像的偏移比例
            img_rect = QRectF(self.offset_x, self.offset_y, 
                             self.pixmap.width() * self.scale_factor, 
                             self.pixmap.height() * self.scale_factor)
            
            if img_rect.contains(pos):
                # 计算鼠标在图像上的相对位置(0-1)
                rel_x = (pos.x() - self.offset_x) / (self.pixmap.width() * self.scale_factor)
                rel_y = (pos.y() - self.offset_y) / (self.pixmap.height() * self.scale_factor)
                
                # 获取滚轮增量
                delta = event.angleDelta().y()
                
                # 保存旧的缩放因子
                old_scale = self.scale_factor
                
                # 根据滚轮方向放大或缩小
                if delta > 0:
                    self.scale_factor = min(self.scale_factor * 1.1, self.max_scale)
                else:
                    self.scale_factor = max(self.scale_factor / 1.1, self.min_scale)
                
                # 更新滑块值
                self.zoom_slider.setValue(int(self.scale_factor * 100))
                self.zoom_label.setText(f"{int(self.scale_factor * 100)}%")
                
                # 调整偏移量，使鼠标指向的点保持在同一位置
                scale_change = self.scale_factor / old_scale
                new_img_width = self.pixmap.width() * self.scale_factor
                new_img_height = self.pixmap.height() * self.scale_factor
                
                self.offset_x = pos.x() - rel_x * new_img_width
                self.offset_y = pos.y() - rel_y * new_img_height
                
                # 更新界面
                self.update()
            
            event.accept()
    
    def resizeEvent(self, event):
        """窗口大小变化事件"""
        if self.pixmap and not self.pixmap.isNull():
            # 如果是首次显示或重置后，居中图像
            if self.offset_x == 0 and self.offset_y == 0:
                self.center_image()
        super().resizeEvent(event)
    
    def keyPressEvent(self, event):
        """键盘按键事件"""
        if event.key() == Qt.Key_Space:
            # 空格键临时切换到平移模式
            self.setCursor(Qt.OpenHandCursor)
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            # 加号键放大
            self.zoom_in()
        elif event.key() == Qt.Key_Minus:
            # 减号键缩小
            self.zoom_out()
        elif event.key() == Qt.Key_0:
            # 0键重置视图
            self.reset_view()
        super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        """键盘释放事件"""
        if event.key() == Qt.Key_Space:
            # 空格键释放，恢复光标
            if self.pan_btn.isChecked():
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        super().keyReleaseEvent(event) 