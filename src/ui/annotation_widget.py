from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QLabel, QListWidget, QListWidgetItem, QScrollArea, 
                           QFrame, QSplitter, QFileDialog, QMessageBox, QMenu,
                           QAction, QInputDialog, QComboBox, QShortcut)
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF, QSizeF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush, QCursor, QKeySequence
import os
import glob
import xml.etree.ElementTree as ET
from datetime import datetime

class AnnotationCanvas(QWidget):
    """图像标注画布"""
    
    # 定义信号
    box_created = pyqtSignal(QRectF, str)  # 创建了新的标注框
    box_selected = pyqtSignal(int)  # 选中了标注框
    box_modified = pyqtSignal(int, QRectF)  # 修改了标注框
    box_deleted = pyqtSignal(int)  # 删除了标注框
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.image_path = None
        self.scale_factor = 1.0
        self.offset_x = 0  # 图像水平偏移量
        self.offset_y = 0  # 图像垂直偏移量
        self.boxes = []  # [(rect, label, color), ...]
        self.deleted_boxes = []  # 存储被删除的标注框，用于撤销操作
        self.operation_history = []  # 操作历史记录 [(操作类型, 数据), ...]
        self.current_label = None
        
        # 绘制状态
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.selected_box = -1
        self.dragging = False
        self.drag_start_point = None
        self.original_rect = None
        self.panning = False  # 是否正在平移图像
        self.pan_start_point = None  # 平移起始点
        
        # 设置鼠标追踪
        self.setMouseTracking(True)
        
        # 设置焦点策略
        self.setFocusPolicy(Qt.StrongFocus)
        
        # 设置最小尺寸
        self.setMinimumSize(600, 400)
        
        # 设置背景色
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.darkGray)
        self.setPalette(palette)
        
        # 设置属性，使其能够接收鼠标事件
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        
        # 打印调试信息
        print("AnnotationCanvas初始化完成")
        
    def set_image(self, image_path):
        """设置要标注的图像"""
        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            return False
            
        self.image_path = image_path
        print(f"正在加载图像: {image_path}")
        self.pixmap = QPixmap(image_path)
        if self.pixmap.isNull():
            print(f"加载图像失败: {image_path}")
            return False
            
        print(f"图像加载成功: {image_path}, 尺寸: {self.pixmap.width()}x{self.pixmap.height()}")
        
        # 重置状态
        self.boxes = []
        self.selected_box = -1
        self.scale_factor = 1.0
        
        # 更新界面
        self.update()
        return True
        
    def set_current_label(self, label):
        """设置当前标注标签"""
        self.current_label = label
        
    def add_box(self, rect, label):
        """添加标注框"""
        # 生成随机颜色
        color = QColor(
            hash(label) % 256,
            (hash(label) * 2) % 256,
            (hash(label) * 3) % 256,
            128
        )
        # 记录添加操作
        self.operation_history.append(('add', len(self.boxes)))
        self.boxes.append((rect, label, color))
        self.update()
        
    def clear_boxes(self):
        """清除所有标注框"""
        self.boxes = []
        self.selected_box = -1
        self.update()
        
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景
        painter.fillRect(self.rect(), Qt.darkGray)
        
        # 绘制图像
        if self.pixmap and not self.pixmap.isNull():
            # 计算缩放后的图像尺寸
            scaled_size = self.pixmap.size() * self.scale_factor
            
            # 计算居中位置（考虑偏移量）
            x = (self.width() - scaled_size.width()) / 2 + self.offset_x
            y = (self.height() - scaled_size.height()) / 2 + self.offset_y
            
            # 绘制图像
            painter.drawPixmap(int(x), int(y), int(scaled_size.width()), int(scaled_size.height()), self.pixmap)
            
            # 绘制已有的标注框
            for i, (rect, label, color) in enumerate(self.boxes):
                # 转换为屏幕坐标
                screen_rect = QRectF(
                    x + rect.x() * self.scale_factor,
                    y + rect.y() * self.scale_factor,
                    rect.width() * self.scale_factor,
                    rect.height() * self.scale_factor
                )
                
                # 设置画笔和画刷
                if i == self.selected_box:
                    pen = QPen(Qt.red, 2, Qt.SolidLine)
                else:
                    pen = QPen(color, 2, Qt.SolidLine)
                    
                brush = QBrush(color)
                
                painter.setPen(pen)
                painter.setBrush(brush)
                painter.drawRect(screen_rect)
                
                # 绘制标签
                painter.setPen(Qt.black)
                painter.drawText(screen_rect.topLeft() + QPointF(0, -5), label)
            
            # 绘制正在创建的标注框
            if self.drawing and self.start_point and self.end_point:
                pen = QPen(Qt.green, 2, Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                
                rect = QRectF(self.start_point, self.end_point).normalized()
                painter.drawRect(rect)
                
            # 绘制十字箭头控件
            self.draw_pan_controls(painter)
        else:
            # 没有图像时显示提示
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "请加载图像")
            
    def draw_pan_controls(self, painter):
        """绘制平移控制箭头"""
        # 设置画笔
        pen = QPen(Qt.white, 2, Qt.SolidLine)
        painter.setPen(pen)
        
        # 计算控件位置（右下角）
        control_size = 80
        control_x = self.width() - control_size - 10
        control_y = self.height() - control_size - 10
        
        # 绘制外圆
        painter.setBrush(QBrush(QColor(0, 0, 0, 128)))
        painter.drawEllipse(control_x, control_y, control_size, control_size)
        
        # 绘制十字
        center_x = control_x + control_size / 2
        center_y = control_y + control_size / 2
        
        # 上箭头
        painter.setBrush(QBrush(Qt.white))
        points_up = [
            QPointF(center_x, center_y - control_size / 3),
            QPointF(center_x - 10, center_y - control_size / 5),
            QPointF(center_x + 10, center_y - control_size / 5)
        ]
        painter.drawPolygon(points_up)
        
        # 下箭头
        points_down = [
            QPointF(center_x, center_y + control_size / 3),
            QPointF(center_x - 10, center_y + control_size / 5),
            QPointF(center_x + 10, center_y + control_size / 5)
        ]
        painter.drawPolygon(points_down)
        
        # 左箭头
        points_left = [
            QPointF(center_x - control_size / 3, center_y),
            QPointF(center_x - control_size / 5, center_y - 10),
            QPointF(center_x - control_size / 5, center_y + 10)
        ]
        painter.drawPolygon(points_left)
        
        # 右箭头
        points_right = [
            QPointF(center_x + control_size / 3, center_y),
            QPointF(center_x + control_size / 5, center_y - 10),
            QPointF(center_x + control_size / 5, center_y + 10)
        ]
        painter.drawPolygon(points_right)
        
        # 绘制中心圆
        painter.setBrush(QBrush(QColor(255, 255, 255, 128)))
        painter.drawEllipse(center_x - 5, center_y - 5, 10, 10)
        
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if not self.pixmap or self.pixmap.isNull():
            return
            
        # 检查是否点击了平移控件
        if self.is_pan_control_clicked(event.pos()):
            self.handle_pan_control_click(event.pos())
            return
            
        if event.button() == Qt.LeftButton:
            # 检查是否点击了已有的标注框
            clicked_box = self.box_at_position(event.pos())
            
            if clicked_box >= 0:
                # 选中标注框
                self.selected_box = clicked_box
                self.box_selected.emit(clicked_box)
                self.update()
            else:
                # 开始绘制新标注框
                if self.current_label:
                    self.drawing = True
                    self.start_point = event.pos()
                    self.end_point = event.pos()
                    
        elif event.button() == Qt.RightButton:
            # 检查是否点击了已有的标注框
            clicked_box = self.box_at_position(event.pos())
            
            if clicked_box >= 0:
                # 选中标注框并准备拖动
                self.selected_box = clicked_box
                self.box_selected.emit(clicked_box)
                self.dragging = True
                self.drag_start_point = event.pos()
                self.original_rect = self.boxes[clicked_box][0]
                self.update()
            else:
                # 开始平移图像
                self.panning = True
                self.pan_start_point = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
                
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if not self.pixmap or self.pixmap.isNull():
            return
            
        # 检查是否在平移控件上悬停
        if self.is_pan_control_hovered(event.pos()):
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
            
        if self.drawing:
            # 更新绘制中的标注框
            self.end_point = event.pos()
            self.update()
        elif self.dragging and self.selected_box >= 0:
            # 拖动标注框
            delta = event.pos() - self.drag_start_point
            
            # 计算图像坐标系中的偏移量
            x = (self.width() - self.pixmap.width() * self.scale_factor) / 2 + self.offset_x
            y = (self.height() - self.pixmap.height() * self.scale_factor) / 2 + self.offset_y
            
            delta_x = delta.x() / self.scale_factor
            delta_y = delta.y() / self.scale_factor
            
            # 更新标注框位置
            new_rect = QRectF(
                self.original_rect.x() + delta_x,
                self.original_rect.y() + delta_y,
                self.original_rect.width(),
                self.original_rect.height()
            )
            
            # 确保标注框不超出图像范围
            if (new_rect.left() >= 0 and new_rect.top() >= 0 and
                new_rect.right() <= self.pixmap.width() and
                new_rect.bottom() <= self.pixmap.height()):
                
                # 更新标注框
                label, color = self.boxes[self.selected_box][1:]
                self.boxes[self.selected_box] = (new_rect, label, color)
                
                # 发送信号
                self.box_modified.emit(self.selected_box, new_rect)
                
                self.update()
        elif self.panning:
            # 平移图像
            delta = event.pos() - self.pan_start_point
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self.pan_start_point = event.pos()
            self.update()
            
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if not self.pixmap or self.pixmap.isNull():
            return
            
        if event.button() == Qt.LeftButton and self.drawing:
            # 完成绘制标注框
            self.drawing = False
            
            if self.start_point and self.end_point:
                # 计算图像坐标系中的矩形
                x = (self.width() - self.pixmap.width() * self.scale_factor) / 2 + self.offset_x
                y = (self.height() - self.pixmap.height() * self.scale_factor) / 2 + self.offset_y
                
                screen_rect = QRectF(self.start_point, self.end_point).normalized()
                
                # 转换为图像坐标
                image_rect = QRectF(
                    (screen_rect.x() - x) / self.scale_factor,
                    (screen_rect.y() - y) / self.scale_factor,
                    screen_rect.width() / self.scale_factor,
                    screen_rect.height() / self.scale_factor
                )
                
                # 确保标注框不超出图像范围
                image_rect = image_rect.intersected(QRectF(0, 0, self.pixmap.width(), self.pixmap.height()))
                
                # 检查标注框是否有效
                if image_rect.width() > 5 and image_rect.height() > 5:
                    # 添加标注框
                    self.add_box(image_rect, self.current_label)
                    
                    # 发送信号
                    self.box_created.emit(image_rect, self.current_label)
                
            self.start_point = None
            self.end_point = None
            
        elif event.button() == Qt.RightButton:
            if self.dragging:
                # 完成拖动标注框
                self.dragging = False
                
                # 记录修改操作
                if self.selected_box >= 0:
                    self.operation_history.append(('modify', self.selected_box, self.original_rect))
                    
                self.drag_start_point = None
                self.original_rect = None
            elif self.panning:
                # 完成平移图像
                self.panning = False
                self.pan_start_point = None
                self.setCursor(Qt.ArrowCursor)
                
    def is_pan_control_clicked(self, pos):
        """检查是否点击了平移控件"""
        # 计算控件位置
        control_size = 80
        control_x = self.width() - control_size - 10
        control_y = self.height() - control_size - 10
        
        # 计算中心点
        center_x = control_x + control_size / 2
        center_y = control_y + control_size / 2
        
        # 计算点击位置到中心的距离
        dx = pos.x() - center_x
        dy = pos.y() - center_y
        distance = (dx * dx + dy * dy) ** 0.5
        
        # 如果距离小于控件半径，则认为点击了控件
        return distance <= control_size / 2
        
    def is_pan_control_hovered(self, pos):
        """检查鼠标是否悬停在平移控件上"""
        return self.is_pan_control_clicked(pos)
        
    def handle_pan_control_click(self, pos):
        """处理平移控件点击事件"""
        # 计算控件位置
        control_size = 80
        control_x = self.width() - control_size - 10
        control_y = self.height() - control_size - 10
        
        # 计算中心点
        center_x = control_x + control_size / 2
        center_y = control_y + control_size / 2
        
        # 计算点击位置相对于中心的偏移
        dx = pos.x() - center_x
        dy = pos.y() - center_y
        
        # 根据偏移方向确定平移方向
        if abs(dx) > abs(dy):
            # 水平方向
            if dx > 0:
                # 向右平移
                self.offset_x -= 20
            else:
                # 向左平移
                self.offset_x += 20
        else:
            # 垂直方向
            if dy > 0:
                # 向下平移
                self.offset_y -= 20
            else:
                # 向上平移
                self.offset_y += 20
                
        # 更新界面
        self.update()
        
    def reset_view(self):
        """重置视图"""
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.update()
        
    def wheelEvent(self, event):
        """鼠标滚轮事件"""
        if not self.pixmap or self.pixmap.isNull():
            return
            
        # 获取鼠标位置
        mouse_pos = event.pos()
        
        # 计算鼠标位置相对于图像的偏移
        x = (self.width() - self.pixmap.width() * self.scale_factor) / 2 + self.offset_x
        y = (self.height() - self.pixmap.height() * self.scale_factor) / 2 + self.offset_y
        
        # 计算鼠标在图像上的位置（相对于图像左上角）
        image_x = (mouse_pos.x() - x) / self.scale_factor
        image_y = (mouse_pos.y() - y) / self.scale_factor
        
        # 计算缩放因子
        old_scale = self.scale_factor
        delta = event.angleDelta().y()
        if delta > 0:
            # 放大
            self.scale_factor *= 1.1
        else:
            # 缩小
            self.scale_factor *= 0.9
            
        # 限制缩放范围
        self.scale_factor = max(0.1, min(5.0, self.scale_factor))
        
        # 调整偏移量，使鼠标位置保持不变
        scale_change = self.scale_factor / old_scale
        self.offset_x = mouse_pos.x() - (image_x * self.scale_factor + (self.width() - self.pixmap.width() * self.scale_factor) / 2)
        self.offset_y = mouse_pos.y() - (image_y * self.scale_factor + (self.height() - self.pixmap.height() * self.scale_factor) / 2)
        
        self.update()
        
    def keyPressEvent(self, event):
        """键盘按键事件"""
        if event.key() == Qt.Key_Delete and self.selected_box >= 0:
            # 删除选中的标注框
            deleted_box = self.boxes.pop(self.selected_box)
            # 记录删除操作
            self.operation_history.append(('delete', self.selected_box, deleted_box))
            self.box_deleted.emit(self.selected_box)
            self.selected_box = -1
            self.update()
        elif event.key() == Qt.Key_Escape:
            # 取消绘制或选择
            if self.drawing:
                self.drawing = False
                self.start_point = None
                self.end_point = None
            else:
                self.selected_box = -1
            self.update()
        elif event.key() == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier):
            # Ctrl+Z: 撤销
            self.undo()
        elif event.key() == Qt.Key_R:
            # R: 重置视图
            self.reset_view()
        elif event.key() == Qt.Key_Left:
            # 左箭头: 向左平移
            self.offset_x += 20
            self.update()
        elif event.key() == Qt.Key_Right:
            # 右箭头: 向右平移
            self.offset_x -= 20
            self.update()
        elif event.key() == Qt.Key_Up:
            # 上箭头: 向上平移
            self.offset_y += 20
            self.update()
        elif event.key() == Qt.Key_Down:
            # 下箭头: 向下平移
            self.offset_y -= 20
            self.update()
        
    def box_at_position(self, pos):
        """获取指定位置的标注框索引"""
        if not self.pixmap or self.pixmap.isNull():
            return -1
            
        # 计算图像位置
        x = (self.width() - self.pixmap.width() * self.scale_factor) / 2 + self.offset_x
        y = (self.height() - self.pixmap.height() * self.scale_factor) / 2 + self.offset_y
        
        # 检查每个标注框
        for i, (rect, _, _) in enumerate(self.boxes):
            # 转换为屏幕坐标
            screen_rect = QRectF(
                x + rect.x() * self.scale_factor,
                y + rect.y() * self.scale_factor,
                rect.width() * self.scale_factor,
                rect.height() * self.scale_factor
            )
            
            if screen_rect.contains(pos):
                return i
                
        return -1
        
    def save_annotations(self, output_dir, format_type='voc'):
        """保存标注结果"""
        if not self.pixmap or self.pixmap.isNull() or not self.image_path:
            return False
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        if format_type.lower() == 'voc':
            # 保存为VOC格式
            return self._save_as_voc(output_dir)
        elif format_type.lower() == 'yolo':
            # 保存为YOLO格式
            return self._save_as_yolo(output_dir)
        else:
            return False
            
    def _save_as_voc(self, output_dir):
        """保存为VOC格式"""
        try:
            # 创建XML根元素
            root = ET.Element('annotation')
            
            # 添加基本信息
            folder = ET.SubElement(root, 'folder')
            folder.text = os.path.basename(os.path.dirname(self.image_path))
            
            filename = ET.SubElement(root, 'filename')
            filename.text = os.path.basename(self.image_path)
            
            path = ET.SubElement(root, 'path')
            path.text = self.image_path
            
            source = ET.SubElement(root, 'source')
            database = ET.SubElement(source, 'database')
            database.text = 'Unknown'
            
            # 添加图像尺寸信息
            size = ET.SubElement(root, 'size')
            width = ET.SubElement(size, 'width')
            width.text = str(self.pixmap.width())
            height = ET.SubElement(size, 'height')
            height.text = str(self.pixmap.height())
            depth = ET.SubElement(size, 'depth')
            depth.text = '3'  # 假设是RGB图像
            
            segmented = ET.SubElement(root, 'segmented')
            segmented.text = '0'
            
            # 添加标注框信息
            for rect, label, _ in self.boxes:
                obj = ET.SubElement(root, 'object')
                
                name = ET.SubElement(obj, 'name')
                name.text = label
                
                pose = ET.SubElement(obj, 'pose')
                pose.text = 'Unspecified'
                
                truncated = ET.SubElement(obj, 'truncated')
                truncated.text = '0'
                
                difficult = ET.SubElement(obj, 'difficult')
                difficult.text = '0'
                
                bndbox = ET.SubElement(obj, 'bndbox')
                
                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = str(int(rect.x()))
                
                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = str(int(rect.y()))
                
                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = str(int(rect.x() + rect.width()))
                
                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = str(int(rect.y() + rect.height()))
                
            # 生成XML文件名
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            xml_path = os.path.join(output_dir, f"{base_name}.xml")
            
            # 写入XML文件
            tree = ET.ElementTree(root)
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            
            return True
            
        except Exception as e:
            print(f"保存VOC格式标注失败: {str(e)}")
            return False
            
    def _save_as_yolo(self, output_dir):
        """保存为YOLO格式"""
        try:
            # 获取所有标签
            labels = set()
            for _, label, _ in self.boxes:
                labels.add(label)
                
            # 创建标签映射
            label_map = {label: i for i, label in enumerate(sorted(labels))}
            
            # 生成标注文件名
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            txt_path = os.path.join(output_dir, f"{base_name}.txt")
            
            # 写入标注文件
            with open(txt_path, 'w') as f:
                for rect, label, _ in self.boxes:
                    # 计算归一化坐标
                    x_center = (rect.x() + rect.width() / 2) / self.pixmap.width()
                    y_center = (rect.y() + rect.height() / 2) / self.pixmap.height()
                    width = rect.width() / self.pixmap.width()
                    height = rect.height() / self.pixmap.height()
                    
                    # 写入标注行
                    f.write(f"{label_map[label]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
            # 保存标签映射
            classes_path = os.path.join(output_dir, "classes.txt")
            if not os.path.exists(classes_path):
                with open(classes_path, 'w') as f:
                    for label in sorted(labels):
                        f.write(f"{label}\n")
                        
            return True
            
        except Exception as e:
            print(f"保存YOLO格式标注失败: {str(e)}")
            return False

    def undo(self):
        """撤销最近的操作"""
        if not self.operation_history:
            return False
            
        # 获取最近的操作
        operation = self.operation_history.pop()
        
        if operation[0] == 'add':
            # 撤销添加操作
            index = operation[1]
            if index < len(self.boxes):
                self.boxes.pop()
                self.update()
                return True
        elif operation[0] == 'delete':
            # 撤销删除操作
            index = operation[1]
            box = operation[2]
            self.boxes.insert(index, box)
            self.update()
            return True
        elif operation[0] == 'modify':
            # 撤销修改操作
            index = operation[1]
            original_rect = operation[2]
            if index < len(self.boxes):
                rect, label, color = self.boxes[index]
                self.boxes[index] = (original_rect, label, color)
                self.update()
                return True
                
        return False

class AnnotationWidget(QWidget):
    """图像标注工具主界面"""
    
    # 定义信号
    status_updated = pyqtSignal(str)
    annotation_completed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
        # 初始化变量
        self.image_folder = None
        self.output_folder = None
        self.image_files = []
        self.current_index = -1
        self.class_names = []
        self.annotation_format = 'voc'  # 默认VOC格式
        
        # 连接信号
        self.status_updated.connect(self.update_status_label)
        
        # 添加快捷键
        self.setup_shortcuts()
        
    def init_ui(self):
        """初始化UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 图像列表
        self.image_list_widget = QListWidget()
        self.image_list_widget.setSelectionMode(QListWidget.SingleSelection)
        left_layout.addWidget(QLabel("图像列表:"))
        left_layout.addWidget(self.image_list_widget)
        
        # 标签选择
        left_layout.addWidget(QLabel("标签选择:"))
        self.label_combo = QComboBox()
        left_layout.addWidget(self.label_combo)
        
        # 导航按钮
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一张")
        self.next_btn = QPushButton("下一张")
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        left_layout.addLayout(nav_layout)
        
        # 操作按钮
        op_layout = QHBoxLayout()
        self.save_btn = QPushButton("保存标注")
        self.undo_btn = QPushButton("撤销操作")
        self.undo_btn.setToolTip("撤销最近的删除操作")
        op_layout.addWidget(self.save_btn)
        op_layout.addWidget(self.undo_btn)
        left_layout.addLayout(op_layout)
        
        # 添加重置视图按钮
        self.reset_view_btn = QPushButton("重置视图")
        self.reset_view_btn.setToolTip("重置图像缩放和位置 (R)")
        self.reset_view_btn.clicked.connect(self.reset_view)
        left_layout.addWidget(self.reset_view_btn)
        
        # 格式选择
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("保存格式:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["VOC", "YOLO"])
        format_layout.addWidget(self.format_combo)
        left_layout.addLayout(format_layout)
        
        # 添加左侧面板到分割器
        splitter.addWidget(left_panel)
        
        # 右侧标注画布
        self.canvas = AnnotationCanvas()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.canvas)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(600)  # 设置最小宽度
        scroll_area.setMinimumHeight(400)  # 设置最小高度
        scroll_area.setAlignment(Qt.AlignCenter)  # 居中对齐
        splitter.addWidget(scroll_area)
        
        # 设置分割器比例
        splitter.setSizes([200, 800])
        
        # 状态栏
        self.status_label = QLabel("就绪")
        main_layout.addWidget(self.status_label)
        
        # 绑定事件
        self.image_list_widget.currentRowChanged.connect(self.load_image)
        self.label_combo.currentTextChanged.connect(self.canvas.set_current_label)
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.save_btn.clicked.connect(self.save_annotations)
        self.undo_btn.clicked.connect(self.undo_operation)
        self.format_combo.currentTextChanged.connect(self.set_annotation_format)
        
    def setup_shortcuts(self):
        """设置快捷键"""
        # Ctrl+Z: 撤销
        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self.undo_operation)
        
        # R: 重置视图
        reset_view_shortcut = QShortcut(QKeySequence("R"), self)
        reset_view_shortcut.activated.connect(self.reset_view)
        
    def set_image_folder(self, folder_path):
        """设置图像文件夹"""
        if not folder_path:
            print("图像文件夹路径为空")
            self.status_updated.emit("图像文件夹路径为空")
            return False
            
        # 打印调试信息
        print(f"设置图像文件夹: {folder_path}")
        self.status_updated.emit(f"设置图像文件夹: {folder_path}")
        
        if not os.path.exists(folder_path):
            print(f"图像文件夹不存在: {folder_path}")
            self.status_updated.emit(f"图像文件夹不存在: {folder_path}")
            return False
            
        self.image_folder = folder_path
        
        # 加载图像文件
        self.image_files = []
        
        # 尝试列出文件夹中的所有文件
        try:
            all_files = os.listdir(folder_path)
            print(f"文件夹中的所有文件: {all_files}")
            
            # 手动筛选图像文件
            for file in all_files:
                file_lower = file.lower()
                if (file_lower.endswith('.jpg') or file_lower.endswith('.jpeg') or 
                    file_lower.endswith('.png') or file_lower.endswith('.bmp')):
                    full_path = os.path.join(folder_path, file)
                    self.image_files.append(full_path)
                    print(f"添加图像文件: {full_path}")
        except Exception as e:
            print(f"列出文件夹内容时出错: {str(e)}")
            self.status_updated.emit(f"列出文件夹内容时出错: {str(e)}")
            
        # 如果手动筛选没有找到文件，尝试使用glob
        if not self.image_files:
            try:
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    pattern = os.path.join(folder_path, ext)
                    found_files = glob.glob(pattern)
                    print(f"搜索模式 {pattern}: 找到 {len(found_files)} 个文件")
                    self.image_files.extend(found_files)
            except Exception as e:
                print(f"使用glob搜索文件时出错: {str(e)}")
                self.status_updated.emit(f"使用glob搜索文件时出错: {str(e)}")
            
        if not self.image_files:
            print(f"在文件夹 {folder_path} 中未找到图像文件")
            self.status_updated.emit("未找到图像文件")
            return False
            
        # 更新图像列表
        try:
            self.image_list_widget.clear()
            for image_file in self.image_files:
                self.image_list_widget.addItem(os.path.basename(image_file))
                
            # 设置当前索引
            self.current_index = 0
            
            # 设置当前行（这可能会触发currentRowChanged信号，从而调用load_image方法）
            self.image_list_widget.setCurrentRow(self.current_index)
            
            # 为确保图像被加载，显式调用load_image方法
            self.load_image(self.current_index)
            
            self.status_updated.emit(f"已加载 {len(self.image_files)} 张图像")
            print(f"已加载 {len(self.image_files)} 张图像")
            return True
        except Exception as e:
            print(f"更新图像列表时出错: {str(e)}")
            self.status_updated.emit(f"更新图像列表时出错: {str(e)}")
            return False
        
    def set_output_folder(self, folder_path):
        """设置输出文件夹"""
        self.output_folder = folder_path
        os.makedirs(folder_path, exist_ok=True)
        self.status_updated.emit(f"标注结果将保存到: {folder_path}")
        return True
        
    def set_class_names(self, class_names):
        """设置类别名称"""
        self.class_names = class_names
        
        # 更新标签下拉框
        self.label_combo.clear()
        self.label_combo.addItems(class_names)
        
        if class_names:
            self.canvas.set_current_label(class_names[0])
            
        self.status_updated.emit(f"已加载 {len(class_names)} 个类别")
        return True
        
    def set_annotation_format(self, format_type):
        """设置标注格式"""
        self.annotation_format = format_type.lower()
        self.status_updated.emit(f"标注格式已设置为: {format_type}")
        
    def load_image(self, index):
        """加载指定索引的图像"""
        print(f"尝试加载图像，索引: {index}, 图像文件数量: {len(self.image_files)}")
        
        if index < 0 or index >= len(self.image_files):
            print(f"索引超出范围: {index}")
            return
            
        self.current_index = index
        image_path = self.image_files[index]
        print(f"加载图像: {image_path}")
        
        # 确保图像文件存在
        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            self.status_updated.emit(f"图像文件不存在: {os.path.basename(image_path)}")
            return
            
        # 尝试加载图像
        if self.canvas.set_image(image_path):
            print(f"图像加载成功: {image_path}")
            self.status_updated.emit(f"已加载图像: {os.path.basename(image_path)}")
            
            # 强制更新画布
            self.canvas.update()
        else:
            print(f"图像加载失败: {image_path}")
            self.status_updated.emit(f"加载图像失败: {os.path.basename(image_path)}")
            
    def prev_image(self):
        """加载上一张图像"""
        if self.current_index > 0:
            self.image_list_widget.setCurrentRow(self.current_index - 1)
            
    def next_image(self):
        """加载下一张图像"""
        if self.current_index < len(self.image_files) - 1:
            self.image_list_widget.setCurrentRow(self.current_index + 1)
            
    def save_annotations(self):
        """保存当前图像的标注结果"""
        if not self.output_folder:
            self.status_updated.emit("请先设置输出文件夹")
            return
            
        if self.canvas.save_annotations(self.output_folder, self.annotation_format):
            self.status_updated.emit(f"已保存标注结果: {os.path.basename(self.image_files[self.current_index])}")
            
            # 自动加载下一张图像
            self.next_image()
        else:
            self.status_updated.emit("保存标注结果失败")
            
    def update_status_label(self, message):
        """更新状态标签"""
        try:
            if hasattr(self, 'status_label'):
                self.status_label.setText(message)
            print(f"标注工具状态: {message}")
        except Exception as e:
            print(f"更新标注工具状态时出错: {str(e)}")
            
    def undo_operation(self):
        """撤销操作"""
        if self.canvas.undo():
            self.status_updated.emit("已撤销最近的操作")
        else:
            self.status_updated.emit("没有可撤销的操作")

    def reset_view(self):
        """重置图像视图"""
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.reset_view()
            self.status_updated.emit("视图已重置") 