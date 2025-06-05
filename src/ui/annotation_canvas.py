from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QCursor
import os
import xml.etree.ElementTree as ET

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
        
        # 设置十字光标
        self.setCursor(Qt.CrossCursor)
        
        # 打印调试信息
        print("AnnotationCanvas初始化完成")
        
    def enterEvent(self, event):
        """鼠标进入事件"""
        # 如果有图像，使用十字光标，否则使用默认光标
        if self.pixmap and not self.pixmap.isNull():
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        
    def leaveEvent(self, event):
        """鼠标离开事件"""
        # 恢复默认光标
        self.setCursor(Qt.ArrowCursor)
        
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
        
        # 计算初始缩放因子以适应窗口
        if self.width() > 0 and self.height() > 0:
            # 计算宽度和高度的缩放比例
            width_ratio = self.width() / self.pixmap.width()
            height_ratio = self.height() / self.pixmap.height()
            
            # 使用较小的缩放比例，确保图像完全显示在窗口中
            self.scale_factor = min(width_ratio, height_ratio) * 0.9  # 留出10%的边距
        
        # 计算居中位置
        scaled_width = int(self.pixmap.width() * self.scale_factor)
        scaled_height = int(self.pixmap.height() * self.scale_factor)
        self.offset_x = int((self.width() - scaled_width) / 2)
        self.offset_y = int((self.height() - scaled_height) / 2)
        
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
                
                # 将图像坐标转换为屏幕坐标
                start_screen = QPointF(
                    x + self.start_point.x() * self.scale_factor,
                    y + self.start_point.y() * self.scale_factor
                )
                end_screen = QPointF(
                    x + self.end_point.x() * self.scale_factor,
                    y + self.end_point.y() * self.scale_factor
                )
                
                # 使用屏幕坐标绘制矩形
                rect = QRectF(start_screen, end_screen).normalized()
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
                # 开始拖动
                self.dragging = True
                self.drag_start_point = self.screen_to_image_coords(event.pos())
                self.original_rect = self.boxes[clicked_box][0]
                self.update()
            else:
                # 检查点击是否在图像范围内
                image_rect = self.get_image_rect()
                if image_rect.contains(event.pos()):
                    # 开始绘制新标注框
                    self.drawing = True
                    self.start_point = self.screen_to_image_coords(event.pos())
                    self.end_point = self.start_point
                
        elif event.button() == Qt.RightButton:
            # 开始平移图像
            self.panning = True
            self.pan_start_point = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if not self.pixmap or self.pixmap.isNull():
            return
            
        if self.drawing and self.start_point:
            # 检查是否在图像范围内移动
            image_rect = self.get_image_rect()
            if image_rect.contains(event.pos()):
                # 更新标注框大小
                self.end_point = self.screen_to_image_coords(event.pos())
                self.update()
            
        elif self.panning and self.pan_start_point:
            # 平移图像
            delta = event.pos() - self.pan_start_point
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self.pan_start_point = event.pos()
            self.update()
            
        elif self.dragging and self.original_rect:
            # 移动选中的标注框
            delta = self.screen_to_image_coords(event.pos()) - self.drag_start_point
            rect, label, color = self.boxes[self.selected_box]
            new_rect = QRectF(
                self.original_rect.x() + delta.x(),
                self.original_rect.y() + delta.y(),
                self.original_rect.width(),
                self.original_rect.height()
            )
            self.boxes[self.selected_box] = (new_rect, label, color)
            self.update()
            
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            if self.drawing and self.start_point and self.end_point:
                # 完成标注框绘制
                rect = QRectF(self.start_point, self.end_point).normalized()
                if rect.width() > 5 and rect.height() > 5:  # 最小尺寸限制
                    self.add_box(rect, self.current_label)
                    self.box_created.emit(rect, self.current_label)
                self.drawing = False
                self.start_point = None
                self.end_point = None
                self.update()
            elif self.dragging:
                # 结束拖动
                self.dragging = False
                self.drag_start_point = None
                self.original_rect = None
                self.update()
                
        elif event.button() == Qt.RightButton:
            # 结束平移
            self.panning = False
            self.pan_start_point = None
            self.setCursor(Qt.ArrowCursor)
            
    def wheelEvent(self, event):
        """鼠标滚轮事件"""
        if not self.pixmap or self.pixmap.isNull():
            return
            
        # 计算缩放因子
        delta = event.angleDelta().y()
        if delta > 0:
            self.scale_factor *= 1.1
        else:
            self.scale_factor /= 1.1
            
        # 限制缩放范围
        self.scale_factor = max(0.1, min(10.0, self.scale_factor))
        
        self.update()
        
    def keyPressEvent(self, event):
        """键盘按下事件"""
        if event.key() == Qt.Key_Delete and self.selected_box >= 0:
            # 删除选中的标注框
            self.delete_selected_box()
        elif event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            # 撤销操作
            self.undo()
            
    def delete_selected_box(self):
        """删除选中的标注框"""
        if self.selected_box >= 0:
            # 记录删除操作
            self.operation_history.append(('delete', self.selected_box))
            self.deleted_boxes.append(self.boxes[self.selected_box])
            del self.boxes[self.selected_box]
            self.selected_box = -1
            self.update()
            
    def undo(self):
        """撤销最近的操作"""
        if not self.operation_history:
            return False
            
        operation_type, data = self.operation_history.pop()
        
        if operation_type == 'add':
            # 撤销添加操作
            self.deleted_boxes.append(self.boxes.pop())
        elif operation_type == 'delete':
            # 撤销删除操作
            self.boxes.insert(data, self.deleted_boxes.pop())
            
        self.update()
        return True
        
    def reset_view(self):
        """重置视图"""
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.update()
        
    def save_annotations(self, output_folder, format_type='voc'):
        """保存标注结果到指定文件夹"""
        if not self.image_path:
            print("错误: 没有图像可供标注")
            return False
            
        image_name = os.path.basename(self.image_path)
        base_name = os.path.splitext(image_name)[0]
        
        # 获取图像尺寸
        if self.pixmap:
            image_width = self.pixmap.width()
            image_height = self.pixmap.height()
        else:
            print("错误: 无法获取图像尺寸")
            return False
            
        try:
            # 确保boxes属性存在
            if not hasattr(self, 'boxes'):
                self.boxes = []
                print("警告: 标注框列表不存在，已初始化为空列表")
            
            # 根据格式保存
            if format_type.lower() == 'voc':
                result = self.save_voc_format(output_folder, base_name, image_width, image_height)
            else:  # 默认为YOLO格式
                result = self.save_yolo_format(output_folder, base_name, image_width, image_height)
            
            # 打印保存路径信息，便于调试
            print(f"标注文件已保存到: {output_folder}")
            print(f"图像名称: {image_name}, 基本名称: {base_name}")
            print(f"保存格式: {format_type}, 图像尺寸: {image_width}x{image_height}")
            print(f"标注框数量: {len(self.boxes)}")
            
            return result
        except Exception as e:
            print(f"保存标注结果时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def save_voc_format(self, output_folder, base_name, image_width, image_height):
        """保存为VOC格式的XML文件"""
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        # 设置XML文件路径
        xml_path = os.path.join(output_folder, f"{base_name}.xml")
        
        # 创建XML根元素
        root = ET.Element("annotation")
        
        # 添加基本信息
        ET.SubElement(root, "folder").text = os.path.basename(os.path.dirname(self.image_path))
        ET.SubElement(root, "filename").text = os.path.basename(self.image_path)
        ET.SubElement(root, "path").text = self.image_path
        
        # 添加源信息
        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "Unknown"
        
        # 添加大小信息
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(image_width)
        ET.SubElement(size, "height").text = str(image_height)
        ET.SubElement(size, "depth").text = "3"
        
        # 添加分割信息
        ET.SubElement(root, "segmented").text = "0"
        
        # 添加每个对象
        for rect, label, color in self.boxes:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = label
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            
            # 添加边界框信息
            bbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bbox, "xmin").text = str(int(rect.x()))
            ET.SubElement(bbox, "ymin").text = str(int(rect.y()))
            ET.SubElement(bbox, "xmax").text = str(int(rect.x() + rect.width()))
            ET.SubElement(bbox, "ymax").text = str(int(rect.y() + rect.height()))
        
        # 保存XML文件
        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        
        print(f"VOC格式标注文件已保存到: {xml_path}")
        return True
        
    def save_yolo_format(self, output_folder, base_name, image_width, image_height):
        """保存为YOLO格式的TXT文件"""
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        # 设置TXT文件路径
        txt_path = os.path.join(output_folder, f"{base_name}.txt")
        
        # 如果没有标注框，创建一个空文件表示无标注
        if not self.boxes:
            open(txt_path, 'w').close()
            print(f"创建了空的YOLO格式标注文件: {txt_path} (无标注)")
            return True
            
        # 写入标注
        with open(txt_path, 'w') as f:
            for rect, label, color in self.boxes:
                # 计算YOLO格式的参数
                x_center = (rect.x() + rect.width() / 2) / image_width
                y_center = (rect.y() + rect.height() / 2) / image_height
                width = rect.width() / image_width
                height = rect.height() / image_height
                
                # 写入标签文件
                f.write(f"{label} {x_center} {y_center} {width} {height}\n")
        
        print(f"YOLO格式标注文件已保存到: {txt_path}")
        return True
        
    def box_at_position(self, pos):
        """获取指定位置的标注框索引"""
        if not self.pixmap or self.pixmap.isNull():
            return -1
            
        # 转换屏幕坐标到图像坐标
        image_pos = self.screen_to_image_coords(pos)
        
        # 检查每个标注框
        for i, (rect, _, _) in enumerate(self.boxes):
            if rect.contains(image_pos):
                return i
                
        return -1
        
    def screen_to_image_coords(self, pos):
        """将屏幕坐标转换为图像坐标"""
        if not self.pixmap or self.pixmap.isNull():
            return QPointF()
            
        # 计算图像在屏幕上的位置
        scaled_size = self.pixmap.size() * self.scale_factor
        x_offset = (self.width() - scaled_size.width()) / 2 + self.offset_x
        y_offset = (self.height() - scaled_size.height()) / 2 + self.offset_y
        
        # 计算鼠标位置相对于图像的偏移
        relative_x = pos.x() - x_offset
        relative_y = pos.y() - y_offset
        
        # 通过缩放因子将屏幕坐标转换为图像坐标
        image_x = relative_x / self.scale_factor
        image_y = relative_y / self.scale_factor
        
        # 限制坐标在图像范围内
        image_x = max(0, min(image_x, self.pixmap.width()))
        image_y = max(0, min(image_y, self.pixmap.height()))
        
        return QPointF(image_x, image_y)
        
    def is_pan_control_clicked(self, pos):
        """检查是否点击了平移控件"""
        control_size = 80
        control_x = self.width() - control_size - 10
        control_y = self.height() - control_size - 10
        
        return (control_x <= pos.x() <= control_x + control_size and
                control_y <= pos.y() <= control_y + control_size)
                
    def handle_pan_control_click(self, pos):
        """处理平移控件点击"""
        control_size = 80
        control_x = self.width() - control_size - 10
        control_y = self.height() - control_size - 10
        
        # 计算相对于控件中心的偏移
        center_x = control_x + control_size / 2
        center_y = control_y + control_size / 2
        dx = pos.x() - center_x
        dy = pos.y() - center_y
        
        # 根据点击位置决定平移方向
        if abs(dx) > abs(dy):
            # 水平平移
            if dx > 0:
                self.offset_x += 20
            else:
                self.offset_x -= 20
        else:
            # 垂直平移
            if dy > 0:
                self.offset_y += 20
            else:
                self.offset_y -= 20
                
        self.update()

    def get_image_rect(self):
        """获取图像在屏幕上的矩形区域"""
        if not self.pixmap or self.pixmap.isNull():
            return QRectF()
            
        # 计算缩放后的图像尺寸
        scaled_size = self.pixmap.size() * self.scale_factor
        
        # 计算居中位置（考虑偏移量）
        x = (self.width() - scaled_size.width()) / 2 + self.offset_x
        y = (self.height() - scaled_size.height()) / 2 + self.offset_y
        
        return QRectF(x, y, scaled_size.width(), scaled_size.height()) 