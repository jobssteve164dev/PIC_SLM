from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QListWidget, QListWidgetItem, QGroupBox, QGridLayout,
                           QSizePolicy, QLineEdit, QInputDialog, QMessageBox, QRadioButton,
                           QButtonGroup, QStackedWidget, QComboBox, QScrollArea, QFrame, QSplitter,
                           QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF, QSizeF
from PyQt5.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QColor, QBrush, QCursor, QKeySequence
import os
import glob
import xml.etree.ElementTree as ET
from datetime import datetime
from .base_tab import BaseTab
import json

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
        """保存标注结果"""
        if not self.image_path or not self.boxes:
            return False
            
        try:
            # 创建输出文件夹
            os.makedirs(output_folder, exist_ok=True)
            
            # 获取图像信息
            image_name = os.path.basename(self.image_path)
            image_width = self.pixmap.width()
            image_height = self.pixmap.height()
            
            if format_type == 'voc':
                # 保存为VOC格式
                self.save_voc_format(output_folder, image_name, image_width, image_height)
            else:
                # 保存为YOLO格式
                self.save_yolo_format(output_folder, image_name, image_width, image_height)
                
            return True
            
        except Exception as e:
            print(f"保存标注失败: {str(e)}")
            return False
            
    def save_voc_format(self, output_folder, image_name, image_width, image_height):
        """保存为VOC格式"""
        # 创建XML文件
        root = ET.Element('annotation')
        
        # 添加基本信息
        folder = ET.SubElement(root, 'folder')
        folder.text = os.path.basename(output_folder)
        
        filename = ET.SubElement(root, 'filename')
        filename.text = image_name
        
        path = ET.SubElement(root, 'path')
        path.text = self.image_path
        
        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(image_width)
        height = ET.SubElement(size, 'height')
        height.text = str(image_height)
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'
        
        # 添加标注框
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
            
        # 保存XML文件
        xml_name = os.path.splitext(image_name)[0] + '.xml'
        xml_path = os.path.join(output_folder, xml_name)
        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        
    def save_yolo_format(self, output_folder, image_name, image_width, image_height):
        """保存为YOLO格式"""
        # 创建标签文件
        txt_name = os.path.splitext(image_name)[0] + '.txt'
        txt_path = os.path.join(output_folder, txt_name)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            for rect, label, _ in self.boxes:
                # 转换为YOLO格式（归一化坐标）
                x_center = (rect.x() + rect.width() / 2) / image_width
                y_center = (rect.y() + rect.height() / 2) / image_height
                width = rect.width() / image_width
                height = rect.height() / image_height
                
                # 写入标签文件
                f.write(f"{label} {x_center} {y_center} {width} {height}\n")
                
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

class AnnotationTab(BaseTab):
    """标注标签页，负责图像标注功能"""
    
    # 定义信号
    annotation_started = pyqtSignal(str)
    open_validation_folder_signal = pyqtSignal(str)
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent, main_window)
        self.defect_classes = []  # 缺陷类别列表
        self.detection_classes = []  # 目标检测类别列表
        self.processed_folder = ""  # 处理后的图片文件夹
        self.detection_folder = ""  # 目标检测图像文件夹
        self.annotation_folder = ""  # 标注输出文件夹
        self.init_ui()
        
        # 尝试从配置文件加载默认路径
        config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
        print(f"尝试加载配置文件: {config_file}")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    print(f"成功加载配置: {config}")
                    # 加载默认处理后文件夹和标注文件夹
                    if 'default_processed_folder' in config and config['default_processed_folder']:
                        self.processed_folder = config['default_processed_folder']
                        print(f"设置处理后文件夹: {self.processed_folder}")
                    if 'default_annotation_folder' in config and config['default_annotation_folder']:
                        self.annotation_folder = config['default_annotation_folder']
                        print(f"设置标注文件夹: {self.annotation_folder}")
            except Exception as e:
                print(f"加载配置失败: {str(e)}")
                import traceback
                traceback.print_exc()
        
    def init_ui(self):
        """初始化UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self.scroll_content)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 添加标题
        title_label = QLabel("图像标注")
        title_label.setFont(QFont('微软雅黑', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 添加标注模式选择
        mode_group = QGroupBox("标注模式")
        mode_layout = QHBoxLayout()
        
        # 创建单选按钮
        self.classification_radio = QRadioButton("图片分类")
        self.detection_radio = QRadioButton("目标检测")
        
        # 创建按钮组
        self.mode_button_group = QButtonGroup()
        self.mode_button_group.addButton(self.classification_radio, 0)
        self.mode_button_group.addButton(self.detection_radio, 1)
        self.classification_radio.setChecked(True)  # 默认选择图片分类
        
        # 添加到布局
        mode_layout.addWidget(self.classification_radio)
        mode_layout.addWidget(self.detection_radio)
        mode_layout.addStretch()
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)
        
        # 创建堆叠部件用于切换不同的标注界面
        self.stacked_widget = QStackedWidget()
        
        # 创建图片分类标注界面
        self.classification_widget = QWidget()
        self.init_classification_ui()
        
        # 创建目标检测标注界面
        self.detection_widget = QWidget()
        self.init_detection_ui()
        
        # 添加到堆叠部件
        self.stacked_widget.addWidget(self.classification_widget)
        self.stacked_widget.addWidget(self.detection_widget)
        
        # 添加到主布局
        main_layout.addWidget(self.stacked_widget)
        
        # 连接信号
        self.mode_button_group.buttonClicked.connect(self.on_mode_changed)
        
    def init_classification_ui(self):
        """初始化图像分类标注界面"""
        # 创建主布局
        main_layout = QVBoxLayout(self.classification_widget)
        
        # 创建顶部控制面板
        control_layout = QHBoxLayout()
        
        # 图像文件夹选择组
        folder_group = QGroupBox("处理后的图片文件夹")
        folder_layout = QHBoxLayout()
        
        self.processed_path_edit = QLineEdit()
        self.processed_path_edit.setReadOnly(True)
        self.processed_path_edit.setPlaceholderText("请选择处理后的图片文件夹")
        
        folder_btn = QPushButton("浏览...")
        folder_btn.clicked.connect(self.select_processed_folder)
        
        folder_layout.addWidget(self.processed_path_edit)
        folder_layout.addWidget(folder_btn)
        
        folder_group.setLayout(folder_layout)
        control_layout.addWidget(folder_group)
        
        # 类别管理组
        class_group = QGroupBox("缺陷类别")
        class_layout = QVBoxLayout()
        
        # 添加类别列表
        self.class_list = QListWidget()
        self.class_list.setMaximumHeight(100)
        class_layout.addWidget(self.class_list)
        
        # 添加按钮组
        btn_layout = QHBoxLayout()
        
        add_class_btn = QPushButton("添加类别")
        add_class_btn.clicked.connect(self.add_defect_class)
        btn_layout.addWidget(add_class_btn)
        
        remove_class_btn = QPushButton("删除类别")
        remove_class_btn.clicked.connect(self.remove_defect_class)
        btn_layout.addWidget(remove_class_btn)
        
        # 添加未分类选项
        self.create_unclassified_checkbox = QCheckBox("创建未分类文件夹")
        self.create_unclassified_checkbox.setChecked(True)
        btn_layout.addWidget(self.create_unclassified_checkbox)
        
        class_layout.addLayout(btn_layout)
        class_group.setLayout(class_layout)
        control_layout.addWidget(class_group)
        
        main_layout.addLayout(control_layout)
        
        # 创建开始标注按钮
        self.annotation_btn = QPushButton("开始分类标注")
        self.annotation_btn.setMinimumHeight(40)
        self.annotation_btn.clicked.connect(self.start_annotation)
        self.annotation_btn.setEnabled(False)  # 默认禁用
        
        main_layout.addWidget(self.annotation_btn)
        
        # 添加打开验证集文件夹按钮
        self.val_folder_btn = QPushButton("打开验证集文件夹")
        self.val_folder_btn.setMinimumHeight(40)
        self.val_folder_btn.clicked.connect(self.open_validation_folder)
        self.val_folder_btn.setEnabled(False)  # 默认禁用
        
        main_layout.addWidget(self.val_folder_btn)
        
        # 添加使用说明
        help_text = """
        <b>图像分类标注步骤:</b>
        <ol>
            <li>选择处理后的图片文件夹</li>
            <li>添加需要的缺陷类别</li>
            <li>点击"开始分类标注"按钮</li>
            <li>系统将在训练集(train)和验证集(val)文件夹中分别创建类别文件夹</li>
            <li>在弹出的文件浏览器中，将训练集图片拖放到对应的类别文件夹中</li>
            <li>点击"打开验证集文件夹"按钮，对验证集图片进行同样的标注</li>
        </ol>
        
        <b>提示:</b>
        <ul>
            <li>可以选择是否创建"未分类"文件夹，用于存放暂时无法分类的图片</li>
            <li>分类完成后，各个缺陷类别的图片将位于对应类别文件夹中，便于后续训练</li>
            <li>务必确保训练集和验证集都完成了分类标注，以保证模型训练效果</li>
        </ul>
        """
        
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_label.setTextFormat(Qt.RichText)
        help_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        
        main_layout.addWidget(help_label)
        main_layout.addStretch()
        
        # 自动设置默认处理后文件夹路径
        if self.processed_folder:
            self.processed_path_edit.setText(self.processed_folder)
            self.check_annotation_ready()
        
    def init_detection_ui(self):
        """初始化目标检测标注界面"""
        # 创建主布局
        main_layout = QVBoxLayout(self.detection_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 图像文件夹选择组
        folder_group = QGroupBox("图像文件夹")
        folder_layout = QGridLayout()
        
        self.detection_path_edit = QLineEdit()
        self.detection_path_edit.setReadOnly(True)
        self.detection_path_edit.setPlaceholderText("请选择图像文件夹")
        
        folder_btn = QPushButton("浏览...")
        folder_btn.clicked.connect(self.select_detection_folder)
        
        folder_layout.addWidget(QLabel("文件夹:"), 0, 0)
        folder_layout.addWidget(self.detection_path_edit, 0, 1)
        folder_layout.addWidget(folder_btn, 0, 2)
        
        folder_group.setLayout(folder_layout)
        left_layout.addWidget(folder_group)
        
        # 输出文件夹选择组
        output_group = QGroupBox("标注输出文件夹")
        output_layout = QGridLayout()
        
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setReadOnly(True)
        self.output_path_edit.setPlaceholderText("请选择标注输出文件夹")
        
        output_btn = QPushButton("浏览...")
        output_btn.clicked.connect(self.select_output_folder)
        
        output_layout.addWidget(QLabel("文件夹:"), 0, 0)
        output_layout.addWidget(self.output_path_edit, 0, 1)
        output_layout.addWidget(output_btn, 0, 2)
        
        output_group.setLayout(output_layout)
        left_layout.addWidget(output_group)
        
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
        self.undo_btn = QPushButton("撤销操作")
        self.reset_view_btn = QPushButton("重置视图")
        op_layout.addWidget(self.undo_btn)
        op_layout.addWidget(self.reset_view_btn)
        left_layout.addLayout(op_layout)
        
        # 添加保存标注按钮（单独一行）
        self.save_btn = QPushButton("保存标注")
        self.save_btn.setMinimumHeight(30)  # 设置按钮更高一些
        left_layout.addWidget(self.save_btn)
        
        # 格式选择
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("保存格式:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["VOC", "YOLO"])
        format_layout.addWidget(self.format_combo)
        left_layout.addLayout(format_layout)
        
        # 添加类别管理组
        class_group = QGroupBox("目标类别")
        class_layout = QVBoxLayout()
        
        # 添加类别列表
        self.detection_class_list = QListWidget()
        self.detection_class_list.setMinimumHeight(100)
        class_layout.addWidget(self.detection_class_list)
        
        # 添加按钮组
        class_btn_layout = QHBoxLayout()
        
        add_class_btn = QPushButton("添加类别")
        add_class_btn.clicked.connect(self.add_detection_class)
        class_btn_layout.addWidget(add_class_btn)
        
        remove_class_btn = QPushButton("删除类别")
        remove_class_btn.clicked.connect(self.remove_detection_class)
        class_btn_layout.addWidget(remove_class_btn)
        
        class_layout.addLayout(class_btn_layout)
        class_group.setLayout(class_layout)
        left_layout.addWidget(class_group)
        
        # 添加左侧面板到分割器
        splitter.addWidget(left_panel)
        
        # 右侧标注画布
        self.annotation_canvas = AnnotationCanvas()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.annotation_canvas)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(600)
        scroll_area.setMinimumHeight(400)
        scroll_area.setAlignment(Qt.AlignCenter)
        splitter.addWidget(scroll_area)
        
        # 设置分割器比例
        splitter.setSizes([200, 800])
        
        # 添加分割器到主布局
        main_layout.addWidget(splitter)
        
        # 状态标签
        self.detection_status_label = QLabel("就绪")
        main_layout.addWidget(self.detection_status_label)
        
        # 绑定事件
        self.image_list_widget.currentRowChanged.connect(self.load_image)
        self.label_combo.currentTextChanged.connect(self.on_label_changed)
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.save_btn.clicked.connect(self.save_annotations)
        self.undo_btn.clicked.connect(self.undo_operation)
        self.format_combo.currentTextChanged.connect(self.set_annotation_format)
        self.reset_view_btn.clicked.connect(self.reset_view)
        
        # 初始化变量
        self.current_index = -1
        self.image_files = []
        self.annotation_format = 'voc'  # 默认VOC格式
        
        # 打印当前实例变量状态，帮助调试
        print(f"初始化目标检测UI时的路径状态：")
        print(f"  processed_folder = {self.processed_folder}")
        print(f"  annotation_folder = {self.annotation_folder}")
        
        # 自动设置默认路径
        if self.processed_folder:
            print(f"设置detection_path_edit文本为: {self.processed_folder}")
            self.detection_path_edit.setText(self.processed_folder)
        
        if self.annotation_folder:
            print(f"设置output_path_edit文本为: {self.annotation_folder}")
            self.output_path_edit.setText(self.annotation_folder)
            
        # 如果两个路径都已设置，自动加载图像
        if self.processed_folder and self.annotation_folder:
            print(f"自动加载图像文件，文件夹: {self.processed_folder}")
            self.load_image_files(self.processed_folder)
            self.check_detection_ready()
        
    def on_mode_changed(self, button):
        """标注模式改变时调用"""
        if button == self.classification_radio:
            self.stacked_widget.setCurrentIndex(0)
        else:
            # 切换到目标检测模式时重新应用路径
            self.stacked_widget.setCurrentIndex(1)
            
            # 确保在切换到目标检测模式时应用默认路径
            if self.processed_folder:
                self.detection_path_edit.setText(self.processed_folder)
            
            if self.annotation_folder:
                self.output_path_edit.setText(self.annotation_folder)
                
            # 如果两个路径都已设置但图像列表为空，则尝试加载图像
            if self.processed_folder and self.annotation_folder and len(self.image_files) == 0:
                self.load_image_files(self.processed_folder)
                self.check_detection_ready()
    
    def select_folder(self, title, path_edit, check_callback):
        """通用的文件夹选择方法"""
        folder = QFileDialog.getExistingDirectory(self, title)
        if folder:
            path_edit.setText(folder)
            check_callback()
            
    def add_class(self, title, prompt, class_list, list_widget, check_callback):
        """通用的添加类别方法"""
        class_name, ok = QInputDialog.getText(self, title, prompt)
        if ok and class_name:
            # 检查是否已存在
            if class_name in class_list:
                QMessageBox.warning(self, "警告", f"类别 '{class_name}' 已存在!")
                return
            class_list.append(class_name)
            list_widget.addItem(class_name)
            check_callback()
            
    def remove_class(self, list_widget, check_callback):
        """通用的删除类别方法"""
        current_item = list_widget.currentItem()
        if current_item:
            class_name = current_item.text()
            # 从对应的列表中移除类别
            if list_widget == self.class_list:
                self.defect_classes.remove(class_name)
            elif list_widget == self.detection_class_list:
                self.detection_classes.remove(class_name)
            list_widget.takeItem(list_widget.row(current_item))
            check_callback()
            
    def select_processed_folder(self):
        """选择处理后的图片文件夹"""
        self.select_folder(
            "选择处理后的图片文件夹",
            self.processed_path_edit,
            lambda: [setattr(self, 'processed_folder', self.processed_path_edit.text()), self.check_annotation_ready()]
        )
    
    def select_detection_folder(self):
        """选择目标检测图像文件夹"""
        self.select_folder(
            "选择图像文件夹",
            self.detection_path_edit,
            lambda: [
                setattr(self, 'detection_folder', self.detection_path_edit.text()),
                self.load_image_files(self.detection_path_edit.text()), 
                self.check_detection_ready()
            ]
        )
    
    def select_output_folder(self):
        """选择标注输出文件夹"""
        self.select_folder(
            "选择标注输出文件夹",
            self.output_path_edit,
            lambda: [
                setattr(self, 'annotation_folder', self.output_path_edit.text()),
                self.check_detection_ready()
            ]
        )
    
    def check_annotation_ready(self):
        """检查是否可以开始图片分类标注"""
        is_ready = bool(self.processed_folder and self.defect_classes)
        self.annotation_btn.setEnabled(is_ready)
        self.val_folder_btn.setEnabled(is_ready)  # 同时启用/禁用验证集按钮
    
    def check_detection_ready(self):
        """检查是否可以开始目标检测标注"""
        has_image_folder = bool(self.detection_path_edit.text())
        has_classes = self.detection_class_list.count() > 0
        
        # 更新标签下拉框
        if has_classes:
            self.update_label_combo()
            
        # 更新按钮状态
        self.save_btn.setEnabled(has_image_folder and has_classes and self.current_index >= 0)
        self.prev_btn.setEnabled(has_image_folder and self.current_index > 0)
        self.next_btn.setEnabled(has_image_folder and self.current_index < len(self.image_files) - 1)
    
    def add_defect_class(self):
        """添加缺陷类别"""
        self.add_class(
            "添加缺陷类别",
            "请输入缺陷类别名称:",
            self.defect_classes,
            self.class_list,
            self.check_annotation_ready
        )
    
    def add_detection_class(self):
        """添加目标检测类别"""
        self.add_class(
            "添加目标类别",
            "请输入目标类别名称:",
            self.detection_classes,  # 使用detection_classes列表
            self.detection_class_list,
            lambda: [self.update_label_combo(), self.check_detection_ready()]
        )
    
    def remove_defect_class(self):
        """删除缺陷类别"""
        self.remove_class(self.class_list, self.check_annotation_ready)
    
    def remove_detection_class(self):
        """删除目标检测类别"""
        self.remove_class(
            self.detection_class_list,
            lambda: [self.update_label_combo(), self.check_detection_ready()]
        )
    
    def start_annotation(self):
        """开始图片分类标注"""
        if not self.processed_folder:
            QMessageBox.warning(self, "警告", "请先选择处理后的图片文件夹!")
            return
            
        if not self.defect_classes:
            QMessageBox.warning(self, "警告", "请先添加至少一个缺陷类别!")
            return
            
        # 创建分类文件夹
        if not self.create_classification_folders():
            return  # 如果创建文件夹失败，终止操作
        
        # 发出标注开始信号
        self.annotation_started.emit(self.processed_folder)
        self.update_status("开始图像标注...")
    
    def create_classification_folders(self):
        """创建分类文件夹"""
        try:
            created_count = 0
            
            # 指定train和val文件夹路径
            dataset_folder = os.path.join(self.processed_folder, 'dataset')
            train_folder = os.path.join(dataset_folder, 'train')
            val_folder = os.path.join(dataset_folder, 'val')
            
            # 检查必要的文件夹是否存在
            if not os.path.exists(train_folder) or not os.path.exists(val_folder):
                QMessageBox.warning(self, "警告", "数据集文件夹结构不完整，请先完成数据预处理步骤")
                return False

            # 在train文件夹中为每个类别创建文件夹
            for class_name in self.defect_classes:
                train_class_folder = os.path.join(train_folder, class_name)
                if not os.path.exists(train_class_folder):
                    os.makedirs(train_class_folder)
                    created_count += 1
                
                # 同时在val文件夹中也创建对应的类别文件夹
                val_class_folder = os.path.join(val_folder, class_name)
                if not os.path.exists(val_class_folder):
                    os.makedirs(val_class_folder)
                    created_count += 1
                    
            # 根据复选框状态决定是否创建未分类文件夹
            if self.create_unclassified_checkbox.isChecked():
                # 在train和val文件夹中创建未分类文件夹
                train_unclassified_folder = os.path.join(train_folder, "未分类")
                if not os.path.exists(train_unclassified_folder):
                    os.makedirs(train_unclassified_folder)
                    created_count += 1
                
                val_unclassified_folder = os.path.join(val_folder, "未分类")
                if not os.path.exists(val_unclassified_folder):
                    os.makedirs(val_unclassified_folder)
                    created_count += 1
                
            # 添加成功提示
            if created_count > 0:
                QMessageBox.information(self, "成功", f"已成功创建 {created_count} 个分类文件夹")
            else:
                QMessageBox.information(self, "提示", "所有分类文件夹已存在，无需重新创建")
                
            return True
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建分类文件夹失败: {str(e)}")
            return False
            
    def load_image_files(self, folder):
        """加载图像文件列表"""
        self.image_files = []
        
        # 支持的图像格式
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            pattern = os.path.join(folder, ext)
            self.image_files.extend(glob.glob(pattern))
            
        # 更新图像列表
        self.image_list_widget.clear()
        for image_file in self.image_files:
            self.image_list_widget.addItem(os.path.basename(image_file))
            
        # 如果有图像，选择第一张
        if self.image_files:
            self.current_index = 0
            self.image_list_widget.setCurrentRow(0)
            self.update_detection_status(f"已加载 {len(self.image_files)} 张图像")
        else:
            self.current_index = -1
            self.update_detection_status("未找到图像文件")
            
    def load_image(self, index):
        """加载指定索引的图像"""
        if index < 0 or index >= len(self.image_files):
            return
            
        self.current_index = index
        image_path = self.image_files[index]
        
        # 加载图像到画布
        if self.annotation_canvas.set_image(image_path):
            self.update_detection_status(f"已加载图像: {os.path.basename(image_path)}")
            self.check_detection_ready()
        else:
            self.update_detection_status(f"加载图像失败: {os.path.basename(image_path)}")
            
    def prev_image(self):
        """加载上一张图像"""
        if self.current_index > 0:
            self.image_list_widget.setCurrentRow(self.current_index - 1)
            
    def next_image(self):
        """加载下一张图像"""
        if self.current_index < len(self.image_files) - 1:
            self.image_list_widget.setCurrentRow(self.current_index + 1)
            
    def update_label_combo(self):
        """更新标签下拉框"""
        self.label_combo.clear()
        
        # 添加所有类别
        for i in range(self.detection_class_list.count()):
            self.label_combo.addItem(self.detection_class_list.item(i).text())
            
        # 如果有类别，设置当前标签
        if self.label_combo.count() > 0:
            self.annotation_canvas.set_current_label(self.label_combo.itemText(0))
            
    def on_label_changed(self, label):
        """标签改变时调用"""
        if label:
            self.annotation_canvas.set_current_label(label)
            
    def save_annotations(self):
        """保存当前图像的标注结果"""
        output_folder = self.output_path_edit.text()
        
        if not output_folder:
            # 如果未指定输出文件夹，使用图像文件夹下的annotations子文件夹
            image_folder = os.path.dirname(self.image_files[self.current_index])
            output_folder = os.path.join(image_folder, "annotations")
            os.makedirs(output_folder, exist_ok=True)
            
        # 保存标注
        if self.annotation_canvas.save_annotations(output_folder, self.annotation_format):
            self.update_detection_status(f"已保存标注结果: {os.path.basename(self.image_files[self.current_index])}")
            
            # 自动加载下一张图像
            self.next_image()
        else:
            self.update_detection_status("保存标注结果失败")
            
    def undo_operation(self):
        """撤销操作"""
        if self.annotation_canvas.undo():
            self.update_detection_status("已撤销最近的操作")
        else:
            self.update_detection_status("没有可撤销的操作")
            
    def reset_view(self):
        """重置视图"""
        self.annotation_canvas.reset_view()
        self.update_detection_status("视图已重置")
        
    def set_annotation_format(self, format_type):
        """设置标注格式"""
        self.annotation_format = format_type.lower()
        self.update_detection_status(f"标注格式已设置为: {format_type}")
        
    def update_detection_status(self, message):
        """更新目标检测状态标签"""
        self.detection_status_label.setText(message)
        if hasattr(self, 'main_window'):
            self.main_window.update_status(message)

    def apply_config(self, config):
        """应用配置设置"""
        try:
            # 加载默认处理后文件夹和标注文件夹
            if 'default_processed_folder' in config and config['default_processed_folder']:
                self.processed_folder = config['default_processed_folder']
                # 如果处理后文件夹输入框已创建，则设置文本
                if hasattr(self, 'processed_path_edit'):
                    self.processed_path_edit.setText(self.processed_folder)
                # 如果目标检测路径输入框已创建，也设置相同的路径
                if hasattr(self, 'detection_path_edit'):
                    self.detection_path_edit.setText(self.processed_folder)
                    # 如果当前在目标检测模式，加载图像
                    if self.stacked_widget.currentIndex() == 1:
                        self.load_image_files(self.processed_folder)
            
            # 加载默认标注文件夹
            if 'default_annotation_folder' in config and config['default_annotation_folder']:
                self.annotation_folder = config['default_annotation_folder']
                # 如果标注输出文件夹输入框已创建，则设置文本
                if hasattr(self, 'output_path_edit'):
                    self.output_path_edit.setText(self.annotation_folder)
            
            # 加载默认类别
            if 'default_classes' in config and config['default_classes']:
                # 清除现有类别
                self.defect_classes = []
                if hasattr(self, 'class_list'):
                    self.class_list.clear()
                
                # 添加新类别到分类标注
                for class_name in config['default_classes']:
                    self.defect_classes.append(class_name)
                    if hasattr(self, 'class_list'):
                        self.class_list.addItem(class_name)
                
                # 同时添加到目标检测类别
                self.detection_classes = []
                if hasattr(self, 'detection_class_list'):
                    self.detection_class_list.clear()
                    
                for class_name in config['default_classes']:
                    self.detection_classes.append(class_name)
                    if hasattr(self, 'detection_class_list'):
                        self.detection_class_list.addItem(class_name)
                
                # 更新标签下拉框
                if hasattr(self, 'label_combo'):
                    self.update_label_combo()
            
            # 检查状态并更新UI
            if hasattr(self, 'check_annotation_ready'):
                self.check_annotation_ready()
            if hasattr(self, 'check_detection_ready'):
                self.check_detection_ready()
                
            return True
        except Exception as e:
            print(f"应用配置到标注标签页时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False 

    def open_validation_folder(self):
        """打开验证集文件夹"""
        if not self.processed_folder:
            QMessageBox.warning(self, "警告", "请先选择处理后的图片文件夹!")
            return
            
        # 发出打开验证集文件夹信号
        self.open_validation_folder_signal.emit(self.processed_folder)
        self.update_status("正在打开验证集文件夹...") 