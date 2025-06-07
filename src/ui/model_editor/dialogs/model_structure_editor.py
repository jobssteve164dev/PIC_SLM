"""
模型结构编辑器主对话框
"""
import json
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                           QLabel, QPushButton, QComboBox, QLineEdit,
                           QMessageBox, QFileDialog, QWidget, QFrame)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QBrush

from ..graphics_scene import NetworkGraphicsScene, NetworkGraphicsView
from ..utils.constants import LAYER_TYPES, LAYER_COLORS
from ..utils.model_extractor import ModelExtractor
from .layer_parameter_dialog import LayerParameterDialog
from .editor_functions import EditorFunctions


class ModelStructureEditor(QDialog, EditorFunctions):
    """模型结构编辑器对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layers = []  # 存储所有层信息
        self.connections = []  # 存储层之间的连接
        self.selected_layer = None
        self.model_extractor = ModelExtractor(self)
        
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
        
        # 添加导入预训练模型按钮
        import_model_button = QPushButton("导入预训练模型")
        import_model_button.clicked.connect(self.import_pretrained_model)
        tools_layout.addWidget(import_model_button)
        
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
        for layer_type, color in sorted(LAYER_COLORS.items()):
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
        self.edit_group = QGroupBox("编辑区域")
        edit_layout = QVBoxLayout(self.edit_group)
        
        # 创建场景
        self.scene = NetworkGraphicsScene(self)
        
        # 创建视图
        self.view = NetworkGraphicsView(self.scene)
        edit_layout.addWidget(self.view)
        
        main_layout.addWidget(self.edit_group)
        
        # 设置布局比例
        main_layout.setStretch(0, 1)  # 工具栏占1
        main_layout.setStretch(1, 4)  # 编辑区域占4
        
        # 创建控制按钮和导航指示器
        self.create_control_widgets()
        
    def create_control_widgets(self):
        """创建控制控件，添加到编辑区域"""
        # 缩放控制按钮
        self.create_zoom_controls()
        # 导航指示器
        self.create_navigation_indicator()
        
    def create_zoom_controls(self):
        """创建缩放控制按钮"""
        # 创建缩放控制容器
        self.zoom_container = QWidget(self.edit_group)
        self.zoom_container.setObjectName("zoomControls")
        
        # 设置样式
        self.zoom_container.setStyleSheet("""
            QWidget#zoomControls {
                background-color: rgba(240, 240, 240, 180);
                border: 1px solid #aaaaaa;
                border-radius: 5px;
            }
            QPushButton {
                font-weight: bold;
                min-width: 30px;
                min-height: 30px;
                padding: 5px;
                border-radius: 4px;
                background-color: white;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        
        # 创建布局
        zoom_layout = QVBoxLayout(self.zoom_container)
        zoom_layout.setContentsMargins(4, 4, 4, 4)
        zoom_layout.setSpacing(4)
        
        # 创建放大按钮
        self.zoom_in_btn = QPushButton("+", self.zoom_container)
        self.zoom_in_btn.setToolTip("放大")
        self.zoom_in_btn.clicked.connect(self.view.zoom_in)
        
        # 创建缩小按钮
        self.zoom_out_btn = QPushButton("-", self.zoom_container)
        self.zoom_out_btn.setToolTip("缩小")
        self.zoom_out_btn.clicked.connect(self.view.zoom_out)
        
        # 创建重置按钮
        self.zoom_reset_btn = QPushButton("⟲", self.zoom_container)
        self.zoom_reset_btn.setToolTip("重置视图")
        self.zoom_reset_btn.clicked.connect(self.view.reset_view)
        
        # 创建导航开关按钮
        self.nav_toggle_btn = QPushButton("🧭", self.zoom_container)
        self.nav_toggle_btn.setToolTip("显示/隐藏导航指示器")
        self.nav_toggle_btn.clicked.connect(self.toggle_navigation)
        
        # 添加按钮到布局
        zoom_layout.addWidget(self.zoom_in_btn)
        zoom_layout.addWidget(self.zoom_out_btn)
        zoom_layout.addWidget(self.zoom_reset_btn)
        zoom_layout.addWidget(self.nav_toggle_btn)
        
        # 调整大小以适应内容
        self.zoom_container.adjustSize()
        
        # 初始位置（右上角）
        self.position_zoom_controls()
        
        # 确保控件在顶层显示
        self.zoom_container.raise_()
        
    def create_navigation_indicator(self):
        """创建导航指示器，显示当前视图在整个场景中的位置"""
        # 创建导航指示器
        self.nav_indicator = QWidget(self.edit_group)
        self.nav_indicator.setObjectName("navIndicator")
        self.nav_indicator.setFixedSize(120, 120)
        self.nav_indicator.setStyleSheet("""
            QWidget#navIndicator {
                background-color: rgba(240, 240, 240, 180);
                border: 1px solid #aaaaaa;
                border-radius: 5px;
            }
        """)
        
        # 创建布局
        nav_layout = QVBoxLayout(self.nav_indicator)
        nav_layout.setContentsMargins(5, 5, 5, 5)
        
        # 创建导航视图标签
        self.nav_label = QLabel(self.nav_indicator)
        self.nav_label.setMinimumSize(110, 110)
        self.nav_label.setAlignment(Qt.AlignCenter)
        self.nav_label.setText("可视区域")
        nav_layout.addWidget(self.nav_label)
        
        # 默认隐藏导航指示器
        self.nav_indicator.hide()
        
        # 初始位置（左下角）
        self.position_navigation_indicator()
        
        # 确保控件在顶层显示
        self.nav_indicator.raise_()
        
    def toggle_navigation(self):
        """切换导航指示器显示状态"""
        if self.nav_indicator.isVisible():
            self.nav_indicator.hide()
        else:
            self.nav_indicator.show()
            self.update_navigation_indicator()
    
    def position_zoom_controls(self):
        """设置缩放控制按钮位置"""
        # 获取编辑区域的大小
        edit_rect = self.edit_group.rect()
        margin = 15
        
        # 设置位置（右上角）
        self.zoom_container.move(
            edit_rect.width() - self.zoom_container.width() - margin,
            margin
        )
    
    def position_navigation_indicator(self):
        """设置导航指示器位置"""
        # 获取编辑区域的大小
        edit_rect = self.edit_group.rect()
        margin = 15
        
        # 设置位置（左下角）
        self.nav_indicator.move(
            margin,
            edit_rect.height() - self.nav_indicator.height() - margin
        )
    
    def update_navigation_indicator(self):
        """更新导航指示器显示的内容"""
        if not self.nav_indicator.isVisible():
            return
            
        # 获取当前视口和场景矩形
        viewport_rect = self.view.viewport().rect()
        scene_rect = self.view.sceneRect()
        
        # 计算当前视口在场景中的位置
        viewport_scene_rect = self.view.mapToScene(viewport_rect).boundingRect()
        
        # 获取场景中所有项的边界矩形
        items_rect = self.scene.itemsBoundingRect()
        
        # 更新场景矩形，确保它包含所有项
        effective_rect = scene_rect.united(items_rect)
        
        # 计算导航指示器中的显示比例
        nav_width = self.nav_label.width()
        nav_height = self.nav_label.height()
        
        # 创建导航图像
        nav_image = QPixmap(nav_width, nav_height)
        nav_image.fill(Qt.white)
        
        # 绘制整个场景和当前视口位置
        painter = QPainter(nav_image)
        painter.setPen(QPen(Qt.lightGray, 1))
        
        # 缩放因子，确保整个有效区域能显示在导航图中
        scale_x = nav_width / effective_rect.width()
        scale_y = nav_height / effective_rect.height()
        scale = min(scale_x, scale_y) * 0.9  # 留些边距
        
        # 计算绘制位置的偏移量，使内容居中
        x_offset = (nav_width - effective_rect.width() * scale) / 2
        y_offset = (nav_height - effective_rect.height() * scale) / 2
        
        # 绘制所有项的边界
        painter.setPen(QPen(Qt.darkGray, 1))
        item_x = x_offset + (items_rect.x() - effective_rect.x()) * scale
        item_y = y_offset + (items_rect.y() - effective_rect.y()) * scale
        item_w = items_rect.width() * scale
        item_h = items_rect.height() * scale
        painter.drawRect(int(item_x), int(item_y), int(item_w), int(item_h))
        
        # 绘制当前视口位置
        viewport_x = x_offset + (viewport_scene_rect.x() - effective_rect.x()) * scale
        viewport_y = y_offset + (viewport_scene_rect.y() - effective_rect.y()) * scale
        viewport_w = viewport_scene_rect.width() * scale
        viewport_h = viewport_scene_rect.height() * scale
        
        # 绘制当前可视区域
        painter.setBrush(QBrush(QColor(100, 100, 255, 100)))
        painter.drawRect(int(viewport_x), int(viewport_y), int(viewport_w), int(viewport_h))
        
        # 结束绘制
        painter.end()
        
        # 更新导航标签
        self.nav_label.setPixmap(nav_image)
    
    def resizeEvent(self, event):
        """处理窗口大小变化事件"""
        super().resizeEvent(event)
        
        # 重新定位控件
        self.position_zoom_controls()
        self.position_navigation_indicator()
        
        # 更新导航指示器
        self.update_navigation_indicator() 