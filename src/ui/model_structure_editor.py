import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                           QLabel, QPushButton, QComboBox, QSpinBox, 
                           QDoubleSpinBox, QLineEdit, QScrollArea, QWidget,
                           QGridLayout, QMenu, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush
import json

class LayerWidget(QWidget):
    """表示单个网络层的小部件"""
    
    # 定义信号
    layer_selected = pyqtSignal(dict)  # 层被选中时发出信号
    layer_modified = pyqtSignal(dict)  # 层参数被修改时发出信号
    layer_deleted = pyqtSignal(str)    # 层被删除时发出信号
    
    def __init__(self, layer_info, parent=None):
        super().__init__(parent)
        self.layer_info = layer_info
        self.is_selected = False
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 层名称标签
        self.name_label = QLabel(self.layer_info['name'])
        self.name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.name_label)
        
        # 层类型标签
        type_label = QLabel(self.layer_info['type'])
        type_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(type_label)
        
        # 设置样式
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 5px;
            }
            QLabel {
                background: none;
                border: none;
            }
        """)
        
        # 设置固定大小
        self.setFixedSize(120, 80)
        
    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        if event.button() == Qt.LeftButton:
            self.is_selected = not self.is_selected
            self.update()
            self.layer_selected.emit(self.layer_info)
        elif event.button() == Qt.RightButton:
            self.show_context_menu(event.pos())
            
    def paintEvent(self, event):
        """绘制部件"""
        super().paintEvent(event)
        if self.is_selected:
            painter = QPainter(self)
            pen = QPen(QColor("#3399ff"), 2)
            painter.setPen(pen)
            painter.drawRect(1, 1, self.width()-2, self.height()-2)
            
    def show_context_menu(self, pos):
        """显示右键菜单"""
        menu = QMenu(self)
        
        # 编辑操作
        edit_action = menu.addAction("编辑参数")
        edit_action.triggered.connect(self.edit_parameters)
        
        # 删除操作
        delete_action = menu.addAction("删除")
        delete_action.triggered.connect(self.delete_layer)
        
        menu.exec_(self.mapToGlobal(pos))
        
    def edit_parameters(self):
        """编辑层参数"""
        dialog = LayerParameterDialog(self.layer_info, self)
        if dialog.exec_() == QDialog.Accepted:
            self.layer_info.update(dialog.get_parameters())
            self.layer_modified.emit(self.layer_info)
            
    def delete_layer(self):
        """删除层"""
        reply = QMessageBox.question(self, '确认删除', 
                                   f'确定要删除层 {self.layer_info["name"]} 吗？',
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.layer_deleted.emit(self.layer_info['name'])

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
            current_kernel = f"{self.layer_info.get('kernel_size', 3)}x{self.layer_info.get('kernel_size', 3)}"
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
            current_pool = f"{self.layer_info.get('kernel_size', 2)}x{self.layer_info.get('kernel_size', 2)}"
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
        
        tools_group.setLayout(tools_layout)
        main_layout.addWidget(tools_group)
        
        # 右侧编辑区域
        edit_group = QGroupBox("编辑区域")
        self.edit_layout = QVBoxLayout()
        
        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.layer_layout = QVBoxLayout(scroll_widget)
        scroll.setWidget(scroll_widget)
        self.edit_layout.addWidget(scroll)
        
        edit_group.setLayout(self.edit_layout)
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
                'type': type_combo.currentText()
            }
            
            # 创建层部件
            layer_widget = LayerWidget(layer_info)
            layer_widget.layer_selected.connect(self.on_layer_selected)
            layer_widget.layer_modified.connect(self.on_layer_modified)
            layer_widget.layer_deleted.connect(self.on_layer_deleted)
            
            self.layers.append(layer_info)
            self.layer_layout.addWidget(layer_widget)
            
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
            connection = {
                'from': from_combo.currentText(),
                'to': to_combo.currentText()
            }
            self.connections.append(connection)
            self.update_connection_display()
            
    def update_connection_display(self):
        """更新连接显示"""
        # TODO: 实现连接的可视化显示
        pass
        
    def clear_all(self):
        """清除所有层和连接"""
        reply = QMessageBox.question(self, '确认清除', 
                                   '确定要清除所有层和连接吗？',
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.layers.clear()
            self.connections.clear()
            # 清除所有层部件
            while self.layer_layout.count():
                item = self.layer_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                    
    def import_structure(self):
        """导入模型结构"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "导入模型结构", "", "JSON文件 (*.json)")
            
        if file_name:
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.layers = data.get('layers', [])
                    self.connections = data.get('connections', [])
                    
                # 清除现有显示
                self.clear_all()
                
                # 重新创建层部件
                for layer_info in self.layers:
                    layer_widget = LayerWidget(layer_info)
                    layer_widget.layer_selected.connect(self.on_layer_selected)
                    layer_widget.layer_modified.connect(self.on_layer_modified)
                    layer_widget.layer_deleted.connect(self.on_layer_deleted)
                    self.layer_layout.addWidget(layer_widget)
                    
                self.update_connection_display()
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导入失败: {str(e)}")
                
    def export_structure(self):
        """导出模型结构"""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "导出模型结构", "", "JSON文件 (*.json)")
            
        if file_name:
            try:
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
        
    def on_layer_modified(self, layer_info):
        """处理层修改事件"""
        for i, layer in enumerate(self.layers):
            if layer['name'] == layer_info['name']:
                self.layers[i] = layer_info
                break
                
    def on_layer_deleted(self, layer_name):
        """处理层删除事件"""
        # 删除层信息
        self.layers = [layer for layer in self.layers if layer['name'] != layer_name]
        
        # 删除相关连接
        self.connections = [conn for conn in self.connections 
                          if conn['from'] != layer_name and conn['to'] != layer_name]
        
        # 删除层部件
        for i in range(self.layer_layout.count()):
            widget = self.layer_layout.itemAt(i).widget()
            if isinstance(widget, LayerWidget) and widget.layer_info['name'] == layer_name:
                widget.deleteLater()
                break
                
        self.update_connection_display()
        
    def get_model_structure(self):
        """获取模型结构定义"""
        return {
            'layers': self.layers,
            'connections': self.connections
        } 