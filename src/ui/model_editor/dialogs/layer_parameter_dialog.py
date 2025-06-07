"""
层参数编辑对话框
"""
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                           QLabel, QPushButton, QComboBox, QSpinBox, 
                           QDoubleSpinBox, QGridLayout)


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