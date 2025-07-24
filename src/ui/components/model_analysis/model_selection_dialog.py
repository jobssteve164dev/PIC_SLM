"""
模型选择对话框组件
用于模型工厂中选择多个模型进行对比分析
"""

import os
import json
from typing import List, Dict, Any, Optional
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QFileDialog, QListWidget, QListWidgetItem, QMessageBox, QComboBox,
    QDialogButtonBox, QSplitter, QTextEdit, QCheckBox, QFrame, QGridLayout
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon


class ModelSelectionDialog(QDialog):
    """模型选择对话框"""
    
    # 定义信号
    models_selected = pyqtSignal(list)  # 发送选中的模型信息列表
    
    def __init__(self, parent=None, title="选择对比模型"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(800, 600)
        
        # 初始化数据
        self.models_dir = ""
        self.test_data_dir = ""
        self.available_models = []
        self.selected_models_data = []
        
        # 支持的模型架构
        self.classification_architectures = [
            "MobileNetV2", "MobileNetV3", "ResNet18", "ResNet34", "ResNet50", 
            "ResNet101", "ResNet152", "EfficientNetB0", "EfficientNetB1", 
            "EfficientNetB2", "EfficientNetB3", "EfficientNetB4", "VGG16", 
            "VGG19", "DenseNet121", "DenseNet169", "DenseNet201", "InceptionV3", "Xception"
        ]
        
        self.detection_architectures = [
            "YOLOv5", "YOLOv8", "YOLOv7", "YOLOv6", "YOLOv4", "YOLOv3",
            "SSD", "SSD512", "SSD300", "Faster R-CNN", "Mask R-CNN",
            "RetinaNet", "DETR"
        ]
        
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # 添加标题说明
        title_label = QLabel("模型对比分析 - 自定义模型选择")
        title_label.setFont(QFont('微软雅黑', 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：模型配置和选择
        left_widget = self.create_left_panel()
        splitter.addWidget(left_widget)
        
        # 右侧：选中模型预览
        right_widget = self.create_right_panel()
        splitter.addWidget(right_widget)
        
        # 设置分割器比例
        splitter.setSizes([500, 300])
        layout.addWidget(splitter)
        
        # 底部按钮
        button_layout = self.create_button_layout()
        layout.addLayout(button_layout)
        
    def create_left_panel(self):
        """创建左侧面板"""
        widget = QFrame()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 模型目录配置组
        config_group = QGroupBox("模型配置")
        config_layout = QGridLayout()
        
        # 模型目录选择
        config_layout.addWidget(QLabel("模型目录:"), 0, 0)
        self.models_path_label = QLabel("请选择包含模型的目录")
        self.models_path_label.setStyleSheet("QLabel { border: 1px solid gray; padding: 5px; background-color: white; }")
        config_layout.addWidget(self.models_path_label, 0, 1)
        
        models_btn = QPushButton("浏览...")
        models_btn.clicked.connect(self.select_models_dir)
        config_layout.addWidget(models_btn, 0, 2)
        
        # 测试集目录选择
        config_layout.addWidget(QLabel("测试集目录:"), 1, 0)
        self.test_data_path_label = QLabel("请选择测试集数据目录")
        self.test_data_path_label.setStyleSheet("QLabel { border: 1px solid gray; padding: 5px; background-color: white; }")
        config_layout.addWidget(self.test_data_path_label, 1, 1)
        
        test_data_btn = QPushButton("浏览...")
        test_data_btn.clicked.connect(self.select_test_data_dir)
        config_layout.addWidget(test_data_btn, 1, 2)
        
        # 模型类型选择
        config_layout.addWidget(QLabel("模型类型:"), 2, 0)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["分类模型", "检测模型"])
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
        config_layout.addWidget(self.model_type_combo, 2, 1)
        
        # 模型架构选择
        config_layout.addWidget(QLabel("模型架构:"), 2, 2)
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(self.classification_architectures)
        config_layout.addWidget(self.arch_combo, 2, 3)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # 可用模型列表组
        models_group = QGroupBox("可用模型")
        models_layout = QVBoxLayout()
        
        # 刷新按钮
        refresh_layout = QHBoxLayout()
        refresh_btn = QPushButton("刷新模型列表")
        refresh_btn.clicked.connect(self.refresh_model_list)
        refresh_layout.addWidget(refresh_btn)
        refresh_layout.addStretch()
        models_layout.addLayout(refresh_layout)
        
        # 模型列表
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.MultiSelection)
        self.model_list.setMinimumHeight(200)
        self.model_list.itemSelectionChanged.connect(self.on_model_selection_changed)
        models_layout.addWidget(self.model_list)
        
        # 选择控制按钮
        select_layout = QHBoxLayout()
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(self.select_all_models)
        clear_btn = QPushButton("清除选择")
        clear_btn.clicked.connect(self.clear_selection)
        
        select_layout.addWidget(select_all_btn)
        select_layout.addWidget(clear_btn)
        select_layout.addStretch()
        models_layout.addLayout(select_layout)
        
        models_group.setLayout(models_layout)
        layout.addWidget(models_group)
        
        return widget
        
    def create_right_panel(self):
        """创建右侧面板"""
        widget = QFrame()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 选中模型预览组
        preview_group = QGroupBox("选中模型预览")
        preview_layout = QVBoxLayout()
        
        # 统计信息
        self.selection_info_label = QLabel("未选择模型")
        self.selection_info_label.setFont(QFont('微软雅黑', 10, QFont.Bold))
        self.selection_info_label.setAlignment(Qt.AlignCenter)
        self.selection_info_label.setStyleSheet("color: #666; padding: 10px;")
        preview_layout.addWidget(self.selection_info_label)
        
        # 选中模型详情
        self.selected_models_text = QTextEdit()
        self.selected_models_text.setReadOnly(True)
        self.selected_models_text.setMaximumHeight(300)
        self.selected_models_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
            }
        """)
        preview_layout.addWidget(self.selected_models_text)
        
        # 高级选项
        advanced_group = QGroupBox("高级选项")
        advanced_layout = QVBoxLayout()
        
        self.include_metrics_cb = QCheckBox("包含性能指标模拟")
        self.include_metrics_cb.setChecked(True)
        self.include_metrics_cb.setToolTip("为选中的模型生成模拟的性能指标数据")
        advanced_layout.addWidget(self.include_metrics_cb)
        
        self.auto_analyze_cb = QCheckBox("自动进行LLM分析")
        self.auto_analyze_cb.setChecked(True)
        self.auto_analyze_cb.setToolTip("选择模型后自动调用LLM进行对比分析")
        advanced_layout.addWidget(self.auto_analyze_cb)
        
        advanced_group.setLayout(advanced_layout)
        preview_layout.addWidget(advanced_group)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        return widget
        
    def create_button_layout(self):
        """创建底部按钮布局"""
        layout = QHBoxLayout()
        
        # 添加帮助按钮
        help_btn = QPushButton("帮助")
        help_btn.clicked.connect(self.show_help)
        layout.addWidget(help_btn)
        
        layout.addStretch()
        
        # 标准对话框按钮
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self
        )
        button_box.accepted.connect(self.accept_selection)
        button_box.rejected.connect(self.reject)
        
        # 自定义按钮文本
        button_box.button(QDialogButtonBox.Ok).setText("开始对比分析")
        button_box.button(QDialogButtonBox.Cancel).setText("取消")
        
        layout.addWidget(button_box)
        return layout
        
    def select_models_dir(self):
        """选择模型目录"""
        folder = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if folder:
            self.models_dir = folder
            self.models_path_label.setText(folder)
            self.refresh_model_list()
            
    def select_test_data_dir(self):
        """选择测试集数据目录"""
        folder = QFileDialog.getExistingDirectory(self, "选择测试集数据目录")
        if folder:
            self.test_data_dir = folder
            self.test_data_path_label.setText(folder)
            
    def on_model_type_changed(self, model_type):
        """模型类型变化时更新架构列表"""
        self.arch_combo.clear()
        if model_type == "分类模型":
            self.arch_combo.addItems(self.classification_architectures)
        else:  # 检测模型
            self.arch_combo.addItems(self.detection_architectures)
            
    def refresh_model_list(self):
        """刷新模型列表"""
        self.model_list.clear()
        self.available_models.clear()
        
        if not self.models_dir or not os.path.exists(self.models_dir):
            QMessageBox.warning(self, "警告", "请先选择有效的模型目录")
            return
            
        try:
            # 查找模型文件
            model_extensions = ('.h5', '.pb', '.tflite', '.pth', '.pt')
            for file in os.listdir(self.models_dir):
                if file.lower().endswith(model_extensions):
                    model_path = os.path.join(self.models_dir, file)
                    model_info = {
                        'name': file,
                        'path': model_path,
                        'size': os.path.getsize(model_path),
                        'type': self.model_type_combo.currentText(),
                        'architecture': self.arch_combo.currentText()
                    }
                    self.available_models.append(model_info)
                    
                    # 添加到列表
                    item = QListWidgetItem(f"{file} ({self._format_file_size(model_info['size'])})")
                    item.setData(Qt.UserRole, model_info)
                    self.model_list.addItem(item)
                    
            if not self.available_models:
                QMessageBox.information(self, "提示", "在指定目录中未找到模型文件")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"刷新模型列表失败: {str(e)}")
            
    def _format_file_size(self, size_bytes):
        """格式化文件大小"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f} MB"
        else:
            return f"{size_bytes/(1024**3):.1f} GB"
            
    def select_all_models(self):
        """选择所有模型"""
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            item.setSelected(True)
            
    def clear_selection(self):
        """清除所有选择"""
        self.model_list.clearSelection()
        
    def on_model_selection_changed(self):
        """模型选择变化时更新预览"""
        selected_items = self.model_list.selectedItems()
        self.selected_models_data = []
        
        for item in selected_items:
            model_info = item.data(Qt.UserRole)
            if model_info:
                self.selected_models_data.append(model_info)
                
        # 更新统计信息
        count = len(self.selected_models_data)
        if count == 0:
            self.selection_info_label.setText("未选择模型")
            self.selected_models_text.clear()
        elif count == 1:
            self.selection_info_label.setText("已选择 1 个模型（不足对比，需至少2个）")
            self.selection_info_label.setStyleSheet("color: #ff6b6b; padding: 10px; font-weight: bold;")
        else:
            self.selection_info_label.setText(f"已选择 {count} 个模型，可以进行对比分析")
            self.selection_info_label.setStyleSheet("color: #28a745; padding: 10px; font-weight: bold;")
            
        # 更新详情文本
        self.update_preview_text()
        
    def update_preview_text(self):
        """更新预览文本"""
        if not self.selected_models_data:
            self.selected_models_text.setText("请从左侧列表中选择要对比的模型...")
            return
            
        preview_text = "选中的模型详情：\n" + "="*50 + "\n\n"
        
        for i, model in enumerate(self.selected_models_data, 1):
            preview_text += f"{i}. {model['name']}\n"
            preview_text += f"   路径: {model['path']}\n"
            preview_text += f"   大小: {self._format_file_size(model['size'])}\n"
            preview_text += f"   类型: {model['type']}\n"
            preview_text += f"   架构: {model['architecture']}\n\n"
            
        if len(self.selected_models_data) >= 2:
            preview_text += "✅ 可以进行对比分析\n"
            preview_text += f"将对比 {len(self.selected_models_data)} 个模型的性能指标"
        else:
            preview_text += "⚠️ 需要至少选择2个模型才能进行对比分析"
            
        self.selected_models_text.setText(preview_text)
        
    def accept_selection(self):
        """确认选择"""
        if len(self.selected_models_data) < 2:
            QMessageBox.warning(self, "警告", "请至少选择2个模型进行对比分析")
            return
            
        if not self.test_data_dir:
            reply = QMessageBox.question(
                self, "确认", 
                "未设置测试集目录，将使用模拟数据进行对比。是否继续？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
                
        # 为选中的模型生成完整的对比数据
        comparison_data = self.generate_comparison_data()
        
        # 发送信号
        self.models_selected.emit(comparison_data)
        self.accept()
        
    def generate_comparison_data(self):
        """为选中的模型生成对比数据（准备实际评估所需的信息）"""
        comparison_data = []
        
        for model in self.selected_models_data:
            model_data = {
                'model_name': model['name'].replace('.pth', '').replace('.h5', '').replace('.pt', ''),
                'model_path': model['path'],
                'architecture': model['architecture'],
                'model_type': model['type'],
                'model_size_mb': round(model['size'] / (1024*1024), 1),
                'test_data_dir': self.test_data_dir,
                'needs_evaluation': True,  # 标记需要本地评估
                'include_detailed_metrics': self.include_metrics_cb.isChecked()
            }
            
            comparison_data.append(model_data)
            
        return comparison_data
        
    def show_help(self):
        """显示帮助信息"""
        help_text = """
模型对比分析帮助

使用说明：
1. 选择模型目录：包含训练好的模型文件的文件夹
2. 选择测试集目录：用于评估模型性能的测试数据（可选）
3. 设置模型类型和架构：确保与实际模型匹配
4. 从可用模型列表中选择至少2个模型
5. 点击"开始对比分析"进行对比

支持的模型格式：
- PyTorch: .pth, .pt
- TensorFlow: .h5, .pb
- TensorFlow Lite: .tflite

注意事项：
- 至少需要选择2个模型才能进行对比
- 如果未设置测试集目录，将使用模拟数据
- 建议选择相同类型和架构的模型进行对比
- 对比结果将在模型工厂的聊天界面中显示
        """
        
        QMessageBox.information(self, "帮助", help_text)
        
    def get_selected_models(self):
        """获取选中的模型数据"""
        return self.selected_models_data 