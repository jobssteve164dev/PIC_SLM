"""
模型结构编辑器功能扩展
"""
import json
import torch
import torchvision.models as models
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QComboBox, QLineEdit, QMessageBox, 
                           QFileDialog)
from PyQt5.QtCore import Qt, QPointF

from ..utils.constants import LAYER_TYPES, CLASSIFICATION_MODELS, DETECTION_MODELS


class EditorFunctions:
    """编辑器功能混入类"""
    
    def add_layer(self):
        """添加新层"""
        dialog = QDialog(self)
        dialog.setWindowTitle("添加层")
        layout = QVBoxLayout(dialog)
        
        # 层类型选择
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("层类型:"))
        type_combo = QComboBox()
        type_combo.addItems(LAYER_TYPES)
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
                'type': type_combo.currentText(),
                'position': {'x': 0, 'y': 0}  # 初始位置
            }
            
            # 添加到层列表
            self.layers.append(layer_info)
            
            # 添加到场景
            self.scene.add_layer(layer_info)
            
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
            from_layer = from_combo.currentText()
            to_layer = to_combo.currentText()
            
            # 检查连接是否已存在
            for conn in self.connections:
                if conn['from'] == from_layer and conn['to'] == to_layer:
                    QMessageBox.warning(self, "警告", "该连接已存在")
                    return
            
            # 添加连接
            connection = {
                'from': from_layer,
                'to': to_layer
            }
            self.connections.append(connection)
            
            # 添加到场景
            self.scene.add_connection(from_layer, to_layer)
            
    def clear_all(self):
        """清除所有层和连接"""
        reply = QMessageBox.question(self, '确认清除', 
                                   '确定要清除所有层和连接吗？',
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            # 清除数据
            self.layers.clear()
            self.connections.clear()
            self.selected_layer = None
            
            # 清除场景
            self.scene.clear_all()
                    
    def import_structure(self):
        """导入模型结构"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "导入模型结构", "", "JSON文件 (*.json)")
            
        if file_name:
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 清除现有内容
                self.clear_all()
                
                # 加载层
                for layer_info in data.get('layers', []):
                    self.layers.append(layer_info)
                    
                    # 如果没有位置信息，添加默认位置
                    if 'position' not in layer_info:
                        layer_info['position'] = {'x': 0, 'y': 0}
                        
                    # 添加到场景
                    pos = QPointF(layer_info['position']['x'], layer_info['position']['y'])
                    self.scene.add_layer(layer_info, pos)
                    
                # 加载连接
                for connection in data.get('connections', []):
                    self.connections.append(connection)
                    
                    # 添加到场景
                    self.scene.add_connection(connection['from'], connection['to'])
                    
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导入失败: {str(e)}")
                
    def export_structure(self):
        """导出模型结构"""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "导出模型结构", "", "JSON文件 (*.json)")
            
        if file_name:
            try:
                # 更新层的位置信息
                for layer in self.layers:
                    layer_item = self.scene.layer_items.get(layer['name'])
                    if layer_item:
                        pos = layer_item.pos()
                        layer['position'] = {'x': pos.x(), 'y': pos.y()}
                
                data = {
                    'layers': self.layers,
                    'connections': self.connections
                }
                with open(file_name, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                QMessageBox.information(self, "成功", "模型结构已成功导出")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
                
    def import_pretrained_model(self):
        """导入预训练模型并提取其结构"""
        try:
            import torch
            import torchvision.models as models
            from torch import nn
        except ImportError:
            QMessageBox.critical(self, "错误", "无法导入PyTorch库，请确保已安装PyTorch和torchvision")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("导入预训练模型")
        layout = QVBoxLayout(dialog)
        
        # 模型架构类型选择
        arch_layout = QHBoxLayout()
        arch_layout.addWidget(QLabel("模型架构类型:"))
        arch_combo = QComboBox()
        arch_combo.addItems(["分类模型", "目标检测模型"])
        arch_layout.addWidget(arch_combo)
        layout.addLayout(arch_layout)
        
        # 模型选择
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("选择模型:"))
        model_combo = QComboBox()
        
        # 初始设置为分类模型
        model_combo.addItems(CLASSIFICATION_MODELS)
        model_layout.addWidget(model_combo)
        layout.addLayout(model_layout)
        
        # 自定义模型文件选择
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("模型文件:"))
        file_edit = QLineEdit()
        file_edit.setEnabled(False)
        file_button = QPushButton("浏览...")
        file_button.setEnabled(False)
        
        def toggle_file_controls():
            is_custom = model_combo.currentText() == "自定义模型文件"
            file_edit.setEnabled(is_custom)
            file_button.setEnabled(is_custom)
        
        # 架构类型变化时更新模型列表
        def update_model_list():
            selected_arch = arch_combo.currentText()
            current_text = model_combo.currentText()
            model_combo.clear()
            
            if selected_arch == "分类模型":
                model_combo.addItems(CLASSIFICATION_MODELS)
            else:  # 目标检测模型
                model_combo.addItems(DETECTION_MODELS)
            
            # 尝试保持之前的选择
            index = model_combo.findText(current_text)
            if index >= 0:
                model_combo.setCurrentIndex(index)
            
            toggle_file_controls()
        
        arch_combo.currentTextChanged.connect(update_model_list)
        model_combo.currentTextChanged.connect(toggle_file_controls)
        
        def browse_file():
            file_name, _ = QFileDialog.getOpenFileName(
                dialog, "选择模型文件", "", "PyTorch模型 (*.pt *.pth)")
            if file_name:
                file_edit.setText(file_name)
                
        file_button.clicked.connect(browse_file)
        
        file_layout.addWidget(file_edit)
        file_layout.addWidget(file_button)
        layout.addLayout(file_layout)
        
        # 按钮
        button_layout = QHBoxLayout()
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(dialog.accept)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        if dialog.exec_() != QDialog.Accepted:
            return
            
        # 加载选择的模型
        arch_type = arch_combo.currentText()
        model_name = model_combo.currentText()
        model = None
        
        try:
            if model_name == "自定义模型文件":
                model_path = file_edit.text()
                if not model_path:
                    QMessageBox.warning(self, "警告", "请选择一个模型文件")
                    return
                
                try:
                    model = torch.load(model_path, map_location=torch.device('cpu'))
                    if isinstance(model, dict) and 'state_dict' in model:
                        QMessageBox.warning(self, "警告", "文件包含模型状态字典，但没有模型结构定义，无法导入")
                        return
                    elif isinstance(model, dict):
                        QMessageBox.warning(self, "警告", "文件可能仅包含权重，但没有模型结构定义，无法导入")
                        return
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"加载模型文件失败: {str(e)}")
                    return
            else:
                model = self._load_predefined_model(arch_type, model_name)
                    
            if model is None:
                QMessageBox.critical(self, "错误", "无法创建模型")
                return
                
            # 提取模型结构
            self.model_extractor.extract_model_structure(model, f"{arch_type}: {model_name}", self)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            QMessageBox.critical(self, "错误", f"处理模型时出错: {str(e)}\n\n详细信息:\n{error_details}")
    
    def _load_predefined_model(self, arch_type, model_name):
        """加载预定义模型"""
        try:
            if arch_type == "分类模型":
                return self._load_classification_model(model_name)
            else:
                return self._load_detection_model(model_name)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载{model_name}模型失败: {str(e)}")
            return None
    
    def _load_classification_model(self, model_name):
        """加载分类模型"""
        if model_name == "ResNet18":
            return models.resnet18(pretrained=False)
        elif model_name == "ResNet34":
            return models.resnet34(pretrained=False)
        elif model_name == "ResNet50":
            return models.resnet50(pretrained=False)
        elif model_name == "ResNet101":
            return models.resnet101(pretrained=False)
        elif model_name == "ResNet152":
            return models.resnet152(pretrained=False)
        elif model_name == "VGG16":
            return models.vgg16(pretrained=False)
        elif model_name == "VGG19":
            return models.vgg19(pretrained=False)
        elif model_name == "DenseNet121":
            return models.densenet121(pretrained=False)
        elif model_name == "DenseNet169":
            return models.densenet169(pretrained=False)
        elif model_name == "DenseNet201":
            return models.densenet201(pretrained=False)
        elif model_name == "MobileNetV2":
            return models.mobilenet_v2(pretrained=False)
        elif model_name == "MobileNetV3Small":
            return models.mobilenet_v3_small(pretrained=False)
        elif model_name == "MobileNetV3Large":
            return models.mobilenet_v3_large(pretrained=False)
        else:
            # 尝试加载其他模型
            try:
                if model_name.startswith("EfficientNet"):
                    if model_name == "EfficientNetB0":
                        from torchvision.models import efficientnet_b0
                        return efficientnet_b0(pretrained=False)
                    elif model_name == "EfficientNetB1":
                        from torchvision.models import efficientnet_b1
                        return efficientnet_b1(pretrained=False)
                    elif model_name == "EfficientNetB2":
                        from torchvision.models import efficientnet_b2
                        return efficientnet_b2(pretrained=False)
            except (ImportError, AttributeError):
                QMessageBox.critical(self, "错误", f"您的torchvision版本不支持{model_name}，请选择其他模型")
                return None
        return None
    
    def _load_detection_model(self, model_name):
        """加载检测模型"""
        try:
            if model_name.startswith("YOLOX"):
                QMessageBox.warning(self, "提示", f"无法载入YOLOX原始模型\n将创建替代模型结构以供显示")
                return self.model_extractor.create_dummy_yolox(model_name)
            elif model_name == "FasterRCNN_ResNet50_FPN":
                try:
                    return models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
                except:
                    return models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=False)
            elif model_name == "FasterRCNN_MobileNetV3_Large_FPN":
                return models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
            elif model_name == "RetinaNet_ResNet50_FPN":
                return models.detection.retinanet_resnet50_fpn(pretrained=False)
            elif model_name == "SSD300_VGG16":
                return models.detection.ssd300_vgg16(pretrained=False)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载目标检测模型失败: {str(e)}")
            return None
        return None
                
    def on_layer_selected(self, layer_info):
        """处理层选中事件"""
        self.selected_layer = layer_info
        
    def update_layer_position(self, layer_name, pos):
        """更新层位置信息"""
        for layer in self.layers:
            if layer['name'] == layer_name:
                layer['position'] = {'x': pos.x(), 'y': pos.y()}
                break
                
    def edit_layer_parameters(self, layer_info):
        """编辑层参数"""
        from .layer_parameter_dialog import LayerParameterDialog
        
        dialog = LayerParameterDialog(layer_info, self)
        if dialog.exec_() == QDialog.Accepted:
            # 更新层参数
            updated_info = dialog.get_parameters()
            
            # 查找并更新层信息
            for i, layer in enumerate(self.layers):
                if layer['name'] == layer_info['name']:
                    self.layers[i].update(updated_info)
                    
                    # 更新图形项
                    layer_item = self.scene.layer_items.get(layer_info['name'])
                    if layer_item:
                        layer_item.layer_info.update(updated_info)
                        layer_item.update_style()
                        layer_item.update_param_text()
                        layer_item.update()
                    break
        
    def delete_layer(self, layer_name):
        """删除层"""
        reply = QMessageBox.question(self, '确认删除', 
                                   f'确定要删除层 {layer_name} 吗？',
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            # 从层列表删除
            self.layers = [layer for layer in self.layers if layer['name'] != layer_name]
            
            # 从连接列表删除相关连接
            self.connections = [conn for conn in self.connections 
                              if conn['from'] != layer_name and conn['to'] != layer_name]
            
            # 从场景删除
            self.scene.remove_layer(layer_name)
            
    def get_model_structure(self):
        """获取模型结构定义"""
        # 更新层的位置信息
        for layer in self.layers:
            layer_item = self.scene.layer_items.get(layer['name'])
            if layer_item:
                pos = layer_item.pos()
                layer['position'] = {'x': pos.x(), 'y': pos.y()}
                
        return {
            'layers': self.layers,
            'connections': self.connections
        } 