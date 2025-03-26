from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                           QFileDialog, QMessageBox, QHBoxLayout, QComboBox,
                           QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox, QGridLayout, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import logging
import json

class SensitivityAnalysisWidget(QWidget):
    """敏感性分析组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.image = None
        self.image_tensor = None
        self.class_names = []
        self.model_file = None
        self.class_info_file = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 说明标签
        info_label = QLabel(
            "敏感性分析可以帮助您理解模型对输入变化的敏感程度。\n"
            "选择一张图片和目标类别后，将显示不同扰动下的预测变化。"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 创建模型文件选择组
        model_group = QGroupBox("模型文件")
        model_layout = QGridLayout()
        
        # 添加模型类型选择
        model_layout.addWidget(QLabel("模型类型:"), 0, 0)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["分类模型", "检测模型"])
        self.model_type_combo.currentIndexChanged.connect(self.switch_model_type)
        model_layout.addWidget(self.model_type_combo, 0, 1, 1, 1)
        
        # 添加模型架构选择
        model_layout.addWidget(QLabel("模型架构:"), 0, 2)
        self.model_arch_combo = QComboBox()
        self.model_arch_combo.addItems([
            "MobileNetV2", "MobileNetV3", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
            "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4",
            "VGG16", "VGG19", "DenseNet121", "DenseNet169", "DenseNet201", "InceptionV3", "Xception"
        ])
        model_layout.addWidget(self.model_arch_combo, 0, 3, 1, 1)
        
        # 模型文件选择
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText("请选择训练好的模型文件")
        
        model_btn = QPushButton("浏览...")
        model_btn.clicked.connect(self.select_model_file)
        
        model_layout.addWidget(QLabel("模型文件:"), 1, 0)
        model_layout.addWidget(self.model_path_edit, 1, 1, 1, 2)
        model_layout.addWidget(model_btn, 1, 3)
        
        # 类别信息文件
        self.class_info_path_edit = QLineEdit()
        self.class_info_path_edit.setReadOnly(True)
        self.class_info_path_edit.setPlaceholderText("请选择类别信息文件")
        
        class_info_btn = QPushButton("浏览...")
        class_info_btn.clicked.connect(self.select_class_info_file)
        
        model_layout.addWidget(QLabel("类别信息:"), 2, 0)
        model_layout.addWidget(self.class_info_path_edit, 2, 1, 1, 2)
        model_layout.addWidget(class_info_btn, 2, 3)
        
        # 加载模型按钮
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.load_model)
        self.load_model_btn.setEnabled(False)
        model_layout.addWidget(self.load_model_btn, 3, 0, 1, 4)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 图片选择按钮
        self.select_image_btn = QPushButton("选择图片")
        self.select_image_btn.clicked.connect(self.select_image)
        layout.addWidget(self.select_image_btn)
        
        # 创建参数设置表单
        param_layout = QFormLayout()
        
        # 类别选择下拉框
        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self.update_analysis)
        param_layout.addRow("目标类别:", self.class_combo)
        
        # 扰动范围设置
        self.perturbation_range = QDoubleSpinBox()
        self.perturbation_range.setRange(0.01, 1.0)
        self.perturbation_range.setValue(0.1)
        self.perturbation_range.setSingleStep(0.01)
        self.perturbation_range.valueChanged.connect(self.update_analysis)
        param_layout.addRow("扰动范围:", self.perturbation_range)
        
        # 扰动步数设置
        self.num_steps = QSpinBox()
        self.num_steps.setRange(10, 100)
        self.num_steps.setValue(20)
        self.num_steps.valueChanged.connect(self.update_analysis)
        param_layout.addRow("扰动步数:", self.num_steps)
        
        layout.addLayout(param_layout)
        
        # 创建水平布局用于显示原始图片和敏感性曲线
        image_layout = QHBoxLayout()
        
        # 显示原始图片
        self.original_image_label = QLabel("原始图片")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(300, 300)
        self.original_image_label.setStyleSheet("border: 1px solid gray;")
        image_layout.addWidget(self.original_image_label)
        
        # 显示敏感性曲线
        self.sensitivity_label = QLabel("敏感性曲线")
        self.sensitivity_label.setAlignment(Qt.AlignCenter)
        self.sensitivity_label.setMinimumSize(300, 300)
        self.sensitivity_label.setStyleSheet("border: 1px solid gray;")
        image_layout.addWidget(self.sensitivity_label)
        
        layout.addLayout(image_layout)
        
        # 添加弹性空间
        layout.addStretch()
        
    def switch_model_type(self, index):
        """切换模型类型"""
        if index == 0:  # 分类模型
            self.model_arch_combo.clear()
            self.model_arch_combo.addItems([
                "MobileNetV2", "MobileNetV3", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
                "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4",
                "VGG16", "VGG19", "DenseNet121", "DenseNet169", "DenseNet201", "InceptionV3", "Xception"
            ])
        else:  # 检测模型
            self.model_arch_combo.clear()
            self.model_arch_combo.addItems([
                "YOLOv5", "YOLOv8", "YOLOv7", "YOLOv6", "YOLOv4", "YOLOv3",
                "SSD", "SSD512", "SSD300", "Faster R-CNN", "Mask R-CNN",
                "RetinaNet", "DETR"
            ])
    
    def select_model_file(self):
        """选择模型文件"""
        file, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "模型文件 (*.h5 *.pb *.tflite *.pt *.pth);;所有文件 (*)")
        if file:
            self.model_file = file
            self.model_path_edit.setText(file)
            self.check_model_ready()
    
    def select_class_info_file(self):
        """选择类别信息文件"""
        file, _ = QFileDialog.getOpenFileName(self, "选择类别信息文件", "", "JSON文件 (*.json);;所有文件 (*)")
        if file:
            self.class_info_file = file
            self.class_info_path_edit.setText(file)
            self.check_model_ready()
    
    def check_model_ready(self):
        """检查是否可以加载模型"""
        is_ready = bool(self.model_file and self.class_info_file)
        self.load_model_btn.setEnabled(is_ready)
        return is_ready
        
    def load_model(self):
        """加载模型"""
        if not self.model_file or not self.class_info_file:
            QMessageBox.warning(self, "警告", "请先选择模型文件和类别信息文件!")
            return
            
        try:
            # 获取模型类型和架构
            model_type = self.model_type_combo.currentText()
            model_arch = self.model_arch_combo.currentText()
            
            # 创建模型信息字典
            model_info = {
                "model_path": self.model_file,
                "class_info_path": self.class_info_file,
                "model_type": model_type,
                "model_arch": model_arch
            }
            
            # 寻找主窗口引用
            main_window = None
            parent = self.parent()
            
            # 逐级向上查找主窗口
            while parent:
                if hasattr(parent, 'worker') and hasattr(parent.worker, 'predictor'):
                    main_window = parent
                    break
                elif hasattr(parent, 'main_window'):
                    main_window = parent.main_window
                    break
                parent = parent.parent()
                
            if main_window and hasattr(main_window, 'worker') and hasattr(main_window.worker, 'predictor'):
                # 使用找到的主窗口加载模型
                main_window.worker.predictor.load_model_with_info(model_info)
                
                # 获取加载后的模型
                self.model = main_window.worker.predictor.model
                
                # 加载类别名称
                try:
                    with open(self.class_info_file, 'r', encoding='utf-8') as f:
                        class_data = json.load(f)
                    
                    # 保存原始类别数据
                    self.class_names = class_data
                    
                    # 根据类别数据类型准备下拉框内容
                    class_display_names = []
                    if isinstance(class_data, list):
                        # 如果是列表，直接使用
                        class_display_names = class_data
                    elif isinstance(class_data, dict):
                        # 如果是字典，使用值或键
                        if all(isinstance(v, str) for v in class_data.values()):
                            # 值是字符串，使用值
                            class_display_names = list(class_data.values())
                        else:
                            # 否则使用键
                            class_display_names = [str(k) for k in class_data.keys()]
                    
                    # 确保至少有一个类别
                    if not class_display_names:
                        class_display_names = ["类别0"]
                        self.logger.warning("类别信息为空，使用默认类别")
                    
                    # 更新类别下拉框
                    self.class_combo.clear()
                    self.class_combo.addItems(class_display_names)
                    
                    # 记录类别加载信息
                    self.logger.info(f"已加载 {len(class_display_names)} 个类别")
                    self.logger.info(f"类别信息格式: {type(class_data).__name__}")
                    
                except Exception as e:
                    self.logger.error(f"读取类别信息文件失败: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    QMessageBox.critical(self, "错误", f"读取类别信息文件失败: {str(e)}")
                
                # 模型加载成功
                QMessageBox.information(self, "成功", f"模型 {model_arch} 加载成功！\n现在可以进行敏感性分析。")
                
                # 禁用加载模型按钮，表示模型已加载
                self.load_model_btn.setEnabled(False)
                
                # 如果已经选择了图像，更新分析
                if self.image_tensor is not None and self.class_combo.count() > 0:
                    self.update_analysis()
            else:
                QMessageBox.critical(self, "错误", "无法访问模型加载器，请检查应用配置")
                
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            return
    
    def select_image(self):
        """选择图片"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择图片",
                "",
                "图片文件 (*.png *.jpg *.jpeg *.bmp)"
            )
            
            if file_path:
                # 加载图片
                self.image = Image.open(file_path)
                # 转换为RGB模式
                if self.image.mode != 'RGB':
                    self.image = self.image.convert('RGB')
                # 调整大小
                self.image = self.image.resize((224, 224))
                # 转换为tensor
                self.image_tensor = torch.from_numpy(np.array(self.image)).float()
                self.image_tensor = self.image_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
                
                # 显示原始图片
                self.display_image(self.image, self.original_image_label)
                
                # 如果模型已加载，则更新分析
                if self.model is not None:
                    self.update_analysis()
                    
        except Exception as e:
            self.logger.error(f"选择图片时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"选择图片时出错: {str(e)}")
            
    def display_image(self, image, label):
        """显示图片"""
        try:
            # 将PIL图片转换为QPixmap
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            qimage = QImage.fromData(buffer.getvalue())
            pixmap = QPixmap.fromImage(qimage)
            
            # 调整大小以适应标签
            scaled_pixmap = pixmap.scaled(
                label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.logger.error(f"显示图片时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"显示图片时出错: {str(e)}")
            
    def perform_sensitivity_analysis(self, target_class):
        """执行敏感性分析"""
        try:
            if self.model is None or self.image_tensor is None:
                return None, None
                
            # 将图片移到正确的设备上
            device = next(self.model.parameters()).device
            image_tensor = self.image_tensor.to(device)
            
            # 获取扰动参数
            perturbation_range = self.perturbation_range.value()
            num_steps = self.num_steps.value()
            
            # 创建扰动序列
            perturbations = torch.linspace(-perturbation_range, perturbation_range, num_steps)
            predictions = []
            
            # 测试一次模型输出，确保能正确处理
            with torch.no_grad():
                test_output = self.model(image_tensor)
                
                # 检查输出类型和形状
                if not isinstance(test_output, torch.Tensor):
                    self.logger.warning(f"模型输出类型异常: {type(test_output)}")
                    QMessageBox.warning(self, "警告", f"模型输出类型异常: {type(test_output)}")
                    return None, None
                
                # 检查输出维度和类别索引
                if test_output.dim() < 2:
                    self.logger.warning(f"模型输出维度异常: {test_output.shape}")
                    QMessageBox.warning(self, "警告", f"模型输出维度异常: {test_output.shape}")
                    return None, None
                
                if target_class >= test_output.size(1):
                    self.logger.warning(f"类别索引 {target_class} 超出范围 (0-{test_output.size(1)-1})")
                    QMessageBox.warning(self, "警告", f"类别索引 {target_class} 超出范围 (0-{test_output.size(1)-1})")
                    return None, None
            
            # 对每个扰动值进行预测
            for delta in perturbations:
                try:
                    # 创建扰动后的图片
                    perturbed_image = image_tensor.clone()
                    # 添加扰动
                    perturbed_image = perturbed_image + delta
                    # 确保像素值在[0,1]范围内
                    perturbed_image = torch.clamp(perturbed_image, 0, 1)
                    
                    # 获取预测
                    with torch.no_grad():
                        output = self.model(perturbed_image)
                        
                        # 确保输出是张量且维度正确
                        if not isinstance(output, torch.Tensor):
                            self.logger.warning(f"扰动值 {delta} 的模型输出类型异常")
                            predictions.append(0.0)
                            continue
                            
                        if output.dim() < 2 or target_class >= output.size(1):
                            self.logger.warning(f"扰动值 {delta} 的模型输出维度异常或类别索引超出范围")
                            predictions.append(0.0)
                            continue
                        
                        # 安全地计算softmax和获取目标类别的概率
                        try:
                            probs = torch.softmax(output, dim=1)
                            prob_value = probs[0, target_class].item()
                            predictions.append(prob_value)
                        except Exception as inner_e:
                            self.logger.error(f"计算概率时出错: {str(inner_e)}")
                            self.logger.error(f"输出形状: {output.shape}, 目标类别: {target_class}")
                            predictions.append(0.0)
                        
                except Exception as e:
                    self.logger.error(f"在扰动值 {delta} 处预测失败: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    # 继续下一个扰动值，而不是完全失败
                    predictions.append(0.0)  # 添加一个占位符值
            
            # 检查是否有足够的有效预测
            if len(predictions) < num_steps // 2:
                self.logger.warning("有效预测数量不足")
                QMessageBox.warning(self, "警告", "大部分预测失败，结果可能不准确")
            elif all(p == 0.0 for p in predictions):
                self.logger.warning("所有预测值都为0，分析结果可能不准确")
                QMessageBox.warning(self, "警告", "所有预测值都为0，分析结果可能不准确")
            
            return perturbations.numpy(), predictions
            
        except Exception as e:
            self.logger.error(f"执行敏感性分析时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"执行敏感性分析时出错: {str(e)}")
            return None, None
            
    def update_analysis(self):
        """更新敏感性分析显示"""
        try:
            if self.class_combo.currentIndex() < 0:
                self.logger.warning("未选择有效的类别，无法执行分析")
                return
                
            if not self.model:
                self.logger.warning("模型未加载，无法执行敏感性分析")
                return
                
            if self.image_tensor is None:  # 修正条件逻辑
                self.logger.warning("图像未加载，无法执行敏感性分析")
                return
                
            target_class = self.class_combo.currentIndex()
            self.logger.info(f"执行目标类别 {target_class} 的敏感性分析")
            self.logger.info(f"图像张量形状: {self.image_tensor.shape}")
            self.logger.info(f"类别总数: {self.class_combo.count()}")
            
            # 获取类别名称，支持字典或列表类型的class_names
            class_name = self.get_class_name(target_class)
            self.logger.info(f"使用类别名称: {class_name}")
            
            perturbations, predictions = self.perform_sensitivity_analysis(target_class)
            
            if perturbations is not None and predictions is not None and len(predictions) > 0:
                self.logger.info(f"生成敏感性分析结果: {len(predictions)} 个预测点")
                # 创建matplotlib图形
                plt.figure(figsize=(10, 6))
                plt.plot(perturbations, predictions, 'b-', label='预测概率')
                plt.title(f'类别 {class_name} 的敏感性分析')
                plt.xlabel('输入扰动')
                plt.ylabel('预测概率')
                plt.grid(True)
                plt.legend()
                
                # 保存图形为临时文件而不是内存缓冲区
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        temp_filename = tmp.name
                        plt.savefig(temp_filename, format='png', bbox_inches='tight', pad_inches=0)
                        plt.close()
                        
                        # 读取临时文件并显示
                        pixmap = QPixmap(temp_filename)
                        
                        # 清理临时文件
                        import os
                        try:
                            os.unlink(temp_filename)
                        except:
                            pass
                except Exception as img_e:
                    # 回退到内存缓冲区方法
                    self.logger.warning(f"使用临时文件失败，回退到内存缓冲区: {str(img_e)}")
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
                    plt.close()
                    buffer.seek(0)
                    
                    # 显示敏感性曲线
                    qimage = QImage.fromData(buffer.getvalue())
                    pixmap = QPixmap.fromImage(qimage)
                
                # 调整大小以适应标签
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        self.sensitivity_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    
                    self.sensitivity_label.setPixmap(scaled_pixmap)
                    self.logger.info("敏感性分析图像已显示")
                else:
                    self.logger.error("生成的图像为空")
                    QMessageBox.warning(self, "警告", "无法生成敏感性分析图像")
            else:
                self.logger.warning("敏感性分析未返回有效数据")
                QMessageBox.warning(self, "警告", "敏感性分析未能生成有效数据")
                
        except Exception as e:
            self.logger.error(f"更新敏感性分析显示时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"更新敏感性分析显示时出错: {str(e)}")
            
    def debug_class_names(self):
        """调试类别名称"""
        try:
            self.logger.info("类别名称调试信息:")
            self.logger.info(f"类别名称类型: {type(self.class_names).__name__}")
            if isinstance(self.class_names, list):
                self.logger.info(f"类别名称列表: {self.class_names[:10]}...")
            elif isinstance(self.class_names, dict):
                keys = list(self.class_names.keys())[:10]
                self.logger.info(f"类别名称字典键: {keys}...")
                values = [self.class_names[k] for k in keys]
                self.logger.info(f"类别名称字典值: {values}...")
            else:
                self.logger.info(f"类别名称内容: {str(self.class_names)[:100]}...")
                
            self.logger.info(f"下拉框项目数: {self.class_combo.count()}")
            items = [self.class_combo.itemText(i) for i in range(min(10, self.class_combo.count()))]
            self.logger.info(f"下拉框前10项: {items}")
        except Exception as e:
            self.logger.error(f"调试类别名称时出错: {str(e)}")
    
    def get_class_name(self, class_index):
        """安全地获取类别名称，支持多种类别名称格式"""
        try:
            # 输出调试信息
            self.debug_class_names()
            
            # 如果是列表
            if isinstance(self.class_names, list) and 0 <= class_index < len(self.class_names):
                return self.class_names[class_index]
            # 如果是字典，尝试使用整数键
            elif isinstance(self.class_names, dict) and class_index in self.class_names:
                return self.class_names[class_index]
            # 如果是字典，尝试使用字符串键
            elif isinstance(self.class_names, dict) and str(class_index) in self.class_names:
                return self.class_names[str(class_index)]
            # 如果都失败，使用下拉框中的文本
            elif 0 <= class_index < self.class_combo.count():
                return self.class_combo.itemText(class_index)
            # 如果都不行，使用默认文本
            else:
                return f"类别{class_index}"
        except Exception as e:
            self.logger.error(f"获取类别名称出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return f"类别{class_index}"  # 提供一个默认值
    
    def set_model(self, model, class_names=None):
        """设置模型和类别名称"""
        self.model = model
        if class_names is not None:
            self.class_names = class_names
            self.class_combo.clear()
            self.class_combo.addItems(class_names)
            # 只在有图片和类别时才更新解释
            if self.image_tensor is not None and len(self.class_names) > 0 and self.isVisible():
                self.update_analysis() 