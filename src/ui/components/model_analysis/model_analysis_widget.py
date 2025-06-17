import logging
import json
import traceback
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QFont
from PIL import Image

from .worker import ModelAnalysisWorker
from .visualization_utils import (display_image, display_feature_visualization,
                                 display_gradcam, display_lime_explanation,
                                 display_sensitivity_analysis)
from .model_loader import load_model_from_main_window, load_class_names, preprocess_image
from .ui_components import (create_model_section, create_image_section,
                          create_analysis_section, create_results_section)


class ModelAnalysisWidget(QWidget):
    """整合的模型分析组件"""
    
    status_updated = pyqtSignal(str)
    
    def __init__(self, parent=None):
        try:
            print("ModelAnalysisWidget: 开始初始化...")
            super().__init__(parent)
            self.model = None
            self.image = None
            self.image_tensor = None
            self.class_names = []
            self.model_file = None
            self.class_info_file = None
            
            print("ModelAnalysisWidget: 创建工作线程...")
            # 创建工作线程
            self.worker = ModelAnalysisWorker()
            self.worker.analysis_finished.connect(self.on_analysis_finished)
            self.worker.progress_updated.connect(self.on_progress_updated)
            self.worker.status_updated.connect(self.status_updated)
            self.worker.error_occurred.connect(self.on_error_occurred)
            
            # 存储当前结果用于resize时重新显示
            self.current_results = {}
            
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            
            print("ModelAnalysisWidget: 初始化UI...")
            self.init_ui()
            
            # 注册事件过滤器处理resize事件
            self.installEventFilter(self)
            
            print("ModelAnalysisWidget: 初始化完成")
            
        except Exception as e:
            print(f"ModelAnalysisWidget: 初始化失败 - {str(e)}")
            traceback.print_exc()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 说明标签
        info_label = QLabel(
            "模型分析工具集：一次加载模型和图片，选择需要的分析方法。\n"
            "支持特征可视化、GradCAM、LIME解释和敏感性分析。"
        )
        info_label.setWordWrap(True)
        info_label.setFont(QFont("Arial", 10))
        layout.addWidget(info_label)
        
        # 创建模型加载区域
        self.model_section = create_model_section(self)
        layout.addWidget(self.model_section['group'])
        
        # 连接模型相关信号
        self.model_section['model_type_combo'].currentIndexChanged.connect(self.switch_model_type)
        self.model_section['model_btn'].clicked.connect(self.select_model_file)
        self.model_section['class_info_btn'].clicked.connect(self.select_class_info_file)
        self.model_section['load_model_btn'].clicked.connect(self.load_model)
        
        # 创建图片选择区域
        self.image_section = create_image_section(self)
        layout.addWidget(self.image_section['group'])
        
        # 连接图片相关信号
        self.image_section['select_image_btn'].clicked.connect(self.select_image)
        
        # 创建分析选择区域
        self.analysis_section = create_analysis_section(self)
        layout.addWidget(self.analysis_section['group'])
        
        # 连接分析相关信号
        self.analysis_section['start_analysis_btn'].clicked.connect(self.start_analysis)
        self.analysis_section['stop_analysis_btn'].clicked.connect(self.stop_analysis)
        
        # 创建结果显示区域
        self.results_section = create_results_section(self)
        layout.addWidget(self.results_section['group'])
        
    def switch_model_type(self, index):
        """切换模型类型"""
        if index == 0:  # 分类模型
            self.model_section['model_arch_combo'].clear()
            self.model_section['model_arch_combo'].addItems([
                "MobileNetV2", "MobileNetV3", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
                "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4",
                "VGG16", "VGG19", "DenseNet121", "DenseNet169", "DenseNet201", "InceptionV3", "Xception"
            ])
        else:  # 检测模型
            self.model_section['model_arch_combo'].clear()
            self.model_section['model_arch_combo'].addItems([
                "YOLOv5", "YOLOv8", "YOLOv7", "YOLOv6", "YOLOv4", "YOLOv3",
                "SSD", "SSD512", "SSD300", "Faster R-CNN", "Mask R-CNN",
                "RetinaNet", "DETR"
            ])
    
    def select_model_file(self):
        """选择模型文件"""
        file, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "模型文件 (*.h5 *.pb *.tflite *.pt *.pth);;所有文件 (*)")
        if file:
            self.model_file = file
            self.model_section['model_path_edit'].setText(file)
            self.check_ready_state()
    
    def select_class_info_file(self):
        """选择类别信息文件"""
        file, _ = QFileDialog.getOpenFileName(self, "选择类别信息文件", "", "JSON文件 (*.json);;所有文件 (*)")
        if file:
            self.class_info_file = file
            self.model_section['class_info_path_edit'].setText(file)
            self.check_ready_state()
    
    def select_image(self):
        """选择图片"""
        file, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)")
        if file:
            try:
                # 加载图片
                self.image = Image.open(file).convert('RGB')
                self.image_section['image_path_edit'].setText(file)
                
                # 显示图片
                display_image(self.image, self.image_section['original_image_label'])
                
                # 转换为tensor
                self.image_tensor = preprocess_image(self.image)
                
                self.check_ready_state()
                
            except Exception as e:
                QMessageBox.warning(self, "错误", f"无法加载图片: {str(e)}")
    
    def load_model(self):
        """加载模型"""
        if not self.model_file or not self.class_info_file:
            QMessageBox.warning(self, "警告", "请先选择模型文件和类别信息文件!")
            return
            
        try:
            # 获取模型类型和架构
            model_type = self.model_section['model_type_combo'].currentText()
            model_arch = self.model_section['model_arch_combo'].currentText()
            
            # 创建模型信息字典
            model_info = {
                "model_path": self.model_file,
                "class_info_path": self.class_info_file,
                "model_type": model_type,
                "model_arch": model_arch
            }
            
            # 加载模型
            self.model = load_model_from_main_window(self, model_info)
            
            # 加载类别名称
            self.class_names = load_class_names(self.class_info_file)
            
            # 更新类别下拉框
            self.analysis_section['class_combo'].clear()
            self.analysis_section['class_combo'].addItems(self.class_names)
            
            QMessageBox.information(self, "成功", "模型加载成功!")
            self.check_ready_state()
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
            
    def check_ready_state(self):
        """检查是否准备就绪"""
        model_ready = bool(self.model_file and self.class_info_file)
        self.model_section['load_model_btn'].setEnabled(model_ready)
        
        analysis_ready = bool(self.model and self.image)
        self.analysis_section['start_analysis_btn'].setEnabled(analysis_ready)
    
    def start_analysis(self):
        """开始分析"""
        if not self.model or not self.image:
            QMessageBox.warning(self, "警告", "请先加载模型和选择图片!")
            return
            
        # 检查是否选择了至少一种分析方法
        selected_methods = []
        if self.analysis_section['feature_checkbox'].isChecked():
            selected_methods.append("特征可视化")
        if self.analysis_section['gradcam_checkbox'].isChecked():
            selected_methods.append("GradCAM")
        if self.analysis_section['lime_checkbox'].isChecked():
            selected_methods.append("LIME解释")
        if self.analysis_section['sensitivity_checkbox'].isChecked():
            selected_methods.append("敏感性分析")
            
        if not selected_methods:
            QMessageBox.warning(self, "警告", "请至少选择一种分析方法!")
            return
            
        # 获取目标类别
        target_class = self.analysis_section['class_combo'].currentIndex()
        
        # 准备分析参数
        analysis_params = {
            'num_superpixels': self.analysis_section['num_superpixels'].value(),
            'num_samples': self.analysis_section['num_samples'].value(),
            'perturbation_range': self.analysis_section['perturbation_range'].value(),
            'num_steps': self.analysis_section['num_steps'].value()
        }
        
        # 显示进度条
        self.analysis_section['progress_bar'].setVisible(True)
        self.analysis_section['start_analysis_btn'].setEnabled(False)
        self.analysis_section['stop_analysis_btn'].setEnabled(True)
        
        # 逐个执行分析
        self.current_methods = selected_methods.copy()
        self.execute_next_analysis(target_class, analysis_params)
    
    def stop_analysis(self):
        """停止分析"""
        self.worker.stop_analysis()
        self.analysis_section['stop_analysis_btn'].setEnabled(False)
        self.status_updated.emit("正在停止分析...")
    
    def execute_next_analysis(self, target_class, analysis_params):
        """执行下一个分析"""
        if not self.current_methods:
            # 所有分析完成
            self.analysis_section['progress_bar'].setVisible(False)
            self.analysis_section['start_analysis_btn'].setEnabled(True)
            self.analysis_section['stop_analysis_btn'].setEnabled(False)
            QMessageBox.information(self, "完成", "所有分析已完成!")
            return
            
        # 获取下一个分析方法
        analysis_type = self.current_methods.pop(0)
        
        # 设置工作线程任务
        self.worker.set_analysis_task(
            analysis_type, self.model, self.image, self.image_tensor,
            self.class_names, target_class, analysis_params
        )
        
        # 启动分析
        self.worker.start()
    
    @pyqtSlot(str, object)
    def on_analysis_finished(self, analysis_type, result):
        """处理分析完成事件"""
        try:
            # 保存结果用于重新显示
            self.current_results[analysis_type] = result
            
            # 获取当前选择的类别名称
            current_class_idx = self.analysis_section['class_combo'].currentIndex()
            current_class_name = self.class_names[current_class_idx] if current_class_idx < len(self.class_names) else f"类别{current_class_idx}"
            
            if analysis_type == "特征可视化":
                display_feature_visualization(result, self.results_section['feature_viewer'])
            elif analysis_type == "GradCAM":
                display_gradcam(result, self.image, self.results_section['gradcam_viewer'], current_class_name)
            elif analysis_type == "LIME解释":
                display_lime_explanation(result, self.image, self.results_section['lime_viewer'], current_class_idx, self.class_names)
            elif analysis_type == "敏感性分析":
                display_sensitivity_analysis(result, self.results_section['sensitivity_viewer'], current_class_name)
                
            # 继续下一个分析
            target_class = self.analysis_section['class_combo'].currentIndex()
            analysis_params = {
                'num_superpixels': self.analysis_section['num_superpixels'].value(),
                'num_samples': self.analysis_section['num_samples'].value(),
                'perturbation_range': self.analysis_section['perturbation_range'].value(),
                'num_steps': self.analysis_section['num_steps'].value()
            }
            self.execute_next_analysis(target_class, analysis_params)
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"显示{analysis_type}结果失败: {str(e)}")
    
    @pyqtSlot(int)
    def on_progress_updated(self, progress):
        """更新进度"""
        self.analysis_section['progress_bar'].setValue(progress)
    
    @pyqtSlot(str)
    def on_error_occurred(self, error_msg):
        """处理错误"""
        QMessageBox.critical(self, "分析错误", error_msg)
        self.analysis_section['progress_bar'].setVisible(False)
        self.analysis_section['start_analysis_btn'].setEnabled(True)
        self.analysis_section['stop_analysis_btn'].setEnabled(False)
    
    def eventFilter(self, obj, event):
        """事件过滤器，处理resize事件"""
        if event.type() == event.Resize and obj == self:
            # 延迟重新显示图片，避免频繁更新
            QTimer.singleShot(200, self.refresh_image_displays)
        return super().eventFilter(obj, event)
    
    def refresh_image_displays(self):
        """刷新所有图片显示"""
        try:
            # 重新显示所有已有的分析结果
            current_class_idx = self.analysis_section['class_combo'].currentIndex()
            current_class_name = self.class_names[current_class_idx] if current_class_idx < len(self.class_names) else f"类别{current_class_idx}"
            
            for analysis_type, result in self.current_results.items():
                if analysis_type == "特征可视化":
                    display_feature_visualization(result, self.results_section['feature_viewer'])
                elif analysis_type == "GradCAM":
                    display_gradcam(result, self.image, self.results_section['gradcam_viewer'], current_class_name)
                elif analysis_type == "LIME解释":
                    display_lime_explanation(result, self.image, self.results_section['lime_viewer'], current_class_idx, self.class_names)
                elif analysis_type == "敏感性分析":
                    display_sensitivity_analysis(result, self.results_section['sensitivity_viewer'], current_class_name)
                    
            # 重新显示原始图片
            if self.image:
                display_image(self.image, self.image_section['original_image_label'])
                
        except Exception as e:
            self.logger.error(f"刷新图片显示失败: {str(e)}")
    
    def set_model(self, model, class_names=None):
        """从外部设置模型"""
        self.model = model
        if class_names:
            self.class_names = class_names
            self.analysis_section['class_combo'].clear()
            self.analysis_section['class_combo'].addItems(self.class_names)
        self.check_ready_state() 