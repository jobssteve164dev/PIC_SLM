from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QGroupBox, QGridLayout, QListWidget,
                           QListWidgetItem, QMessageBox, QTableWidget, QTableWidgetItem,
                           QHeaderView, QTabWidget, QTextEdit, QProgressBar, QSplitter,
                           QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont
import os
import json
import sys
import time
import torch
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_recall_fscore_support, roc_auc_score, 
                           average_precision_score, accuracy_score)
from sklearn.preprocessing import label_binarize
from collections import defaultdict


class ModelEvaluationThread(QThread):
    """模型评估线程"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    evaluation_finished = pyqtSignal(dict)
    evaluation_error = pyqtSignal(str)
    
    def __init__(self, model_path, model_arch, class_info_path, test_dir, task_type):
        super().__init__()
        self.model_path = model_path
        self.model_arch = model_arch
        self.class_info_path = class_info_path
        self.test_dir = test_dir
        self.task_type = task_type
        
    def run(self):
        """执行模型评估"""
        try:
            # 添加src目录到sys.path
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
            from predictor import Predictor
            
            # 创建预测器
            predictor = Predictor()
            
            # 加载模型
            model_info = {
                'model_path': self.model_path,
                'class_info_path': self.class_info_path,
                'model_type': self.task_type,
                'model_arch': self.model_arch
            }
            
            self.status_updated.emit("正在加载模型...")
            predictor.load_model_with_info(model_info)
            
            # 执行评估
            if self.task_type == "分类模型":
                result = self._evaluate_classification_model(predictor, self.test_dir)
            else:
                result = self._evaluate_detection_model(predictor, self.test_dir)
                
            self.evaluation_finished.emit(result)
            
        except Exception as e:
            import traceback
            self.evaluation_error.emit(f"评估失败: {str(e)}\n{traceback.format_exc()}")
    
    def _evaluate_classification_model(self, predictor, test_dir):
        """评估分类模型"""
        self.status_updated.emit("正在收集测试样本...")
        
        # 收集测试样本
        test_samples = []
        class_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        
        for class_dir in class_dirs:
            class_path = os.path.join(test_dir, class_dir)
            image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                test_samples.append((img_path, class_dir))
        
        if not test_samples:
            raise Exception("未找到测试样本")
        
        self.status_updated.emit(f"找到 {len(test_samples)} 个测试样本")
        
        # 执行预测
        y_true = []
        y_pred = []
        y_scores = []  # 用于计算AUC
        inference_times = []
        
        predictor.model.eval()
        total_samples = len(test_samples)
        
        with torch.no_grad():
            for i, (img_path, true_label) in enumerate(test_samples):
                # 更新进度
                progress = int((i + 1) / total_samples * 100)
                self.progress_updated.emit(progress)
                self.status_updated.emit(f"正在预测 {i+1}/{total_samples}")
                
                # 预测
                start_time = time.time()
                result = predictor.predict_image(img_path)
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
                
                if result and result['predictions']:
                    predicted_label = result['predictions'][0]['class_name']
                    predicted_prob = result['predictions'][0]['probability'] / 100.0
                    
                    y_true.append(true_label)
                    y_pred.append(predicted_label)
                    
                    # 获取所有类别的概率分数
                    class_scores = {}
                    for pred in result['predictions']:
                        class_scores[pred['class_name']] = pred['probability'] / 100.0
                    
                    # 为每个类别填充分数（未出现的类别设为0）
                    scores_vector = []
                    for class_name in predictor.class_names:
                        scores_vector.append(class_scores.get(class_name, 0.0))
                    y_scores.append(scores_vector)
        
        # 计算各种评估指标
        self.status_updated.emit("正在计算评估指标...")
        
        # 基本指标
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        # 每个类别的指标
        class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=predictor.class_names, zero_division=0
        )
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=predictor.class_names)
        
        # 分类报告
        class_report = classification_report(y_true, y_pred, target_names=predictor.class_names, output_dict=True, zero_division=0)
        
        # AUC计算（多分类）
        try:
            y_true_binary = label_binarize([predictor.class_names.index(label) for label in y_true], 
                                         classes=range(len(predictor.class_names)))
            y_scores_array = np.array(y_scores)
            
            if len(predictor.class_names) == 2:
                # 二分类
                auc_score = roc_auc_score(y_true_binary[:, 1], y_scores_array[:, 1])
            else:
                # 多分类
                auc_score = roc_auc_score(y_true_binary, y_scores_array, multi_class='ovr', average='weighted')
        except Exception as e:
            print(f"计算AUC时出错: {e}")
            auc_score = 0.0
        
        # 平均精度分数
        try:
            if len(predictor.class_names) == 2:
                ap_score = average_precision_score(y_true_binary[:, 1], y_scores_array[:, 1])
            else:
                ap_score = average_precision_score(y_true_binary, y_scores_array, average='weighted')
        except Exception as e:
            print(f"计算AP时出错: {e}")
            ap_score = 0.0
        
        # 计算模型参数数量
        params_count = sum(p.numel() for p in predictor.model.parameters())
        
        # 平均推理时间
        avg_inference_time = np.mean(inference_times) * 1000  # 转换为毫秒
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'ap_score': ap_score,
            'params_count': params_count,
            'avg_inference_time': avg_inference_time,
            'total_samples': len(test_samples),
            'confusion_matrix': cm.tolist(),
            'class_names': predictor.class_names,
            'class_precision': class_precision.tolist(),
            'class_recall': class_recall.tolist(),
            'class_f1': class_f1.tolist(),
            'class_support': class_support.tolist(),
            'classification_report': class_report,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_scores': y_scores
        }
    
    def _evaluate_detection_model(self, predictor, test_dir):
        """评估检测模型（简化实现）"""
        # 检测模型评估相对复杂，这里提供基础框架
        test_img_dir = os.path.join(test_dir, 'images', 'val')
        total_time = 0.0
        total_samples = 0
        
        if os.path.exists(test_img_dir):
            image_files = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            for i, img_file in enumerate(image_files[:50]):  # 限制样本数量
                self.progress_updated.emit(int((i + 1) / min(50, len(image_files)) * 100))
                self.status_updated.emit(f"正在评估 {i+1}/{min(50, len(image_files))}")
                
                img_path = os.path.join(test_img_dir, img_file)
                start_time = time.time()
                predictor.predict_image(img_path)
                end_time = time.time()
                
                inference_time = end_time - start_time
                total_time += inference_time
                total_samples += 1
        
        params_count = sum(p.numel() for p in predictor.model.parameters())
        avg_inference_time = (total_time / total_samples) * 1000 if total_samples > 0 else 0
        
        return {
            'params_count': params_count,
            'avg_inference_time': avg_inference_time,
            'total_samples': total_samples
        }


class EnhancedModelEvaluationWidget(QWidget):
    """增强的模型评估组件"""
    
    status_updated = pyqtSignal(str)
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.models_dir = ""
        self.test_data_dir = ""
        self.models_list = []
        self.evaluation_results = {}
        
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        # 主布局 - 直接使用组件的布局，不需要额外的滚动区域
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 创建分割器
        splitter = QSplitter(Qt.Vertical)
        
        # 上半部分：模型选择和基本信息
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        
        # 模型目录选择组
        models_group = QGroupBox("模型目录")
        models_layout = QGridLayout()
        
        self.models_path_edit = QLabel()
        self.models_path_edit.setStyleSheet("QLabel { border: 1px solid gray; padding: 5px; }")
        self.models_path_edit.setText("请选择包含模型的目录")
        
        models_btn = QPushButton("浏览...")
        models_btn.clicked.connect(self.select_models_dir)
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_model_list)
        
        models_layout.addWidget(QLabel("模型目录:"), 0, 0)
        models_layout.addWidget(self.models_path_edit, 0, 1)
        models_layout.addWidget(models_btn, 0, 2)
        models_layout.addWidget(refresh_btn, 0, 3)
        
        # 测试集目录选择
        self.test_data_path_edit = QLabel()
        self.test_data_path_edit.setStyleSheet("QLabel { border: 1px solid gray; padding: 5px; }")
        self.test_data_path_edit.setText("请选择测试集数据目录")
        
        test_data_btn = QPushButton("浏览...")
        test_data_btn.clicked.connect(self.select_test_data_dir)
        
        models_layout.addWidget(QLabel("测试集目录:"), 1, 0)
        models_layout.addWidget(self.test_data_path_edit, 1, 1)
        models_layout.addWidget(test_data_btn, 1, 2)
        
        models_group.setLayout(models_layout)
        top_layout.addWidget(models_group)
        
        # 模型列表组
        list_group = QGroupBox("可用模型")
        list_layout = QVBoxLayout()
        
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.MultiSelection)
        self.model_list.setMinimumHeight(120)
        list_layout.addWidget(self.model_list)
        
        # 评估按钮和进度条
        eval_layout = QHBoxLayout()
        self.evaluate_btn = QPushButton("评估选中模型")
        self.evaluate_btn.clicked.connect(self.evaluate_models)
        
        self.compare_btn = QPushButton("对比选中模型")
        self.compare_btn.clicked.connect(self.compare_models)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        eval_layout.addWidget(self.evaluate_btn)
        eval_layout.addWidget(self.compare_btn)
        eval_layout.addWidget(self.progress_bar)
        
        list_layout.addLayout(eval_layout)
        list_group.setLayout(list_layout)
        top_layout.addWidget(list_group)
        
        splitter.addWidget(top_widget)
        
        # 下半部分：评估结果展示
        self.results_tabs = QTabWidget()
        
        # 总览标签页
        self.overview_tab = self.create_overview_tab()
        self.results_tabs.addTab(self.overview_tab, "评估总览")
        
        # 详细指标标签页
        self.metrics_tab = self.create_metrics_tab()
        self.results_tabs.addTab(self.metrics_tab, "详细指标")
        
        # 混淆矩阵标签页
        self.confusion_tab = self.create_confusion_tab()
        self.results_tabs.addTab(self.confusion_tab, "混淆矩阵")
        
        # 分类报告标签页
        self.report_tab = self.create_report_tab()
        self.results_tabs.addTab(self.report_tab, "分类报告")
        
        # 模型对比标签页
        self.comparison_tab = self.create_comparison_tab()
        self.results_tabs.addTab(self.comparison_tab, "模型对比")
        
        splitter.addWidget(self.results_tabs)
        
        # 设置分割器比例
        splitter.setSizes([300, 500])
        
        layout.addWidget(splitter)
    
    def create_overview_tab(self):
        """创建总览标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 基本指标表格
        self.overview_table = QTableWidget(0, 2)
        self.overview_table.setHorizontalHeaderLabels(["指标", "值"])
        self.overview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.overview_table.setMaximumHeight(300)
        
        layout.addWidget(QLabel("模型评估概览"))
        layout.addWidget(self.overview_table)
        
        # 状态标签
        self.status_label = QLabel("请选择模型进行评估")
        self.status_label.setStyleSheet("QLabel { color: blue; font-weight: bold; }")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        return widget
    
    def create_metrics_tab(self):
        """创建详细指标标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 每个类别的详细指标表格
        self.metrics_table = QTableWidget(0, 5)
        self.metrics_table.setHorizontalHeaderLabels(["类别", "精确率", "召回率", "F1分数", "支持样本数"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        layout.addWidget(QLabel("各类别详细指标"))
        layout.addWidget(self.metrics_table)
        
        return widget
    
    def create_confusion_tab(self):
        """创建混淆矩阵标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 创建matplotlib图表
        self.confusion_figure = Figure(figsize=(10, 8))  # 增大尺寸
        self.confusion_canvas = FigureCanvas(self.confusion_figure)
        self.confusion_canvas.setMinimumHeight(500)  # 设置最小高度
        
        layout.addWidget(QLabel("混淆矩阵"))
        layout.addWidget(self.confusion_canvas)
        
        return widget
    
    def create_report_tab(self):
        """创建分类报告标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 文本显示区域
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setFont(QFont("Courier", 10))
        
        layout.addWidget(QLabel("详细分类报告"))
        layout.addWidget(self.report_text)
        
        return widget
    
    def create_comparison_tab(self):
        """创建模型对比标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 对比表格
        self.comparison_table = QTableWidget(0, 0)
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # 移除高度限制，让表格直接展开显示所有内容
        self.comparison_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.comparison_table.verticalHeader().setVisible(False)  # 隐藏行号
        
        layout.addWidget(QLabel("模型对比结果"))
        layout.addWidget(self.comparison_table)
        
        # 对比图表
        self.comparison_figure = Figure(figsize=(14, 10))  # 增大图表尺寸
        self.comparison_canvas = FigureCanvas(self.comparison_figure)
        self.comparison_canvas.setMinimumHeight(600)  # 设置最小高度
        
        layout.addWidget(QLabel("性能对比图"))
        layout.addWidget(self.comparison_canvas)
        
        return widget
    
    def select_models_dir(self):
        """选择模型目录"""
        folder = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if folder:
            self.models_dir = folder
            self.models_path_edit.setText(folder)
            self.refresh_model_list()
    
    def select_test_data_dir(self):
        """选择测试集数据目录"""
        folder = QFileDialog.getExistingDirectory(self, "选择测试集数据目录")
        if folder:
            self.test_data_dir = folder
            self.test_data_path_edit.setText(folder)
    
    def refresh_model_list(self):
        """刷新模型列表"""
        if not self.models_dir:
            return
            
        try:
            self.model_list.clear()
            self.models_list = []
            
            # 查找目录中的所有模型文件
            for file in os.listdir(self.models_dir):
                if file.endswith(('.h5', '.pb', '.tflite', '.pth')):
                    self.models_list.append(file)
                    self.model_list.addItem(file)
            
            if not self.models_list:
                QMessageBox.information(self, "提示", "未找到模型文件")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"刷新模型列表失败: {str(e)}")
    
    def evaluate_models(self):
        """评估选中的模型"""
        selected_items = self.model_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请选择要评估的模型")
            return
        
        if len(selected_items) == 1:
            # 单个模型评估
            self.evaluate_single_model(selected_items[0].text())
        else:
            # 多个模型评估
            self.evaluate_multiple_models([item.text() for item in selected_items])
    
    def compare_models(self):
        """对比选中的模型"""
        selected_items = self.model_list.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.warning(self, "警告", "请至少选择两个模型进行对比")
            return
        
        model_names = [item.text() for item in selected_items]
        self.compare_multiple_models(model_names)
    
    def evaluate_single_model(self, model_filename):
        """评估单个模型"""
        
        # 检查是否选择了测试集目录
        if not self.test_data_dir:
            QMessageBox.warning(self, "警告", "请选择测试集数据目录")
            return
        
        if not os.path.exists(self.test_data_dir):
            QMessageBox.warning(self, "错误", "测试集目录不存在")
            return
        
        model_path = os.path.join(self.models_dir, model_filename)
        
        # 获取配置信息
        config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'config.json')
        
        class_info_path = ""
        
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                class_info_path = config.get('default_class_info_file', '')
        
        # 验证类别信息文件
        if not class_info_path or not os.path.exists(class_info_path):
            json_files = [f for f in os.listdir(self.models_dir) if f.endswith('_classes.json') or f == 'class_info.json']
            if json_files:
                class_info_path = os.path.join(self.models_dir, json_files[0])
            else:
                QMessageBox.warning(self, "错误", "找不到类别信息文件")
                return
        
        # 确定任务类型
        test_dir = self.test_data_dir
        
        # 检查目录结构来判断任务类型
        if self._is_classification_dataset(test_dir):
            task_type = "分类模型"
        elif self._is_detection_dataset(test_dir):
            task_type = "检测模型"
        else:
            QMessageBox.warning(self, "错误", "无法识别测试数据集的格式\n\n分类数据集应该包含按类别命名的子文件夹\n检测数据集应该包含images文件夹")
            return
        
        # 猜测模型架构
        model_arch = self._guess_model_architecture(model_filename)
        
        # 创建并启动评估线程
        self.evaluation_thread = ModelEvaluationThread(
            model_path, model_arch, class_info_path, test_dir, task_type
        )
        
        # 连接信号
        self.evaluation_thread.progress_updated.connect(self.progress_bar.setValue)
        self.evaluation_thread.status_updated.connect(self.update_status)
        self.evaluation_thread.evaluation_finished.connect(self.on_evaluation_finished)
        self.evaluation_thread.evaluation_error.connect(self.on_evaluation_error)
        
        # 启动评估
        self.evaluate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.evaluation_thread.start()
    
    def evaluate_multiple_models(self, model_names):
        """评估多个模型"""
        self.current_batch_models = model_names
        self.batch_results = {}
        self.current_batch_index = 0
        
        self.update_status(f"开始批量评估 {len(model_names)} 个模型...")
        self.evaluate_next_model_in_batch()
    
    def evaluate_next_model_in_batch(self):
        """评估批次中的下一个模型"""
        if self.current_batch_index >= len(self.current_batch_models):
            # 所有模型评估完成
            self.on_batch_evaluation_finished()
            return
        
        model_name = self.current_batch_models[self.current_batch_index]
        self.update_status(f"正在评估模型 {self.current_batch_index + 1}/{len(self.current_batch_models)}: {model_name}")
        
        # 启动单个模型评估
        self.evaluate_single_model(model_name)
    
    def compare_multiple_models(self, model_names):
        """对比多个模型"""
        # 检查是否已有评估结果
        missing_results = []
        for model_name in model_names:
            if model_name not in self.evaluation_results:
                missing_results.append(model_name)
        
        if missing_results:
            reply = QMessageBox.question(
                self, "确认", 
                f"以下模型尚未评估：\n{', '.join(missing_results)}\n\n是否先评估这些模型？",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # 先评估缺失的模型，然后进行对比
                self.pending_comparison_models = model_names
                self.evaluate_multiple_models(missing_results)
            return
        
        # 所有模型都有结果，直接进行对比
        self.display_model_comparison(model_names)
    
    def on_batch_evaluation_finished(self):
        """批量评估完成处理"""
        self.update_status(f"批量评估完成，共评估 {len(self.current_batch_models)} 个模型")
        
        # 如果有待对比的模型，进行对比
        if hasattr(self, 'pending_comparison_models'):
            self.display_model_comparison(self.pending_comparison_models)
            delattr(self, 'pending_comparison_models')
        
        # 显示批量结果总览
        self.display_batch_results()
    
    def display_batch_results(self):
        """显示批量评估结果"""
        if not hasattr(self, 'current_batch_models'):
            return
        
        # 切换到对比标签页
        self.results_tabs.setCurrentWidget(self.comparison_tab)
        
        # 显示所有评估模型的对比
        self.display_model_comparison(self.current_batch_models)
    
    def display_model_comparison(self, model_names):
        """显示模型对比结果"""
        # 准备对比数据
        comparison_data = []
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score', 'params_count', 'avg_inference_time']
        metric_names = ['准确率', '精确率', '召回率', 'F1分数', 'AUC分数', '参数数量', '推理时间(ms)']
        
        for model_name in model_names:
            if model_name in self.evaluation_results:
                result = self.evaluation_results[model_name]
                row_data = [model_name]
                for metric in metrics:
                    value = result.get(metric, 0)
                    if metric == 'params_count':
                        row_data.append(f"{value:,}")
                    elif metric == 'avg_inference_time':
                        row_data.append(f"{value:.2f}")
                    else:
                        row_data.append(f"{value:.4f}")
                comparison_data.append(row_data)
        
        # 更新对比表格
        self.comparison_table.setRowCount(len(comparison_data))
        self.comparison_table.setColumnCount(len(metrics) + 1)
        self.comparison_table.setHorizontalHeaderLabels(['模型名称'] + metric_names)
        
        for i, row_data in enumerate(comparison_data):
            for j, value in enumerate(row_data):
                self.comparison_table.setItem(i, j, QTableWidgetItem(str(value)))
        
        # 自动调整表格大小以适应内容
        self.comparison_table.resizeRowsToContents()
        self.comparison_table.resizeColumnsToContents()
        
        # 设置表格高度以适应所有行
        total_height = self.comparison_table.horizontalHeader().height()
        for i in range(self.comparison_table.rowCount()):
            total_height += self.comparison_table.rowHeight(i)
        total_height += self.comparison_table.frameWidth() * 2
        self.comparison_table.setFixedHeight(total_height)
        
        # 绘制对比图表
        self.plot_model_comparison(model_names)
    
    def plot_model_comparison(self, model_names):
        """绘制模型对比图表"""
        self.comparison_figure.clear()
        
        # 准备数据
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['准确率', '精确率', '召回率', 'F1分数']
        
        model_data = {}
        for model_name in model_names:
            if model_name in self.evaluation_results:
                result = self.evaluation_results[model_name]
                model_data[model_name] = [result.get(metric, 0) for metric in metrics_to_plot]
        
        if not model_data:
            return
        
        # 创建子图
        ax1 = self.comparison_figure.add_subplot(2, 2, 1)
        ax2 = self.comparison_figure.add_subplot(2, 2, 2)
        ax3 = self.comparison_figure.add_subplot(2, 2, 3)
        ax4 = self.comparison_figure.add_subplot(2, 2, 4)
        
        axes = [ax1, ax2, ax3, ax4]
        
        # 绘制每个指标的对比
        x_pos = range(len(model_names))
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            ax = axes[i]
            values = [model_data[model_name][i] for model_name in model_names]
            
            bars = ax.bar(x_pos, values, color=colors)
            ax.set_title(label)
            ax.set_xlabel('模型')
            ax.set_ylabel('分数')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in model_names], rotation=45)
            ax.set_ylim(0, 1)
            
            # 在柱子上显示数值
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        self.comparison_figure.tight_layout()
        self.comparison_canvas.draw()
    
    def _guess_model_architecture(self, model_filename):
        """根据文件名猜测模型架构"""
        filename_lower = model_filename.lower()
        
        if "resnet18" in filename_lower:
            return "ResNet18"
        elif "resnet34" in filename_lower:
            return "ResNet34"
        elif "resnet50" in filename_lower:
            return "ResNet50"
        elif "mobilenet" in filename_lower:
            return "MobileNetV2"
        elif "efficientnet" in filename_lower:
            return "EfficientNetB0"
        elif "vgg16" in filename_lower:
            return "VGG16"
        else:
            return "ResNet18"
    
    def _is_classification_dataset(self, data_dir):
        """判断是否为分类数据集"""
        try:
            items = os.listdir(data_dir)
            # 检查是否有子目录（类别文件夹）
            subdirs = [item for item in items if os.path.isdir(os.path.join(data_dir, item))]
            
            if len(subdirs) > 0:
                # 检查子目录中是否包含图片文件
                for subdir in subdirs[:3]:  # 只检查前3个子目录
                    subdir_path = os.path.join(data_dir, subdir)
                    files = os.listdir(subdir_path)
                    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    if image_files:
                        return True
            return False
        except:
            return False
    
    def _is_detection_dataset(self, data_dir):
        """判断是否为检测数据集"""
        try:
            # 检查是否包含images文件夹
            images_dir = os.path.join(data_dir, 'images')
            if os.path.exists(images_dir):
                # 检查images文件夹中是否有val或test子文件夹
                val_dir = os.path.join(images_dir, 'val')
                test_dir = os.path.join(images_dir, 'test')
                return os.path.exists(val_dir) or os.path.exists(test_dir)
            return False
        except:
            return False
    
    def update_status(self, message):
        """更新状态"""
        self.status_label.setText(message)
        self.status_updated.emit(message)
    
    def on_evaluation_finished(self, result):
        """评估完成处理"""
        self.evaluate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # 处理批量评估
        if hasattr(self, 'current_batch_models') and hasattr(self, 'current_batch_index'):
            model_name = self.current_batch_models[self.current_batch_index]
            self.evaluation_results[model_name] = result
            self.batch_results[model_name] = result
            
            self.update_status(f"模型 {model_name} 评估完成")
            
            # 继续下一个模型
            self.current_batch_index += 1
            self.evaluate_next_model_in_batch()
        else:
            # 单个模型评估
            selected_items = self.model_list.selectedItems()
            if selected_items:
                model_name = selected_items[0].text()
                self.evaluation_results[model_name] = result
                self.display_results(result)
                self.update_status(f"模型 {model_name} 评估完成")
    
    def on_evaluation_error(self, error_msg):
        """评估错误处理"""
        self.evaluate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "评估错误", error_msg)
        self.update_status("评估失败")
    
    def display_results(self, result):
        """显示评估结果"""
        # 更新总览表格
        self.update_overview_table(result)
        
        # 更新详细指标表格
        if 'class_names' in result:
            self.update_metrics_table(result)
            
            # 更新混淆矩阵
            self.update_confusion_matrix(result)
            
            # 更新分类报告
            self.update_classification_report(result)
    
    def update_overview_table(self, result):
        """更新总览表格"""
        overview_data = [
            ("准确率", f"{result.get('accuracy', 0):.4f}"),
            ("精确率 (加权平均)", f"{result.get('precision', 0):.4f}"),
            ("召回率 (加权平均)", f"{result.get('recall', 0):.4f}"),
            ("F1分数 (加权平均)", f"{result.get('f1_score', 0):.4f}"),
            ("AUC分数", f"{result.get('auc_score', 0):.4f}"),
            ("平均精度分数", f"{result.get('ap_score', 0):.4f}"),
            ("模型参数数量", f"{result.get('params_count', 0):,}"),
            ("平均推理时间", f"{result.get('avg_inference_time', 0):.2f} ms"),
            ("测试样本总数", f"{result.get('total_samples', 0):,}")
        ]
        
        self.overview_table.setRowCount(len(overview_data))
        
        for i, (metric, value) in enumerate(overview_data):
            self.overview_table.setItem(i, 0, QTableWidgetItem(metric))
            self.overview_table.setItem(i, 1, QTableWidgetItem(value))
    
    def update_metrics_table(self, result):
        """更新详细指标表格"""
        class_names = result['class_names']
        class_precision = result['class_precision']
        class_recall = result['class_recall']
        class_f1 = result['class_f1']
        class_support = result['class_support']
        
        self.metrics_table.setRowCount(len(class_names))
        
        for i, class_name in enumerate(class_names):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(class_name))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{class_precision[i]:.4f}"))
            self.metrics_table.setItem(i, 2, QTableWidgetItem(f"{class_recall[i]:.4f}"))
            self.metrics_table.setItem(i, 3, QTableWidgetItem(f"{class_f1[i]:.4f}"))
            self.metrics_table.setItem(i, 4, QTableWidgetItem(f"{int(class_support[i])}"))
    
    def update_confusion_matrix(self, result):
        """更新混淆矩阵图表"""
        self.confusion_figure.clear()
        ax = self.confusion_figure.add_subplot(111)
        
        cm = np.array(result['confusion_matrix'])
        class_names = result['class_names']
        
        # 使用seaborn绘制热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_title('混淆矩阵')
        ax.set_xlabel('预测类别')
        ax.set_ylabel('真实类别')
        
        self.confusion_figure.tight_layout()
        self.confusion_canvas.draw()
    
    def update_classification_report(self, result):
        """更新分类报告"""
        if 'classification_report' in result:
            report_dict = result['classification_report']
            
            # 格式化分类报告
            report_text = "分类报告\n" + "="*50 + "\n\n"
            
            # 添加每个类别的详细信息
            for class_name in result['class_names']:
                if class_name in report_dict:
                    class_data = report_dict[class_name]
                    report_text += f"类别: {class_name}\n"
                    report_text += f"  精确率: {class_data['precision']:.4f}\n"
                    report_text += f"  召回率: {class_data['recall']:.4f}\n"
                    report_text += f"  F1分数: {class_data['f1-score']:.4f}\n"
                    report_text += f"  支持样本数: {int(class_data['support'])}\n\n"
            
            # 添加总体统计
            if 'accuracy' in report_dict:
                report_text += f"总体准确率: {report_dict['accuracy']:.4f}\n\n"
            
            if 'macro avg' in report_dict:
                macro_avg = report_dict['macro avg']
                report_text += "宏平均:\n"
                report_text += f"  精确率: {macro_avg['precision']:.4f}\n"
                report_text += f"  召回率: {macro_avg['recall']:.4f}\n"
                report_text += f"  F1分数: {macro_avg['f1-score']:.4f}\n\n"
            
            if 'weighted avg' in report_dict:
                weighted_avg = report_dict['weighted avg']
                report_text += "加权平均:\n"
                report_text += f"  精确率: {weighted_avg['precision']:.4f}\n"
                report_text += f"  召回率: {weighted_avg['recall']:.4f}\n"
                report_text += f"  F1分数: {weighted_avg['f1-score']:.4f}\n"
            
            self.report_text.setPlainText(report_text)
    
    def apply_config(self, config):
        """应用配置"""
        if not config:
            return
            
        # 设置模型评估目录
        if 'default_model_eval_dir' in config:
            model_dir = config['default_model_eval_dir']
            if os.path.exists(model_dir):
                self.models_path_edit.setText(model_dir)
                self.models_dir = model_dir
                self.refresh_model_list()
        
        # 设置默认测试集目录
        if 'default_test_data_dir' in config:
            test_dir = config['default_test_data_dir']
            if os.path.exists(test_dir):
                self.test_data_path_edit.setText(test_dir)
                self.test_data_dir = test_dir
        elif 'default_output_folder' in config:
            # 尝试从输出文件夹中查找测试集
            output_folder = config['default_output_folder']
            possible_test_dirs = [
                os.path.join(output_folder, 'dataset', 'val'),
                os.path.join(output_folder, 'dataset', 'test'),
                os.path.join(output_folder, 'detection_data'),
                os.path.join(output_folder, 'test_data')
            ]
            
            for test_dir in possible_test_dirs:
                if os.path.exists(test_dir):
                    self.test_data_path_edit.setText(test_dir)
                    self.test_data_dir = test_dir
                    break 