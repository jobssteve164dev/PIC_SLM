from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
                           QProgressBar, QMessageBox, QGroupBox, QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal
import os
import sys
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns
from collections import Counter
import cv2
from scipy import spatial

class DatasetEvaluationTab(QWidget):
    """数据集评估标签页，负责数据集质量分析和评估功能"""
    
    # 定义信号
    status_updated = pyqtSignal(str)  # 状态更新信号
    progress_updated = pyqtSignal(int)  # 进度更新信号
    
    def __init__(self, tabs, parent=None):
        super().__init__(parent)
        self.tabs = tabs
        self.dataset_path = ""
        self.init_ui()
        
    def init_ui(self):
        """初始化UI界面"""
        main_layout = QVBoxLayout()
        
        # 数据集选择区域
        dataset_group = QGroupBox("数据集选择")
        dataset_layout = QHBoxLayout()
        
        self.dataset_path_label = QLabel("未选择数据集")
        self.dataset_path_label.setStyleSheet("color: gray;")
        dataset_layout.addWidget(self.dataset_path_label)
        
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_dataset)
        dataset_layout.addWidget(browse_btn)
        
        dataset_group.setLayout(dataset_layout)
        main_layout.addWidget(dataset_group)
        
        # 评估选项区域
        options_group = QGroupBox("评估选项")
        options_layout = QVBoxLayout()
        
        # 评估类型选择
        eval_type_layout = QHBoxLayout()
        eval_type_layout.addWidget(QLabel("评估类型:"))
        self.eval_type_combo = QComboBox()
        self.eval_type_combo.addItems(["分类数据集", "目标检测数据集"])
        eval_type_layout.addWidget(self.eval_type_combo)
        options_layout.addLayout(eval_type_layout)
        
        # 评估指标选择
        metrics_layout = QHBoxLayout()
        metrics_layout.addWidget(QLabel("评估指标:"))
        self.metrics_combo = QComboBox()
        self.metrics_combo.addItems([
            "数据分布分析",
            "图像质量分析",
            "标注质量分析",
            "特征分布分析"
        ])
        metrics_layout.addWidget(self.metrics_combo)
        options_layout.addLayout(metrics_layout)
        
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("")
        main_layout.addWidget(self.status_label)
        
        # 评估按钮
        self.evaluate_btn = QPushButton("开始评估")
        self.evaluate_btn.clicked.connect(self.start_evaluation)
        self.evaluate_btn.setEnabled(False)
        main_layout.addWidget(self.evaluate_btn)
        
        # 结果显示区域
        results_group = QGroupBox("评估结果")
        results_layout = QVBoxLayout()
        
        # 创建包含图形和表格的水平布局
        results_horizontal_layout = QHBoxLayout()
        
        # 左侧：图表区域
        chart_layout = QVBoxLayout()
        
        # 创建matplotlib图形 - 减小图形高度，使比例更合适
        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 6))
        
        # 设置全局字体大小
        plt.rcParams.update({'font.size': 8})  # 减小全局字体大小
        
        self.canvas = FigureCanvas(self.figure)
        self.figure.tight_layout(pad=3.0)  # 调整图表间距
        chart_layout.addWidget(self.canvas)
        
        # 右侧：表格区域和指标说明
        table_layout = QVBoxLayout()
        
        # 数据表格 - 设置最大高度，防止表格过大
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["指标", "值"])
        self.result_table.setMaximumHeight(200)  # 限制表格高度
        self.result_table.horizontalHeader().setStretchLastSection(True)  # 拉伸最后一列填充
        table_layout.addWidget(self.result_table)
        
        # 添加指标说明区域
        self.metrics_info_label = QLabel("指标说明：")
        self.metrics_info_label.setWordWrap(True)  # 允许文本换行
        self.metrics_info_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        self.metrics_info_label.setMinimumHeight(150)  # 设置最小高度
        table_layout.addWidget(self.metrics_info_label)
        
        # 将图表和表格添加到水平布局
        results_horizontal_layout.addLayout(chart_layout, 2)  # 图表占2/3
        results_horizontal_layout.addLayout(table_layout, 1)  # 表格占1/3
        
        results_layout.addLayout(results_horizontal_layout)
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)
        
        self.setLayout(main_layout)
        
    def browse_dataset(self):
        """浏览选择数据集文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择数据集文件夹", self.dataset_path or "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if folder_path:
            self.dataset_path = folder_path
            self.dataset_path_label.setText(folder_path)
            self.evaluate_btn.setEnabled(True)
            self.update_status(f"已选择数据集: {folder_path}")
            
            # 自动更新配置文件，保存选择的数据集路径
            try:
                config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config.json')
                
                # 读取当前配置
                current_config = {}
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        current_config = json.load(f)
                
                # 更新数据集路径
                current_config['default_dataset_dir'] = folder_path
                
                # 保存更新后的配置
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(current_config, f, ensure_ascii=False, indent=4)
                    
                self.update_status(f"已选择数据集并更新配置: {folder_path}")
            except Exception as e:
                print(f"更新配置文件失败: {str(e)}")
                self.update_status(f"已选择数据集，但更新配置失败: {str(e)}")
            
    def start_evaluation(self):
        """开始数据集评估"""
        if not self.dataset_path:
            QMessageBox.warning(self, "警告", "请先选择数据集文件夹!")
            return
            
        eval_type = self.eval_type_combo.currentText()
        metric_type = self.metrics_combo.currentText()
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.evaluate_btn.setEnabled(False)
        
        try:
            if eval_type == "分类数据集":
                self.evaluate_classification_dataset(metric_type)
            else:
                self.evaluate_detection_dataset(metric_type)
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"评估过程中出错: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.evaluate_btn.setEnabled(True)
            
    def evaluate_classification_dataset(self, metric_type):
        """评估分类数据集"""
        dataset_path = self.dataset_path
        train_path = os.path.join(dataset_path, "train")
        val_path = os.path.join(dataset_path, "val")
        
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise ValueError("数据集结构不正确，缺少train或val文件夹")
            
        if metric_type == "数据分布分析":
            self.analyze_class_distribution(train_path, val_path)
        elif metric_type == "图像质量分析":
            self.analyze_image_quality(train_path)
        elif metric_type == "标注质量分析":
            self.analyze_annotation_quality(train_path)
        elif metric_type == "特征分布分析":
            self.analyze_feature_distribution(train_path)
            
    def evaluate_detection_dataset(self, metric_type):
        """评估目标检测数据集"""
        dataset_path = self.dataset_path
        train_path = os.path.join(dataset_path, "train")
        val_path = os.path.join(dataset_path, "val")
        
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise ValueError("数据集结构不正确，缺少train或val文件夹")
            
        if metric_type == "数据分布分析":
            self.analyze_detection_distribution(train_path, val_path)
        elif metric_type == "图像质量分析":
            self.analyze_detection_image_quality(train_path)
        elif metric_type == "标注质量分析":
            self.analyze_detection_annotation_quality(train_path)
        elif metric_type == "特征分布分析":
            self.analyze_detection_feature_distribution(train_path)
            
    def plot_with_adjusted_font(self, ax, title, xlabel, ylabel, x_rotation=45):
        """使用调整过的字体大小绘制图表"""
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.tick_params(axis='x', rotation=x_rotation)
        
        # 自动调整刻度数量，避免拥挤
        if x_rotation > 0:
            # 对于旋转了的x轴刻度，减少标签数量
            ax.locator_params(axis='x', nbins=6)

    def analyze_class_distribution(self, train_path, val_path):
        """分析分类数据集的类别分布"""
        # 清空图表
        self.ax1.clear()
        self.ax2.clear()
        
        # 统计训练集类别分布
        train_classes = {}
        for class_name in os.listdir(train_path):
            class_path = os.path.join(train_path, class_name)
            if os.path.isdir(class_path):
                train_classes[class_name] = len(os.listdir(class_path))
                
        # 统计验证集类别分布
        val_classes = {}
        for class_name in os.listdir(val_path):
            class_path = os.path.join(val_path, class_name)
            if os.path.isdir(class_path):
                val_classes[class_name] = len(os.listdir(class_path))
                
        # 绘制训练集分布
        self.ax1.bar(train_classes.keys(), train_classes.values())
        self.plot_with_adjusted_font(self.ax1, "训练集类别分布", "类别", "样本数量")
        
        # 绘制验证集分布
        self.ax2.bar(val_classes.keys(), val_classes.values())
        self.plot_with_adjusted_font(self.ax2, "验证集类别分布", "类别", "样本数量")
        
        # 调整图形布局
        self.figure.tight_layout(pad=2.0)
        
        # 计算类别不平衡度
        train_values = np.array(list(train_classes.values()))
        imbalance_ratio = np.max(train_values) / np.min(train_values) if np.min(train_values) > 0 else np.inf
        
        # 更新表格
        self.result_table.setRowCount(4)
        self.result_table.setItem(0, 0, QTableWidgetItem("总类别数"))
        self.result_table.setItem(0, 1, QTableWidgetItem(str(len(train_classes))))
        
        self.result_table.setItem(1, 0, QTableWidgetItem("训练集总样本数"))
        self.result_table.setItem(1, 1, QTableWidgetItem(str(sum(train_classes.values()))))
        
        self.result_table.setItem(2, 0, QTableWidgetItem("验证集总样本数"))
        self.result_table.setItem(2, 1, QTableWidgetItem(str(sum(val_classes.values()))))
        
        self.result_table.setItem(3, 0, QTableWidgetItem("类别不平衡度"))
        self.result_table.setItem(3, 1, QTableWidgetItem(f"{imbalance_ratio:.2f}"))
        
        # 设置表格列宽自适应
        self.result_table.resizeColumnsToContents()
        
        # 更新指标说明
        metrics_info = """
        <b>指标影响说明:</b><br>
        <b>总类别数</b>: 反映模型需要学习的概念数量，类别越多，模型通常越复杂<br>
        <b>样本数量</b>: 一般而言，更多的样本有助于模型学习更稳健的特征<br>
        <b>类别不平衡度</b>: 值越接近1越好，过高表示数据严重不平衡，模型可能偏向样本多的类别<br><br>
        <b>建议:</b><br>
        - 类别不平衡度 < 2: 非常均衡，无需特殊处理<br>
        - 类别不平衡度 2~10: 中度不平衡，可考虑数据增强或权重调整<br>
        - 类别不平衡度 > 10: 严重不平衡，建议采用过采样、欠采样或合成样本
        """
        self.metrics_info_label.setText(metrics_info)
        
        # 更新画布
        self.canvas.draw()
        
    def analyze_image_quality(self, train_path):
        """分析图像质量"""
        # 清空图表
        self.ax1.clear()
        self.ax2.clear()
        
        # 收集图像信息
        image_sizes = []
        image_qualities = []
        
        total_images = sum(len(os.listdir(os.path.join(train_path, class_name))) 
                          for class_name in os.listdir(train_path) 
                          if os.path.isdir(os.path.join(train_path, class_name)))
        
        processed = 0
        for class_name in os.listdir(train_path):
            class_path = os.path.join(train_path, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_name)
                        try:
                            img = Image.open(img_path)
                            image_sizes.append(img.size[0] * img.size[1])
                            
                            # 使用OpenCV计算图像质量
                            img_cv = cv2.imread(img_path)
                            if img_cv is not None:
                                quality = cv2.Laplacian(img_cv, cv2.CV_64F).var()
                                image_qualities.append(quality)
                                
                        except Exception as e:
                            print(f"处理图像 {img_path} 时出错: {str(e)}")
                            
                        processed += 1
                        self.progress_bar.setValue(int(processed / total_images * 100))
                        
        # 绘制图像尺寸分布
        self.ax1.hist(image_sizes, bins=30)
        self.plot_with_adjusted_font(self.ax1, "图像尺寸分布", "像素数量", "频次", x_rotation=0)
        
        # 绘制图像质量分布
        self.ax2.hist(image_qualities, bins=30)
        self.plot_with_adjusted_font(self.ax2, "图像质量分布（清晰度）", "清晰度值", "频次", x_rotation=0)
        
        # 调整图形布局
        self.figure.tight_layout(pad=2.0)
        
        # 更新表格
        self.result_table.setRowCount(4)
        self.result_table.setItem(0, 0, QTableWidgetItem("平均图像尺寸(像素)"))
        self.result_table.setItem(0, 1, QTableWidgetItem(f"{np.mean(image_sizes):.0f}"))
        
        self.result_table.setItem(1, 0, QTableWidgetItem("图像尺寸标准差"))
        self.result_table.setItem(1, 1, QTableWidgetItem(f"{np.std(image_sizes):.0f}"))
        
        self.result_table.setItem(2, 0, QTableWidgetItem("平均清晰度"))
        self.result_table.setItem(2, 1, QTableWidgetItem(f"{np.mean(image_qualities):.2f}"))
        
        self.result_table.setItem(3, 0, QTableWidgetItem("清晰度标准差"))
        self.result_table.setItem(3, 1, QTableWidgetItem(f"{np.std(image_qualities):.2f}"))
        
        # 设置表格列宽自适应
        self.result_table.resizeColumnsToContents()
        
        # 更新指标说明
        metrics_info = """
        <b>指标影响说明:</b><br>
        <b>图像尺寸</b>: 影响特征提取能力，太小可能细节不足，太大会增加计算量<br>
        <b>尺寸标准差</b>: 反映数据集的一致性，过高表示尺寸差异大<br>
        <b>图像清晰度</b>: 值越高表示图像越清晰，清晰图像通常含有更多可学习特征<br>
        <b>清晰度标准差</b>: 反映数据集质量的一致性<br><br>
        <b>建议:</b><br>
        - 尺寸标准差/平均尺寸 < 0.1: 尺寸高度一致，有利于模型学习<br>
        - 平均清晰度 < 100: 较为模糊，可能影响特征识别<br>
        - 平均清晰度 > 500: 较为清晰，有利于模型学习细节
        """
        self.metrics_info_label.setText(metrics_info)
        
        # 更新画布
        self.canvas.draw()
        
    def analyze_annotation_quality(self, train_path):
        """分析标注质量"""
        # 清空图表
        self.ax1.clear()
        self.ax2.clear()
        
        # 收集标注信息
        annotation_sizes = []
        annotation_counts = []
        class_names = []
        
        total_classes = len([d for d in os.listdir(train_path) 
                           if os.path.isdir(os.path.join(train_path, d))])
        
        processed = 0
        for class_name in os.listdir(train_path):
            class_path = os.path.join(train_path, class_name)
            if os.path.isdir(class_path):
                annotation_count = len(os.listdir(class_path))
                annotation_counts.append(annotation_count)
                class_names.append(class_name)
                
                # 计算标注文件大小
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_name)
                        try:
                            size = os.path.getsize(img_path)
                            annotation_sizes.append(size)
                        except Exception as e:
                            print(f"获取文件大小失败 {img_path}: {str(e)}")
                            
                processed += 1
                self.progress_bar.setValue(int(processed / total_classes * 100))
                
        # 绘制标注数量分布
        x = np.arange(len(class_names))
        self.ax1.bar(x, annotation_counts)
        # 如果类别数量多，减少显示的类别名称
        if len(class_names) > 6:
            indices = np.linspace(0, len(class_names)-1, 6, dtype=int)
            self.ax1.set_xticks(indices)
            self.ax1.set_xticklabels([class_names[i] for i in indices])
        else:
            self.ax1.set_xticks(x)
            self.ax1.set_xticklabels(class_names)
            
        self.plot_with_adjusted_font(self.ax1, "各类别标注数量分布", "类别", "标注数量")
        
        # 绘制标注文件大小分布
        self.ax2.hist(np.array(annotation_sizes)/1024, bins=30)  # 转换为KB
        self.plot_with_adjusted_font(self.ax2, "标注文件大小分布", "文件大小 (KB)", "频次", x_rotation=0)
        
        # 调整图形布局
        self.figure.tight_layout(pad=2.0)
        
        # 计算标注一致性指标 - 使用变异系数
        cv = np.std(annotation_counts) / np.mean(annotation_counts) if np.mean(annotation_counts) > 0 else 0
        
        # 更新表格
        self.result_table.setRowCount(5)
        self.result_table.setItem(0, 0, QTableWidgetItem("平均标注数量"))
        self.result_table.setItem(0, 1, QTableWidgetItem(f"{np.mean(annotation_counts):.2f}"))
        
        self.result_table.setItem(1, 0, QTableWidgetItem("最大标注数量"))
        self.result_table.setItem(1, 1, QTableWidgetItem(f"{np.max(annotation_counts)}"))
        
        self.result_table.setItem(2, 0, QTableWidgetItem("最小标注数量"))
        self.result_table.setItem(2, 1, QTableWidgetItem(f"{np.min(annotation_counts)}"))
        
        self.result_table.setItem(3, 0, QTableWidgetItem("平均文件大小"))
        self.result_table.setItem(3, 1, QTableWidgetItem(f"{np.mean(annotation_sizes)/1024:.2f} KB"))
        
        self.result_table.setItem(4, 0, QTableWidgetItem("标注一致性(CV)"))
        self.result_table.setItem(4, 1, QTableWidgetItem(f"{cv:.4f}"))
        
        # 设置表格列宽自适应
        self.result_table.resizeColumnsToContents()
        
        # 更新指标说明
        metrics_info = """
        <b>指标影响说明:</b><br>
        <b>平均标注数量</b>: 影响每个类别的学习充分度<br>
        • < 50: 样本可能不足，考虑数据增强<br>
        • > 1000: 通常足够训练稳定模型<br>
        <b>标注一致性(CV)</b>: 变异系数反映类别间数量平衡性<br>
        • < 0.2: 非常均衡<br>
        • 0.2-0.5: 适度均衡<br>
        • > 0.5: 不均衡，考虑均衡采样策略<br>
        <b>文件大小差异</b>: 反映图像复杂度和质量一致性<br>
        <b>建议</b>: 标注数量少的类别考虑增加样本或使用权重调整
        """
        self.metrics_info_label.setText(metrics_info)
        
        # 更新画布
        self.canvas.draw()
        
    def analyze_feature_distribution(self, train_path):
        """分析特征分布"""
        # 清空图表
        self.ax1.clear()
        self.ax2.clear()
        
        # 收集图像特征
        brightness_values = []
        contrast_values = []
        
        total_images = sum(len(os.listdir(os.path.join(train_path, class_name))) 
                          for class_name in os.listdir(train_path) 
                          if os.path.isdir(os.path.join(train_path, class_name)))
        
        processed = 0
        for class_name in os.listdir(train_path):
            class_path = os.path.join(train_path, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_name)
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                # 计算亮度
                                brightness = np.mean(img)
                                brightness_values.append(brightness)
                                
                                # 计算对比度
                                contrast = np.std(img)
                                contrast_values.append(contrast)
                                
                        except Exception as e:
                            print(f"处理图像 {img_path} 时出错: {str(e)}")
                            
                        processed += 1
                        self.progress_bar.setValue(int(processed / total_images * 100))
                        
        # 绘制亮度分布
        self.ax1.hist(brightness_values, bins=30)
        self.plot_with_adjusted_font(self.ax1, "图像亮度分布", "亮度值", "频次", x_rotation=0)
        
        # 绘制对比度分布
        self.ax2.hist(contrast_values, bins=30)
        self.plot_with_adjusted_font(self.ax2, "图像对比度分布", "对比度值", "频次", x_rotation=0)
        
        # 调整图形布局
        self.figure.tight_layout(pad=2.0)
        
        # 计算亮度和对比度的变异系数（标准差/均值）
        brightness_cv = np.std(brightness_values) / np.mean(brightness_values) if np.mean(brightness_values) > 0 else 0
        contrast_cv = np.std(contrast_values) / np.mean(contrast_values) if np.mean(contrast_values) > 0 else 0
        
        # 更新表格
        self.result_table.setRowCount(4)
        self.result_table.setItem(0, 0, QTableWidgetItem("平均亮度"))
        self.result_table.setItem(0, 1, QTableWidgetItem(f"{np.mean(brightness_values):.2f}"))
        
        self.result_table.setItem(1, 0, QTableWidgetItem("亮度变异系数"))
        self.result_table.setItem(1, 1, QTableWidgetItem(f"{brightness_cv:.4f}"))
        
        self.result_table.setItem(2, 0, QTableWidgetItem("平均对比度"))
        self.result_table.setItem(2, 1, QTableWidgetItem(f"{np.mean(contrast_values):.2f}"))
        
        self.result_table.setItem(3, 0, QTableWidgetItem("对比度变异系数"))
        self.result_table.setItem(3, 1, QTableWidgetItem(f"{contrast_cv:.4f}"))
        
        # 设置表格列宽自适应
        self.result_table.resizeColumnsToContents()
        
        # 更新指标说明
        metrics_info = """
        <b>指标影响说明:</b><br>
        <b>平均亮度</b>: 影响模型对不同光照条件的适应性<br>
        • 过低 (<50) 表示图像较暗，可能丢失细节<br>
        • 过高 (>200) 表示图像过亮，可能导致过曝<br>
        • 理想范围: 80-180<br>
        <b>亮度变异系数</b>: 衡量亮度分布一致性，值越小越一致<br>
        <b>平均对比度</b>: 影响特征的可区分性<br>
        • 过低 (<30) 表示图像平淡，特征不明显<br>
        • 过高 (>80) 可能导致某些区域信息丢失<br>
        <b>建议</b>: 考虑添加数据增强，增加亮度和对比度的多样性
        """
        self.metrics_info_label.setText(metrics_info)
        
        # 更新画布
        self.canvas.draw()
        
    def analyze_detection_distribution(self, train_path, val_path):
        """分析目标检测数据集的分布"""
        # 清空图表
        self.ax1.clear()
        self.ax2.clear()
        
        # 统计训练集和验证集的标注数量
        train_annotations = self.count_detection_annotations(train_path)
        val_annotations = self.count_detection_annotations(val_path)
        
        # 绘制训练集标注分布
        self.ax1.bar(train_annotations.keys(), train_annotations.values())
        self.plot_with_adjusted_font(self.ax1, "训练集类别分布", "类别", "标注数量")
        
        # 绘制验证集标注分布
        self.ax2.bar(val_annotations.keys(), val_annotations.values())
        self.plot_with_adjusted_font(self.ax2, "验证集类别分布", "类别", "标注数量")
        
        # 调整图形布局
        self.figure.tight_layout(pad=2.0)
        
        # 计算类别不平衡度
        train_values = np.array(list(train_annotations.values()))
        imbalance_ratio = np.max(train_values) / np.min(train_values) if np.min(train_values) > 0 else float('inf')
        
        # 计算训练集与验证集分布相似度（使用余弦相似度）
        # 确保所有类别都存在于两个集合中
        all_classes = set(train_annotations.keys()) | set(val_annotations.keys())
        train_vector = [train_annotations.get(cls, 0) for cls in all_classes]
        val_vector = [val_annotations.get(cls, 0) for cls in all_classes]
        
        if sum(train_vector) > 0 and sum(val_vector) > 0:
            similarity = 1 - spatial.distance.cosine(train_vector, val_vector)
        else:
            similarity = 0
        
        # 更新表格
        self.result_table.setRowCount(5)
        self.result_table.setItem(0, 0, QTableWidgetItem("总类别数"))
        self.result_table.setItem(0, 1, QTableWidgetItem(str(len(all_classes))))
        
        self.result_table.setItem(1, 0, QTableWidgetItem("训练集总标注数"))
        self.result_table.setItem(1, 1, QTableWidgetItem(str(sum(train_annotations.values()))))
        
        self.result_table.setItem(2, 0, QTableWidgetItem("验证集总标注数"))
        self.result_table.setItem(2, 1, QTableWidgetItem(str(sum(val_annotations.values()))))
        
        self.result_table.setItem(3, 0, QTableWidgetItem("类别不平衡度"))
        self.result_table.setItem(3, 1, QTableWidgetItem(f"{imbalance_ratio:.2f}" if imbalance_ratio != float('inf') else "∞"))
        
        self.result_table.setItem(4, 0, QTableWidgetItem("训练/验证集相似度"))
        self.result_table.setItem(4, 1, QTableWidgetItem(f"{similarity:.4f}"))
        
        # 设置表格列宽自适应
        self.result_table.resizeColumnsToContents()
        
        # 更新指标说明
        metrics_info = """
        <b>指标影响说明:</b><br>
        <b>类别不平衡度</b>: 影响模型对少数类的检测能力<br>
        • < 2: 极佳的类别平衡<br>
        • 2-5: 良好平衡<br>
        • 5-20: 中度不平衡，可能需要特殊处理<br>
        • > 20: 严重不平衡，建议数据增强或采样调整<br>
        <b>训练/验证集相似度</b>: 衡量分布一致性<br>
        • > 0.9: 非常相似，验证集能很好地反映训练集<br>
        • 0.7-0.9: 良好相似度<br>
        • < 0.7: 分布差异大，验证结果可能不可靠<br>
        <b>建议</b>: 对于不平衡类别，考虑使用focal loss或过采样技术
        """
        self.metrics_info_label.setText(metrics_info)
        
        # 更新画布
        self.canvas.draw()
        
    def count_detection_annotations(self, dataset_path):
        """统计目标检测数据集的标注数量"""
        annotations = {}
        
        # 读取类别信息
        class_info_path = os.path.join(os.path.dirname(dataset_path), "class_info.json")
        if os.path.exists(class_info_path):
            with open(class_info_path, 'r', encoding='utf-8') as f:
                class_info = json.load(f)
                class_names = class_info.get('classes', [])
        else:
            # 如果没有类别信息文件，从标注文件中提取类别
            class_names = set()
            
        # 统计标注数量
        labels_path = os.path.join(dataset_path, "labels")
        if os.path.exists(labels_path):
            for label_file in os.listdir(labels_path):
                if label_file.endswith('.txt'):
                    with open(os.path.join(labels_path, label_file), 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 1:
                                class_id = int(parts[0])
                                if class_id < len(class_names):
                                    class_name = class_names[class_id]
                                else:
                                    class_name = f"class_{class_id}"
                                annotations[class_name] = annotations.get(class_name, 0) + 1
                                
        return annotations
        
    def analyze_detection_image_quality(self, train_path):
        """分析目标检测数据集的图像质量"""
        # 清空图表
        self.ax1.clear()
        self.ax2.clear()
        
        # 收集图像信息
        image_sizes = []
        image_qualities = []
        
        images_path = os.path.join(train_path, "images")
        if not os.path.exists(images_path):
            raise ValueError("未找到图像文件夹")
            
        total_images = len([f for f in os.listdir(images_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        processed = 0
        for img_name in os.listdir(images_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(images_path, img_name)
                try:
                    img = Image.open(img_path)
                    image_sizes.append(img.size[0] * img.size[1])
                    
                    # 使用OpenCV计算图像质量
                    img_cv = cv2.imread(img_path)
                    if img_cv is not None:
                        quality = cv2.Laplacian(img_cv, cv2.CV_64F).var()
                        image_qualities.append(quality)
                        
                except Exception as e:
                    print(f"处理图像 {img_path} 时出错: {str(e)}")
                    
                processed += 1
                self.progress_bar.setValue(int(processed / total_images * 100))
                
        # 绘制图像尺寸分布
        self.ax1.hist(image_sizes, bins=30)
        self.plot_with_adjusted_font(self.ax1, "图像尺寸分布", "像素数量", "频次", x_rotation=0)
        
        # 绘制图像质量分布
        self.ax2.hist(image_qualities, bins=30)
        self.plot_with_adjusted_font(self.ax2, "图像质量分布", "清晰度值", "频次", x_rotation=0)
        
        # 调整图形布局
        self.figure.tight_layout(pad=2.0)
        
        # 更新表格
        self.result_table.setRowCount(4)
        self.result_table.setItem(0, 0, QTableWidgetItem("平均图像尺寸(像素)"))
        self.result_table.setItem(0, 1, QTableWidgetItem(f"{np.mean(image_sizes):.0f}" if image_sizes else "0"))
        
        self.result_table.setItem(1, 0, QTableWidgetItem("图像尺寸标准差"))
        self.result_table.setItem(1, 1, QTableWidgetItem(f"{np.std(image_sizes):.0f}" if image_sizes else "0"))
        
        self.result_table.setItem(2, 0, QTableWidgetItem("平均清晰度"))
        self.result_table.setItem(2, 1, QTableWidgetItem(f"{np.mean(image_qualities):.2f}" if image_qualities else "0"))
        
        self.result_table.setItem(3, 0, QTableWidgetItem("清晰度标准差"))
        self.result_table.setItem(3, 1, QTableWidgetItem(f"{np.std(image_qualities):.2f}" if image_qualities else "0"))
        
        # 设置表格列宽自适应
        self.result_table.resizeColumnsToContents()
        
        # 更新指标说明
        metrics_info = """
        <b>指标影响说明:</b><br>
        <b>图像尺寸</b>: 影响特征提取能力，太小可能细节不足，太大会增加计算量<br>
        <b>尺寸标准差</b>: 反映数据集的一致性，过高表示尺寸差异大<br>
        <b>图像清晰度</b>: 值越高表示图像越清晰，清晰图像通常含有更多可学习特征<br>
        <b>清晰度标准差</b>: 反映数据集质量的一致性<br><br>
        <b>建议:</b><br>
        - 尺寸标准差/平均尺寸 < 0.1: 尺寸高度一致，有利于模型学习<br>
        - 平均清晰度 < 100: 较为模糊，可能影响特征识别<br>
        - 平均清晰度 > 500: 较为清晰，有利于模型学习细节
        """
        self.metrics_info_label.setText(metrics_info)
        
        # 更新画布
        self.canvas.draw()
        
    def analyze_detection_annotation_quality(self, train_path):
        """分析目标检测数据集的标注质量"""
        # 清空图表
        self.ax1.clear()
        self.ax2.clear()
        
        # 收集标注信息
        box_sizes = []
        box_counts = []
        
        labels_path = os.path.join(train_path, "labels")
        if not os.path.exists(labels_path):
            raise ValueError("未找到标注文件夹")
            
        total_labels = len([f for f in os.listdir(labels_path) if f.endswith('.txt')])
        
        processed = 0
        for label_file in os.listdir(labels_path):
            if label_file.endswith('.txt'):
                box_count = 0
                with open(os.path.join(labels_path, label_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # 确保有足够的坐标信息
                            # 计算边界框大小
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            box_size = width * height
                            box_sizes.append(box_size)
                            box_count += 1
                            
                box_counts.append(box_count)
                
                processed += 1
                self.progress_bar.setValue(int(processed / total_labels * 100))
                
        # 绘制每张图像的标注数量分布
        self.ax1.hist(box_counts, bins=min(30, len(set(box_counts))))
        self.plot_with_adjusted_font(self.ax1, "每张图像的标注数量分布", "标注数量", "图像数量", x_rotation=0)
        
        # 绘制边界框大小分布
        self.ax2.hist(box_sizes, bins=30)
        self.plot_with_adjusted_font(self.ax2, "边界框大小分布", "边界框大小(相对面积)", "频次", x_rotation=0)
        
        # 调整图形布局
        self.figure.tight_layout(pad=2.0)
        
        # 计算重要统计量
        empty_images = box_counts.count(0)
        empty_ratio = empty_images / len(box_counts) if box_counts else 0
        small_boxes = sum(1 for size in box_sizes if size < 0.01)  # 小目标定义为相对面积<1%
        small_box_ratio = small_boxes / len(box_sizes) if box_sizes else 0
        
        # 更新表格
        self.result_table.setRowCount(5)
        self.result_table.setItem(0, 0, QTableWidgetItem("平均每图标注数"))
        self.result_table.setItem(0, 1, QTableWidgetItem(f"{np.mean(box_counts):.2f}" if box_counts else "0"))
        
        self.result_table.setItem(1, 0, QTableWidgetItem("无标注图像比例"))
        self.result_table.setItem(1, 1, QTableWidgetItem(f"{empty_ratio:.2%}"))
        
        self.result_table.setItem(2, 0, QTableWidgetItem("小目标比例"))
        self.result_table.setItem(2, 1, QTableWidgetItem(f"{small_box_ratio:.2%}"))
        
        self.result_table.setItem(3, 0, QTableWidgetItem("平均边界框大小"))
        self.result_table.setItem(3, 1, QTableWidgetItem(f"{np.mean(box_sizes):.4f}" if box_sizes else "0"))
        
        self.result_table.setItem(4, 0, QTableWidgetItem("边界框大小方差"))
        self.result_table.setItem(4, 1, QTableWidgetItem(f"{np.var(box_sizes):.6f}" if box_sizes else "0"))
        
        # 设置表格列宽自适应
        self.result_table.resizeColumnsToContents()
        
        # 更新指标说明
        metrics_info = """
        <b>指标影响说明:</b><br>
        <b>平均每图标注数</b>: 反映目标密度<br>
        • 过低(<1)可能导致检测器难以学习足够特征<br>
        • 过高(>10)可能使模型对拥挤场景过度适应<br>
        <b>无标注图像比例</b>: 应尽量避免，通常应<5%<br>
        <b>小目标比例</b>: 影响检测难度<br>
        • >50%: 小目标主导，可能需要专门的小目标检测策略<br>
        • <10%: 大目标主导，通常检测更容易<br>
        <b>边界框大小方差</b>: 目标尺寸一致性<br>
        <b>建议</b>: 小目标比例高时，考虑使用FPN、多尺度训练等策略
        """
        self.metrics_info_label.setText(metrics_info)
        
        # 更新画布
        self.canvas.draw()
        
    def analyze_detection_feature_distribution(self, train_path):
        """分析目标检测数据集的特征分布"""
        # 清空图表
        self.ax1.clear()
        self.ax2.clear()
        
        # 收集图像特征
        brightness_values = []
        contrast_values = []
        
        images_path = os.path.join(train_path, "images")
        if not os.path.exists(images_path):
            raise ValueError("未找到图像文件夹")
            
        total_images = len([f for f in os.listdir(images_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        processed = 0
        for img_name in os.listdir(images_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(images_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        # 计算亮度
                        brightness = np.mean(img)
                        brightness_values.append(brightness)
                        
                        # 计算对比度
                        contrast = np.std(img)
                        contrast_values.append(contrast)
                        
                except Exception as e:
                    print(f"处理图像 {img_path} 时出错: {str(e)}")
                    
                processed += 1
                self.progress_bar.setValue(int(processed / total_images * 100))
                
        # 绘制亮度分布
        self.ax1.hist(brightness_values, bins=30)
        self.plot_with_adjusted_font(self.ax1, "图像亮度分布", "亮度值", "频次", x_rotation=0)
        
        # 绘制对比度分布
        self.ax2.hist(contrast_values, bins=30)
        self.plot_with_adjusted_font(self.ax2, "图像对比度分布", "对比度值", "频次", x_rotation=0)
        
        # 调整图形布局
        self.figure.tight_layout(pad=2.0)
        
        # 计算亮度和对比度的变异系数（标准差/均值）
        brightness_cv = np.std(brightness_values) / np.mean(brightness_values) if np.mean(brightness_values) > 0 else 0
        contrast_cv = np.std(contrast_values) / np.mean(contrast_values) if np.mean(contrast_values) > 0 else 0
        
        # 更新表格
        self.result_table.setRowCount(4)
        self.result_table.setItem(0, 0, QTableWidgetItem("平均亮度"))
        self.result_table.setItem(0, 1, QTableWidgetItem(f"{np.mean(brightness_values):.2f}" if brightness_values else "0"))
        
        self.result_table.setItem(1, 0, QTableWidgetItem("亮度变异系数"))
        self.result_table.setItem(1, 1, QTableWidgetItem(f"{brightness_cv:.4f}"))
        
        self.result_table.setItem(2, 0, QTableWidgetItem("平均对比度"))
        self.result_table.setItem(2, 1, QTableWidgetItem(f"{np.mean(contrast_values):.2f}" if contrast_values else "0"))
        
        self.result_table.setItem(3, 0, QTableWidgetItem("对比度变异系数"))
        self.result_table.setItem(3, 1, QTableWidgetItem(f"{contrast_cv:.4f}"))
        
        # 设置表格列宽自适应
        self.result_table.resizeColumnsToContents()
        
        # 更新指标说明
        metrics_info = """
        <b>指标影响说明:</b><br>
        <b>平均亮度</b>: 影响模型对不同光照条件的适应性<br>
        • 过低 (<50) 表示图像较暗，可能丢失细节<br>
        • 过高 (>200) 表示图像过亮，可能导致过曝<br>
        • 理想范围: 80-180<br>
        <b>亮度变异系数</b>: 衡量亮度分布一致性，值越小越一致<br>
        <b>平均对比度</b>: 影响特征的可区分性<br>
        • 过低 (<30) 表示图像平淡，特征不明显<br>
        • 过高 (>80) 可能导致某些区域信息丢失<br>
        <b>建议</b>: 考虑添加数据增强，增加亮度和对比度的多样性
        """
        self.metrics_info_label.setText(metrics_info)
        
        # 更新画布
        self.canvas.draw()
        
    def update_status(self, message):
        """更新状态信息"""
        self.status_label.setText(message) 
        
    def apply_config(self, config):
        """应用配置到数据集评估标签页"""
        # 从配置中获取默认数据集评估文件夹路径
        default_dataset_path = config.get('default_dataset_dir', '')
        
        if default_dataset_path:
            # 验证路径是否存在
            if os.path.exists(default_dataset_path):
                self.dataset_path = default_dataset_path
                self.dataset_path_label.setText(default_dataset_path)
                self.evaluate_btn.setEnabled(True)
                self.status_updated.emit(f"已加载默认数据集路径: {default_dataset_path}")
            else:
                self.status_updated.emit(f"配置中的数据集路径不存在: {default_dataset_path}")
                
        # 应用其他配置项 - 可根据需要扩展 