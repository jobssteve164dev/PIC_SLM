"""
重构后的数据集评估标签页

将原有的大型组件拆分为多个专门的模块，提高代码的可维护性和可测试性
同时保持所有原有功能完整性
"""

import os
import json
import sys
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
                           QProgressBar, QMessageBox, QGroupBox, QComboBox,
                           QDialog, QDialogButtonBox, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 导入matplotlib配置和标准化函数
from src.utils.matplotlib_config import suppress_matplotlib_warnings

# 抑制matplotlib警告
suppress_matplotlib_warnings()

# 导入拆分的组件模块
from .base_tab import BaseTab
from .components.dataset_evaluation import ClassificationAnalyzer, DetectionAnalyzer
from .components.dataset_evaluation import WeightGenerator
from .components.dataset_evaluation import ChartManager
from .components.dataset_evaluation import ResultDisplayManager
from .components.dataset_evaluation import DatasetEvaluationThread


class DatasetEvaluationTab(BaseTab):
    """重构后的数据集评估标签页
    
    主要改进:
    1. 将分析逻辑拆分到专门的分析器模块
    2. 将图表绘制拆分到图表管理器
    3. 将结果显示拆分到结果显示管理器
    4. 将权重生成拆分到权重生成器
    5. 保持所有原有功能完整性
    """
    
    # 定义信号
    status_updated = pyqtSignal(str)  # 状态更新信号
    progress_updated = pyqtSignal(int)  # 进度更新信号
    
    def __init__(self, tabs, parent=None):
        super().__init__(parent, parent)  # 调用BaseTab的构造函数
        self.tabs = tabs
        self.dataset_path = ""
        
        # 初始化各个管理器
        self.weight_generator = WeightGenerator()
        self.chart_manager = ChartManager()
        self.result_display_manager = ResultDisplayManager()
        
        # 当前分析结果缓存
        self.current_analyzer = None
        self.current_class_weights = None
        self.current_class_counts = None
        
        # 评估线程
        self.evaluation_thread = None
        
        self.init_ui()
        self.connect_signals()
        
        # 使用新的智能配置系统
        config = self.get_config_from_manager()
        if config:
            self.apply_config(config)
        
    def init_ui(self):
        """初始化UI界面"""
        # 使用BaseTab提供的滚动内容区域
        main_layout = QVBoxLayout(self.scroll_content)
        
        # 数据集选择区域
        dataset_group = self.create_dataset_selection_group()
        main_layout.addWidget(dataset_group)
        
        # 评估选项区域
        options_group = self.create_evaluation_options_group()
        main_layout.addWidget(options_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("")
        main_layout.addWidget(self.status_label)
        
        # 评估按钮
        button_layout = QHBoxLayout()
        self.evaluate_btn = QPushButton("开始评估")
        self.evaluate_btn.clicked.connect(self.start_evaluation)
        self.evaluate_btn.setEnabled(False)
        button_layout.addWidget(self.evaluate_btn)
        
        # 停止按钮
        self.stop_btn = QPushButton("停止评估")
        self.stop_btn.clicked.connect(self.stop_evaluation)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        main_layout.addLayout(button_layout)
        
        # 结果显示区域
        results_group = self.create_results_display_group()
        main_layout.addWidget(results_group)
        
    def create_dataset_selection_group(self):
        """创建数据集选择区域"""
        dataset_group = QGroupBox("数据集选择")
        dataset_layout = QHBoxLayout()
        
        self.dataset_path_label = QLabel("未选择数据集")
        self.dataset_path_label.setStyleSheet("color: gray;")
        dataset_layout.addWidget(self.dataset_path_label)
        
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_dataset)
        dataset_layout.addWidget(browse_btn)
        
        dataset_group.setLayout(dataset_layout)
        return dataset_group
        
    def create_evaluation_options_group(self):
        """创建评估选项区域"""
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
            "特征分布分析",
            "生成类别权重"
        ])
        metrics_layout.addWidget(self.metrics_combo)
        options_layout.addLayout(metrics_layout)
        
        options_group.setLayout(options_layout)
        return options_group
        
    def create_results_display_group(self):
        """创建结果显示区域"""
        results_group = QGroupBox("评估结果")
        results_layout = QVBoxLayout()
        
        # 创建包含图形和表格的水平布局
        results_horizontal_layout = QHBoxLayout()
        
        # 左侧：图表区域
        chart_layout = QVBoxLayout()
        
        # 创建matplotlib图形
        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 6))
        plt.rcParams.update({'font.size': 8})
        
        self.canvas = FigureCanvas(self.figure)
        self.figure.tight_layout(pad=3.0)
        chart_layout.addWidget(self.canvas)
        
        # 设置图表管理器
        self.chart_manager.setup_figure(self.canvas)
        
        # 右侧：表格区域和指标说明
        table_layout = QVBoxLayout()
        
        # 数据表格
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["指标", "值"])
        self.result_table.setMaximumHeight(200)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.result_table)
        
        # 指标说明区域
        self.metrics_info_label = QLabel("指标说明：")
        self.metrics_info_label.setWordWrap(True)
        self.metrics_info_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        self.metrics_info_label.setMinimumHeight(150)
        table_layout.addWidget(self.metrics_info_label)
        
        # 设置结果显示管理器
        self.result_display_manager.setup_table(self.result_table, self.metrics_info_label)
        
        # 将图表和表格添加到水平布局
        results_horizontal_layout.addLayout(chart_layout, 2)  # 图表占2/3
        results_horizontal_layout.addLayout(table_layout, 1)  # 表格占1/3
        
        results_layout.addLayout(results_horizontal_layout)
        results_group.setLayout(results_layout)
        
        return results_group
        
    def connect_signals(self):
        """连接信号和槽"""
        # 连接权重生成器信号
        self.weight_generator.status_updated.connect(self.update_status)
        
        # 连接图表管理器信号
        self.chart_manager.chart_updated.connect(self.canvas.draw)
        
        # 连接结果显示管理器信号
        self.result_display_manager.result_updated.connect(self.on_result_updated)
        
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
            
            # 自动更新配置文件
            self.update_config_file(folder_path)
            
    def update_config_file(self, dataset_path):
        """更新配置文件，保存数据集路径"""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                'config.json'
            )
            
            # 读取当前配置
            current_config = {}
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    current_config = json.load(f)
            
            # 更新数据集路径
            current_config['default_dataset_dir'] = dataset_path
            
            # 保存更新后的配置
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(current_config, f, ensure_ascii=False, indent=4)
                
            self.update_status(f"已选择数据集并更新配置: {dataset_path}")
        except Exception as e:
            print(f"更新配置文件失败: {str(e)}")
            self.update_status(f"已选择数据集，但更新配置失败: {str(e)}")
            
    def start_evaluation(self):
        """开始数据集评估（使用独立线程）"""
        if not self.dataset_path:
            QMessageBox.warning(self, "警告", "请先选择数据集文件夹!")
            return
            
        # 如果已有线程在运行，先停止
        if self.evaluation_thread and self.evaluation_thread.isRunning():
            self.stop_evaluation()
            return
            
        eval_type = self.eval_type_combo.currentText()
        metric_type = self.metrics_combo.currentText()
        
        # 更新UI状态
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.evaluate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # 创建并启动评估线程
        self.evaluation_thread = DatasetEvaluationThread(
            self.dataset_path, eval_type, metric_type
        )
        
        # 连接线程信号
        self.evaluation_thread.progress_updated.connect(self.progress_bar.setValue)
        self.evaluation_thread.status_updated.connect(self.update_status)
        self.evaluation_thread.evaluation_finished.connect(self.on_evaluation_finished)
        self.evaluation_thread.evaluation_error.connect(self.on_evaluation_error)
        self.evaluation_thread.evaluation_stopped.connect(self.on_evaluation_stopped)
        
        # 启动线程
        self.evaluation_thread.start()
        
    def stop_evaluation(self):
        """停止数据集评估"""
        if self.evaluation_thread and self.evaluation_thread.isRunning():
            self.evaluation_thread.stop()
            self.update_status("正在停止评估...")
            
    # 原来的同步评估方法已移至独立线程中执行
    # def evaluate_classification_dataset(self, metric_type):
    #     """评估分类数据集（已弃用，使用线程版本）"""
    #     pass
            
    # 原来的同步评估方法已移至独立线程中执行
    # def evaluate_detection_dataset(self, metric_type):
    #     """评估目标检测数据集（已弃用，使用线程版本）"""
    #     pass
            
    def analyze_classification_distribution(self):
        """分析分类数据集分布"""
        result = self.current_analyzer.analyze_data_distribution()
        
        # 使用图表管理器绘制图表
        self.chart_manager.plot_classification_distribution(
            result['train_classes'], 
            result['val_classes']
        )
        
        # 使用结果显示管理器显示结果
        self.result_display_manager.display_classification_distribution_results(result)
        
    def analyze_classification_image_quality(self):
        """分析分类数据集图像质量"""
        result = self.current_analyzer.analyze_image_quality()
        
        # 使用图表管理器绘制图表
        self.chart_manager.plot_image_quality(
            result['image_sizes'],
            result['image_qualities']
        )
        
        # 使用结果显示管理器显示结果
        self.result_display_manager.display_image_quality_results(result)
        
    def analyze_classification_annotation_quality(self):
        """分析分类数据集标注质量"""
        result = self.current_analyzer.analyze_annotation_quality()
        
        # 使用图表管理器绘制图表
        self.chart_manager.plot_annotation_quality(
            result['class_names'],
            result['annotation_counts'],
            result['annotation_sizes']
        )
        
        # 使用结果显示管理器显示结果
        self.result_display_manager.display_annotation_quality_results(result)
        
    def analyze_classification_feature_distribution(self):
        """分析分类数据集特征分布"""
        result = self.current_analyzer.analyze_feature_distribution()
        
        # 使用图表管理器绘制图表
        self.chart_manager.plot_feature_distribution(
            result['brightness_values'],
            result['contrast_values']
        )
        
        # 使用结果显示管理器显示结果
        self.result_display_manager.display_feature_distribution_results(result)
        
    def generate_classification_weights(self):
        """生成分类数据集权重"""
        # 获取类别分布
        distribution_result = self.current_analyzer.analyze_data_distribution()
        train_classes = distribution_result['train_classes']
        
        if not train_classes:
            QMessageBox.warning(self, "警告", "训练集中没有找到类别文件夹")
            return
            
        # 使用权重生成器生成权重
        weight_strategies = self.weight_generator.generate_weights(train_classes)
        
        # 计算统计信息
        stats = self.weight_generator.calculate_weight_stats(weight_strategies, train_classes)
        
        # 缓存权重信息供导出使用
        self.current_class_weights = weight_strategies
        self.current_class_counts = train_classes
        
        # 使用图表管理器绘制权重分布
        balanced_weights = [weight_strategies['balanced'][name] for name in train_classes.keys()]
        self.chart_manager.plot_weight_distribution(
            list(train_classes.keys()),
            list(train_classes.values()),
            balanced_weights
        )
        
        # 使用结果显示管理器显示结果
        self.result_display_manager.display_weight_generation_results(
            train_classes, weight_strategies, stats
        )
        
        # 添加导出按钮
        self.add_export_weights_button()
        
    def analyze_detection_distribution(self):
        """分析目标检测数据集分布"""
        result = self.current_analyzer.analyze_data_distribution()
        
        # 使用图表管理器绘制图表
        self.chart_manager.plot_detection_distribution(
            result['train_annotations'],
            result['val_annotations']
        )
        
        # 使用结果显示管理器显示结果
        self.result_display_manager.display_detection_distribution_results(result)
        
    def analyze_detection_image_quality(self):
        """分析目标检测数据集图像质量"""
        result = self.current_analyzer.analyze_image_quality()
        
        # 使用图表管理器绘制图表
        self.chart_manager.plot_image_quality(
            result['image_sizes'],
            result['image_qualities']
        )
        
        # 使用结果显示管理器显示结果
        self.result_display_manager.display_image_quality_results(result)
        
    def analyze_detection_annotation_quality(self):
        """分析目标检测数据集标注质量"""
        result = self.current_analyzer.analyze_annotation_quality()
        
        # 使用图表管理器绘制图表
        self.chart_manager.plot_detection_annotation_quality(
            result['box_counts'],
            result['box_sizes']
        )
        
        # 使用结果显示管理器显示结果
        self.result_display_manager.display_detection_annotation_quality_results(result)
        
    def analyze_detection_feature_distribution(self):
        """分析目标检测数据集特征分布"""
        result = self.current_analyzer.analyze_feature_distribution()
        
        # 使用图表管理器绘制图表
        self.chart_manager.plot_feature_distribution(
            result['brightness_values'],
            result['contrast_values']
        )
        
        # 使用结果显示管理器显示结果
        self.result_display_manager.display_feature_distribution_results(result)
        
    def add_export_weights_button(self):
        """添加导出权重按钮"""
        # 检查是否已经存在导出按钮
        if hasattr(self, 'export_weights_btn'):
            return
            
        # 在结果区域下方添加导出按钮
        self.export_weights_btn = QPushButton("导出类别权重配置")
        self.export_weights_btn.clicked.connect(self.export_class_weights)
        self.export_weights_btn.setEnabled(True)
        
        # 将按钮添加到主布局
        main_layout = self.layout()
        main_layout.insertWidget(main_layout.count() - 1, self.export_weights_btn)
        
    def export_class_weights(self):
        """导出类别权重配置"""
        if not hasattr(self, 'current_class_weights') or not self.current_class_weights:
            QMessageBox.warning(self, "警告", "请先生成类别权重!")
            return
            
        # 创建策略选择对话框
        selected_strategy = self.show_strategy_selection_dialog()
        if not selected_strategy:
            return
            
        # 获取保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "保存类别权重配置", 
            f"class_weights_{selected_strategy}.json", 
            "JSON文件 (*.json)"
        )
        
        if not file_path:
            return
            
        # 使用权重生成器导出配置
        success = self.weight_generator.export_weights_config(
            self.dataset_path,
            self.current_class_counts,
            self.current_class_weights,
            selected_strategy,
            file_path
        )
        
        if success:
            strategy_names = {
                'balanced': 'Balanced (推荐)',
                'inverse': 'Inverse (逆频率)', 
                'log_inverse': 'Log Inverse (对数逆频率)',
                'normalized': 'Normalized (归一化)'
            }
            
            QMessageBox.information(
                self, 
                "导出成功", 
                f"类别权重配置已导出到:\n{file_path}\n\n"
                f"策略: {strategy_names.get(selected_strategy, selected_strategy)}\n"
                f"类别数: {len(self.current_class_counts)}\n"
                f"总样本数: {sum(self.current_class_counts.values())}\n\n"
                f"使用方法:\n"
                f"1. 在设置界面导入此配置文件\n"
                f"2. 或手动复制权重值到设置界面\n"
                f"3. 选择对应的权重策略进行训练"
            )
            
    def show_strategy_selection_dialog(self):
        """显示权重策略选择对话框"""
        strategies = list(self.current_class_weights.keys())
        strategy_names = {
            'balanced': 'Balanced (推荐)',
            'inverse': 'Inverse (逆频率)', 
            'log_inverse': 'Log Inverse (对数逆频率)',
            'normalized': 'Normalized (归一化)'
        }
        
        dialog = QDialog(self)
        dialog.setWindowTitle("选择权重策略")
        dialog.setModal(True)
        
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("请选择要导出的权重策略:"))
        
        button_group = QButtonGroup(dialog)
        radio_buttons = {}
        
        for strategy in strategies:
            display_name = strategy_names.get(strategy, strategy)
            radio_btn = QRadioButton(display_name)
            if strategy == 'balanced':  # 默认选择balanced
                radio_btn.setChecked(True)
            button_group.addButton(radio_btn)
            radio_buttons[strategy] = radio_btn
            layout.addWidget(radio_btn)
        
        # 添加对话框按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec_() != QDialog.Accepted:
            return None
            
        # 获取选择的策略
        for strategy, radio_btn in radio_buttons.items():
            if radio_btn.isChecked():
                return strategy
                
        return None
        
    def on_result_updated(self):
        """结果更新时的回调"""
        # 可以在这里添加额外的处理逻辑
        pass
        
    def update_status(self, message):
        """更新状态信息"""
        self.status_label.setText(message)
        self.status_updated.emit(message)
        
    def _do_apply_config(self, config):
        """实现具体的配置应用逻辑 - 智能配置系统"""
        print(f"DatasetEvaluationTab: 智能应用配置，包含 {len(config)} 个配置项")
        
        # 从配置中获取默认数据集评估文件夹路径
        default_dataset_path = config.get('default_dataset_dir', '')
        
        if default_dataset_path:
            # 验证路径是否存在
            if os.path.exists(default_dataset_path):
                self.dataset_path = default_dataset_path
                if hasattr(self, 'dataset_path_label'):
                    self.dataset_path_label.setText(default_dataset_path)
                if hasattr(self, 'evaluate_btn'):
                    self.evaluate_btn.setEnabled(True)
                print(f"DatasetEvaluationTab: 已加载默认数据集路径: {default_dataset_path}")
                self.status_updated.emit(f"已加载默认数据集路径: {default_dataset_path}")
            else:
                print(f"DatasetEvaluationTab: 配置中的数据集路径不存在: {default_dataset_path}")
                self.status_updated.emit(f"配置中的数据集路径不存在: {default_dataset_path}")
                
        print("DatasetEvaluationTab: 智能配置应用完成")
                
    def get_component_info(self):
        """获取组件信息 - 用于调试和监控"""
        return {
            'dataset_path': self.dataset_path,
            'current_analyzer_type': type(self.current_analyzer).__name__ if self.current_analyzer else None,
            'has_weights': self.current_class_weights is not None,
            'chart_info': self.chart_manager.get_chart_summary(),
            'ui_state': {
                'eval_type': self.eval_type_combo.currentText(),
                'metric_type': self.metrics_combo.currentText(),
                'progress_visible': self.progress_bar.isVisible(),
                'evaluate_enabled': self.evaluate_btn.isEnabled()
            }
        }
        
    def on_evaluation_finished(self, result):
        """评估完成回调"""
        try:
            # 更新UI状态
            self.progress_bar.setVisible(False)
            self.evaluate_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            
            # 处理评估结果
            eval_type = result.get('eval_type', '')
            metric_type = result.get('metric_type', '')
            analysis_type = result.get('analysis_type', '')
            
            if eval_type == "分类数据集":
                self._handle_classification_result(result, analysis_type)
            else:
                self._handle_detection_result(result, analysis_type)
                
            self.update_status(f"评估完成: {metric_type}")
            
        except Exception as e:
            self.on_evaluation_error(f"处理评估结果时出错: {str(e)}")
            
    def on_evaluation_error(self, error_msg):
        """评估错误回调"""
        # 更新UI状态
        self.progress_bar.setVisible(False)
        self.evaluate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # 显示错误信息
        QMessageBox.critical(self, "评估错误", error_msg)
        self.update_status("评估失败")
        
    def on_evaluation_stopped(self):
        """评估停止回调"""
        # 更新UI状态
        self.progress_bar.setVisible(False)
        self.evaluate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        self.update_status("评估已停止")
        
    def _handle_classification_result(self, result, analysis_type):
        """处理分类数据集评估结果"""
        if analysis_type == 'distribution':
            # 使用图表管理器绘制图表
            self.chart_manager.plot_classification_distribution(
                result['train_classes'], 
                result['val_classes']
            )
            # 使用结果显示管理器显示结果
            self.result_display_manager.display_classification_distribution_results(result)
            
        elif analysis_type == 'image_quality':
            self.chart_manager.plot_image_quality_analysis(
                result['image_sizes'],
                result['image_qualities'],
                result['brightness_values'],
                result['contrast_values']
            )
            self.result_display_manager.display_image_quality_results(result)
            
        elif analysis_type == 'annotation_quality':
            self.chart_manager.plot_annotation_quality_analysis(
                result['class_names'],
                result['annotation_counts']
            )
            self.result_display_manager.display_annotation_quality_results(result)
            
        elif analysis_type == 'feature_distribution':
            self.chart_manager.plot_feature_distribution_analysis(
                result['brightness_values'],
                result['contrast_values']
            )
            self.result_display_manager.display_feature_distribution_results(result)
            
        elif analysis_type == 'weight_generation':
            # 缓存权重结果
            self.current_class_weights = result['class_weights']
            self.current_class_counts = result['train_classes']
            
            # 显示权重生成结果
            self.chart_manager.plot_classification_distribution(
                result['train_classes'], 
                result['val_classes']
            )
            self.chart_manager.plot_weight_distribution(
                result['class_weights']
            )
            self.result_display_manager.display_weight_generation_results(result)
            
            # 添加导出按钮
            self.add_export_weights_button()
            
    def _handle_detection_result(self, result, analysis_type):
        """处理检测数据集评估结果"""
        if analysis_type == 'distribution':
            self.chart_manager.plot_detection_distribution(
                result.get('class_distribution', {}),
                result.get('bbox_count', 0),
                result.get('image_count', 0)
            )
            self.result_display_manager.display_detection_distribution_results(result)
            
        elif analysis_type == 'image_quality':
            self.chart_manager.plot_detection_image_quality_analysis(
                result.get('image_sizes', []),
                result.get('image_qualities', [])
            )
            self.result_display_manager.display_detection_image_quality_results(result)
            
        elif analysis_type == 'annotation_quality':
            self.chart_manager.plot_detection_annotation_quality_analysis(
                result.get('valid_annotations', 0),
                result.get('invalid_annotations', 0)
            )
            self.result_display_manager.display_detection_annotation_quality_results(result)
            
        elif analysis_type == 'feature_distribution':
            self.chart_manager.plot_detection_feature_distribution_analysis(
                result.get('bbox_aspect_ratios', []),
                result.get('bbox_areas', [])
            )
            self.result_display_manager.display_detection_feature_distribution_results(result) 