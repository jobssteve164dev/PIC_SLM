from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QSizePolicy, QMessageBox
from PyQt5.QtCore import Qt, pyqtSlot
import numpy as np
import logging
import os
import json

from .plot_canvas import PlotCanvas
from .metrics_manager import MetricsManager
from .metrics_explanation import MetricsExplanation

logger = logging.getLogger(__name__)

class TrainingVisualizationWidget(QWidget):
    """训练可视化主组件，整合各个子组件实现训练过程的可视化"""
    
    def __init__(self, parent=None):
        """初始化训练可视化组件"""
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 初始化子组件
        self.metrics_manager = MetricsManager()
        self.plot_canvas = PlotCanvas(figsize=(12, 8), dpi=100)
        self.metrics_explanation = MetricsExplanation()
        
        # 初始化UI
        self.init_ui()
        
        # 记录最后更新时间
        self.last_update_time = 0
        
    def init_ui(self):
        """初始化UI界面"""
        layout = QVBoxLayout(self)
        
        # 控制面板
        control_layout = QHBoxLayout()
        
        # 指标选择
        self.metric_combo = QComboBox()
        self.metric_combo.addItems([
            '损失和准确率总览',  # 修改为更明确的名称
            '仅损失曲线',       
            '仅分类准确率',     
            '仅目标检测mAP',    
            'Precision/Recall',  
            'F1-Score',          
            'mAP50/mAP75',       
            '类别损失/目标损失',  
            '框损失/置信度损失',  
            'ROC-AUC曲线',       
            'Average Precision', 
            'Top-K准确率',       
            '平衡准确率'         
        ])
        self.metric_combo.setCurrentIndex(0)
        self.metric_combo.currentIndexChanged.connect(self.update_display)
        self.metric_combo.currentIndexChanged.connect(self.update_metric_explanation)
        control_layout.addWidget(QLabel('显示指标:'))
        control_layout.addWidget(self.metric_combo)
        
        # 重置按钮
        self.reset_btn = QPushButton('重置图表')
        self.reset_btn.clicked.connect(self.reset_plots)
        control_layout.addWidget(self.reset_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # 添加指标标签
        metrics_layout = QHBoxLayout()
        
        # 训练损失标签
        self.train_loss_label = QLabel('训练损失: 0.0000')
        metrics_layout.addWidget(self.train_loss_label)
        
        # 验证损失标签
        self.val_loss_label = QLabel('验证损失: 0.0000')
        metrics_layout.addWidget(self.val_loss_label)
        
        # mAP标签
        self.map_label = QLabel('mAP: 0.0000')
        metrics_layout.addWidget(self.map_label)
        
        # 学习率标签
        self.lr_label = QLabel('学习率: 0.000000')
        metrics_layout.addWidget(self.lr_label)
        
        # 添加更多标签用于显示其他指标
        self.precision_label = QLabel('精确率: 0.0000')
        metrics_layout.addWidget(self.precision_label)
        
        self.recall_label = QLabel('召回率: 0.0000')
        metrics_layout.addWidget(self.recall_label)
        
        self.f1_label = QLabel('F1-Score: 0.0000')
        metrics_layout.addWidget(self.f1_label)
        
        # 创建第二行指标布局
        metrics_layout2 = QHBoxLayout()
        
        # 添加分类特有指标标签
        self.roc_auc_label = QLabel('ROC-AUC: 0.0000')
        metrics_layout2.addWidget(self.roc_auc_label)
        
        self.avg_precision_label = QLabel('Avg.Precision: 0.0000')
        metrics_layout2.addWidget(self.avg_precision_label)
        
        self.top_k_label = QLabel('Top-K准确率: 0.0000')
        metrics_layout2.addWidget(self.top_k_label)
        
        self.balanced_acc_label = QLabel('平衡准确率: 0.0000')
        metrics_layout2.addWidget(self.balanced_acc_label)
        
        # 添加检测特有指标标签
        self.map50_label = QLabel('mAP50: 0.0000')
        metrics_layout2.addWidget(self.map50_label)
        
        self.map75_label = QLabel('mAP75: 0.0000')
        metrics_layout2.addWidget(self.map75_label)
        
        # 添加检测特有损失标签
        self.class_loss_label = QLabel('类别损失: 0.0000')
        metrics_layout2.addWidget(self.class_loss_label)
        
        self.obj_loss_label = QLabel('目标损失: 0.0000')
        metrics_layout2.addWidget(self.obj_loss_label)
        
        self.box_loss_label = QLabel('框损失: 0.0000')
        metrics_layout2.addWidget(self.box_loss_label)
        
        layout.addLayout(metrics_layout)
        layout.addLayout(metrics_layout2)
        
        # 添加指标说明区域
        explanation_group, self.explanation_text = self.metrics_explanation.create_explanation_widget()
        layout.addWidget(explanation_group)
        
        # 添加图表画布
        canvas = self.plot_canvas.get_canvas()
        layout.addWidget(canvas)
        
        # 创建状态标签，但设置为隐藏
        self.status_label = QLabel('等待训练开始...')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setVisible(False)  # 隐藏这个标签，因为与EvaluationTab中的training_status_label重复
        layout.addWidget(self.status_label)
        
        # 设置整个组件的大小策略为扩展
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumWidth(650)  # 设置组件最小宽度
        
        # 初始化指标说明
        self.update_metric_explanation(self.metric_combo.currentIndex())
        
    def connect_signals(self, trainer):
        """连接训练器的信号"""
        try:
            # 连接实时指标更新信号
            trainer.metrics_updated.connect(self.update_metrics)
            trainer.epoch_finished.connect(self.update_metrics)  # 使用相同的处理函数
            trainer.training_finished.connect(self.on_training_finished)
            trainer.training_error.connect(self.on_training_error)
            trainer.training_stopped.connect(self.on_training_stopped)
            
            self.logger.info("成功连接训练器信号")
        except Exception as e:
            self.logger.error(f"连接训练器信号失败: {str(e)}")
            
    def update_metrics(self, metrics):
        """更新训练指标，并触发UI更新"""
        try:
            # 使用指标管理器处理指标数据
            updated = self.metrics_manager.update_metrics(metrics)
            
            if updated:
                # 更新曲线图表
                self.update_display()
                
                # 更新指标标签文本
                self.update_metric_labels(metrics)
                
        except Exception as e:
            self.logger.error(f"更新训练指标时出错: {str(e)}")
            
    def update_metric_labels(self, metrics):
        """更新指标标签显示"""
        try:
            # 更新基本指标标签
            self.train_loss_label.setText(f"训练损失: {metrics.get('train_loss', 0.0):.4f}")
            self.val_loss_label.setText(f"验证损失: {metrics.get('val_loss', 0.0):.4f}")
            self.lr_label.setText(f"学习率: {metrics.get('learning_rate', 0.0):.6f}")
            
            # 判断任务类型
            is_classification = 'val_accuracy' in metrics
            
            # 根据任务类型更新性能指标标签
            if is_classification:
                self.map_label.setText(f"准确率: {metrics.get('val_accuracy', 0.0):.4f}")
            else:
                self.map_label.setText(f"mAP: {metrics.get('val_map', 0.0):.4f}")
            
            # 更新共用指标标签
            if 'precision' in metrics:
                self.precision_label.setText(f"精确率: {metrics['precision']:.4f}")
            if 'recall' in metrics:
                self.recall_label.setText(f"召回率: {metrics['recall']:.4f}")
            if 'f1_score' in metrics:
                self.f1_label.setText(f"F1-Score: {metrics['f1_score']:.4f}")
                
            # 更新分类特有指标标签
            if is_classification:
                if 'roc_auc' in metrics:
                    self.roc_auc_label.setText(f"ROC-AUC: {metrics['roc_auc']:.4f}")
                if 'average_precision' in metrics:
                    self.avg_precision_label.setText(f"Avg.Precision: {metrics['average_precision']:.4f}")
                if 'top_k_accuracy' in metrics:
                    self.top_k_label.setText(f"Top-K准确率: {metrics['top_k_accuracy']:.4f}")
                if 'balanced_accuracy' in metrics:
                    self.balanced_acc_label.setText(f"平衡准确率: {metrics['balanced_accuracy']:.4f}")
            
            # 更新检测特有指标标签
            else:
                if 'mAP50' in metrics:
                    self.map50_label.setText(f"mAP50: {metrics['mAP50']:.4f}")
                if 'mAP75' in metrics:
                    self.map75_label.setText(f"mAP75: {metrics['mAP75']:.4f}")
                if 'class_loss' in metrics and 'obj_loss' in metrics:
                    self.class_loss_label.setText(f"类别损失: {metrics['class_loss']:.4f}")
                    self.obj_loss_label.setText(f"目标损失: {metrics['obj_loss']:.4f}")
                if 'box_loss' in metrics:
                    self.box_loss_label.setText(f"框损失: {metrics['box_loss']:.4f}")
            
        except Exception as e:
            self.logger.error(f"更新指标标签时出错: {str(e)}")
            
    def update_display(self):
        """根据选择的指标更新图表显示"""
        try:
            # 清除图表
            self.plot_canvas.clear_axes()
            
            # 获取当前选择的指标
            metric_option = self.metric_combo.currentIndex()
            
            # 获取处理后的训练数据
            data = self.metrics_manager.get_aligned_data()
            
            # 如果没有数据，直接返回
            if data is None:
                self.plot_canvas.resize_plot()
                return
                
            # 解包数据
            (epochs, train_losses, val_losses, maps, train_maps,
             precisions, recalls, f1_scores, map50s, map75s,
             class_losses, obj_losses, box_losses,
             roc_aucs, average_precisions, top_k_accuracies, balanced_accuracies) = data
            
            # 获取x轴刻度
            x_ticks = self.plot_canvas.generate_x_ticks(epochs)
            
            # 根据选择的指标显示不同图表
            loss_ax = self.plot_canvas.loss_ax
            acc_ax = self.plot_canvas.acc_ax
            
            if metric_option == 0:  # 损失和准确率总览
                # 上方图显示损失
                loss_ax.set_title('训练和验证损失')
                loss_ax.set_xlabel('训练轮次')
                loss_ax.set_ylabel('损失值')
                
                # 设置网格
                loss_ax.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制损失数据
                if train_losses:
                    loss_ax.plot(epochs, train_losses, 'b-', label='训练损失', marker='o', markersize=4)
                
                # 绘制验证损失
                if val_losses:
                    loss_ax.plot(epochs, val_losses, 'r-', label='验证损失', marker='o', markersize=4)
                
                # 计算并设置y轴范围
                y_range = self.plot_canvas.calculate_y_range(train_losses + val_losses)
                loss_ax.set_ylim(y_range)
                
                loss_ax.legend(loc='upper right')
                loss_ax.set_xticks(x_ticks)
                loss_ax.set_visible(True)
                
                # 下方图显示准确率/mAP
                if any(m > 0 for m in maps):  # 有性能指标数据
                    # 根据任务类型设置图表
                    if self.metrics_manager.task_type == 'classification':
                        acc_ax.set_title('训练和验证准确率')
                        acc_ax.set_ylabel('准确率')
                        val_metric_label = '验证准确率'
                        train_metric_label = '训练准确率'
                    else:  # 默认为detection或未设置
                        acc_ax.set_title('训练和验证mAP')
                        acc_ax.set_ylabel('mAP')
                        val_metric_label = '验证mAP'
                        train_metric_label = '训练mAP'
                    
                    acc_ax.set_xlabel('训练轮次')
                    
                    # 设置网格
                    acc_ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # 绘制验证性能指标数据
                    if maps:
                        acc_ax.plot(epochs, maps, 'g-', label=val_metric_label, marker='o', markersize=4)
                    
                    # 绘制训练性能指标数据
                    if train_maps and any(m > 0 for m in train_maps):
                        acc_ax.plot(epochs, train_maps, 'c-', label=train_metric_label, marker='o', markersize=4)
                    
                    # 计算并设置y轴范围
                    all_metrics = [m for m in maps + train_maps if m is not None and m > 0]
                    y_range = self.plot_canvas.calculate_y_range(all_metrics, is_accuracy=True)
                    acc_ax.set_ylim(y_range)
                    
                    acc_ax.legend(loc='lower right')
                    acc_ax.set_xticks(x_ticks)
                    acc_ax.set_visible(True)
                else:
                    acc_ax.set_visible(False)
                    
            elif metric_option == 1:  # 仅损失曲线
                loss_ax.set_title('训练和验证损失')
                loss_ax.set_xlabel('训练轮次')
                loss_ax.set_ylabel('损失值')
                
                # 设置网格
                loss_ax.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制损失数据
                if train_losses:
                    loss_ax.plot(epochs, train_losses, 'b-', label='训练损失', marker='o', markersize=4)
                
                # 绘制验证损失
                if val_losses:
                    loss_ax.plot(epochs, val_losses, 'r-', label='验证损失', marker='o', markersize=4)
                
                # 计算并设置y轴范围
                y_range = self.plot_canvas.calculate_y_range(train_losses + val_losses)
                loss_ax.set_ylim(y_range)
                
                loss_ax.legend(loc='upper right')
                loss_ax.set_xticks(x_ticks)
                loss_ax.set_visible(True)
                
                acc_ax.set_visible(False)  # 隐藏下方图表
                
            elif metric_option == 2:  # 仅分类准确率
                loss_ax.set_visible(False)  # 隐藏上方图表
                
                # 使用任务类型判断
                if self.metrics_manager.task_type == 'classification':
                    acc_ax.set_title('训练和验证准确率')
                    acc_ax.set_xlabel('训练轮次')
                    acc_ax.set_ylabel('准确率')
                    
                    # 设置网格
                    acc_ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # 绘制验证准确率数据
                    if maps:
                        acc_ax.plot(epochs, maps, 'g-', label='验证准确率', marker='o', markersize=4)
                    
                    # 绘制训练准确率数据
                    if train_maps and any(m > 0 for m in train_maps):
                        acc_ax.plot(epochs, train_maps, 'c-', label='训练准确率', marker='o', markersize=4)
                    
                    # 计算并设置y轴范围
                    all_metrics = [m for m in maps + train_maps if m is not None and m > 0]
                    y_range = self.plot_canvas.calculate_y_range(all_metrics, is_accuracy=True)
                    acc_ax.set_ylim(y_range)
                    
                    acc_ax.legend(loc='lower right')
                    acc_ax.set_xticks(x_ticks)
                    acc_ax.set_visible(True)
                else:
                    # 如果不是分类任务，显示提示信息
                    acc_ax.text(0.5, 0.5, '当前模型不是分类模型', 
                                horizontalalignment='center', verticalalignment='center',
                                transform=acc_ax.transAxes, fontsize=14)
                    acc_ax.set_visible(True)
            
            # 进一步实现其他图表类型的绘制...（省略其余代码，根据需要实现）
            # 这里应该包含剩余的metric_option 3-12的处理代码
            
            # 调整布局并绘制
            self.plot_canvas.resize_plot()
            
        except Exception as e:
            self.logger.error(f"更新图表显示时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_metric_explanation(self, index):
        """更新指标说明文本"""
        explanation_html = self.metrics_explanation.get_explanation(index)
        self.explanation_text.setHtml(explanation_html)
            
    def set_update_frequency(self, frequency):
        """设置更新频率"""
        self.metrics_manager.set_update_frequency(frequency)
            
    def on_training_finished(self):
        """训练完成处理"""
        try:
            self.logger.info("训练完成")
            # 保存训练曲线数据
            self.save_training_data()
        except Exception as e:
            self.logger.error(f"处理训练完成时出错: {str(e)}")
            
    def on_training_error(self, error_msg):
        """训练错误处理"""
        try:
            self.logger.error(f"训练错误: {error_msg}")
            # 显示错误消息
            QMessageBox.critical(self, "训练错误", error_msg)
        except Exception as e:
            self.logger.error(f"处理训练错误时出错: {str(e)}")
            
    def on_training_stopped(self):
        """训练停止处理"""
        try:
            self.logger.info("训练已停止")
            # 保存训练曲线数据
            self.save_training_data()
        except Exception as e:
            self.logger.error(f"处理训练停止时出错: {str(e)}")
            
    def save_training_data(self):
        """保存训练数据"""
        return self.metrics_manager.save_training_data('models')
            
    def resizeEvent(self, event):
        """重写resizeEvent以在窗口大小改变时调整图表布局"""
        super().resizeEvent(event)
        self.plot_canvas.resize_plot()
        
    def reset_plots(self):
        """重置图表数据"""
        # 重置指标管理器
        self.metrics_manager.reset_metrics()
        
        # 重置绘图组件历史记录
        self.plot_canvas.reset_history()
        
        # 重置标签显示
        self.train_loss_label.setText("训练损失: 0.0000")
        self.val_loss_label.setText("验证损失: 0.0000")
        self.map_label.setText("mAP/准确率: 0.0000")  # 通用标签兼容分类和检测
        self.lr_label.setText("学习率: 0.000000")
        
        # 重置共用指标标签
        self.precision_label.setText("精确率: 0.0000")
        self.recall_label.setText("召回率: 0.0000")
        self.f1_label.setText("F1-Score: 0.0000")
        
        # 重置分类特有指标标签
        self.roc_auc_label.setText("ROC-AUC: 0.0000")
        self.avg_precision_label.setText("Avg.Precision: 0.0000")
        self.top_k_label.setText("Top-K准确率: 0.0000")
        self.balanced_acc_label.setText("平衡准确率: 0.0000")
        
        # 重置检测特有指标标签
        self.map50_label.setText("mAP50: 0.0000")
        self.map75_label.setText("mAP75: 0.0000")
        self.class_loss_label.setText("类别损失: 0.0000")
        self.obj_loss_label.setText("目标损失: 0.0000")
        self.box_loss_label.setText("框损失: 0.0000")
        
        # 清空图表并绘制
        self.plot_canvas.clear_axes()
        self.plot_canvas.reset_axes_titles()
        
        # 更新状态文本但保持隐藏状态
        self.status_label.setText("图表已重置，等待训练开始...")
        self.status_label.setVisible(False)  # 确保状态标签保持隐藏 