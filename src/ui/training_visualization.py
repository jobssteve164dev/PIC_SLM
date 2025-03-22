from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QSizePolicy, QMessageBox, QGroupBox, QTextEdit
from PyQt5.QtCore import Qt, pyqtSlot
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
# 在UI组件中使用Qt5Agg后端以正确集成到PyQt中
matplotlib.use('Qt5Agg')
import numpy as np
import os
import webbrowser
import logging
import json
import time

# 创建logger
logger = logging.getLogger(__name__)

# 配置matplotlib使用中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['font.size'] = 12  # 设置字体大小

class TrainingVisualizationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.setup_metrics()
        self.setup_logger()
        
    def setup_logger(self):
        """设置日志记录器"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def setup_metrics(self):
        """初始化训练指标"""
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.maps = []
        self.train_maps = []  # 添加训练准确率/mAP存储
        self.learning_rates = []
        # 添加更多指标存储
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.map50s = []
        self.map75s = []
        self.class_losses = []
        self.obj_losses = []
        self.box_losses = []
        # 添加分类任务特有指标
        self.roc_aucs = []
        self.average_precisions = []
        self.top_k_accuracies = []
        self.balanced_accuracies = []
        # 混淆矩阵数据无法用曲线直接显示
        
        self.update_frequency = 10  # 默认更新频率
        self.last_update_time = time.time()
        
    def init_ui(self):
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
        self.metric_combo.currentIndexChanged.connect(self.update_metric_explanation)  # 添加关联到指标说明的信号
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
        explanation_layout = QVBoxLayout()
        explanation_group = QGroupBox("当前指标说明")
        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setMaximumHeight(150)  # 限制最大高度，避免占用太多空间
        
        # 设置文本编辑器样式
        self.explanation_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                font-family: '微软雅黑', '宋体', sans-serif;
                font-size: 13px;
                line-height: 1.4;
            }
        """)
        
        explanation_inner_layout = QVBoxLayout(explanation_group)
        explanation_inner_layout.addWidget(self.explanation_text)
        explanation_layout.addWidget(explanation_group)
        
        layout.addLayout(explanation_layout)
        
        # 创建图表
        self.figure = Figure()
        
        # 损失子图
        self.loss_ax = self.figure.add_subplot(211)
        self.loss_ax.set_title('训练和验证损失')
        self.loss_ax.set_xlabel('训练轮次')
        self.loss_ax.set_ylabel('损失值')
        
        # mAP子图
        self.acc_ax = self.figure.add_subplot(212)
        self.acc_ax.set_title('验证准确率/mAP')
        self.acc_ax.set_xlabel('训练轮次')
        self.acc_ax.set_ylabel('准确率/mAP')
        
        # 创建画布并设置大小策略为扩展
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)
        
        # 设置整个组件的大小策略为扩展
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.figure.tight_layout()
        
        # 状态标签
        self.status_label = QLabel('等待训练开始...')
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # 初始化指标说明
        self.update_metric_explanation(self.metric_combo.currentIndex())
        
    def connect_signals(self, trainer):
        """连接训练器的信号"""
        try:
            # 连接实时指标更新信号
            trainer.metrics_updated.connect(self.update_metrics)
            trainer.epoch_finished.connect(self.update_epoch_metrics)
            trainer.training_finished.connect(self.on_training_finished)
            trainer.training_error.connect(self.on_training_error)
            trainer.training_stopped.connect(self.on_training_stopped)
            
            self.logger.info("成功连接训练器信号")
        except Exception as e:
            self.logger.error(f"连接训练器信号失败: {str(e)}")
            
    def update_metrics(self, metrics):
        """更新训练指标"""
        try:
            # 验证数据格式
            required_fields = ['epoch', 'train_loss', 'val_loss', 'learning_rate']
            if not all(field in metrics for field in required_fields):
                self.logger.warning(f"缺少必要的训练指标字段: {metrics}")
                return
                
            # 检查更新频率
            current_time = time.time()
            if current_time - self.last_update_time < 1.0 / self.update_frequency:
                return
            self.last_update_time = current_time
            
            # 更新数据
            epoch = int(metrics['epoch'])
            
            # 检查这个epoch是否已存在
            if epoch not in self.epochs:
                self.epochs.append(epoch)
                self.train_losses.append(metrics['train_loss'])
                self.val_losses.append(metrics['val_loss'])
                self.learning_rates.append(metrics['learning_rate'])
                
                # 根据任务类型获取准确率/mAP (兼容分类和检测任务)
                if 'val_map' in metrics:  # 检测任务
                    self.maps.append(metrics['val_map'])
                    self.train_maps.append(metrics.get('train_map', 0.0))  # 添加训练mAP
                elif 'val_accuracy' in metrics:  # 分类任务
                    self.maps.append(metrics['val_accuracy'])
                    self.train_maps.append(metrics.get('train_accuracy', 0.0))  # 添加训练准确率
                else:
                    self.maps.append(0.0)
                    self.train_maps.append(0.0)
                
                # 更新其他指标
                self.precisions.append(metrics.get('precision', 0.0))
                self.recalls.append(metrics.get('recall', 0.0))
                self.f1_scores.append(metrics.get('f1_score', 0.0))
                self.map50s.append(metrics.get('mAP50', 0.0))
                self.map75s.append(metrics.get('mAP75', 0.0))
                self.class_losses.append(metrics.get('class_loss', 0.0))
                self.obj_losses.append(metrics.get('obj_loss', 0.0))
                self.box_losses.append(metrics.get('box_loss', 0.0))
                
                # 更新分类任务特有指标
                self.roc_aucs.append(metrics.get('roc_auc', 0.0))
                self.average_precisions.append(metrics.get('average_precision', 0.0))
                self.top_k_accuracies.append(metrics.get('top_k_accuracy', 0.0))
                self.balanced_accuracies.append(metrics.get('balanced_accuracy', 0.0))
                
            else:
                idx = self.epochs.index(epoch)
                self.train_losses[idx] = metrics['train_loss']
                self.val_losses[idx] = metrics['val_loss']
                self.learning_rates[idx] = metrics['learning_rate']
                
                # 更新准确率/mAP
                if 'val_map' in metrics:
                    self.maps[idx] = metrics['val_map']
                    self.train_maps[idx] = metrics.get('train_map', 0.0)  # 更新训练mAP
                elif 'val_accuracy' in metrics:
                    self.maps[idx] = metrics['val_accuracy']
                    self.train_maps[idx] = metrics.get('train_accuracy', 0.0)  # 更新训练准确率
                
                # 更新其他指标
                if 'precision' in metrics:
                    self.precisions[idx] = metrics['precision']
                if 'recall' in metrics:
                    self.recalls[idx] = metrics['recall']
                if 'f1_score' in metrics:
                    self.f1_scores[idx] = metrics['f1_score']
                if 'mAP50' in metrics:
                    self.map50s[idx] = metrics['mAP50']
                if 'mAP75' in metrics:
                    self.map75s[idx] = metrics['mAP75']
                if 'class_loss' in metrics:
                    self.class_losses[idx] = metrics['class_loss']
                if 'obj_loss' in metrics:
                    self.obj_losses[idx] = metrics['obj_loss']
                if 'box_loss' in metrics:
                    self.box_losses[idx] = metrics['box_loss']
                
                # 更新分类任务特有指标
                if 'roc_auc' in metrics:
                    self.roc_aucs[idx] = metrics['roc_auc']
                if 'average_precision' in metrics:
                    self.average_precisions[idx] = metrics['average_precision']
                if 'top_k_accuracy' in metrics:
                    self.top_k_accuracies[idx] = metrics['top_k_accuracy']
                if 'balanced_accuracy' in metrics:
                    self.balanced_accuracies[idx] = metrics['balanced_accuracy']
            
            # 保存最后一次更新的指标以便后续使用
            self.last_metrics = metrics.copy()
            
            # 更新曲线
            self.update_display()
            
            # 更新指标显示
            self.update_metric_labels(metrics)
            
        except Exception as e:
            self.logger.error(f"更新训练指标时出错: {str(e)}")
            
    def update_metric_labels(self, metrics):
        """更新指标标签"""
        try:
            # 更新基本指标标签
            self.train_loss_label.setText(f"训练损失: {metrics['train_loss']:.4f}")
            self.val_loss_label.setText(f"验证损失: {metrics['val_loss']:.4f}")
            self.lr_label.setText(f"学习率: {metrics['learning_rate']:.6f}")
            
            # 判断任务类型
            is_classification = 'val_accuracy' in metrics
            
            # 根据任务类型更新性能指标标签
            if is_classification:
                self.map_label.setText(f"准确率: {metrics['val_accuracy']:.4f}")
            else:
                self.map_label.setText(f"mAP: {metrics['val_map']:.4f}")
            
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
        """根据选择的指标更新显示"""
        try:
            # 清除旧图
            self.loss_ax.clear()
            self.acc_ax.clear()
            
            # 获取当前选择的指标
            metric_option = self.metric_combo.currentIndex()
            
            # 确保所有数据列表长度一致
            min_len = min(len(self.epochs), len(self.train_losses))
            epochs = self.epochs[:min_len]
            train_losses = self.train_losses[:min_len]
            val_losses = self.val_losses[:min_len]
            maps = self.maps[:min_len]
            train_maps = self.train_maps[:min_len] if hasattr(self, 'train_maps') and len(self.train_maps) > 0 else []
            
            # 确保附加指标列表长度一致
            precisions = self.precisions[:min_len] if len(self.precisions) > 0 else []
            recalls = self.recalls[:min_len] if len(self.recalls) > 0 else []
            f1_scores = self.f1_scores[:min_len] if len(self.f1_scores) > 0 else []
            map50s = self.map50s[:min_len] if len(self.map50s) > 0 else []
            map75s = self.map75s[:min_len] if len(self.map75s) > 0 else []
            class_losses = self.class_losses[:min_len] if len(self.class_losses) > 0 else []
            obj_losses = self.obj_losses[:min_len] if len(self.obj_losses) > 0 else []
            box_losses = self.box_losses[:min_len] if len(self.box_losses) > 0 else []
            
            # 分类特有指标数据
            roc_aucs = self.roc_aucs[:min_len] if len(self.roc_aucs) > 0 else []
            average_precisions = self.average_precisions[:min_len] if len(self.average_precisions) > 0 else []
            top_k_accuracies = self.top_k_accuracies[:min_len] if len(self.top_k_accuracies) > 0 else []
            balanced_accuracies = self.balanced_accuracies[:min_len] if len(self.balanced_accuracies) > 0 else []
            
            # 设置x轴范围
            max_epoch = max(epochs) if epochs else 1
            x_ticks = np.arange(0, max_epoch + 1, max(1, max_epoch // 10))
            
            # 根据选择的指标显示图表
            if metric_option == 0:  # 损失和准确率总览
                # 上方图显示损失
                self.loss_ax.set_title('训练和验证损失')
                self.loss_ax.set_xlabel('训练轮次')
                self.loss_ax.set_ylabel('损失值')
                
                # 设置网格
                self.loss_ax.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制损失数据
                if train_losses:
                    self.loss_ax.plot(epochs, train_losses, 'b-', label='训练损失', marker='o', markersize=4)
                
                # 绘制验证损失
                if val_losses:
                    self.loss_ax.plot(epochs, val_losses, 'r-', label='验证损失', marker='o', markersize=4)
                    
                    # 设置y轴范围
                    filtered_val_losses = list(filter(None, val_losses))
                    if train_losses and filtered_val_losses:
                        y_min = min(min(train_losses), min(filtered_val_losses))
                        y_max = max(max(train_losses), max(filtered_val_losses))
                        y_margin = (y_max - y_min) * 0.1
                        self.loss_ax.set_ylim([max(0, y_min - y_margin), y_max + y_margin])
                    elif train_losses:
                        y_min = min(train_losses)
                        y_max = max(train_losses)
                        y_margin = (y_max - y_min) * 0.1
                        self.loss_ax.set_ylim([max(0, y_min - y_margin), y_max + y_margin])
                    elif filtered_val_losses:
                        y_min = min(filtered_val_losses)
                        y_max = max(filtered_val_losses)
                        y_margin = (y_max - y_min) * 0.1
                        self.loss_ax.set_ylim([max(0, y_min - y_margin), y_max + y_margin])
                
                self.loss_ax.legend(loc='upper right')
                self.loss_ax.set_xticks(x_ticks)
                self.loss_ax.set_visible(True)
                
                # 下方图显示准确率/mAP（根据任务类型判断）
                if any(m > 0 for m in maps):  # 有性能指标数据
                    # 是分类任务还是检测任务
                    if 'val_accuracy' in self.last_metrics:
                        self.acc_ax.set_title('训练和验证准确率')
                        self.acc_ax.set_ylabel('准确率')
                        val_metric_label = '验证准确率'
                        train_metric_label = '训练准确率'
                    else:
                        self.acc_ax.set_title('训练和验证mAP')
                        self.acc_ax.set_ylabel('mAP')
                        val_metric_label = '验证mAP'
                        train_metric_label = '训练mAP'
                    
                    self.acc_ax.set_xlabel('训练轮次')
                    
                    # 设置网格
                    self.acc_ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # 绘制验证性能指标数据
                    if maps:
                        self.acc_ax.plot(epochs, maps, 'g-', label=val_metric_label, marker='o', markersize=4)
                    
                    # 绘制训练性能指标数据
                    if train_maps and any(m > 0 for m in train_maps):
                        self.acc_ax.plot(epochs, train_maps, 'c-', label=train_metric_label, marker='o', markersize=4)
                        
                    # 设置y轴范围 - 考虑训练和验证指标
                    all_accuracies = maps.copy()
                    if train_maps:
                        all_accuracies.extend(train_maps)
                    
                    if all_accuracies:
                        # 增加安全检查，确保过滤后的列表不为空
                        filtered_accuracies = list(filter(lambda x: x > 0, all_accuracies))
                        if filtered_accuracies:
                            y_min = min(filtered_accuracies)
                            y_max = max(all_accuracies)
                            y_margin = (y_max - y_min) * 0.1
                            self.acc_ax.set_ylim([max(0, y_min - y_margin), min(1.0, y_max + y_margin)])
                        else:
                            # 如果没有大于0的准确率，使用默认范围
                            self.acc_ax.set_ylim([0, 1.0])
                    
                    self.acc_ax.legend(loc='lower right')
                    self.acc_ax.set_xticks(x_ticks)
                    self.acc_ax.set_visible(True)
                else:
                    self.acc_ax.set_visible(False)
            
            elif metric_option == 1:  # 仅损失曲线
                self.loss_ax.set_title('训练和验证损失')
                self.loss_ax.set_xlabel('训练轮次')
                self.loss_ax.set_ylabel('损失值')
                
                # 设置网格
                self.loss_ax.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制损失数据
                if train_losses:
                    self.loss_ax.plot(epochs, train_losses, 'b-', label='训练损失', marker='o', markersize=4)
                
                # 绘制验证损失
                if val_losses:
                    self.loss_ax.plot(epochs, val_losses, 'r-', label='验证损失', marker='o', markersize=4)
                    
                    # 设置y轴范围
                    filtered_val_losses = list(filter(None, val_losses))
                    if train_losses and filtered_val_losses:
                        y_min = min(min(train_losses), min(filtered_val_losses))
                        y_max = max(max(train_losses), max(filtered_val_losses))
                        y_margin = (y_max - y_min) * 0.1
                        self.loss_ax.set_ylim([max(0, y_min - y_margin), y_max + y_margin])
                    elif train_losses:
                        y_min = min(train_losses)
                        y_max = max(train_losses)
                        y_margin = (y_max - y_min) * 0.1
                        self.loss_ax.set_ylim([max(0, y_min - y_margin), y_max + y_margin])
                    elif filtered_val_losses:
                        y_min = min(filtered_val_losses)
                        y_max = max(filtered_val_losses)
                        y_margin = (y_max - y_min) * 0.1
                        self.loss_ax.set_ylim([max(0, y_min - y_margin), y_max + y_margin])
                
                self.loss_ax.legend(loc='upper right')
                self.loss_ax.set_xticks(x_ticks)
                self.loss_ax.set_visible(True)
                
                self.acc_ax.set_visible(False)  # 隐藏下方图表
                
            elif metric_option == 2:  # 仅分类准确率
                self.loss_ax.set_visible(False)  # 隐藏上方图表
                
                self.acc_ax.set_title('训练和验证准确率')
                self.acc_ax.set_xlabel('训练轮次')
                self.acc_ax.set_ylabel('准确率')
                
                # 设置网格
                self.acc_ax.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制验证准确率数据
                if maps:
                    self.acc_ax.plot(epochs, maps, 'g-', label='验证准确率', marker='o', markersize=4)
                
                # 绘制训练准确率数据
                if train_maps and any(m > 0 for m in train_maps):
                    self.acc_ax.plot(epochs, train_maps, 'c-', label='训练准确率', marker='o', markersize=4)
                    
                # 设置y轴范围
                all_accuracies = maps.copy()
                if train_maps:
                    all_accuracies.extend(train_maps)
                
                if all_accuracies:
                    # 增加安全检查，确保过滤后的列表不为空
                    filtered_accuracies = list(filter(lambda x: x > 0, all_accuracies))
                    if filtered_accuracies:
                        y_min = min(filtered_accuracies)
                        y_max = max(all_accuracies)
                        y_margin = (y_max - y_min) * 0.1
                        self.acc_ax.set_ylim([max(0, y_min - y_margin), min(1.0, y_max + y_margin)])
                    else:
                        # 如果没有大于0的准确率，使用默认范围
                        self.acc_ax.set_ylim([0, 1.0])
                
                self.acc_ax.legend(loc='lower right')
                self.acc_ax.set_xticks(x_ticks)
                self.acc_ax.set_visible(True)
                
            elif metric_option == 3:  # 仅目标检测mAP
                self.loss_ax.set_visible(False)  # 隐藏上方图表
                
                self.acc_ax.set_title('训练和验证mAP')
                self.acc_ax.set_xlabel('训练轮次')
                self.acc_ax.set_ylabel('mAP')
                
                # 设置网格
                self.acc_ax.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制验证mAP数据
                if maps:
                    self.acc_ax.plot(epochs, maps, 'g-', label='验证mAP', marker='o', markersize=4)
                
                # 绘制训练mAP数据
                if train_maps and any(m > 0 for m in train_maps):
                    self.acc_ax.plot(epochs, train_maps, 'c-', label='训练mAP', marker='o', markersize=4)
                    
                    # 设置y轴范围
                all_maps = maps.copy()
                if train_maps:
                    all_maps.extend(train_maps)
                
                if all_maps:
                    # 增加安全检查，确保过滤后的列表不为空
                    filtered_maps = list(filter(lambda x: x > 0, all_maps))
                    if filtered_maps:
                        y_min = min(filtered_maps)
                        y_max = max(all_maps)
                        y_margin = (y_max - y_min) * 0.1
                        self.acc_ax.set_ylim([max(0, y_min - y_margin), min(1.0, y_max + y_margin)])
                    else:
                        # 如果没有大于0的mAP，使用默认范围
                        self.acc_ax.set_ylim([0, 1.0])
                
                self.acc_ax.legend(loc='lower right')
                self.acc_ax.set_xticks(x_ticks)
                self.acc_ax.set_visible(True)
            
            elif metric_option == 4:  # Precision/Recall
                self.loss_ax.set_visible(False)  # 隐藏上方图表
                
                self.acc_ax.set_title('精确率和召回率')
                self.acc_ax.set_xlabel('训练轮次')
                self.acc_ax.set_ylabel('值')
                
                # 设置网格
                self.acc_ax.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制Precision数据
                if precisions:
                    self.acc_ax.plot(epochs, precisions, 'g-', label='精确率', marker='o', markersize=4)
                
                # 绘制Recall数据
                if recalls:
                    self.acc_ax.plot(epochs, recalls, 'm-', label='召回率', marker='o', markersize=4)
                
                self.acc_ax.legend(loc='lower right')
                self.acc_ax.set_xticks(x_ticks)
                self.acc_ax.set_ylim([0, 1.0])
                self.acc_ax.set_visible(True)
            
            elif metric_option == 5:  # F1-Score
                self.loss_ax.set_visible(False)  # 隐藏上方图表
                
                self.acc_ax.set_title('F1-Score')
                self.acc_ax.set_xlabel('训练轮次')
                self.acc_ax.set_ylabel('F1-Score')
                
                # 设置网格
                self.acc_ax.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制F1-Score数据
                if f1_scores:
                    self.acc_ax.plot(epochs, f1_scores, 'c-', label='F1-Score', marker='o', markersize=4)
                
                self.acc_ax.legend(loc='lower right')
                self.acc_ax.set_xticks(x_ticks)
                self.acc_ax.set_ylim([0, 1.0])
                self.acc_ax.set_visible(True)
            
            elif metric_option == 6:  # mAP50/mAP75
                self.loss_ax.set_visible(False)  # 隐藏上方图表
                
                self.acc_ax.set_title('mAP50和mAP75')
                self.acc_ax.set_xlabel('训练轮次')
                self.acc_ax.set_ylabel('mAP值')
                
                # 设置网格
                self.acc_ax.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制mAP50数据
                if map50s:
                    self.acc_ax.plot(epochs, map50s, 'r-', label='mAP50', marker='o', markersize=4)
                
                # 绘制mAP75数据
                if map75s:
                    self.acc_ax.plot(epochs, map75s, 'b-', label='mAP75', marker='o', markersize=4)
                
                self.acc_ax.legend(loc='lower right')
                self.acc_ax.set_xticks(x_ticks)
                self.acc_ax.set_ylim([0, 1.0])
                self.acc_ax.set_visible(True)
            
            elif metric_option == 7:  # 类别损失/目标损失
                self.loss_ax.set_title('类别损失和目标损失')
                self.loss_ax.set_xlabel('训练轮次')
                self.loss_ax.set_ylabel('损失值')
                
                # 设置网格
                self.loss_ax.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制类别损失
                if class_losses:
                    self.loss_ax.plot(epochs, class_losses, 'm-', label='类别损失', marker='o', markersize=4)
                
                # 绘制目标损失
                if obj_losses:
                    self.loss_ax.plot(epochs, obj_losses, 'c-', label='目标损失', marker='o', markersize=4)
                
                self.loss_ax.legend(loc='upper right')
                self.loss_ax.set_xticks(x_ticks)
                self.loss_ax.set_visible(True)
            
            elif metric_option == 8:  # 框损失/置信度损失
                self.loss_ax.set_title('框损失和置信度损失')
                self.loss_ax.set_xlabel('训练轮次')
                self.loss_ax.set_ylabel('损失值')
                
                # 设置网格
                self.loss_ax.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制框损失
                if box_losses:
                    self.loss_ax.plot(epochs, box_losses, 'y-', label='框损失', marker='o', markersize=4)
                
                # 可以添加置信度损失如果有的话
                
                self.loss_ax.legend(loc='upper right')
                self.loss_ax.set_xticks(x_ticks)
                self.loss_ax.set_visible(True)
            
            elif metric_option == 9:  # ROC-AUC曲线
                self.loss_ax.set_visible(False)  # 隐藏上方图表
                
                self.acc_ax.set_title('ROC-AUC曲线')
                self.acc_ax.set_xlabel('训练轮次')
                self.acc_ax.set_ylabel('ROC-AUC')
                
                # 设置网格
                self.acc_ax.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制ROC-AUC数据
                if roc_aucs:
                    self.acc_ax.plot(epochs, roc_aucs, 'c-', label='ROC-AUC', marker='o', markersize=4)
                
                self.acc_ax.legend(loc='lower right')
                self.acc_ax.set_xticks(x_ticks)
                self.acc_ax.set_ylim([0, 1.0])
                self.acc_ax.set_visible(True)
            
            elif metric_option == 10:  # Average Precision
                self.loss_ax.set_visible(False)  # 隐藏上方图表
                
                self.acc_ax.set_title('Average Precision')
                self.acc_ax.set_xlabel('训练轮次')
                self.acc_ax.set_ylabel('Average Precision')
                
                # 设置网格
                self.acc_ax.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制Average Precision数据
                if average_precisions:
                    self.acc_ax.plot(epochs, average_precisions, 'm-', label='Average Precision', marker='o', markersize=4)
                
                self.acc_ax.legend(loc='lower right')
                self.acc_ax.set_xticks(x_ticks)
                self.acc_ax.set_ylim([0, 1.0])
                self.acc_ax.set_visible(True)
            
            elif metric_option == 11:  # Top-K准确率
                self.loss_ax.set_visible(False)  # 隐藏上方图表
                
                self.acc_ax.set_title('Top-K准确率')
                self.acc_ax.set_xlabel('训练轮次')
                self.acc_ax.set_ylabel('Top-K准确率')
                
                # 设置网格
                self.acc_ax.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制Top-K准确率数据
                if top_k_accuracies:
                    self.acc_ax.plot(epochs, top_k_accuracies, 'y-', label='Top-K准确率', marker='o', markersize=4)
                
                self.acc_ax.legend(loc='lower right')
                self.acc_ax.set_xticks(x_ticks)
                self.acc_ax.set_ylim([0, 1.0])
                self.acc_ax.set_visible(True)
            
            elif metric_option == 12:  # 平衡准确率
                self.loss_ax.set_visible(False)  # 隐藏上方图表
                
                self.acc_ax.set_title('平衡准确率')
                self.acc_ax.set_xlabel('训练轮次')
                self.acc_ax.set_ylabel('平衡准确率')
                
                # 设置网格
                self.acc_ax.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制平衡准确率数据
                if balanced_accuracies:
                    self.acc_ax.plot(epochs, balanced_accuracies, 'g-', label='平衡准确率', marker='o', markersize=4)
                
                self.acc_ax.legend(loc='lower right')
                self.acc_ax.set_xticks(x_ticks)
                self.acc_ax.set_ylim([0, 1.0])
                self.acc_ax.set_visible(True)
            
            # 重新调整布局和绘制
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"更新图表显示时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def set_update_frequency(self, frequency):
        """设置更新频率"""
        try:
            self.update_frequency = max(1, min(frequency, 100))  # 限制频率范围
            self.logger.info(f"更新频率设置为: {self.update_frequency}")
        except Exception as e:
            self.logger.error(f"设置更新频率时出错: {str(e)}")
            
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
        try:
            data = {
                'epochs': self.epochs,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'maps': self.maps,
                'learning_rates': self.learning_rates,
                'precisions': self.precisions,
                'recalls': self.recalls,
                'f1_scores': self.f1_scores,
                'map50s': self.map50s,
                'map75s': self.map75s,
                'class_losses': self.class_losses,
                'obj_losses': self.obj_losses,
                'box_losses': self.box_losses,
                'roc_aucs': self.roc_aucs,
                'average_precisions': self.average_precisions,
                'top_k_accuracies': self.top_k_accuracies,
                'balanced_accuracies': self.balanced_accuracies
            }
            
            # 保存为JSON文件
            save_path = os.path.join('models', 'training_data.json')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(data, f)
                
            self.logger.info(f"训练数据已保存到: {save_path}")
        except Exception as e:
            self.logger.error(f"保存训练数据时出错: {str(e)}")
            
    def resizeEvent(self, event):
        """重写resizeEvent以在窗口大小改变时调整图表布局"""
        super().resizeEvent(event)
        self.figure.tight_layout()
        self.canvas.draw()
        
    def reset_plots(self):
        """重置图表数据"""
        # 重置基本训练指标
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.maps = []
        self.train_maps = []  # 重置训练准确率/mAP
        self.learning_rates = []
        
        # 重置共用指标
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        
        # 重置检测任务特有指标
        self.map50s = []
        self.map75s = []
        self.class_losses = []
        self.obj_losses = []
        self.box_losses = []
        
        # 重置分类任务特有指标
        self.roc_aucs = []
        self.average_precisions = []
        self.top_k_accuracies = []
        self.balanced_accuracies = []
        
        # 清除最后一次指标记录
        if hasattr(self, 'last_metrics'):
            self.last_metrics = {}
        
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
        
        # 清空图表
        self.loss_ax.clear()
        self.acc_ax.clear()
        
        # 重置图表标题和标签
        self.loss_ax.set_title('训练和验证损失')
        self.loss_ax.set_xlabel('训练轮次')
        self.loss_ax.set_ylabel('损失值')
        
        self.acc_ax.set_title('验证性能指标')  # 修改为通用标题
        self.acc_ax.set_xlabel('训练轮次')
        self.acc_ax.set_ylabel('指标值')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        # 更新状态
        self.status_label.setText("图表已重置，等待训练开始...")

    def update_metric_explanation(self, index):
        """根据选择的指标更新说明文本"""
        explanations = {
            0: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">损失和准确率总览</h3>
                <p>同时显示训练损失、验证损失、训练准确率和验证准确率曲线，提供整体训练进度视图。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>训练损失</b>：模型在训练集上的误差，值越小表示模型在训练数据上拟合得越好。</li>
                <li><b>验证损失</b>：模型在验证集上的误差，是评估模型泛化能力的重要指标。</li>
                <li><b>训练准确率</b>：模型在训练集上的预测准确率。</li>
                <li><b>验证准确率</b>：模型在验证集上的预测准确率，反映模型泛化能力。</li>
                </ul>
                </div>""",
                
            1: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">仅损失曲线</h3>
                <p>仅显示训练损失和验证损失曲线，专注于损失变化趋势。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>训练损失</b>：模型在训练集上的误差，值越小表示模型在训练数据上拟合得越好。</li>
                <li><b>验证损失</b>：模型在验证集上的误差，是评估模型泛化能力的重要指标。</li>
                </ul>
                <p>两者的变化趋势可以帮助判断模型是否过拟合。验证损失上升而训练损失下降，表明模型可能过拟合了训练数据。</p>
                </div>""",
                
            2: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">仅分类准确率</h3>
                <p>仅显示分类模型的验证准确率曲线。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>准确率</b>：正确预测的样本数占总样本数的比例，是分类任务的主要评价指标。</li>
                </ul>
                <p>准确率越高表示模型性能越好，曲线上升表明模型不断改进，平稳则表明训练可能达到瓶颈。</p>
                </div>""",
                
            3: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">仅目标检测mAP</h3>
                <p>仅显示目标检测模型的验证mAP曲线。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>mAP</b>：平均精度均值，是目标检测任务的标准评价指标。</li>
                </ul>
                <p>mAP综合考虑了检测的精确性和完整性，值越高表示检测性能越好。</p>
                </div>""",
                
            4: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">Precision/Recall</h3>
                <p>显示精确率和召回率曲线。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>精确率(Precision)</b>：正确预测为正例的数量占所有预测为正例的比例。</li>
                <li><b>召回率(Recall)</b>：正确预测为正例的数量占所有实际正例的比例。</li>
                </ul>
                <p>这两个指标通常是此消彼长的关系，需要在应用中找到合适的平衡点。</p>
                </div>""",
                
            5: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">F1-Score</h3>
                <p>显示F1分数曲线。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>F1-Score</b>：精确率和召回率的调和平均值，平衡了这两个指标的重要性。</li>
                </ul>
                <p>F1-Score是0到1之间的值，越接近1表示模型的精确率和召回率都较高。</p>
                <p>对于类别不平衡的数据集，F1-Score通常比简单的准确率更能反映模型性能。</p>
                </div>""",
                
            6: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">mAP50/mAP75</h3>
                <p>显示在不同IOU阈值下的mAP曲线。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>mAP50</b>：IOU阈值为0.5时的平均精度均值。</li>
                <li><b>mAP75</b>：IOU阈值为0.75时的平均精度均值，要求更加严格。</li>
                </ul>
                <p>IOU阈值越高，对目标检测位置精度的要求越高。mAP75更能反映模型的精细定位能力。</p>
                </div>""",
                
            7: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">类别损失/目标损失</h3>
                <p>显示目标检测中的分类分支和目标存在性损失曲线。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>类别损失</b>：目标检测中负责将物体分类到正确类别的损失函数。</li>
                <li><b>目标损失</b>：检测是否存在物体的置信度损失函数。</li>
                </ul>
                <p>这些损失值的变化可以帮助诊断检测模型在分类和定位方面的问题。</p>
                </div>""",
                
            8: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">框损失/置信度损失</h3>
                <p>显示目标检测中的边界框回归和置信度损失曲线。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>框损失</b>：边界框位置和尺寸预测的损失函数。</li>
                <li><b>置信度损失</b>：模型对预测框置信度的损失函数。</li>
                </ul>
                <p>框损失下降表明模型定位能力提升，置信度损失下降表明模型对预测的确信度提高。</p>
                </div>""",
                
            9: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">ROC-AUC曲线</h3>
                <p>显示ROC曲线下的面积变化。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>ROC-AUC</b>：ROC曲线下的面积，衡量分类器的性能，越接近1越好。</li>
                </ul>
                <p>AUC值为0.5表示模型性能与随机猜测相当；接近1表示模型有很强的区分能力。</p>
                <p>AUC对不平衡数据集不敏感，即使在正负样本比例变化时也能保持稳定。</p>
                </div>""",
                
            10: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">Average Precision</h3>
                <p>显示PR曲线下的面积变化。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>平均精度</b>：PR曲线下的面积，在不平衡数据集上比AUC更有参考价值。</li>
                </ul>
                <p>该指标综合考虑了精确率和召回率，特别适用于正负样本不平衡的情况。</p>
                <p>在多类别任务中，可以计算每个类别的AP然后取平均值(mAP)。</p>
                </div>""",
                
            11: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">Top-K准确率</h3>
                <p>显示模型预测正确类别位于前K个预测中的比例。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>Top-K准确率</b>：预测的前K个类别中包含正确类别的比例。</li>
                </ul>
                <p>对于大量类别的分类任务，Top-K准确率通常比Top-1准确率更为宽松。</p>
                <p>常见的K值有1、3、5等，对于复杂分类任务，Top-5准确率是一个常用指标。</p>
                </div>""",
                
            12: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">平衡准确率</h3>
                <p>显示考虑了类别不平衡的准确率指标。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>平衡准确率</b>：每个类别的准确率的平均值，而不是简单地计算正确样本比例。</li>
                </ul>
                <p>对于类别不平衡的数据集，这个指标比普通准确率更公平。</p>
                <p>它给予所有类别相同的权重，防止模型偏向主导类别。</p>
                </div>"""
        }
        
        self.explanation_text.setHtml(explanations.get(index, "<div style='margin:5px;'><p>无相关说明</p></div>"))


class TensorBoardWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tensorboard_dir = None
        self.init_ui()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 说明标签
        info_label = QLabel("TensorBoard是一个强大的可视化工具，可以帮助您更深入地分析模型训练过程。")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 状态标签
        self.status_label = QLabel("TensorBoard日志目录: 未设置")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        # 使用说明
        usage_label = QLabel(
            "使用说明:\n"
            "1. 在模型训练选项卡中启用TensorBoard选项\n"
            "2. 开始训练模型\n"
            "3. 训练过程中或训练完成后，使用上方的TensorBoard控件\n"
            "4. TensorBoard将在浏览器中打开，显示详细的训练指标和可视化"
        )
        usage_label.setWordWrap(True)
        usage_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        layout.addWidget(usage_label)
        
        layout.addStretch()
        
    def update_tensorboard(self, metric_name, value, step):
        """接收并处理来自训练器的TensorBoard更新信号"""
        try:
            self.logger.info(f"接收到TensorBoard更新: {metric_name}={value:.4f} (step={step})")
            
            # 更新状态标签，显示最近的指标更新
            self.status_label.setText(
                f"TensorBoard日志目录: {self.tensorboard_dir or '未设置'}\n"
                f"最新指标: {metric_name} = {value:.4f} (step={step})"
            )
            
            # 如果TensorBoard目录未设置，尝试使用默认目录
            if not self.tensorboard_dir:
                default_dir = os.path.join('runs', 'detection')
                if os.path.exists(default_dir):
                    self.set_tensorboard_dir(default_dir)
                    self.logger.info(f"自动设置TensorBoard目录: {default_dir}")
            
        except Exception as e:
            self.logger.error(f"处理TensorBoard更新时出错: {str(e)}")
        
    def set_tensorboard_dir(self, directory):
        """设置TensorBoard日志目录"""
        if directory and os.path.exists(directory):
            self.tensorboard_dir = directory
            self.status_label.setText(f"TensorBoard日志目录: {directory}")
        else:
            self.status_label.setText(f"TensorBoard日志目录不存在: {directory}") 