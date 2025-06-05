from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QComboBox, QSizePolicy, QMessageBox, QGroupBox, QTextEdit)
from PyQt5.QtCore import Qt, pyqtSlot
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
import logging

# 导入分离的组件
from .metrics_data_manager import MetricsDataManager
from .chart_renderer import ChartRenderer
from .metric_explanations import MetricExplanations
from .tensorboard_widget import TensorBoardWidget

# 在UI组件中使用Qt5Agg后端以正确集成到PyQt中
matplotlib.use('Qt5Agg')

# 配置matplotlib使用中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['font.size'] = 12  # 设置字体大小


class TrainingVisualizationWidget(QWidget):
    """训练可视化主组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = self.setup_logger()
        
        # 初始化数据管理器
        self.data_manager = MetricsDataManager()
        
        # 初始化UI
        self.init_ui()
        
        # 初始化图表渲染器
        self.chart_renderer = ChartRenderer(self.figure, self.loss_ax, self.acc_ax)
        
        # 初始化完成后调用一次更新显示，确保默认视图正确
        self.update_display()
        
    def setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 控制面板
        control_layout = QHBoxLayout()
        
        # 指标选择
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(MetricExplanations.get_metric_names())
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
        
        # 指标标签区域
        self.create_metric_labels(layout)
        
        # 添加指标说明区域
        self.create_explanation_area(layout)
        
        # 创建图表
        self.create_charts(layout)
        
        # 创建状态标签（隐藏）
        self.status_label = QLabel('等待训练开始...')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)
        
        # 初始化指标说明
        self.update_metric_explanation(self.metric_combo.currentIndex())
        
    def create_metric_labels(self, layout):
        """创建指标标签区域"""
        # 第一行指标布局
        metrics_layout = QHBoxLayout()
        
        # 基础指标标签
        self.train_loss_label = QLabel('训练损失: 0.0000')
        metrics_layout.addWidget(self.train_loss_label)
        
        self.val_loss_label = QLabel('验证损失: 0.0000')
        metrics_layout.addWidget(self.val_loss_label)
        
        self.map_label = QLabel('mAP: 0.0000')
        metrics_layout.addWidget(self.map_label)
        
        self.lr_label = QLabel('学习率: 0.000000')
        metrics_layout.addWidget(self.lr_label)
        
        # 通用指标标签
        self.precision_label = QLabel('精确率: 0.0000')
        metrics_layout.addWidget(self.precision_label)
        
        self.recall_label = QLabel('召回率: 0.0000')
        metrics_layout.addWidget(self.recall_label)
        
        self.f1_label = QLabel('F1-Score: 0.0000')
        metrics_layout.addWidget(self.f1_label)
        
        layout.addLayout(metrics_layout)
        
        # 第二行指标布局
        metrics_layout2 = QHBoxLayout()
        
        # 分类特有指标标签
        self.roc_auc_label = QLabel('ROC-AUC: 0.0000')
        metrics_layout2.addWidget(self.roc_auc_label)
        
        self.avg_precision_label = QLabel('Avg.Precision: 0.0000')
        metrics_layout2.addWidget(self.avg_precision_label)
        
        self.top_k_label = QLabel('Top-K准确率: 0.0000')
        metrics_layout2.addWidget(self.top_k_label)
        
        self.balanced_acc_label = QLabel('平衡准确率: 0.0000')
        metrics_layout2.addWidget(self.balanced_acc_label)
        
        # 检测特有指标标签
        self.map50_label = QLabel('mAP50: 0.0000')
        metrics_layout2.addWidget(self.map50_label)
        
        self.map75_label = QLabel('mAP75: 0.0000')
        metrics_layout2.addWidget(self.map75_label)
        
        self.class_loss_label = QLabel('类别损失: 0.0000')
        metrics_layout2.addWidget(self.class_loss_label)
        
        self.obj_loss_label = QLabel('目标损失: 0.0000')
        metrics_layout2.addWidget(self.obj_loss_label)
        
        self.box_loss_label = QLabel('框损失: 0.0000')
        metrics_layout2.addWidget(self.box_loss_label)
        
        layout.addLayout(metrics_layout2)
        
    def create_explanation_area(self, layout):
        """创建指标说明区域"""
        explanation_layout = QVBoxLayout()
        explanation_group = QGroupBox("当前指标说明")
        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setMaximumHeight(150)
        
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
        
    def create_charts(self, layout):
        """创建图表区域"""
        # 增大图表默认大小，设置更宽的图形
        self.figure = Figure(figsize=(12, 8), dpi=100)
        
        # 设置更大的左右边距，防止截断
        self.figure.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
        
        # 损失子图
        self.loss_ax = self.figure.add_subplot(211)
        self.loss_ax.set_title('训练和验证损失')
        self.loss_ax.set_xlabel('训练轮次')
        self.loss_ax.set_ylabel('损失值')
        
        # mAP子图
        self.acc_ax = self.figure.add_subplot(212)
        self.acc_ax.set_title('训练和验证准确率/mAP')
        self.acc_ax.set_xlabel('训练轮次')
        self.acc_ax.set_ylabel('准确率/mAP')
        
        # 创建画布并设置大小策略为扩展
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumWidth(600)
        layout.addWidget(self.canvas)
        
        # 设置整个组件的大小策略为扩展
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumWidth(650)
        
        # 使用tight_layout但保留足够的边距
        self.figure.tight_layout(pad=2.0)
        
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
            # 使用数据管理器更新指标
            if self.data_manager.update_metrics(metrics):
                # 更新图表显示
                self.update_display()
                
                # 更新指标标签
                self.update_metric_labels(metrics)
                
        except Exception as e:
            self.logger.error(f"更新训练指标时出错: {str(e)}")
            
    def update_epoch_metrics(self, metrics):
        """更新epoch完成后的指标"""
        self.update_metrics(metrics)
        
    def update_metric_labels(self, metrics):
        """更新指标标签"""
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
            self.precision_label.setText(f"精确率: {metrics.get('precision', 0.0):.4f}")
            self.recall_label.setText(f"召回率: {metrics.get('recall', 0.0):.4f}")
            self.f1_label.setText(f"F1-Score: {metrics.get('f1_score', 0.0):.4f}")
                
            # 更新分类特有指标标签
            if is_classification:
                self.roc_auc_label.setText(f"ROC-AUC: {metrics.get('roc_auc', 0.0):.4f}")
                self.avg_precision_label.setText(f"Avg.Precision: {metrics.get('average_precision', 0.0):.4f}")
                self.top_k_label.setText(f"Top-K准确率: {metrics.get('top_k_accuracy', 0.0):.4f}")
                self.balanced_acc_label.setText(f"平衡准确率: {metrics.get('balanced_accuracy', 0.0):.4f}")
            
            # 更新检测特有指标标签
            else:
                self.map50_label.setText(f"mAP50: {metrics.get('mAP50', 0.0):.4f}")
                self.map75_label.setText(f"mAP75: {metrics.get('mAP75', 0.0):.4f}")
                self.class_loss_label.setText(f"类别损失: {metrics.get('class_loss', 0.0):.4f}")
                self.obj_loss_label.setText(f"目标损失: {metrics.get('obj_loss', 0.0):.4f}")
                self.box_loss_label.setText(f"框损失: {metrics.get('box_loss', 0.0):.4f}")
            
        except Exception as e:
            self.logger.error(f"更新指标标签时出错: {str(e)}")
            
    def update_display(self):
        """根据选择的指标更新显示"""
        try:
            # 清除旧图
            self.chart_renderer.clear_charts()
            
            # 获取显示数据
            data = self.data_manager.get_data_for_display()
            if data is None:
                # 重置坐标轴并绘制
                self.chart_renderer.finalize_layout()
                self.canvas.draw()
                return
            
            # 获取当前选择的指标
            metric_option = self.metric_combo.currentIndex()
            task_type = self.data_manager.task_type
            historical_ranges = self.data_manager.get_historical_ranges()
            
            # 根据选择的指标渲染图表
            if metric_option == 0:  # 损失和准确率总览
                self.chart_renderer.render_overview_chart(data, task_type, historical_ranges)
                
            elif metric_option == 1:  # 仅损失曲线
                self.chart_renderer.render_loss_only_chart(data, historical_ranges)
                
            elif metric_option == 2:  # 仅分类准确率
                self.chart_renderer.render_accuracy_only_chart(data, task_type)
                
            elif metric_option == 3:  # 仅目标检测mAP
                self.chart_renderer.render_map_only_chart(data, task_type)
                
            elif metric_option == 4:  # Precision/Recall
                self.chart_renderer.render_precision_recall_chart(data)
                
            elif metric_option == 5:  # F1-Score
                self.chart_renderer.render_single_metric_chart(
                    data, 'f1_scores', 'F1-Score', 'F1-Score', 'F1-Score'
                )
                
            elif metric_option == 6:  # mAP50/mAP75
                self.chart_renderer.render_dual_metric_chart(
                    data, 'map50s', 'map75s', 'mAP50和mAP75', 'mAP值', 'mAP50', 'mAP75'
                )
                
            elif metric_option == 7:  # 类别损失/目标损失
                self.chart_renderer.render_loss_metrics_chart(
                    data, 'class_losses', 'obj_losses', '类别损失和目标损失', '类别损失', '目标损失'
                )
                
            elif metric_option == 8:  # 框损失/置信度损失
                self.chart_renderer.render_loss_metrics_chart(
                    data, 'box_losses', 'obj_losses', '框损失', '框损失', '目标损失', color1='y-', color2='c-'
                )
                
            elif metric_option == 9:  # ROC-AUC曲线
                self.chart_renderer.render_single_metric_chart(
                    data, 'roc_aucs', 'ROC-AUC曲线', 'ROC-AUC', 'ROC-AUC'
                )
                
            elif metric_option == 10:  # Average Precision
                self.chart_renderer.render_single_metric_chart(
                    data, 'average_precisions', 'Average Precision', 'Average Precision', 'Average Precision', 'm-'
                )
                
            elif metric_option == 11:  # Top-K准确率
                self.chart_renderer.render_single_metric_chart(
                    data, 'top_k_accuracies', 'Top-K准确率', 'Top-K准确率', 'Top-K准确率', 'y-'
                )
                
            elif metric_option == 12:  # 平衡准确率
                self.chart_renderer.render_single_metric_chart(
                    data, 'balanced_accuracies', '平衡准确率', '平衡准确率', '平衡准确率', 'g-'
                )
            
            # 完成布局调整
            self.chart_renderer.finalize_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"更新图表显示时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def update_metric_explanation(self, index):
        """根据选择的指标更新说明文本"""
        explanation = MetricExplanations.get_explanation(index)
        self.explanation_text.setHtml(explanation)
        
    def set_update_frequency(self, frequency):
        """设置更新频率"""
        self.data_manager.set_update_frequency(frequency)
        
    def on_training_finished(self):
        """训练完成处理"""
        try:
            self.logger.info("训练完成")
            # 保存训练曲线数据
            self.data_manager.save_training_data()
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
            self.data_manager.save_training_data()
        except Exception as e:
            self.logger.error(f"处理训练停止时出错: {str(e)}")
            
    def reset_plots(self):
        """重置图表数据"""
        # 重置数据管理器
        self.data_manager.reset_data()
        
        # 重置标签显示
        self.train_loss_label.setText("训练损失: 0.0000")
        self.val_loss_label.setText("验证损失: 0.0000")
        self.map_label.setText("mAP/准确率: 0.0000")
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
        self.chart_renderer.clear_charts()
        
        # 重置图表标题和标签
        self.loss_ax.set_title('训练和验证损失')
        self.loss_ax.set_xlabel('训练轮次')
        self.loss_ax.set_ylabel('损失值')
        
        self.acc_ax.set_title('训练和验证准确率/mAP')
        self.acc_ax.set_xlabel('训练轮次')
        self.acc_ax.set_ylabel('准确率/mAP')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        # 更新状态文本但保持隐藏状态
        self.status_label.setText("图表已重置，等待训练开始...")
        self.status_label.setVisible(False)
        
    def resizeEvent(self, event):
        """重写resizeEvent以在窗口大小改变时调整图表布局"""
        super().resizeEvent(event)
        # 确保在大小变化时保持足够的边距
        self.figure.tight_layout(pad=2.0)
        self.figure.subplots_adjust(left=0.1, right=0.95)
        self.canvas.draw() 