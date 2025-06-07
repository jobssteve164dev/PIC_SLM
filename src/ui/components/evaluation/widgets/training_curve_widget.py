from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from .training_visualization_widget import TrainingVisualizationWidget


class TrainingCurveWidget(QWidget):
    """实时训练曲线组件，负责显示训练过程中的实时曲线和状态"""
    
    status_updated = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 初始化训练指标存储
        self._init_training_metrics()
        
        self.init_ui()
        
    def _init_training_metrics(self):
        """初始化训练指标存储变量"""
        # 基础指标
        self.last_train_loss = 0.0
        self.last_val_loss = 0.0
        
        # 分类任务指标
        self.last_val_accuracy = 0.0
        self.last_train_accuracy = 0.0
        self.last_roc_auc = 0.0
        self.last_average_precision = 0.0
        self.last_top_k_accuracy = 0.0
        self.last_balanced_accuracy = 0.0
        
        # 检测任务指标
        self.last_val_map = 0.0
        self.last_train_map = 0.0
        self.last_map50 = 0.0
        self.last_map75 = 0.0
        self.last_class_loss = 0.0
        self.last_obj_loss = 0.0
        self.last_box_loss = 0.0
        
        # 共用指标
        self.last_precision = 0.0
        self.last_recall = 0.0
        self.last_f1_score = 0.0
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)  # 添加一些边距
        
        # 添加说明标签
        info_label = QLabel("实时训练曲线可视化")
        info_label.setFont(QFont('微软雅黑', 12, QFont.Bold))
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        # 添加训练开始/停止按键的提示
        control_tip = QLabel("训练停止条件：当验证损失在多个轮次后不再下降，或达到设定的最大轮次。")
        control_tip.setWordWrap(True)
        layout.addWidget(control_tip)
        
        # 添加训练可视化组件
        self.training_visualization = TrainingVisualizationWidget()
        
        # 确保训练可视化组件有足够的最小宽度
        self.training_visualization.setMinimumWidth(800)
        
        # 创建内部滚动区域来容纳训练可视化组件，以确保UI的正确显示
        visualization_scroll = QScrollArea()
        visualization_scroll.setWidgetResizable(True)
        visualization_scroll.setWidget(self.training_visualization)
        visualization_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        visualization_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        visualization_scroll.setMinimumHeight(600)  # 设置滚动区域的最小高度
        visualization_scroll.setMinimumWidth(820)  # 设置滚动区域的最小宽度，稍大于内部组件
        
        # 将滚动区域添加到布局中
        layout.addWidget(visualization_scroll)
        
        # 添加当前训练状态标签
        self.training_status_label = QLabel("等待训练开始...")
        self.training_status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.training_status_label)
        
        # 设置整个训练曲线区域的最小宽度
        self.setMinimumWidth(850)
    
    def update_training_visualization(self, data):
        """更新训练可视化"""
        try:
            # 数据格式转换
            is_train = data.get('phase') == 'train'
            epoch = data.get('epoch', 0)
            loss = float(data.get('loss', 0))
            learning_rate = float(data.get('learning_rate', 0.001))
            
            # 根据任务类型获取性能指标
            is_classification = 'accuracy' in data
            
            # 更新对应的损失值和指标值
            if is_train:
                self.last_train_loss = loss
                
                # 更新训练阶段指标
                if is_classification:
                    self.last_train_accuracy = float(data.get('accuracy', 0.0))
                else:
                    self.last_train_map = float(data.get('mAP', 0.0))
            else:
                self.last_val_loss = loss
                
                # 根据任务类型更新相应指标
                if is_classification:
                    # 分类任务指标
                    self.last_val_accuracy = float(data.get('accuracy', 0.0))
                    self.last_roc_auc = float(data.get('roc_auc', 0.0))
                    self.last_average_precision = float(data.get('average_precision', 0.0))
                    self.last_top_k_accuracy = float(data.get('top_k_accuracy', 0.0))
                    self.last_balanced_accuracy = float(data.get('balanced_accuracy', 0.0))
                else:
                    # 检测任务指标
                    self.last_val_map = float(data.get('mAP', 0.0))
                    self.last_map50 = float(data.get('mAP50', 0.0))
                    self.last_map75 = float(data.get('mAP75', 0.0))
                    self.last_class_loss = float(data.get('class_loss', 0.0))
                    self.last_obj_loss = float(data.get('obj_loss', 0.0))
                    self.last_box_loss = float(data.get('box_loss', 0.0))
                
                # 共用指标
                self.last_precision = float(data.get('precision', 0.0))
                self.last_recall = float(data.get('recall', 0.0))
                self.last_f1_score = float(data.get('f1_score', 0.0))
            
            # 构建TrainingVisualizationWidget期望的指标格式
            metrics = {
                'epoch': epoch,
                'train_loss': self.last_train_loss,
                'val_loss': self.last_val_loss,
                'learning_rate': learning_rate,
                'precision': self.last_precision,
                'recall': self.last_recall,
                'f1_score': self.last_f1_score
            }
            
            # 根据任务类型添加相应的性能指标
            if is_classification:
                # 添加分类特有指标
                metrics['val_accuracy'] = self.last_val_accuracy
                metrics['train_accuracy'] = self.last_train_accuracy
                metrics['roc_auc'] = self.last_roc_auc
                metrics['average_precision'] = self.last_average_precision
                metrics['top_k_accuracy'] = self.last_top_k_accuracy
                metrics['balanced_accuracy'] = self.last_balanced_accuracy
            else:
                # 添加检测特有指标
                metrics['val_map'] = self.last_val_map
                metrics['train_map'] = self.last_train_map
                metrics['mAP50'] = self.last_map50
                metrics['mAP75'] = self.last_map75
                metrics['class_loss'] = self.last_class_loss
                metrics['obj_loss'] = self.last_obj_loss
                metrics['box_loss'] = self.last_box_loss
            
            # 更新可视化
            self.training_visualization.update_metrics(metrics)
            
            # 更新状态标签
            phase_text = "训练" if is_train else "验证"
            
            # 根据任务类型显示不同的状态文本
            if is_classification:
                accuracy_value = float(data.get('accuracy', 0.0))
                status_text = f"轮次 {epoch}: {phase_text}损失 = {loss:.4f}, {phase_text}准确率 = {accuracy_value:.4f}"
                
                # 添加分类额外指标到状态
                if not is_train and self.last_precision > 0:
                    status_text += f", 精确率 = {self.last_precision:.4f}, 召回率 = {self.last_recall:.4f}"
            else:
                map_value = float(data.get('mAP', 0.0))
                status_text = f"轮次 {epoch}: {phase_text}损失 = {loss:.4f}, {phase_text}mAP = {map_value:.4f}"
                
                # 添加检测额外指标到状态
                if not is_train and self.last_map50 > 0:
                    status_text += f", mAP50 = {self.last_map50:.4f}, mAP75 = {self.last_map75:.4f}"
            
            self.training_status_label.setText(status_text)
            
        except Exception as e:
            import traceback
            print(f"更新训练可视化时出错: {str(e)}")
            print(traceback.format_exc())
    
    def reset_training_visualization(self):
        """重置训练可视化"""
        # 重置数据存储
        self._init_training_metrics()
        
        # 调用TrainingVisualizationWidget的reset_plots方法
        if hasattr(self.training_visualization, 'reset_plots'):
            self.training_visualization.reset_plots()
        
        self.training_status_label.setText("等待训练开始...")
    
    def setup_trainer(self, trainer):
        """设置训练器并连接信号"""
        try:
            if trainer is not None:
                # 直接连接TrainingVisualizationWidget和训练器
                if hasattr(self.training_visualization, 'connect_signals'):
                    self.training_visualization.connect_signals(trainer)
                
                # 记录设置成功的日志
                print(f"已成功设置训练器并连接信号到训练可视化组件")
                return True
        except Exception as e:
            import traceback
            print(f"设置训练器时出错: {str(e)}")
            print(traceback.format_exc())
            return False 