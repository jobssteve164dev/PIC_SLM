from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTabWidget
import webbrowser
import logging

# 导入拆分后的组件
from .components.training_charts.training_visualization_widget import TrainingVisualizationWidget
from .components.training_charts.tensorboard_widget import TensorBoardWidget

# 创建logger
logger = logging.getLogger(__name__)

class TrainingVisualization(QWidget):
    """训练可视化界面，包含训练曲线和TensorBoard可视化"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.init_ui()
        
    def init_ui(self):
        """初始化UI界面"""
        layout = QVBoxLayout(self)
        
        # 创建选项卡组件
        tab_widget = QTabWidget()
        
        # 创建训练曲线选项卡
        self.visualization_widget = TrainingVisualizationWidget()
        tab_widget.addTab(self.visualization_widget, "训练曲线")
        
        # 创建TensorBoard选项卡
        self.tensorboard_widget = TensorBoardWidget()
        tab_widget.addTab(self.tensorboard_widget, "TensorBoard")
        
        # 添加选项卡组件到主布局
        layout.addWidget(tab_widget)
        
    def connect_signals(self, trainer):
        """连接训练器的信号"""
        # 连接训练曲线组件信号
        self.visualization_widget.connect_signals(trainer)
        
        # 如果有TensorBoard相关信号也可以在这里连接
        if hasattr(trainer, 'tensorboard_updated'):
            trainer.tensorboard_updated.connect(self.tensorboard_widget.update_tensorboard)
            
    def update_metrics(self, metrics):
        """更新训练指标"""
        self.visualization_widget.update_metrics(metrics)
            
    def set_update_frequency(self, frequency):
        """设置更新频率"""
        self.visualization_widget.set_update_frequency(frequency)
            
    def on_training_finished(self):
        """训练完成处理"""
        self.visualization_widget.on_training_finished()
        
        # 通知TensorBoard训练已完成
        if hasattr(self, 'tensorboard_widget'):
            self.tensorboard_widget.update_tensorboard("training", 0, -1)  # -1表示训练结束
            
    def on_training_error(self, error_msg):
        """训练错误处理"""
        self.visualization_widget.on_training_error(error_msg)
            
    def on_training_stopped(self):
        """训练停止处理"""
        self.visualization_widget.on_training_stopped()
            
    def save_training_data(self):
        """保存训练数据"""
        return self.visualization_widget.save_training_data()
        
    def launch_tensorboard(self, port=6006):
        """启动TensorBoard服务"""
        return self.tensorboard_widget.launch_tensorboard(port)
        
    def set_tensorboard_dir(self, directory):
        """设置TensorBoard日志目录"""
        self.tensorboard_widget.set_tensorboard_dir(directory)
        
    def reset_plots(self):
        """重置图表"""
        self.visualization_widget.reset_plots() 