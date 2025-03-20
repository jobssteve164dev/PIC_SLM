import logging
import time
import os
import json
from PyQt5.QtWidgets import QWidget, QMessageBox

class TrainingCurveWidget(QWidget):
    """训练曲线组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_curves()
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
        self.learning_rates = []
        self.update_frequency = 10  # 默认更新频率
        self.last_update_time = time.time()
        
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
            required_fields = ['epoch', 'train_loss', 'val_loss', 'val_map', 'learning_rate']
            if not all(field in metrics for field in required_fields):
                self.logger.warning(f"缺少必要的训练指标字段: {metrics}")
                return
                
            # 检查更新频率
            current_time = time.time()
            if current_time - self.last_update_time < 1.0 / self.update_frequency:
                return
            self.last_update_time = current_time
            
            # 更新数据
            self.epochs.append(metrics['epoch'])
            self.train_losses.append(metrics['train_loss'])
            self.val_losses.append(metrics['val_loss'])
            self.maps.append(metrics['val_map'])
            self.learning_rates.append(metrics['learning_rate'])
            
            # 更新曲线
            self.update_curves()
            
            # 更新指标显示
            self.update_metric_labels(metrics)
            
        except Exception as e:
            self.logger.error(f"更新训练指标时出错: {str(e)}")
            
    def update_curves(self):
        """更新所有曲线"""
        try:
            # 更新训练损失曲线
            self.train_loss_curve.setData(self.epochs, self.train_losses)
            
            # 更新验证损失曲线
            self.val_loss_curve.setData(self.epochs, self.val_losses)
            
            # 更新mAP曲线
            self.map_curve.setData(self.epochs, self.maps)
            
            # 更新学习率曲线
            self.lr_curve.setData(self.epochs, self.learning_rates)
            
            # 自动调整坐标轴范围
            self.plot_widget.setXRange(min(self.epochs), max(self.epochs))
            
        except Exception as e:
            self.logger.error(f"更新曲线时出错: {str(e)}")
            
    def update_metric_labels(self, metrics):
        """更新指标标签"""
        try:
            # 更新训练损失标签
            self.train_loss_label.setText(f"训练损失: {metrics['train_loss']:.4f}")
            
            # 更新验证损失标签
            self.val_loss_label.setText(f"验证损失: {metrics['val_loss']:.4f}")
            
            # 更新mAP标签
            self.map_label.setText(f"mAP: {metrics['val_map']:.4f}")
            
            # 更新学习率标签
            self.lr_label.setText(f"学习率: {metrics['learning_rate']:.6f}")
            
        except Exception as e:
            self.logger.error(f"更新指标标签时出错: {str(e)}")
            
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
                'learning_rates': self.learning_rates
            }
            
            # 保存为JSON文件
            save_path = os.path.join('models', 'training_data.json')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(data, f)
                
            self.logger.info(f"训练数据已保存到: {save_path}")
        except Exception as e:
            self.logger.error(f"保存训练数据时出错: {str(e)}") 