from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QSizePolicy, QMessageBox
from PyQt5.QtCore import Qt, pyqtSlot
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
import numpy as np
import os
import webbrowser
import logging
import json
import time

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
        self.learning_rates = []
        self.update_frequency = 10  # 默认更新频率
        self.last_update_time = time.time()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 控制面板
        control_layout = QHBoxLayout()
        
        # 指标选择
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(['损失和mAP', '仅损失', '仅mAP'])
        self.metric_combo.setCurrentIndex(0)
        self.metric_combo.currentIndexChanged.connect(self.update_display)
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
        
        layout.addLayout(metrics_layout)
        
        # 创建图表
        self.figure = Figure()
        
        # 损失子图
        self.loss_ax = self.figure.add_subplot(211)
        self.loss_ax.set_title('训练和验证损失')
        self.loss_ax.set_xlabel('训练轮次')
        self.loss_ax.set_ylabel('损失值')
        
        # mAP子图
        self.acc_ax = self.figure.add_subplot(212)
        self.acc_ax.set_title('验证mAP')
        self.acc_ax.set_xlabel('训练轮次')
        self.acc_ax.set_ylabel('mAP')
        
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
            epoch = int(metrics['epoch'])
            if epoch not in self.epochs:
                self.epochs.append(epoch)
                self.train_losses.append(metrics['train_loss'])
                self.val_losses.append(metrics['val_loss'])
                self.maps.append(metrics['val_map'])
                self.learning_rates.append(metrics['learning_rate'])
            else:
                idx = self.epochs.index(epoch)
                self.train_losses[idx] = metrics['train_loss']
                self.val_losses[idx] = metrics['val_loss']
                self.maps[idx] = metrics['val_map']
                self.learning_rates[idx] = metrics['learning_rate']
            
            # 更新曲线
            self.update_display()
            
            # 更新指标显示
            self.update_metric_labels(metrics)
            
        except Exception as e:
            self.logger.error(f"更新训练指标时出错: {str(e)}")
            
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
            learning_rates = self.learning_rates[:min_len]
            
            # 设置x轴范围
            max_epoch = max(epochs) if epochs else 1
            x_ticks = np.arange(0, max_epoch + 1, max(1, max_epoch // 10))
            
            # 根据选择的指标显示图表
            if metric_option == 0 or metric_option == 1:  # 显示损失
                self.loss_ax.set_title('训练和验证损失')
                self.loss_ax.set_xlabel('训练轮次')
                self.loss_ax.set_ylabel('损失值')
                
                # 设置网格
                self.loss_ax.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制损失数据
                if train_losses:
                    self.loss_ax.plot(epochs, train_losses, 'b-', label='训练损失', marker='o', markersize=4)
                    
                    # 设置y轴范围
                    y_min = min(min(train_losses), min(filter(None, val_losses)))
                    y_max = max(max(train_losses), max(filter(None, val_losses)))
                    y_margin = (y_max - y_min) * 0.1
                    self.loss_ax.set_ylim([max(0, y_min - y_margin), y_max + y_margin])
                
                # 绘制验证损失
                if val_losses:
                    self.loss_ax.plot(epochs, val_losses, 'r-', label='验证损失', marker='o', markersize=4)
                
                self.loss_ax.legend(loc='upper right')
                self.loss_ax.set_xticks(x_ticks)
                self.loss_ax.set_visible(True)
            else:
                self.loss_ax.set_visible(False)
            
            if metric_option == 0 or metric_option == 2:  # 显示mAP
                self.acc_ax.set_title('验证mAP')
                self.acc_ax.set_xlabel('训练轮次')
                self.acc_ax.set_ylabel('mAP')
                
                # 设置网格
                self.acc_ax.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制mAP数据
                if maps:
                    self.acc_ax.plot(epochs, maps, 'g-', label='验证mAP', marker='o', markersize=4)
                    
                    # 设置y轴范围
                    y_min = min(maps)
                    y_max = max(maps)
                    y_margin = (y_max - y_min) * 0.1
                    self.acc_ax.set_ylim([max(0, y_min - y_margin), min(1, y_max + y_margin)])
                
                self.acc_ax.legend(loc='lower right')
                self.acc_ax.set_xticks(x_ticks)
                self.acc_ax.set_visible(True)
            else:
                self.acc_ax.set_visible(False)
            
            # 调整布局并刷新画布
            self.figure.tight_layout()
            self.canvas.draw()
            self.canvas.flush_events()
            
        except Exception as e:
            self.logger.error(f"显示训练图表时出错: {str(e)}")
            self.status_label.setText(f'显示图表出错: {str(e)}')
            
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
            
    def resizeEvent(self, event):
        """重写resizeEvent以在窗口大小改变时调整图表布局"""
        super().resizeEvent(event)
        self.figure.tight_layout()
        self.canvas.draw()
        
    def reset_plots(self):
        """重置所有图表数据"""
        self.train_losses = []
        self.val_losses = []
        self.maps = []
        self.learning_rates = []
        self.epochs = []
        self.status_label.setText('图表已重置，等待训练开始...')
        self.update_display()


class TensorBoardWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tensorboard_dir = None
        self.init_ui()
        
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
        
        # 启动TensorBoard按钮
        self.launch_btn = QPushButton("启动TensorBoard")
        self.launch_btn.clicked.connect(self.launch_tensorboard)
        self.launch_btn.setEnabled(False)
        layout.addWidget(self.launch_btn)
        
        # 使用说明
        usage_label = QLabel(
            "使用说明:\n"
            "1. 在模型训练选项卡中启用TensorBoard选项\n"
            "2. 开始训练模型\n"
            "3. 训练过程中或训练完成后，点击'启动TensorBoard'按钮\n"
            "4. TensorBoard将在浏览器中打开，显示详细的训练指标和可视化"
        )
        usage_label.setWordWrap(True)
        usage_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        layout.addWidget(usage_label)
        
        layout.addStretch()
        
    def set_tensorboard_dir(self, directory):
        """设置TensorBoard日志目录"""
        if directory and os.path.exists(directory):
            self.tensorboard_dir = directory
            self.status_label.setText(f"TensorBoard日志目录: {directory}")
            self.launch_btn.setEnabled(True)
        else:
            self.status_label.setText(f"TensorBoard日志目录不存在: {directory}")
            self.launch_btn.setEnabled(False)
            
    def launch_tensorboard(self):
        """启动TensorBoard"""
        if not self.tensorboard_dir or not os.path.exists(self.tensorboard_dir):
            self.status_label.setText("错误: TensorBoard日志目录不存在")
            return
            
        try:
            # 使用subprocess启动TensorBoard
            import subprocess
            import threading
            
            def run_tensorboard():
                try:
                    # 启动TensorBoard进程
                    process = subprocess.Popen(
                        ["tensorboard", "--logdir", self.tensorboard_dir, "--port", "6006"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    # 等待一段时间让TensorBoard启动
                    import time
                    time.sleep(3)
                    
                    # 在浏览器中打开TensorBoard
                    webbrowser.open("http://localhost:6006")
                    
                    # 更新状态
                    self.status_label.setText(f"TensorBoard已启动，访问地址: http://localhost:6006")
                except Exception as e:
                    self.status_label.setText(f"启动TensorBoard时出错: {str(e)}")
            
            # 在新线程中启动TensorBoard
            threading.Thread(target=run_tensorboard).start()
            
        except Exception as e:
            self.status_label.setText(f"启动TensorBoard时出错: {str(e)}") 