from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox
from PyQt5.QtCore import Qt, pyqtSlot
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import os
import webbrowser

class TrainingVisualizationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.epochs = []
        self.current_epoch = 0
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 控制面板
        control_layout = QHBoxLayout()
        
        # 指标选择
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(['损失和准确率', '仅损失', '仅准确率'])
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
        
        # 创建图表
        self.figure = Figure(figsize=(5, 8))
        
        # 损失子图
        self.loss_ax = self.figure.add_subplot(211)
        self.loss_ax.set_title('训练和验证损失')
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        
        # 准确率子图
        self.acc_ax = self.figure.add_subplot(212)
        self.acc_ax.set_title('训练和验证准确率')
        self.acc_ax.set_xlabel('Epoch')
        self.acc_ax.set_ylabel('Accuracy')
        
        # 创建画布
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.figure.tight_layout()
        
        # 状态标签
        self.status_label = QLabel('等待训练开始...')
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
    @pyqtSlot(dict)
    def update_plots(self, epoch_results):
        """更新训练图表"""
        phase = epoch_results['phase']
        epoch = epoch_results['epoch']
        loss = epoch_results['loss']
        accuracy = epoch_results['accuracy']
        
        if phase == 'train':
            if epoch not in self.epochs:
                self.epochs.append(epoch)
                self.current_epoch = epoch
            self.train_losses.append(loss)
            self.train_accs.append(accuracy)
            self.status_label.setText(f'Epoch {epoch}: 训练损失 = {loss:.4f}, 训练准确率 = {accuracy:.4f}')
        else:  # val
            self.val_losses.append(loss)
            self.val_accs.append(accuracy)
            self.status_label.setText(f'Epoch {epoch}: 验证损失 = {loss:.4f}, 验证准确率 = {accuracy:.4f}')
            
        self.update_display()
        
    def update_display(self):
        """根据选择的指标更新显示"""
        # 清除旧图
        self.loss_ax.clear()
        self.acc_ax.clear()
        
        # 获取当前选择的指标
        metric_option = self.metric_combo.currentIndex()
        
        # 根据选择的指标显示图表
        if metric_option == 0 or metric_option == 1:  # 显示损失
            # 重新设置损失图标题
            self.loss_ax.set_title('训练和验证损失')
            self.loss_ax.set_xlabel('Epoch')
            self.loss_ax.set_ylabel('Loss')
            
            # 绘制损失数据
            if self.train_losses:
                self.loss_ax.plot(self.epochs, self.train_losses, 'b-', label='训练损失')
            if self.val_losses:
                self.loss_ax.plot(self.epochs, self.val_losses, 'r-', label='验证损失')
            self.loss_ax.legend()
            self.loss_ax.set_visible(True)
        else:
            self.loss_ax.set_visible(False)
        
        if metric_option == 0 or metric_option == 2:  # 显示准确率
            # 重新设置准确率图标题
            self.acc_ax.set_title('训练和验证准确率')
            self.acc_ax.set_xlabel('Epoch')
            self.acc_ax.set_ylabel('Accuracy')
            
            # 绘制准确率数据
            if self.train_accs:
                self.acc_ax.plot(self.epochs, self.train_accs, 'b-', label='训练准确率')
            if self.val_accs:
                self.acc_ax.plot(self.epochs, self.val_accs, 'r-', label='验证准确率')
            self.acc_ax.legend()
            self.acc_ax.set_visible(True)
        else:
            self.acc_ax.set_visible(False)
        
        # 更新画布
        self.figure.tight_layout()
        self.canvas.draw()
        
    def reset_plots(self):
        """重置所有图表数据"""
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.epochs = []
        self.current_epoch = 0
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