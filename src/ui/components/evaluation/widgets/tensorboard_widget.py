from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
import logging
import os
import subprocess


class TensorBoardWidget(QWidget):
    """TensorBoard集成组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tensorboard_dir = None
        self.init_ui()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)
        
        # 说明标签
        info_label = QLabel("TensorBoard是一个强大的可视化工具，可以帮助您更深入地分析模型训练过程。")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #2c3e50; font-weight: bold; padding: 5px;")
        layout.addWidget(info_label)
        
        # 状态标签
        self.status_label = QLabel("TensorBoard日志目录: 未设置")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #7f8c8d; padding: 5px; background-color: #f8f9fa; border-radius: 3px;")
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
        usage_label.setStyleSheet("""
            background-color: #e8f4fd; 
            padding: 10px; 
            border-radius: 5px;
            border-left: 4px solid #3498db;
            color: #2c3e50;
        """)
        layout.addWidget(usage_label)
        
        # 设置固定高度，避免占用过多空间
        self.setMaximumHeight(150)
        
    def update_tensorboard(self, metric_name, value, step):
        """接收并处理来自训练器的TensorBoard更新信号"""
        try:
            # 检查是否是训练结束信号（step为-1表示训练结束）
            if step == -1:
                self.logger.info(f"收到训练结束信号，正在刷新TensorBoard")
                self.status_label.setText(
                    f"TensorBoard日志目录: {self.tensorboard_dir or '未设置'}\n"
                    f"训练已完成，所有指标已记录"
                )
                return
                
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
            
    def ensure_no_tensorboard_process(self):
        """确保没有TensorBoard进程在运行"""
        try:
            self.logger.info("正在确保TensorBoard进程已终止")
            if os.name == 'nt':  # Windows
                subprocess.call(['taskkill', '/F', '/IM', 'tensorboard.exe'], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:  # Linux/Mac
                subprocess.call("pkill -f tensorboard", shell=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            self.logger.error(f"终止TensorBoard进程失败: {str(e)}")
    
    def closeEvent(self, event):
        """组件关闭事件"""
        self.ensure_no_tensorboard_process()
        super().closeEvent(event) 