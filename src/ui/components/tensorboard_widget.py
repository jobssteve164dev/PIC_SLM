from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
import os
import subprocess
import logging
import webbrowser

logger = logging.getLogger(__name__)

class TensorBoardWidget(QWidget):
    """TensorBoard集成组件，用于管理TensorBoard服务和展示"""
    
    def __init__(self, parent=None):
        """初始化TensorBoard组件"""
        super().__init__(parent)
        self.tensorboard_dir = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.init_ui()
        
    def init_ui(self):
        """初始化UI界面"""
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
            self.logger.info(f"设置TensorBoard目录: {directory}")
        else:
            self.status_label.setText(f"TensorBoard日志目录不存在: {directory}")
            self.logger.warning(f"TensorBoard目录不存在: {directory}")
            
    def launch_tensorboard(self, port=6006):
        """启动TensorBoard服务"""
        if not self.tensorboard_dir or not os.path.exists(self.tensorboard_dir):
            self.logger.error("无法启动TensorBoard: 目录不存在")
            return False
            
        try:
            # 确保没有其他TensorBoard进程
            self.ensure_no_tensorboard_process()
            
            # 启动TensorBoard进程
            cmd = ['tensorboard', '--logdir', self.tensorboard_dir, '--port', str(port)]
            subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 在浏览器中打开TensorBoard
            webbrowser.open(f'http://localhost:{port}')
            
            self.logger.info(f"TensorBoard已在端口{port}启动")
            return True
        except Exception as e:
            self.logger.error(f"启动TensorBoard失败: {str(e)}")
            return False
            
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