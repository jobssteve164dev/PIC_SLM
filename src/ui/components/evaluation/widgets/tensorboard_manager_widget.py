from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QGroupBox, QGridLayout, QMessageBox, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import os
import subprocess
import webbrowser
from .tensorboard_widget import TensorBoardWidget
from .tensorboard_params_guide_widget import TensorBoardParamsGuideWidget


class TensorBoardManagerWidget(QWidget):
    """TensorBoard管理组件，负责TensorBoard的启动、停止和管理功能"""
    
    status_updated = pyqtSignal(str)
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.log_dir = ""
        self.tensorboard_process = None
        
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # 创建日志目录选择组
        log_group = QGroupBox("TensorBoard日志目录")
        log_layout = QGridLayout()
        
        self.log_path_edit = QLabel()
        self.log_path_edit.setStyleSheet("QLabel { border: 1px solid gray; padding: 5px; }")
        self.log_path_edit.setText("请选择TensorBoard日志目录")
        
        log_btn = QPushButton("浏览...")
        log_btn.clicked.connect(self.select_log_dir)
        
        log_layout.addWidget(QLabel("日志目录:"), 0, 0)
        log_layout.addWidget(self.log_path_edit, 0, 1)
        log_layout.addWidget(log_btn, 0, 2)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group, 0)  # 不拉伸
        
        # 创建控制按钮组
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("启动TensorBoard")
        self.start_btn.clicked.connect(self.start_tensorboard)
        self.start_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止TensorBoard")
        self.stop_btn.clicked.connect(self.stop_tensorboard)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        layout.addLayout(control_layout, 0)  # 不拉伸
        
        # 创建TensorBoard嵌入视图
        self.tensorboard_widget = TensorBoardWidget()
        layout.addWidget(self.tensorboard_widget, 0)  # 不拉伸
        
        # 添加状态标签
        self.tb_status_label = QLabel("TensorBoard未启动")
        self.tb_status_label.setAlignment(Qt.AlignCenter)
        self.tb_status_label.setStyleSheet("""
            QLabel {
                color: #7f8c8d;
                padding: 8px;
                background-color: #f8f9fa;
                border-radius: 4px;
                margin: 5px 0px;
            }
        """)
        layout.addWidget(self.tb_status_label, 0)  # 不拉伸
        
        # 添加参数监控说明控件
        self.create_params_guide_section(layout)
    
    def select_log_dir(self):
        """选择TensorBoard日志目录"""
        folder = QFileDialog.getExistingDirectory(self, "选择TensorBoard日志目录")
        if folder:
            self.log_dir = folder
            self.log_path_edit.setText(folder)
            self.start_btn.setEnabled(True)
            self.tensorboard_widget.set_tensorboard_dir(folder)
            
            # 如果有主窗口并且有设置标签页，则更新设置
            if self.main_window and hasattr(self.main_window, 'settings_tab'):
                self.main_window.settings_tab.default_tensorboard_log_dir_edit.setText(folder)
    
    def start_tensorboard(self):
        """启动TensorBoard"""
        if not self.log_dir:
            QMessageBox.warning(self, "警告", "请先选择TensorBoard日志目录!")
            return
            
        try:
            # 检查TensorBoard是否已经在运行
            if self.tensorboard_process and self.tensorboard_process.poll() is None:
                QMessageBox.warning(self, "警告", "TensorBoard已经在运行!")
                return
            
            # 确定要监控的日志目录
            log_dir = self.log_dir
            
            # 如果指向的是model_save_dir，则尝试找到其下的tensorboard_logs目录
            tensorboard_parent = os.path.join(log_dir, 'tensorboard_logs')
            if os.path.exists(tensorboard_parent) and os.path.isdir(tensorboard_parent):
                # 使用tensorboard_logs作为根目录，这样可以显示所有训练运行
                log_dir = tensorboard_parent
                self.status_updated.emit(f"已找到tensorboard_logs目录: {log_dir}")
                
            # 启动TensorBoard进程
            port = 6006  # 默认TensorBoard端口
            cmd = f"tensorboard --logdir={log_dir} --port={port}"
            
            self.status_updated.emit(f"启动TensorBoard，命令: {cmd}")
            
            if os.name == 'nt':  # Windows
                self.tensorboard_process = subprocess.Popen(
                    cmd, shell=True, creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:  # Linux/Mac
                self.tensorboard_process = subprocess.Popen(
                    cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            
            # 更新UI状态
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            # 更新TensorBoard小部件
            self.tensorboard_widget.set_tensorboard_dir(log_dir)
            
            # 打开网页浏览器
            webbrowser.open(f"http://localhost:{port}")
            
            self.tb_status_label.setText(f"TensorBoard已启动，端口: {port}，日志目录: {log_dir}")
            self.status_updated.emit(f"TensorBoard已启动，端口: {port}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动TensorBoard失败: {str(e)}")
    
    def stop_tensorboard(self):
        """停止TensorBoard"""
        try:
            if self.tensorboard_process:
                # 终止TensorBoard进程
                if os.name == 'nt':  # Windows
                    # 先尝试使用进程ID终止
                    try:
                        subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.tensorboard_process.pid)])
                    except Exception as e:
                        self.status_updated.emit(f"通过PID终止TensorBoard失败: {str(e)}")
                    
                    # 再查找并终止所有TensorBoard进程
                    try:
                        subprocess.call(['taskkill', '/F', '/IM', 'tensorboard.exe'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    except Exception as e:
                        self.status_updated.emit(f"终止所有TensorBoard进程失败: {str(e)}")
                else:  # Linux/Mac
                    try:
                        self.tensorboard_process.terminate()
                        self.tensorboard_process.wait(timeout=5)  # 等待最多5秒
                        if self.tensorboard_process.poll() is None:  # 如果进程仍在运行
                            self.tensorboard_process.kill()  # 强制终止
                    except Exception as e:
                        self.status_updated.emit(f"终止TensorBoard进程失败: {str(e)}")
                    
                    # 查找并终止所有TensorBoard进程
                    try:
                        subprocess.call("pkill -f tensorboard", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    except Exception as e:
                        self.status_updated.emit(f"终止所有TensorBoard进程失败: {str(e)}")
                
                self.tensorboard_process = None
            else:
                # 即使没有记录的tensorboard_process，也尝试查找和终止TensorBoard进程
                if os.name == 'nt':  # Windows
                    try:
                        subprocess.call(['taskkill', '/F', '/IM', 'tensorboard.exe'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    except Exception:
                        pass  # 忽略错误，因为这只是一个额外的安全措施
                else:  # Linux/Mac
                    try:
                        subprocess.call("pkill -f tensorboard", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    except Exception:
                        pass  # 忽略错误
            
            # 更新UI状态
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            
            self.tb_status_label.setText("TensorBoard已停止")
            self.status_updated.emit("TensorBoard已停止")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"停止TensorBoard失败: {str(e)}")
    
    def apply_config(self, config):
        """应用配置"""
        if not config:
            return
            
        # 设置TensorBoard日志目录
        if 'default_tensorboard_log_dir' in config:
            log_dir = config['default_tensorboard_log_dir']
            if os.path.exists(log_dir):
                self.log_path_edit.setText(log_dir)
                self.log_dir = log_dir
                self.start_btn.setEnabled(True)
                self.tensorboard_widget.set_tensorboard_dir(log_dir)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 确保在关闭窗口时停止TensorBoard进程
        self.stop_tensorboard()
        super().closeEvent(event) 
        
    def create_params_guide_section(self, parent_layout):
        """创建参数监控说明区域"""
        # 创建参数说明组
        guide_group = QGroupBox("📖 TensorBoard 参数监控详细说明")
        guide_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #34495e;
                border-radius: 8px;
                margin: 10px 0px;
                padding-top: 20px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                color: #34495e;
                font-size: 12pt;
            }
        """)
        
        guide_layout = QVBoxLayout()
        
        # 创建滚动区域来容纳参数说明控件
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #f0f0f0;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #bdc3c7;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #95a5a6;
            }
        """)
        
        # 创建参数监控说明控件
        self.params_guide_widget = TensorBoardParamsGuideWidget()
        scroll_area.setWidget(self.params_guide_widget)
        
        # 设置滚动区域的合理高度，让它能充分利用可用空间
        scroll_area.setMinimumHeight(400)
        # 不设置最大高度限制，让它能够自适应
        
        guide_layout.addWidget(scroll_area)
        guide_group.setLayout(guide_layout)
        parent_layout.addWidget(guide_group, 1)  # 设置拉伸因子为1，让它占用剩余空间
    
    def __del__(self):
        """析构方法，确保在对象被销毁时停止TensorBoard进程"""
        try:
            self.stop_tensorboard()
        except:
            # 在析构时忽略异常
            pass 