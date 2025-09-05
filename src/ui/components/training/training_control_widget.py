from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal
import os
import platform
import subprocess


class TrainingControlWidget(QWidget):
    """训练控制组件，包含训练按钮和状态显示"""
    
    # 定义信号
    training_started = pyqtSignal()
    training_stopped = pyqtSignal()
    help_requested = pyqtSignal()
    model_folder_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        # 创建底部控制区域
        self.setMaximumHeight(140)  # 增加底部区域高度以容纳模型名称备注
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建模型名称备注区域
        model_note_layout = QHBoxLayout()
        model_note_layout.setSpacing(10)
        
        model_note_label = QLabel("模型命名备注:")
        model_note_label.setToolTip("添加到训练输出模型文件名中的备注")
        model_note_layout.addWidget(model_note_label)
        
        self.model_note_edit = QLineEdit()
        self.model_note_edit.setPlaceholderText("可选: 添加模型命名备注")
        self.model_note_edit.setToolTip("这个备注将添加到输出模型文件名中，方便识别不同训练的模型")
        model_note_layout.addWidget(self.model_note_edit)
        
        layout.addLayout(model_note_layout)
        
        # 创建训练按钮布局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)  # 增加按钮之间的间距
        
        # 设置按钮样式
        button_style = """
            QPushButton {
                font-size: 14px;
                font-weight: bold;
                padding: 4px;
                min-height: 30px;
                border-radius: 4px;
                border: 1px solid #BDBDBD;
            }
            QPushButton:enabled {
                color: #333333;
            }
            QPushButton:disabled {
                color: #757575;
            }
            QPushButton:hover:enabled {
                background-color: #F5F5F5;
            }
        """
        
        self.train_btn = QPushButton("开始训练")
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self.on_train_clicked)
        
        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.on_stop_clicked)
        
        self.help_btn = QPushButton("训练帮助")
        self.help_btn.clicked.connect(self.on_help_clicked)
        
        # 添加打开模型保存文件夹按钮
        self.open_model_folder_btn = QPushButton("打开模型文件夹")
        self.open_model_folder_btn.setEnabled(False)  # 默认禁用
        self.open_model_folder_btn.clicked.connect(self.on_open_model_folder_clicked)
        
        # 检查模型文件夹是否存在，如果存在则启用按钮
        self.check_model_folder()
        
        # 设置按钮样式
        self.train_btn.setStyleSheet(button_style)
        self.stop_btn.setStyleSheet(button_style)
        self.help_btn.setStyleSheet(button_style)
        self.open_model_folder_btn.setStyleSheet(button_style)
        
        # 添加按钮到布局，设置拉伸因子使其平均分配空间
        button_layout.addWidget(self.train_btn, 1)
        button_layout.addWidget(self.stop_btn, 1)
        button_layout.addWidget(self.help_btn, 1)
        button_layout.addWidget(self.open_model_folder_btn, 1)
        
        # 添加训练状态标签
        self.training_status_label = QLabel("等待训练开始...")
        self.training_status_label.setAlignment(Qt.AlignCenter)
        self.training_status_label.setStyleSheet("font-size: 12px; color: #666;")
        
        # 将按钮和状态标签添加到布局
        layout.addLayout(button_layout)
        layout.addWidget(self.training_status_label)
    
    def check_model_folder(self):
        """检查模型文件夹是否存在"""
        try:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_dir = os.path.join(root_dir, "models", "saved_models")
            if os.path.exists(model_dir) and os.path.isdir(model_dir):
                self.open_model_folder_btn.setEnabled(True)
        except Exception as e:
            print(f"检查模型文件夹时出错: {str(e)}")
    
    def on_train_clicked(self):
        """训练按钮点击事件"""
        self.training_started.emit()
    
    def on_stop_clicked(self):
        """停止按钮点击事件"""
        reply = QMessageBox.question(
            self, 
            "确认停止", 
            "确定要停止训练吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.training_stopped.emit()
    
    def on_help_clicked(self):
        """帮助按钮点击事件"""
        self.help_requested.emit()
    
    def on_open_model_folder_clicked(self):
        """打开模型文件夹按钮点击事件"""
        self.model_folder_requested.emit()
    
    def set_training_ready(self, ready):
        """设置训练准备状态"""
        self.train_btn.setEnabled(ready)
    
    def set_training_started(self):
        """设置训练开始状态"""
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.training_status_label.setText("正在训练...")
    
    def set_training_stopped(self, is_intelligent_restart=False):
        """设置训练停止状态"""
        if is_intelligent_restart:
            # 智能训练重启中，不显示"训练已停止"
            self.training_status_label.setText("智能训练重启中...")
            # 保持按钮状态不变，因为训练实际上还在继续
        else:
            # 真正的训练停止
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.training_status_label.setText("训练已停止")
    
    def set_training_finished(self):
        """设置训练完成状态"""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.open_model_folder_btn.setEnabled(True)  # 训练完成后启用打开模型文件夹按钮
        self.training_status_label.setText("训练完成")
    
    def set_training_error(self, error_msg):
        """设置训练错误状态"""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.open_model_folder_btn.setEnabled(False)  # 训练出错时禁用打开模型文件夹按钮
        self.training_status_label.setText(f"训练出错: {error_msg}")
    
    def update_status(self, message):
        """更新状态信息"""
        self.training_status_label.setText(message)
    
    def get_model_note(self):
        """获取模型名称备注"""
        return self.model_note_edit.text().strip()
    
    def open_model_folder(self):
        """打开保存模型的文件夹"""
        try:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_dir = os.path.join(root_dir, "models", "saved_models")
            
            if os.path.exists(model_dir):
                if platform.system() == "Windows":
                    os.startfile(model_dir)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", model_dir])
                else:  # Linux
                    subprocess.run(["xdg-open", model_dir])
                return True
            else:
                QMessageBox.warning(self, "文件夹未找到", f"模型保存文件夹不存在: {model_dir}")
                return False
        except Exception as e:
            QMessageBox.warning(self, "错误", f"无法打开模型文件夹: {str(e)}")
            return False 