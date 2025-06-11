from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout,
                           QSizePolicy, QLineEdit, QCheckBox, QMessageBox, QRadioButton, QButtonGroup,
                           QStackedWidget, QScrollArea, QListWidget, QDialog, QTextBrowser)
from PyQt5.QtCore import Qt, pyqtSignal, QUrl
from PyQt5.QtGui import QFont, QDesktopServices
import os
import sys
import platform
from .base_tab import BaseTab
from .components.training import (
    TrainingHelpDialog,
    LayerConfigWidget,
    ModelDownloadDialog,
    ClassificationTrainingWidget,
    DetectionTrainingWidget,
    TrainingControlWidget,
    TrainingConfigSelector,
    ConfigApplier
)
import json
import subprocess
from src.config_loader import ConfigLoader


class TrainingTab(BaseTab):
    """训练标签页，负责模型训练功能"""
    
    # 定义信号
    training_started = pyqtSignal()
    training_progress_updated = pyqtSignal(dict)  # 修改为只接收dict参数
    training_stopped = pyqtSignal()  # 添加训练停止信号
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent, main_window)
        self.annotation_folder = ""
        self.task_type = "classification"  # 默认为图片分类任务
        self.init_ui()
        
        # 尝试直接从配置文件加载默认输出文件夹
        self.load_default_config()
        
        # 初始化权重配置显示
        self.refresh_weight_config()
        
    def load_default_config(self):
        """加载默认配置"""
        try:
            config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config.json')
            print(f"训练标签页直接加载配置文件: {config_file}")
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if 'default_output_folder' in config and config['default_output_folder']:
                        self.annotation_folder = config['default_output_folder']
                        print(f"训练标签页直接设置标注文件夹: {self.annotation_folder}")
                        
                        # 设置相应的路径输入框
                        if hasattr(self, 'classification_widget'):
                            dataset_folder = os.path.join(self.annotation_folder, 'dataset')
                            self.classification_widget.set_folder_path(dataset_folder)
                            print(f"直接设置分类路径输入框: {dataset_folder}")
                        
                        if hasattr(self, 'detection_widget'):
                            detection_folder = os.path.join(self.annotation_folder, 'detection_data')
                            self.detection_widget.set_folder_path(detection_folder)
                            print(f"直接设置检测路径输入框: {detection_folder}")
                        
                        # 检查是否可以开始训练
                        self.check_training_ready()
        except Exception as e:
            print(f"训练标签页直接加载配置文件出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def apply_config(self, config):
        """应用配置设置到训练标签页"""
        try:
            print(f"训练标签页apply_config被调用，配置内容：{config}")
            
            # 加载默认输出文件夹路径作为标注文件夹
            if 'default_output_folder' in config and config['default_output_folder']:
                print(f"发现默认输出文件夹配置: {config['default_output_folder']}")
                self.annotation_folder = config['default_output_folder']
                
                # 设置相应的路径输入框
                if hasattr(self, 'classification_widget'):
                    dataset_folder = os.path.join(self.annotation_folder, 'dataset')
                    self.classification_widget.set_folder_path(dataset_folder)
                    print(f"设置分类路径输入框: {dataset_folder}")
                
                if hasattr(self, 'detection_widget'):
                    detection_folder = os.path.join(self.annotation_folder, 'detection_data')
                    self.detection_widget.set_folder_path(detection_folder)
                    print(f"设置检测路径输入框: {detection_folder}")
                
                # 检查是否可以开始训练
                self.check_training_ready()
                
            # 刷新权重配置显示
            self.refresh_weight_config()
                
        except Exception as e:
            print(f"训练标签页apply_config出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def init_ui(self):
        """初始化UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self.scroll_content)
        
        # 添加标题
        title_label = QLabel("模型训练")
        title_label.setFont(QFont('微软雅黑', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 添加训练任务类型选择
        task_group = QGroupBox("训练任务")
        task_layout = QHBoxLayout()
        
        # 创建单选按钮
        self.classification_radio = QRadioButton("图片分类")
        self.detection_radio = QRadioButton("目标检测")
        
        # 创建按钮组
        self.task_button_group = QButtonGroup()
        self.task_button_group.addButton(self.classification_radio, 0)
        self.task_button_group.addButton(self.detection_radio, 1)
        self.classification_radio.setChecked(True)  # 默认选择图片分类
        
        # 添加到布局
        task_layout.addWidget(self.classification_radio)
        task_layout.addWidget(self.detection_radio)
        task_layout.addStretch()
        task_group.setLayout(task_layout)
        main_layout.addWidget(task_group)
        
        # 添加训练配置选择器组件
        self.config_selector = TrainingConfigSelector(self)
        self.config_selector.config_applied.connect(self.on_config_applied)
        main_layout.addWidget(self.config_selector)
        
        # 创建堆叠部件用于切换不同的训练界面
        self.stacked_widget = QStackedWidget()
        
        # 创建图片分类训练界面
        self.classification_widget = ClassificationTrainingWidget(self)
        self.classification_widget.folder_changed.connect(self.on_folder_changed)
        self.classification_widget.params_changed.connect(self.on_params_changed)
        
        # 创建目标检测训练界面
        self.detection_widget = DetectionTrainingWidget(self)
        self.detection_widget.folder_changed.connect(self.on_folder_changed)
        self.detection_widget.params_changed.connect(self.on_params_changed)
        
        # 添加到堆叠部件
        self.stacked_widget.addWidget(self.classification_widget)
        self.stacked_widget.addWidget(self.detection_widget)
        
        # 添加到主布局
        main_layout.addWidget(self.stacked_widget)
        
        # 创建训练控制组件
        self.control_widget = TrainingControlWidget(self)
        self.control_widget.training_started.connect(self.train_model)
        self.control_widget.training_stopped.connect(self.stop_training)
        self.control_widget.help_requested.connect(self.show_training_help)
        self.control_widget.model_folder_requested.connect(self.open_model_folder)
        
        # 将控制组件添加到滚动布局
        main_layout.addWidget(self.control_widget)
        
        # 连接信号
        self.task_button_group.buttonClicked.connect(self.on_task_changed)

    def on_task_changed(self, button):
        """训练任务改变时调用"""
        if button == self.classification_radio:
            self.stacked_widget.setCurrentIndex(0)
            self.task_type = "classification"
            # 自动设置分类数据集路径
            self.classification_widget.refresh_folder()
        else:
            self.stacked_widget.setCurrentIndex(1)
            self.task_type = "detection"
            # 自动设置目标检测数据集路径
            self.detection_widget.refresh_folder()
            
        self.check_training_ready()
        
        # 刷新权重配置显示
        self.refresh_weight_config()
    
    def on_folder_changed(self, folder_path):
        """文件夹路径改变时调用"""
        self.annotation_folder = folder_path
        self.check_training_ready()
    
    def on_params_changed(self):
        """参数改变时调用"""
        # 可以在这里添加参数验证逻辑
        pass
    
    def on_config_applied(self, config):
        """当配置选择器应用配置时调用"""
        try:
            # 使用ConfigApplier应用配置到训练标签页
            success = ConfigApplier.apply_to_training_tab(config, self)
            
            if success:
                # 重新检查训练准备状态
                self.check_training_ready()
                # 刷新权重配置显示
                self.refresh_weight_config()
                print("训练配置已成功应用到界面")
            else:
                print("应用训练配置失败")
                
        except Exception as e:
            print(f"应用配置时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def check_training_ready(self):
        """检查是否可以开始训练"""
        try:
            # 尝试从配置文件直接加载
            config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    default_output_folder = config.get('default_output_folder', '')
            else:
                # 如果配置文件不存在，尝试从main_window获取
                if hasattr(self, 'main_window') and hasattr(self.main_window, 'config'):
                    default_output_folder = self.main_window.config.get('default_output_folder', '')
                else:
                    default_output_folder = ''
            
            if not default_output_folder:
                self.control_widget.set_training_ready(False)
                self.update_status("请先在设置中配置默认输出文件夹")
                return False
                
            # 根据任务类型检查相应的数据集文件夹
            if self.task_type == "classification":
                dataset_folder = os.path.join(default_output_folder, 'dataset')
                train_folder = os.path.join(dataset_folder, 'train')
                val_folder = os.path.join(dataset_folder, 'val')
                
                # 检查训练集和验证集文件夹是否存在
                if os.path.exists(train_folder) and os.path.exists(val_folder):
                    self.control_widget.set_training_ready(True)
                    self.update_status("分类数据集结构检查完成，可以开始训练")
                    return True
                else:
                    missing_folders = []
                    if not os.path.exists(dataset_folder):
                        missing_folders.append("dataset")
                    else:
                        if not os.path.exists(train_folder):
                            missing_folders.append("train")
                        if not os.path.exists(val_folder):
                            missing_folders.append("val")
                    
                    error_msg = f"缺少必要的文件夹: {', '.join(missing_folders)}"
                    self.update_status(error_msg)
                    self.control_widget.set_training_ready(False)
                    return False
            else:
                detection_data_folder = os.path.join(default_output_folder, 'detection_data')
                train_images = os.path.join(detection_data_folder, 'images', 'train')
                val_images = os.path.join(detection_data_folder, 'images', 'val')
                train_labels = os.path.join(detection_data_folder, 'labels', 'train')
                val_labels = os.path.join(detection_data_folder, 'labels', 'val')
                
                # 检查目标检测数据集的完整性
                if (os.path.exists(train_images) and os.path.exists(val_images) and
                    os.path.exists(train_labels) and os.path.exists(val_labels)):
                    self.control_widget.set_training_ready(True)
                    self.update_status("目标检测数据集结构检查完成，可以开始训练")
                    return True
                else:
                    missing_folders = []
                    if not os.path.exists(detection_data_folder):
                        missing_folders.append("detection_data")
                    else:
                        if not os.path.exists(os.path.join(detection_data_folder, 'images')):
                            missing_folders.append("images")
                        else:
                            if not os.path.exists(train_images):
                                missing_folders.append("images/train")
                            if not os.path.exists(val_images):
                                missing_folders.append("images/val")
                            
                        if not os.path.exists(os.path.join(detection_data_folder, 'labels')):
                            missing_folders.append("labels")
                        else:
                            if not os.path.exists(train_labels):
                                missing_folders.append("labels/train")
                            if not os.path.exists(val_labels):
                                missing_folders.append("labels/val")
                    
                    error_msg = f"缺少必要的文件夹: {', '.join(missing_folders)}"
                    self.update_status(error_msg)
                    self.control_widget.set_training_ready(False)
                    return False
            
        except Exception as e:
            print(f"检查训练准备状态时出错: {str(e)}")
            self.control_widget.set_training_ready(False)
            self.update_status(f"检查数据集结构时出错: {str(e)}")
            return False
    
    def train_model(self):
        """开始训练模型"""
        # 先根据当前任务类型自动刷新数据集目录，确保使用最新的数据集状态
        try:
            # 获取配置文件中的默认输出文件夹
            config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config.json')
            default_output_folder = ""
            
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    default_output_folder = config.get('default_output_folder', '')
            
            if not default_output_folder:
                if hasattr(self, 'main_window') and hasattr(self.main_window, 'config'):
                    default_output_folder = self.main_window.config.get('default_output_folder', '')
                
            if not default_output_folder:
                QMessageBox.warning(self, "错误", "未配置默认输出文件夹，请先在设置中配置")
                return
                
            # 确定数据集路径
            if self.task_type == "classification":
                dataset_folder = os.path.join(default_output_folder, 'dataset')
                if not os.path.exists(dataset_folder):
                    QMessageBox.warning(self, "错误", f"未找到分类数据集文件夹: {dataset_folder}")
                    return
                self.annotation_folder = dataset_folder
                self.classification_widget.set_folder_path(dataset_folder)
                self.update_status(f"已自动刷新分类数据集目录: {dataset_folder}")
            else:
                detection_data_folder = os.path.join(default_output_folder, 'detection_data')
                if not os.path.exists(detection_data_folder):
                    QMessageBox.warning(self, "错误", f"未找到目标检测数据集文件夹: {detection_data_folder}")
                    return
                self.annotation_folder = detection_data_folder
                self.detection_widget.set_folder_path(detection_data_folder)
                self.update_status(f"已自动刷新目标检测数据集目录: {detection_data_folder}")
                
            # 强制刷新，确保训练前再次验证数据集
            self.check_training_ready()
        except Exception as e:
            print(f"自动刷新数据集目录失败: {str(e)}")
            QMessageBox.warning(self, "错误", f"自动刷新数据集目录失败: {str(e)}")
            return
            
        # 再次检查是否满足训练条件
        if not self.check_training_ready():
            QMessageBox.warning(self, "警告", "训练条件检查失败，请确认数据集文件夹结构正确")
            return
            
        # 确保annotation_folder已正确设置且存在
        if not self.annotation_folder or not os.path.exists(self.annotation_folder):
            QMessageBox.warning(self, "错误", f"数据集路径无效: {self.annotation_folder}")
            return
            
        # 打印路径信息以便调试
        print(f"开始训练，使用数据集路径: {self.annotation_folder}")
        
        # 更新UI状态
        self.control_widget.set_training_started()
        self.update_status("正在准备训练...")
        self.update_progress(0)
        
        # 更新训练状态标签
        self.control_widget.update_status(f"正在准备训练，使用数据集: {os.path.basename(self.annotation_folder)}")
        
        # 发射训练开始信号
        self.training_started.emit()
    
    def stop_training(self):
        """停止训练"""
        self.update_status("正在停止训练...")
        self.control_widget.update_status("正在停止训练...")
        
        # 直接发射停止训练信号
        if hasattr(self, 'main_window') and hasattr(self.main_window, 'worker') and hasattr(self.main_window.worker, 'model_trainer'):
            self.main_window.worker.model_trainer.stop()
            self.training_stopped.emit()  # 发射训练停止信号
        else:
            print("无法找到model_trainer对象，停止训练失败")
            QMessageBox.warning(self, "警告", "无法停止训练，请尝试关闭程序")
    
    def show_training_help(self):
        """显示训练帮助"""
        dialog = TrainingHelpDialog(self)
        dialog.exec_()
    
    def open_model_folder(self):
        """打开保存模型的文件夹"""
        return self.control_widget.open_model_folder()
    
    def update_training_progress(self, data_dict):
        """更新训练进度"""
        # 从字典中提取epoch和logs信息
        epoch = data_dict.get('epoch', 0) - 1  # 减1是因为我们的epoch从1开始，但索引从0开始
        logs = data_dict  # 直接使用整个字典作为logs
        
        # 根据当前任务类型获取训练轮数
        if self.task_type == "classification":
            epochs = self.classification_widget.epochs_spin.value()
        else:
            epochs = self.detection_widget.epochs_spin.value()
            
        # 更新进度条
        progress = int((epoch + 1) / epochs * 100)
        self.update_progress(progress)
        
        # 更新状态信息
        status = f"训练中... 轮次: {epoch + 1}/{epochs}"
        if logs:
            status += f" - 损失: {logs.get('loss', 0):.4f}"
            if 'accuracy' in logs:
                status += f" - 准确率: {logs.get('accuracy', 0):.4f}"
            elif 'acc' in logs:
                status += f" - 准确率: {logs.get('acc', 0):.4f}"
            elif 'mAP' in logs:
                status += f" - mAP: {logs.get('mAP', 0):.4f}"
        
        self.update_status(status)
        self.control_widget.update_status(status)
        
        # 直接传递整个data_dict到训练进度更新信号
        # 这个信号被连接到evaluation_tab.update_training_visualization
        self.training_progress_updated.emit(data_dict)
    
    def on_training_finished(self):
        """训练完成时调用"""
        self.control_widget.set_training_finished()
        self.update_status("训练完成")
        self.update_progress(100)
    
    def on_training_error(self, error):
        """训练出错时调用"""
        self.control_widget.set_training_error(str(error))
        self.update_status(f"训练出错: {error}")
        QMessageBox.critical(self, "训练错误", str(error))
    
    def get_training_params(self):
        """获取当前训练参数"""
        if self.task_type == "classification":
            return self.classification_widget.get_training_params()
        else:
            return self.detection_widget.get_training_params()

    def on_model_download_failed(self, model_name, model_link):
        """处理模型下载失败事件"""
        dialog = ModelDownloadDialog(model_name, model_link, self)
        dialog.exec_()
        
    def on_training_stopped(self):
        """训练停止完成时调用"""
        # 检查防止重复调用的标志
        if hasattr(self, '_stopping_in_progress') and self._stopping_in_progress:
            return
            
        # 设置标志防止重复调用
        self._stopping_in_progress = True
        
        try:
            self.control_widget.set_training_stopped()
            self.update_status("训练已停止")
            QMessageBox.information(self, "训练状态", "训练已成功停止")
        finally:
            # 重置标志
            self._stopping_in_progress = False

    def connect_model_trainer_signals(self, model_trainer):
        """连接模型训练器的信号"""
        if hasattr(model_trainer, 'model_download_failed'):
            model_trainer.model_download_failed.connect(self.on_model_download_failed)
        if hasattr(model_trainer, 'training_finished'):
            model_trainer.training_finished.connect(self.on_training_finished)
        if hasattr(model_trainer, 'training_error'):
            model_trainer.training_error.connect(self.on_training_error)
        if hasattr(model_trainer, 'epoch_finished'):
            model_trainer.epoch_finished.connect(self.update_training_progress)
        if hasattr(model_trainer, 'training_stopped'):
            model_trainer.training_stopped.connect(self.on_training_stopped) 

    def goto_annotation_tab(self):
        """切换到标注标签页"""
        try:
            if self.main_window and hasattr(self.main_window, 'tabs'):
                annotation_tab_index = -1
                for i in range(self.main_window.tabs.count()):
                    if self.main_window.tabs.tabText(i) == "图像标注":
                        annotation_tab_index = i
                        break
                
                if annotation_tab_index >= 0:
                    self.main_window.tabs.setCurrentIndex(annotation_tab_index)
                    # 如果有标注标签页的引用，尝试切换到目标检测模式
                    if hasattr(self.main_window, 'annotation_tab'):
                        # 使用正确的属性名：mode_button_group 而不是 mode_radio_group
                        if hasattr(self.main_window.annotation_tab, 'mode_button_group'):
                            # 获取按钮
                            detection_radio = self.main_window.annotation_tab.detection_radio
                            if detection_radio:
                                detection_radio.setChecked(True)
                                # 手动触发模式切换事件
                                self.main_window.annotation_tab.on_mode_changed(detection_radio)
        except Exception as e:
            print(f"切换到标注标签页时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_status(self, message):
        """更新状态信息"""
        if hasattr(self, 'main_window') and hasattr(self.main_window, 'update_status'):
            self.main_window.update_status(message)
        # 更新训练状态标签
        if hasattr(self, 'control_widget'):
            self.control_widget.update_status(message)

    def start_training(self):
        """开始训练模型"""
        try:
            if not self.check_training_ready():
                return
                
            # 创建训练配置
            config = self.get_training_params()
            
            # 发射训练开始信号，携带配置信息
            self.update_status("准备开始训练...")
            
            # 重置图表
            if hasattr(self.main_window, 'evaluation_tab') and hasattr(self.main_window.evaluation_tab, 'reset_training_visualization'):
                self.main_window.evaluation_tab.reset_training_visualization()
                
            # 通知主窗口训练已开始，并传递配置
            self.training_started.emit()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动训练失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def refresh_weight_config(self):
        """刷新类别权重配置信息"""
        try:
            self.update_status("正在刷新权重配置...")
            
            # 刷新当前显示的组件的权重配置
            if self.task_type == "classification":
                if hasattr(self.classification_widget, 'weight_config_widget'):
                    self.classification_widget.weight_config_widget.refresh_weight_config()
            else:
                if hasattr(self.detection_widget, 'weight_config_widget'):
                    self.detection_widget.weight_config_widget.refresh_weight_config()
                    
            self.update_status("权重配置刷新完成")
            
        except Exception as e:
            self.update_status(f"刷新权重配置失败: {str(e)}")
            print(f"刷新权重配置时出错: {str(e)}")
            import traceback
            traceback.print_exc()
