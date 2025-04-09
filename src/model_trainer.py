import os
import torch
from PyQt5.QtCore import QObject, pyqtSignal
import json
import time

# 导入我们刚刚创建的工具模块
from utils.training_thread import TrainingThread
from utils.model_utils import create_model, configure_model_layers  # 原来已经导入
from utils.activation_utils import apply_activation_function, apply_dropout
from utils.data_utils import save_class_info, save_training_info

class ModelTrainer(QObject):
    # 定义信号
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    training_finished = pyqtSignal()
    training_error = pyqtSignal(str)
    epoch_finished = pyqtSignal(dict)
    model_download_failed = pyqtSignal(str, str)  # 模型名称，下载链接
    training_stopped = pyqtSignal()
    metrics_updated = pyqtSignal(dict)
    tensorboard_updated = pyqtSignal(str, float, int)

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.stop_training = False
        self.training_thread = None
        self.detection_trainer = None

    def configure_model(self, model, layer_config):
        """根据层配置调整模型结构"""
        if not layer_config or not layer_config.get('enabled', False):
            return model
            
        return configure_model_layers(model, layer_config)
        
    def train_model_with_config(self, config):
        """使用配置训练模型"""
        try:
            # 创建基础模型
            self.model = create_model(config)
            
            # 如果有层配置，应用层配置
            if 'layer_config' in config:
                self.model = self.configure_model(self.model, config['layer_config'])
            
            # 提取核心训练参数
            task_type = config.get('task_type', 'classification')
            
            # 获取当前时间戳，用于保存模型和配置文件名
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_name = config.get('model_name', 'Unknown')
            model_note = config.get('model_note', '')
            
            # 构建统一的文件名基础部分
            model_filename = f"{model_name}_{timestamp}"
            if model_note:
                model_filename += f"_{model_note}"
            
            # 获取模型保存目录和参数保存目录
            model_save_dir = config.get('model_save_dir', 'models/saved_models')
            param_save_dir = config.get('default_param_save_dir', model_save_dir)  # 如果未指定参数保存目录，则使用模型保存目录
            
            # 标准化路径格式，确保所有路径使用相同的格式
            model_save_dir = os.path.normpath(model_save_dir)
            param_save_dir = os.path.normpath(param_save_dir)
            
            # 确保目录存在
            os.makedirs(model_save_dir, exist_ok=True)
            os.makedirs(param_save_dir, exist_ok=True)
            
            # 在配置中添加保存的文件名信息，供后续使用
            config['model_filename'] = model_filename
            config['timestamp'] = timestamp
            
            # 保存配置文件，使用统一的文件名格式，保存到参数保存目录
            config_file_path = os.path.join(param_save_dir, f"{model_filename}_config.json")
            try:
                with open(config_file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=4)
                self.status_updated.emit(f"训练配置已保存到: {config_file_path}")
            except Exception as e:
                self.training_error.emit(f"保存训练配置文件时发生错误: {str(e)}")
            
            # 根据任务类型选择不同的训练方法
            if task_type == 'classification':
                # 调用分类模型训练线程
                self.status_updated.emit("启动分类模型训练...")
                self.training_thread = TrainingThread(config)
                
                # 连接训练线程信号
                self.training_thread.progress_updated.connect(self.progress_updated)
                self.training_thread.status_updated.connect(self.status_updated)
                self.training_thread.training_finished.connect(self.training_finished)
                self.training_thread.training_error.connect(self.training_error)
                self.training_thread.epoch_finished.connect(self.epoch_finished)
                self.training_thread.model_download_failed.connect(self.model_download_failed)
                self.training_thread.training_stopped.connect(self.training_stopped)
                
                # 启动训练线程
                self.training_thread.start()
                
            elif task_type == 'detection':
                # 使用YOLOv5等目标检测模型的训练逻辑
                self.status_updated.emit("启动目标检测模型训练...")
                
                # 创建DetectionTrainer实例
                from detection_trainer import DetectionTrainer
                
                try:
                    # 初始化检测训练器
                    self.detection_trainer = DetectionTrainer(config)
                    
                    # 连接信号
                    self.detection_trainer.progress_updated.connect(self.progress_updated)
                    self.detection_trainer.status_updated.connect(self.status_updated)
                    self.detection_trainer.training_finished.connect(self.training_finished)
                    self.detection_trainer.training_error.connect(self.training_error)
                    self.detection_trainer.metrics_updated.connect(self.metrics_updated)
                    self.detection_trainer.tensorboard_updated.connect(self.tensorboard_updated)
                    self.detection_trainer.model_download_failed.connect(self.model_download_failed)
                    self.detection_trainer.training_stopped.connect(self.training_stopped)
                    
                    # 启动训练 - 确保使用包含模型文件名的配置
                    self.detection_trainer.start_training(config)
                    
                except Exception as e:
                    self.training_error.emit(f"创建目标检测训练器时出错: {str(e)}")
            else:
                self.training_error.emit(f"不支持的任务类型: {task_type}")
                
        except Exception as e:
            self.training_error.emit(f"训练初始化时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def stop(self):
        """停止训练过程"""
        try:
            if self.training_thread and self.training_thread.isRunning():
                self.training_thread.stop()
                self.training_thread.wait()
            
            if self.detection_trainer:
                self.detection_trainer.stop()
                
            self.stop_training = True
            self.status_updated.emit("训练已停止")
            
        except Exception as e:
            print(f"停止训练时出错: {str(e)}")
        
        # 无论线程是否正常结束，都发射一次训练停止信号
        self.training_stopped.emit()