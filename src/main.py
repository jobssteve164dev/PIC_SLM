import sys
import os
import json

# 将src目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("程序开始执行...")
print(f"Python版本: {sys.version}")
print(f"当前工作目录: {os.getcwd()}")

from PyQt5.QtWidgets import QApplication, QMessageBox, QTabWidget, QVBoxLayout, QWidget, QMainWindow
print("PyQt5模块已导入")
from PyQt5.QtCore import QThread, QObject
from src.ui.main_window import MainWindow
print("MainWindow已导入")
from src.ui.evaluation_tab import EvaluationTab  # 从ui模块导入EvaluationTab
print("EvaluationTab已导入")
from src.data_processor import DataProcessor
print("DataProcessor已导入")
from src.model_trainer import ModelTrainer
print("ModelTrainer已导入")
from src.predictor import Predictor
from src.image_preprocessor import ImagePreprocessor
from src.annotation_tool import AnnotationTool
from src.config_loader import ConfigLoader

# 导入matplotlib配置（这会自动配置matplotlib并抑制警告）
from src.utils.matplotlib_config import suppress_matplotlib_warnings

# 确保matplotlib配置在其他导入之前加载
suppress_matplotlib_warnings()

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas

class Worker(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()
        self.predictor = Predictor()
        self.image_preprocessor = ImagePreprocessor()
        self.annotation_tool = AnnotationTool()

def main():
    # 注册一个退出时的清理函数，确保TensorBoard进程被终止
    import atexit
    
    def cleanup_tensorboard():
        """退出时清理TensorBoard进程"""
        print("程序退出，正在清理TensorBoard进程...")
        try:
            import subprocess
            if os.name == 'nt':  # Windows
                subprocess.call(['taskkill', '/F', '/IM', 'tensorboard.exe'], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:  # Linux/Mac
                subprocess.call("pkill -f tensorboard", shell=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("TensorBoard进程清理完成")
        except Exception as e:
            print(f"清理TensorBoard进程时出错: {str(e)}")
    
    # 注册清理函数
    atexit.register(cleanup_tensorboard)
    
    # 检查配置文件
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
    print(f"配置文件路径: {os.path.abspath(config_path)}")
    print(f"配置文件是否存在: {os.path.exists(config_path)}")
    
    # 创建配置文件
    if not os.path.exists(config_path):
        print("配置文件不存在，创建默认配置")
        default_config = {
            'default_source_folder': '',
            'default_output_folder': '',
            'default_classes': ['划痕', '污点', '缺失', '变形', '异物'],
            'default_model_file': '',
            'default_class_info_file': '',
            'default_model_eval_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'saved_models'),
            'default_model_save_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'saved_models'),
            'default_tensorboard_log_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'tensorboard')
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        os.makedirs(default_config['default_model_eval_dir'], exist_ok=True)
        os.makedirs(default_config['default_model_save_dir'], exist_ok=True)
        os.makedirs(default_config['default_tensorboard_log_dir'], exist_ok=True)
        
        # 保存默认配置
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=4)
        
        print(f"已创建默认配置文件: {config_path}")
    else:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = json.load(f)
                print(f"配置文件内容: {config_content}")
                
                # 检查是否有默认模型评估文件夹，如果没有则添加
                if 'default_model_eval_dir' not in config_content or not config_content['default_model_eval_dir']:
                    print("配置文件中缺少default_model_eval_dir，添加默认值")
                    config_content['default_model_eval_dir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'saved_models')
                    os.makedirs(config_content['default_model_eval_dir'], exist_ok=True)
                    
                    # 保存更新后的配置
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config_content, f, ensure_ascii=False, indent=4)
                    
                    print(f"已更新配置文件，添加默认模型评估文件夹: {config_content['default_model_eval_dir']}")
                
                # 检查是否有默认模型保存文件夹，如果没有则添加
                if 'default_model_save_dir' not in config_content or not config_content['default_model_save_dir']:
                    print("配置文件中缺少default_model_save_dir，添加默认值")
                    config_content['default_model_save_dir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'saved_models')
                    os.makedirs(config_content['default_model_save_dir'], exist_ok=True)
                    
                    # 保存更新后的配置
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config_content, f, ensure_ascii=False, indent=4)
                    
                    print(f"已更新配置文件，添加默认模型保存文件夹: {config_content['default_model_save_dir']}")
                
                # 检查是否有默认TensorBoard日志文件夹，如果没有则添加
                if 'default_tensorboard_log_dir' not in config_content or not config_content['default_tensorboard_log_dir']:
                    print("配置文件中缺少default_tensorboard_log_dir，添加默认值")
                    config_content['default_tensorboard_log_dir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'tensorboard')
                    os.makedirs(config_content['default_tensorboard_log_dir'], exist_ok=True)
                    
                    # 保存更新后的配置
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config_content, f, ensure_ascii=False, indent=4)
                    
                    print(f"已更新配置文件，添加默认TensorBoard日志文件夹: {config_content['default_tensorboard_log_dir']}")
                
                # 检查是否有默认类别，如果没有则添加
                if 'default_classes' not in config_content or not config_content['default_classes']:
                    print("配置文件中缺少default_classes，添加默认值")
                    config_content['default_classes'] = ['划痕', '污点', '缺失', '变形', '异物']
                    
                    # 保存更新后的配置
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config_content, f, ensure_ascii=False, indent=4)
                    
                    print(f"已更新配置文件，添加默认类别: {config_content['default_classes']}")
                
                print(f"默认模型评估文件夹: {config_content.get('default_model_eval_dir', '未设置')}")
                print(f"默认模型保存文件夹: {config_content.get('default_model_save_dir', '未设置')}")
                print(f"默认TensorBoard日志文件夹: {config_content.get('default_tensorboard_log_dir', '未设置')}")
                print(f"默认类别: {config_content.get('default_classes', [])}")
        except Exception as e:
            print(f"读取配置文件出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 加载配置
    config_loader = ConfigLoader(config_path)  # 直接传入已知的配置文件路径
    config = config_loader.get_config() if hasattr(config_loader, 'get_config') else config_loader.load_config()
    ui_config = config_loader.get_ui_config() if hasattr(config_loader, 'get_ui_config') else {}
    
    # 创建应用程序
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    if ui_config:
        app.setStyle(ui_config.get('style', 'Fusion'))
    
    # 创建主窗口
    window = MainWindow()
    
    # 应用配置 - 确保在初始化完成后应用配置
    if config:
        print("正在应用配置...")
        print(f"配置内容: {config}")
        window.apply_config(config)
    else:
        # 如果config_loader没有加载到配置，直接从config.json加载
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                direct_config = json.load(f)
                print("通过直接读取config.json加载配置...")
                print(f"直接配置内容: {direct_config}")
                window.apply_config(direct_config)
        except Exception as e:
            print(f"直接加载配置文件失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
    # 强制应用标注界面的配置，确保目标检测界面路径正确设置
    if hasattr(window, 'annotation_tab') and hasattr(window.annotation_tab, 'apply_config'):
        print("正在强制应用配置到标注界面...")
        if config:
            window.annotation_tab.apply_config(config)
        elif 'direct_config' in locals():
            window.annotation_tab.apply_config(direct_config)
    
    # 强制应用训练界面的配置，确保训练界面标注文件夹路径正确设置
    if hasattr(window, 'training_tab'):
        print(f"训练标签页存在: {window.training_tab}")
        print(f"训练标签页类型: {type(window.training_tab)}")
        
        # 确保训练标签页有apply_config方法
        if hasattr(window.training_tab, 'apply_config'):
            print("训练标签页有apply_config方法，正在应用配置...")
            window.training_tab.apply_config(config)
            print(f"训练标签页配置应用完成，标注文件夹: {window.training_tab.annotation_folder}")
            
            # 检查路径控件是否有文本 - 修复：使用新的组件结构
            if (hasattr(window.training_tab, 'classification_widget') and 
                hasattr(window.training_tab.classification_widget, 'path_edit')):
                print(f"分类路径控件文本: {window.training_tab.classification_widget.path_edit.text()}")
            if (hasattr(window.training_tab, 'detection_widget') and 
                hasattr(window.training_tab.detection_widget, 'path_edit')):
                print(f"检测路径控件文本: {window.training_tab.detection_widget.path_edit.text()}")
        else:
            print("警告: 训练标签页没有apply_config方法")
            
    # 应用评估标签页的配置
    if hasattr(window, 'evaluation_tab') and hasattr(window.evaluation_tab, 'apply_config'):
        print("正在应用配置到评估标签页...")
        if config:
            window.evaluation_tab.apply_config(config)
        elif 'direct_config' in locals():
            window.evaluation_tab.apply_config(direct_config)
    
    # 创建工作线程
    thread = QThread()
    worker = Worker()
    worker.moveToThread(thread)
    thread.start()
    
    # 保存线程引用到窗口对象，以便在窗口关闭时正确处理
    window.worker_thread = thread
    
    # 添加closeEvent方法到MainWindow类
    def closeEvent(self, event):
        # 停止所有正在进行的操作
        if hasattr(self, 'worker'):
            # 停止模型训练
            if hasattr(self.worker, 'model_trainer'):
                self.worker.model_trainer.stop()
            
            # 停止标注工具进程
            if hasattr(self.worker, 'annotation_tool'):
                self.worker.annotation_tool.stop()
                
            # 停止预测器的批处理
            if hasattr(self.worker, 'predictor'):
                self.worker.predictor.stop_batch_processing()
                
            # 停止图像预处理
            if hasattr(self.worker, 'image_preprocessor'):
                self.worker.image_preprocessor.stop()
        
        # 确保TensorBoard进程被终止
        if hasattr(self, 'evaluation_tab') and hasattr(self.evaluation_tab, 'stop_tensorboard'):
            try:
                print("正在停止TensorBoard进程...")
                self.evaluation_tab.stop_tensorboard()
            except Exception as e:
                print(f"停止TensorBoard失败: {str(e)}")
                
        # 额外确保通过操作系统命令终止所有TensorBoard进程
        try:
            print("正在确保所有TensorBoard进程终止...")
            import subprocess
            if os.name == 'nt':  # Windows
                subprocess.call(['taskkill', '/F', '/IM', 'tensorboard.exe'], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:  # Linux/Mac
                subprocess.call("pkill -f tensorboard", shell=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"终止所有TensorBoard进程时发生错误: {str(e)}")
        
        # 等待线程结束并退出
        if hasattr(self, 'worker_thread') and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait(3000)  # 等待最多3秒
            
            # 如果线程仍在运行，则强制终止
            if self.worker_thread.isRunning():
                self.worker_thread.terminate()
                self.worker_thread.wait()
        
        # 接受关闭事件
        event.accept()
    
    # 动态添加closeEvent方法到MainWindow类
    # 如果MainWindow已经有closeEvent方法，则不覆盖
    if not hasattr(MainWindow, 'closeEvent') or MainWindow.closeEvent.__name__ != 'closeEvent':
        MainWindow.closeEvent = closeEvent
    
    # 连接信号和槽
    # 图片预处理信号现在由独立线程处理，不需要Worker中的image_preprocessor连接
    # 创建类别文件夹仍然使用同步方式
    if hasattr(worker, 'image_preprocessor'):
        window.create_class_folders_signal.connect(worker.image_preprocessor.create_class_folders)
    
    # 标注工具信号
    worker.annotation_tool.status_updated.connect(window.update_status)
    worker.annotation_tool.annotation_error.connect(lambda msg: QMessageBox.critical(window, '错误', msg))
    window.annotation_started.connect(lambda folder: worker.annotation_tool.start_annotation(folder))
    # 添加验证集文件夹打开信号连接
    if hasattr(window, 'annotation_tab') and hasattr(window.annotation_tab, 'open_validation_folder_signal'):
        window.annotation_tab.open_validation_folder_signal.connect(lambda folder: worker.annotation_tool.open_validation_folder(folder))
    
    # 模型训练信号
    worker.model_trainer.progress_updated.connect(window.update_progress)
    worker.model_trainer.status_updated.connect(window.update_status)
    worker.model_trainer.training_error.connect(lambda msg: QMessageBox.critical(window, '错误', msg))
    
    # 确保评估标签页存在并正确连接
    if hasattr(window, 'evaluation_tab'):
        if hasattr(window.evaluation_tab, 'update_training_visualization'):
            print("连接训练可视化信号（传统方式）...")
            worker.model_trainer.epoch_finished.connect(window.evaluation_tab.update_training_visualization)
        
        # 添加新增信号的连接
        if hasattr(worker.model_trainer, 'metrics_updated'):
            print("连接实时指标更新信号...")
            worker.model_trainer.metrics_updated.connect(window.evaluation_tab.update_training_visualization)
        
        if hasattr(worker.model_trainer, 'tensorboard_updated') and hasattr(window.evaluation_tab, 'tensorboard_widget'):
            print("连接TensorBoard更新信号...")
            # 如果TensorBoardWidget支持接收tensorboard_updated信号，将其连接
            if hasattr(window.evaluation_tab.tensorboard_widget, 'update_tensorboard'):
                worker.model_trainer.tensorboard_updated.connect(
                    window.evaluation_tab.tensorboard_widget.update_tensorboard)
        
        # 使用新的setup_trainer方法直接连接DetectionTrainer信号
        if hasattr(window.evaluation_tab, 'setup_trainer'):
            print("正在设置训练器的直接连接...")
            # 确保在训练开始时调用setup_trainer
            window.training_started.connect(
                lambda: window.evaluation_tab.setup_trainer(worker.model_trainer.detection_trainer)
                if worker.model_trainer.detection_trainer is not None
                else print("检测训练器尚未初始化，将在训练开始时自动连接")
            )
    else:
        print("警告: 无法连接训练可视化信号，evaluation_tab不存在")
    
    # 添加模型下载失败信号连接
    if hasattr(window, 'training_tab') and hasattr(window.training_tab, 'on_model_download_failed'):
        worker.model_trainer.model_download_failed.connect(window.training_tab.on_model_download_failed)
        
    # 修改训练开始信号连接，改为在工作线程中执行训练
    window.training_started.connect(lambda: window.prepare_training_config(worker.model_trainer))
    
    # 确保模型训练器的信号被正确连接到训练标签页
    if hasattr(window, 'training_tab') and hasattr(window.training_tab, 'connect_model_trainer_signals'):
        window.training_tab.connect_model_trainer_signals(worker.model_trainer)
    
    # 单张预测信号
    worker.predictor.prediction_error.connect(lambda msg: QMessageBox.critical(window, '错误', msg))
    window.prediction_started.connect(lambda: worker.predictor.predict(window.prediction_tab.image_file))
    worker.predictor.prediction_finished.connect(window.update_prediction_result if hasattr(window, 'update_prediction_result') else lambda result: None)
    
    # 批量预测信号 - 连接到prediction_tab而不是batch_prediction_tab
    if hasattr(worker.predictor, 'batch_prediction_progress'):
        # 检查prediction_tab是否存在并包含批量预测功能
        if hasattr(window, 'prediction_tab') and hasattr(window.prediction_tab, 'update_batch_progress'):
            worker.predictor.batch_prediction_progress.connect(window.prediction_tab.update_batch_progress)
            worker.predictor.batch_prediction_status.connect(window.prediction_tab.update_status)
            # 修复：确保结果能正确传递
            worker.predictor.batch_prediction_finished.connect(
                lambda results: window.prediction_tab.batch_prediction_finished(results)
            )
            # 注释掉重复的信号连接，这些已经在main_window.py的connect_signals中连接了
            # window.prediction_tab.batch_prediction_started.connect(window.on_batch_prediction_started)
            # window.prediction_tab.batch_prediction_stopped.connect(window.on_batch_prediction_stopped)
        # 保留对batch_prediction_tab的支持，以便向后兼容
        elif hasattr(window, 'batch_prediction_tab'):
            worker.predictor.batch_prediction_progress.connect(window.batch_prediction_tab.update_prediction_progress)
            worker.predictor.batch_prediction_status.connect(window.batch_prediction_tab.update_status)
            worker.predictor.batch_prediction_finished.connect(window.batch_prediction_tab.prediction_finished)
    
    # 将Worker实例保存到window中，以便在UI中访问
    window.worker = worker
    
    # 显示主窗口
    window.show()
    
    # 定义准备训练配置函数
    def prepare_training_config(self, model_trainer):
        """准备训练配置并发送给工作线程执行"""
        if not hasattr(self, 'training_tab'):
            QMessageBox.critical(self, '错误', '训练标签页不存在')
            return
            
        # 获取训练参数
        params = self.training_tab.get_training_params()
        task_type = params.get('task_type', 'classification')
        
        # 设置模型保存目录
        # 优先使用设置中的模型保存路径
        config_loader = ConfigLoader()
        config = config_loader.get_config()
        model_save_dir = config.get('default_model_save_dir', '')
        
        # 获取参数保存目录
        param_save_dir = config.get('default_param_save_dir', '')
        
        # 获取TensorBoard日志目录
        tensorboard_log_dir = config.get('default_tensorboard_log_dir', '')
        
        # 如果设置中没有指定模型保存目录，则使用默认路径
        if not model_save_dir:
            model_save_dir = os.path.join('models', 'saved_models')
            
        # 如果设置中没有指定参数保存目录，则使用模型保存目录
        if not param_save_dir:
            param_save_dir = model_save_dir
            
        # 如果设置中没有指定TensorBoard日志目录，则使用默认路径
        if not tensorboard_log_dir:
            tensorboard_log_dir = os.path.join('runs', 'tensorboard')
            
        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(param_save_dir, exist_ok=True)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        
        # 获取类别权重配置 - 直接从配置中读取，确保与界面显示一致
        use_class_weights = config.get('use_class_weights', True)
        weight_strategy = config.get('weight_strategy', 'balanced')
        
        # 收集所有可能的权重配置源
        weight_config = {}
        if 'class_weights' in config and config['class_weights']:
            weight_config['class_weights'] = config['class_weights']
        if 'custom_class_weights' in config and config['custom_class_weights']:
            weight_config['custom_class_weights'] = config['custom_class_weights']
        if 'weight_config_file' in config and config['weight_config_file']:
            weight_config['weight_config_file'] = config['weight_config_file']
        if 'all_strategies' in config and config['all_strategies']:
            weight_config['all_strategies'] = config['all_strategies']
        
        # 获取数据目录（根据任务类型选择）
        if task_type == 'classification':
            # 直接使用annotation_folder而不是从控件获取
            data_dir = self.training_tab.annotation_folder
            
            # 用于日志调试
            print(f"准备训练配置，使用分类数据集路径: {data_dir}")
            
            # 确保控件显示的路径与实际使用的路径一致 - 修复：使用新的组件结构
            if hasattr(self.training_tab, 'classification_widget') and hasattr(self.training_tab.classification_widget, 'path_edit'):
                self.training_tab.classification_widget.path_edit.setText(data_dir)
            
            model_name = params.get('model', 'ResNet50')
            batch_size = params.get('batch_size', 32)
            epochs = params.get('epochs', 20)
            learning_rate = params.get('learning_rate', 0.001)
            optimizer = params.get('optimizer', 'Adam')
            use_pretrained = params.get('use_pretrained', True)
            pretrained_path = params.get('pretrained_path', '')  # 获取本地预训练模型路径
            metrics = params.get('metrics', ['accuracy'])  # 直接使用传入的metrics列表
            
            # 添加额外的分类训练参数
            weight_decay = params.get('weight_decay', 0.0001)
            lr_scheduler = params.get('lr_scheduler', 'StepLR')
            use_augmentation = params.get('use_augmentation', True)
            early_stopping = params.get('early_stopping', True)
            early_stopping_patience = params.get('early_stopping_patience', 10)
            gradient_clipping = params.get('gradient_clipping', False)
            gradient_clipping_value = params.get('gradient_clipping_value', 1.0)
            mixed_precision = params.get('mixed_precision', True)
            dropout_rate = params.get('dropout_rate', 0.0)  # 获取dropout率参数
            
        else:  # detection
            # 直接使用annotation_folder而不是从控件获取
            data_dir = self.training_tab.annotation_folder
            
            # 用于日志调试
            print(f"准备训练配置，使用目标检测数据集路径: {data_dir}")
            
            # 确保控件显示的路径与实际使用的路径一致 - 修复：使用新的组件结构
            if hasattr(self.training_tab, 'detection_widget') and hasattr(self.training_tab.detection_widget, 'path_edit'):
                self.training_tab.detection_widget.path_edit.setText(data_dir)
            
            model_name = params.get('model', 'YOLOv5')
            batch_size = params.get('batch_size', 16)
            epochs = params.get('epochs', 50)
            learning_rate = params.get('learning_rate', 0.0005)
            optimizer = params.get('optimizer', 'Adam')
            use_pretrained = params.get('use_pretrained', True)
            pretrained_path = params.get('pretrained_path', '')  # 获取本地预训练模型路径
            metrics = params.get('metrics', ['mAP'])  # 直接使用传入的metrics列表
            
            # 目标检测特有参数
            iou_threshold = params.get('iou_threshold', 0.5)
            conf_threshold = params.get('conf_threshold', 0.25)
            
            # 添加额外的目标检测训练参数
            weight_decay = params.get('weight_decay', 0.0005)
            lr_scheduler = params.get('lr_scheduler', 'StepLR')
            use_augmentation = params.get('use_augmentation', True)
            early_stopping = params.get('early_stopping', True)
            early_stopping_patience = params.get('early_stopping_patience', 10)
            gradient_clipping = params.get('gradient_clipping', False)
            gradient_clipping_value = params.get('gradient_clipping_value', 1.0)
            mixed_precision = params.get('mixed_precision', True)
            dropout_rate = params.get('dropout_rate', 0.0)  # 获取dropout率参数
            use_mosaic = params.get('use_mosaic', True)
            use_multiscale = params.get('use_multiscale', True)
            use_ema = params.get('use_ema', True)
        
        # 创建训练配置
        training_config = {
            'data_dir': data_dir,
            'model_name': model_name,
            'num_epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'model_save_dir': model_save_dir,
            'default_param_save_dir': param_save_dir,  # 添加参数保存目录
            'tensorboard_log_dir': tensorboard_log_dir,  # 添加TensorBoard日志目录
            'task_type': task_type,
            'optimizer': optimizer,
            'use_pretrained': use_pretrained,
            'pretrained_path': pretrained_path,  # 添加本地预训练模型路径
            'metrics': metrics,
            'use_tensorboard': True,
            'model_note': params.get('model_note', ''),  # 添加模型命名备注
            
            # 添加类别权重配置 - 确保与界面显示一致
            'use_class_weights': use_class_weights,
            'weight_strategy': weight_strategy,
            
            # 添加所有共用训练参数
            'weight_decay': weight_decay,
            'lr_scheduler': lr_scheduler,
            'use_augmentation': use_augmentation,
            'early_stopping': early_stopping,
            'early_stopping_patience': early_stopping_patience,
            'gradient_clipping': gradient_clipping,
            'gradient_clipping_value': gradient_clipping_value,
            'mixed_precision': mixed_precision,
            'dropout_rate': dropout_rate,  # 添加dropout率参数
            'activation_function': params.get('activation_function', 'ReLU')  # 添加激活函数参数
        }
        
        # 将权重配置添加到训练配置中
        training_config.update(weight_config)
        
        # 添加任务特有参数
        if task_type == 'detection':
            training_config.update({
                'iou_threshold': iou_threshold,
                'conf_threshold': conf_threshold,
                'resolution': params.get('resolution', '640x640'),
                'nms_threshold': 0.45,  # 默认值
                'use_fpn': True,        # 默认值
                'use_mosaic': use_mosaic,
                'use_multiscale': use_multiscale,
                'use_ema': use_ema
            })
        
        # 输出调试信息，显示传递给训练器的权重配置
        print(f"传递给训练器的权重配置: use_class_weights={use_class_weights}, weight_strategy={weight_strategy}")
        if weight_config:
            print(f"权重配置源: {list(weight_config.keys())}")
        
        # 更新UI状态 - 修复：使用update_status方法代替status_bar.showMessage
        self.update_status(f"开始{task_type}训练：{model_name}")
        
        # 使用工作线程启动训练
        model_trainer.train_model_with_config(training_config)
    
    # 动态添加prepare_training_config方法到MainWindow类
    MainWindow.prepare_training_config = prepare_training_config
    
    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == '__main__':
    # 确保当前工作目录是项目根目录
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main() 