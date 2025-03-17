import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QThread, QObject
from ui.main_window import MainWindow
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from predictor import Predictor
from image_preprocessor import ImagePreprocessor
from annotation_tool import AnnotationTool
from config_loader import ConfigLoader

# 配置matplotlib全局字体设置
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['font.size'] = 12  # 设置字体大小

class Worker(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()
        self.predictor = Predictor()
        self.image_preprocessor = ImagePreprocessor()
        self.annotation_tool = AnnotationTool()

def main():
    # 加载配置
    config_loader = ConfigLoader()
    config = config_loader.get_config() if hasattr(config_loader, 'get_config') else config_loader.load_config()
    ui_config = config_loader.get_ui_config() if hasattr(config_loader, 'get_ui_config') else {}
    
    # 创建应用程序
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    if ui_config:
        app.setStyle(ui_config.get('style', 'Fusion'))
    
    # 创建主窗口
    window = MainWindow()
    
    # 应用配置
    if config:
        window.apply_config(config)
    
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
    # 图片预处理信号
    worker.image_preprocessor.progress_updated.connect(window.update_progress)
    worker.image_preprocessor.status_updated.connect(window.update_status)
    worker.image_preprocessor.preprocessing_error.connect(lambda msg: QMessageBox.critical(window, '错误', msg))
    window.image_preprocessing_started.connect(worker.image_preprocessor.preprocess_images)
    
    # 标注工具信号
    worker.annotation_tool.status_updated.connect(window.update_status)
    worker.annotation_tool.annotation_error.connect(lambda msg: QMessageBox.critical(window, '错误', msg))
    window.annotation_started.connect(lambda folder: worker.annotation_tool.start_annotation(folder))
    
    # 模型训练信号
    worker.model_trainer.progress_updated.connect(window.update_progress)
    worker.model_trainer.status_updated.connect(window.update_status)
    worker.model_trainer.training_error.connect(lambda msg: QMessageBox.critical(window, '错误', msg))
    
    # 修改训练开始信号连接，添加对任务类型的支持
    window.training_started.connect(lambda: start_training(window, worker.model_trainer))
    
    # 如果训练标签页有停止按钮，则连接停止训练信号
    if hasattr(window, 'training_tab') and hasattr(window.training_tab, 'stop_btn'):
        window.training_tab.stop_btn.clicked.connect(worker.model_trainer.stop)
    
    # 单张预测信号
    worker.predictor.prediction_error.connect(lambda msg: QMessageBox.critical(window, '错误', msg))
    window.prediction_started.connect(lambda: worker.predictor.predict())
    worker.predictor.prediction_finished.connect(window.update_prediction_result if hasattr(window, 'update_prediction_result') else lambda result: None)
    
    # 批量预测信号 - 连接到prediction_tab而不是batch_prediction_tab
    if hasattr(worker.predictor, 'batch_prediction_progress'):
        # 检查prediction_tab是否存在并包含批量预测功能
        if hasattr(window, 'prediction_tab') and hasattr(window.prediction_tab, 'update_batch_progress'):
            worker.predictor.batch_prediction_progress.connect(window.prediction_tab.update_batch_progress)
            worker.predictor.batch_prediction_status.connect(window.prediction_tab.update_status)
            worker.predictor.batch_prediction_finished.connect(window.prediction_tab.batch_prediction_finished)
            window.prediction_tab.batch_prediction_started.connect(worker.predictor.batch_predict)
            window.prediction_tab.batch_prediction_stopped.connect(worker.predictor.stop_batch_processing)
        # 保留对batch_prediction_tab的支持，以便向后兼容
        elif hasattr(window, 'batch_prediction_tab'):
            worker.predictor.batch_prediction_progress.connect(window.batch_prediction_tab.update_prediction_progress)
            worker.predictor.batch_prediction_status.connect(window.batch_prediction_tab.update_status)
            worker.predictor.batch_prediction_finished.connect(window.batch_prediction_tab.prediction_finished)
    
    # 将Worker实例保存到window中，以便在UI中访问
    window.worker = worker
    
    # 显示主窗口
    window.show()
    
    # 定义训练启动函数
    def start_training(window, model_trainer):
        """启动模型训练并传递所有必要的参数"""
        if not hasattr(window, 'training_tab'):
            QMessageBox.critical(window, '错误', '训练标签页不存在')
            return
            
        # 获取训练参数
        params = window.training_tab.get_training_params()
        task_type = params.get('task_type', 'classification')
        
        # 设置模型保存目录
        model_save_dir = os.path.join('models', 'saved_models')
        os.makedirs(model_save_dir, exist_ok=True)
        
        # 获取数据目录（根据任务类型选择）
        if task_type == 'classification':
            data_dir = window.training_tab.classification_path_edit.text()
            model_name = params.get('model', 'ResNet50')
            batch_size = params.get('batch_size', 32)
            epochs = params.get('epochs', 20)
            learning_rate = params.get('learning_rate', 0.001)
            optimizer = params.get('optimizer', 'Adam')
            use_pretrained = params.get('use_pretrained', True)
            metrics = params.get('metrics', ['accuracy'])  # 直接使用传入的metrics列表
        else:  # detection
            data_dir = window.training_tab.detection_path_edit.text()
            model_name = params.get('model', 'YOLOv5')
            batch_size = params.get('batch_size', 16)
            epochs = params.get('epochs', 50)
            learning_rate = params.get('learning_rate', 0.0005)
            optimizer = params.get('optimizer', 'Adam')
            use_pretrained = params.get('use_pretrained', True)
            metrics = params.get('metrics', ['mAP'])  # 直接使用传入的metrics列表
            
            # 目标检测特有参数
            iou_threshold = params.get('iou_threshold', 0.5)
            conf_threshold = params.get('conf_threshold', 0.25)
        
        # 创建训练配置
        training_config = {
            'data_dir': data_dir,
            'model_name': model_name,
            'num_epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'model_save_dir': model_save_dir,
            'task_type': task_type,
            'optimizer': optimizer,
            'use_pretrained': use_pretrained,
            'metrics': metrics,
            'use_tensorboard': True
        }
        
        # 添加任务特有参数
        if task_type == 'detection':
            training_config.update({
                'iou_threshold': iou_threshold,
                'conf_threshold': conf_threshold,
                'resolution': params.get('resolution', '640x640'),
                'nms_threshold': 0.45,  # 默认值
                'use_fpn': True         # 默认值
            })
        
        # 启动训练
        window.status_bar.showMessage(f"开始{task_type}训练：{model_name}")
        model_trainer.train_model_with_config(training_config)
    
    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == '__main__':
    # 确保当前工作目录是项目根目录
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main() 