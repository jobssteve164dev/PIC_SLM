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
    config = config_loader.get_config()
    ui_config = config_loader.get_ui_config()
    
    # 创建应用程序
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle(ui_config.get('style', 'Fusion'))
    
    # 创建主窗口
    window = MainWindow()
    
    # 应用配置
    window.apply_config(config)
    
    # 创建工作线程
    thread = QThread()
    worker = Worker()
    worker.moveToThread(thread)
    thread.start()
    
    # 连接信号和槽
    # 图片预处理信号（包含数据集创建）
    worker.image_preprocessor.progress_updated.connect(window.update_progress)
    worker.image_preprocessor.status_updated.connect(window.update_status)
    worker.image_preprocessor.preprocessing_error.connect(lambda msg: QMessageBox.critical(window, '错误', msg))
    window.image_preprocessing_started.connect(worker.image_preprocessor.preprocess_images)
    
    # 标注工具信号
    worker.annotation_tool.status_updated.connect(window.update_status)
    worker.annotation_tool.annotation_error.connect(lambda msg: QMessageBox.critical(window, '错误', msg))
    window.annotation_started.connect(
        lambda folder: worker.annotation_tool.start_labelimg(folder, [window.class_list.item(i).text() for i in range(window.class_list.count())])
        if window.labelimg_radio.isChecked() else
        worker.annotation_tool.start_labelme(folder, [window.class_list.item(i).text() for i in range(window.class_list.count())])
    )
    
    # 模型训练信号
    worker.model_trainer.progress_updated.connect(window.update_progress)
    worker.model_trainer.status_updated.connect(window.update_status)
    worker.model_trainer.training_error.connect(lambda msg: QMessageBox.critical(window, '错误', msg))
    window.training_started.connect(
        lambda: worker.model_trainer.train_model(
            window.annotation_folder,
            window.model_combo.currentText(),
            window.epochs_spin.value(),
            window.batch_size_spin.value(),
            window.lr_spin.value(),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'saved_models')
        )
    )
    window.stop_train_btn.clicked.connect(worker.model_trainer.stop)
    
    # 预测信号
    worker.predictor.prediction_error.connect(lambda msg: QMessageBox.critical(window, '错误', msg))
    window.prediction_started.connect(
        lambda: worker.predictor.predict(window.image_path)
    )
    worker.predictor.prediction_finished.connect(window.update_prediction_result)
    
    # 显示主窗口
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == '__main__':
    # 确保当前工作目录是项目根目录
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main() 