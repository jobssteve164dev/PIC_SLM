from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QGroupBox, QGridLayout, QListWidget,
                           QListWidgetItem, QMessageBox, QTableWidget, QTableWidgetItem,
                           QHeaderView)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import os
import json
import sys
import time
import torch
import torchvision
from PIL import Image


class ModelEvaluationWidget(QWidget):
    """模型评估组件，负责模型评估和比较功能"""
    
    status_updated = pyqtSignal(str)
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.models_dir = ""
        self.models_list = []
        
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 创建模型目录选择组
        models_group = QGroupBox("模型目录")
        models_layout = QGridLayout()
        
        self.models_path_edit = QLabel()
        self.models_path_edit.setStyleSheet("QLabel { border: 1px solid gray; padding: 5px; }")
        self.models_path_edit.setText("请选择包含模型的目录")
        
        models_btn = QPushButton("浏览...")
        models_btn.clicked.connect(self.select_models_dir)
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_model_list)
        
        models_layout.addWidget(QLabel("模型目录:"), 0, 0)
        models_layout.addWidget(self.models_path_edit, 0, 1)
        models_layout.addWidget(models_btn, 0, 2)
        models_layout.addWidget(refresh_btn, 0, 3)
        
        models_group.setLayout(models_layout)
        layout.addWidget(models_group)
        
        # 创建模型列表组
        list_group = QGroupBox("可用模型")
        list_layout = QVBoxLayout()
        
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.NoSelection)  # 禁用选择模式，使用复选框代替
        self.model_list.setMinimumHeight(150)
        list_layout.addWidget(self.model_list)
        
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        # 创建比较按钮
        compare_btn = QPushButton("比较选中模型")
        compare_btn.clicked.connect(self.compare_models)
        layout.addWidget(compare_btn)
        
        # 创建结果表格
        result_group = QGroupBox("比较结果")
        result_layout = QVBoxLayout()
        
        self.result_table = QTableWidget(0, 5)  # 行数将根据选择的模型数量动态调整
        self.result_table.setHorizontalHeaderLabels(["模型名称", "准确率", "损失", "参数量", "推理时间"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        result_layout.addWidget(self.result_table)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
    
    def select_models_dir(self):
        """选择模型目录"""
        folder = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if folder:
            self.models_dir = folder
            self.models_path_edit.setText(folder)
            self.refresh_model_list()
            
            # 如果有主窗口并且有设置标签页，则更新设置
            if self.main_window and hasattr(self.main_window, 'settings_tab'):
                self.main_window.settings_tab.default_model_eval_dir_edit.setText(folder)
    
    def refresh_model_list(self):
        """刷新模型列表"""
        if not self.models_dir:
            return
            
        try:
            self.model_list.clear()
            self.models_list = []
            
            # 查找目录中的所有模型文件
            for file in os.listdir(self.models_dir):
                if file.endswith('.h5') or file.endswith('.pb') or file.endswith('.tflite') or file.endswith('.pth'):
                    self.models_list.append(file)
                    item = QListWidgetItem(file)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)  # 添加复选框
                    item.setCheckState(Qt.Unchecked)  # 默认未选中
                    self.model_list.addItem(item)
            
            if not self.models_list:
                QMessageBox.information(self, "提示", "未找到模型文件，请确保目录中包含.h5、.pb、.tflite或.pth文件")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"刷新模型列表失败: {str(e)}")
            
            # 打印更详细的错误信息以便调试
            import traceback
            print(f"刷新模型列表失败: {str(e)}")
            print(traceback.format_exc())
    
    def compare_models(self):
        """比较选中的模型"""
        selected_models = []
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_models.append(item.text())
        
        if not selected_models:
            QMessageBox.warning(self, "警告", "请先选择要比较的模型!")
            return

        # 验证是否至少选择了一个模型
        if not selected_models:
            QMessageBox.warning(self, "警告", "请选择至少一个模型进行评估")
            return
            
        # 清空结果表格
        self.result_table.setRowCount(0)
        
        self.status_updated.emit("正在评估模型...")
        
        try:
            # 添加src目录到sys.path，确保能导入相关模块
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            
            # 导入预测器和评估工具
            from predictor import Predictor
            
            # 从配置文件获取默认设置
            config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'config.json')
            
            # 读取配置
            data_dir = ""
            class_info_path = ""
            
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    data_dir = config.get('default_output_folder', '')
                    class_info_path = config.get('default_class_info_file', '')
            else:
                # 如果找不到配置文件，使用默认路径
                data_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            
            # 验证类别信息文件是否存在
            if not class_info_path or not os.path.exists(class_info_path):
                # 尝试在模型目录中查找任何类别信息文件
                json_files = [f for f in os.listdir(self.models_dir) if f.endswith('_classes.json') or f == 'class_info.json']
                if json_files:
                    class_info_path = os.path.join(self.models_dir, json_files[0])
                else:
                    QMessageBox.warning(self, "错误", "找不到类别信息文件，请在设置中配置默认类别信息文件")
                    return
            
            # 验证类别信息文件
            try:
                with open(class_info_path, 'r', encoding='utf-8') as f:
                    class_info = json.load(f)
                print(f"使用类别信息文件: {class_info_path}")
                print(f"类别信息: {class_info}")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"无法读取类别信息文件: {str(e)}")
                return
                
            # 判断数据集类型（分类或检测）
            if os.path.exists(os.path.join(data_dir, 'dataset', 'val')):
                # 分类数据集
                test_dir = os.path.join(data_dir, 'dataset', 'val')
                task_type = "分类模型"
            elif os.path.exists(os.path.join(data_dir, 'detection_data')):
                # 检测数据集
                test_dir = os.path.join(data_dir, 'detection_data')
                task_type = "检测模型"
            else:
                QMessageBox.warning(self, "错误", "找不到有效的测试数据集目录")
                return
                
            # 为每个选中的模型进行评估
            for i, model_filename in enumerate(selected_models):
                model_path = os.path.join(self.models_dir, model_filename)
                
                # 创建预测器
                predictor = Predictor()
                
                # 尝试猜测模型架构
                model_arch = self._guess_model_architecture(model_filename)
                
                # 更新状态
                self.status_updated.emit(f"正在评估模型 {model_filename}...")
                
                # 加载模型
                model_info = {
                    'model_path': model_path,
                    'class_info_path': class_info_path,
                    'model_type': task_type,
                    'model_arch': model_arch
                }
                
                try:
                    # 加载模型
                    predictor.load_model_with_info(model_info)
                    
                    # 模型评估
                    result = self._evaluate_model(predictor, test_dir, task_type)
                    
                    # 添加新行到表格
                    row_position = self.result_table.rowCount()
                    self.result_table.insertRow(row_position)
                    
                    # 填充表格
                    self.result_table.setItem(row_position, 0, QTableWidgetItem(model_filename))
                    self.result_table.setItem(row_position, 1, QTableWidgetItem(f"{result['accuracy']:.2%}"))
                    self.result_table.setItem(row_position, 2, QTableWidgetItem(f"{result['loss']:.4f}"))
                    self.result_table.setItem(row_position, 3, QTableWidgetItem(f"{result['params_count']:,}"))
                    self.result_table.setItem(row_position, 4, QTableWidgetItem(f"{result['avg_inference_time']:.2f} ms"))
                    
                except Exception as model_error:
                    import traceback
                    error_msg = f"评估模型 {model_filename} 时出错: {str(model_error)}\n{traceback.format_exc()}"
                    print(error_msg)
                    QMessageBox.warning(self, "错误", error_msg)
                    
                    # 即使出错也添加一行表示该模型
                    row_position = self.result_table.rowCount()
                    self.result_table.insertRow(row_position)
                    self.result_table.setItem(row_position, 0, QTableWidgetItem(model_filename))
                    self.result_table.setItem(row_position, 1, QTableWidgetItem("评估失败"))
                    self.result_table.setItem(row_position, 2, QTableWidgetItem("评估失败"))
                    self.result_table.setItem(row_position, 3, QTableWidgetItem("评估失败"))
                    self.result_table.setItem(row_position, 4, QTableWidgetItem("评估失败"))
            
            self.status_updated.emit(f"已完成 {len(selected_models)} 个模型的评估")
            
        except Exception as e:
            import traceback
            error_msg = f"模型评估过程出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            QMessageBox.critical(self, "错误", error_msg)
    
    def _guess_model_architecture(self, model_filename):
        """根据文件名猜测模型架构"""
        filename_lower = model_filename.lower()
        
        if "resnet18" in filename_lower:
            return "ResNet18"
        elif "resnet34" in filename_lower:
            return "ResNet34"
        elif "resnet50" in filename_lower:
            return "ResNet50"
        elif "mobilenet" in filename_lower:
            return "MobileNetV2"
        elif "efficientnet" in filename_lower:
            return "EfficientNetB0"
        elif "vgg16" in filename_lower:
            return "VGG16"
        else:
            # 默认使用ResNet18
            return "ResNet18"
    
    def _evaluate_model(self, predictor, test_dir, task_type):
        """评估单个模型"""
        accuracy = 0.0
        loss = 0.0
        params_count = 0
        total_time = 0.0
        total_samples = 0
        
        # 计算模型参数数量
        params_count = sum(p.numel() for p in predictor.model.parameters())
        
        # 对分类模型进行评估
        if task_type == "分类模型":
            result = self._evaluate_classification_model(predictor, test_dir)
            accuracy = result['accuracy']
            loss = result['loss']
            total_time = result['total_time']
            total_samples = result['total_samples']
        
        # 对检测模型进行评估
        elif task_type == "检测模型":
            result = self._evaluate_detection_model(predictor, test_dir)
            total_time = result['total_time']
            total_samples = result['total_samples']
        
        # 计算平均推理时间
        avg_inference_time = 0
        if total_samples > 0:
            avg_inference_time = (total_time / total_samples) * 1000  # 转换为毫秒
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'params_count': params_count,
            'avg_inference_time': avg_inference_time
        }
    
    def _evaluate_classification_model(self, predictor, test_dir):
        """评估分类模型"""
        # 准备测试样本
        test_samples = []
        class_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        
        for class_dir in class_dirs:
            class_label = class_dir
            class_path = os.path.join(test_dir, class_dir)
            image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files[:100]:  # 限制每个类别的样本数量
                img_path = os.path.join(class_path, img_file)
                test_samples.append((img_path, class_label))
        
        # 定义损失函数
        criterion = torch.nn.CrossEntropyLoss()
        
        # 评估准确率、损失和推理时间
        correct = 0
        total_loss = 0.0
        total_time = 0.0
        total_samples = 0
        
        # 获取类别到索引的映射
        class_to_idx = {class_name: idx for idx, class_name in enumerate(predictor.class_names)}
        
        # 设置模型为评估模式
        predictor.model.eval()
        
        with torch.no_grad():
            for img_path, true_label in test_samples:
                start_time = time.time()
                
                # 使用预测器获取原始预测结果
                result = predictor.predict_image(img_path)
                
                end_time = time.time()
                
                inference_time = end_time - start_time
                total_time += inference_time
                total_samples += 1
                
                if result:
                    top_prediction = result['predictions'][0]
                    predicted_label = top_prediction['class_name']
                    
                    # 计算损失
                    # 先加载和处理图像，与预测器中相同的方式
                    image = Image.open(img_path).convert('RGB')
                    transform = getattr(predictor, 'transform', None)
                    
                    # 如果预测器没有公开transform属性，则创建一个标准的测试转换
                    if transform is None:
                        transform = torchvision.transforms.Compose([
                            torchvision.transforms.Resize(256),
                            torchvision.transforms.CenterCrop(224),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                    
                    img_tensor = transform(image).unsqueeze(0).to(predictor.device)
                    
                    # 获取目标类别的索引
                    target_idx = class_to_idx.get(true_label)
                    if target_idx is not None:
                        target = torch.tensor([target_idx], device=predictor.device)
                        
                        # 前向传播
                        outputs = predictor.model(img_tensor)
                        
                        # 计算损失
                        batch_loss = criterion(outputs, target)
                        total_loss += batch_loss.item()
                    
                    # 检查预测是否正确
                    if predicted_label == true_label:
                        correct += 1
        
        # 计算平均损失和准确率
        accuracy = 0.0
        loss = 0.0
        if total_samples > 0:
            loss = total_loss / total_samples
            accuracy = correct / total_samples
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'total_time': total_time,
            'total_samples': total_samples
        }
    
    def _evaluate_detection_model(self, predictor, test_dir):
        """评估检测模型"""
        # 检测模型的评估方法与分类模型不同
        # 这里简化实现，只计算推理时间
        test_img_dir = os.path.join(test_dir, 'images', 'val')
        total_time = 0.0
        total_samples = 0
        
        if os.path.exists(test_img_dir):
            image_files = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files[:50]:  # 限制样本数量
                img_path = os.path.join(test_img_dir, img_file)
                start_time = time.time()
                predictor.predict_image(img_path)
                end_time = time.time()
                
                inference_time = end_time - start_time
                total_time += inference_time
                total_samples += 1
        
        return {
            'total_time': total_time,
            'total_samples': total_samples
        }
    
    def apply_config(self, config):
        """应用配置"""
        if not config:
            return
            
        # 设置模型评估目录
        if 'default_model_eval_dir' in config:
            model_dir = config['default_model_eval_dir']
            if os.path.exists(model_dir):
                self.models_path_edit.setText(model_dir)
                self.models_dir = model_dir
                self.refresh_model_list() 