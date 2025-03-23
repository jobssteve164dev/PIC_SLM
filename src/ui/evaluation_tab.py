from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QComboBox, QGroupBox, QGridLayout, QListWidget,
                           QSizePolicy, QLineEdit, QMessageBox, QTableWidget, QTableWidgetItem,
                           QHeaderView, QStackedWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import os
import json
import subprocess
import sys
from .base_tab import BaseTab
from .training_visualization import TensorBoardWidget, TrainingVisualizationWidget
import torch
import torchvision
from PIL import Image
import time

class EvaluationTab(BaseTab):
    """评估标签页，负责模型评估和比较功能，以及TensorBoard可视化"""
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent, main_window)
        self.models_dir = ""
        self.models_list = []
        self.log_dir = ""
        self.tensorboard_process = None
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self.scroll_content)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 添加标题
        title_label = QLabel("模型评估与可视化")
        title_label.setFont(QFont('微软雅黑', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 创建切换按钮组
        switch_layout = QHBoxLayout()
        
        # 添加实时训练曲线按钮
        self.training_curve_btn = QPushButton("实时训练曲线")
        self.training_curve_btn.setCheckable(True)
        self.training_curve_btn.setChecked(True)
        self.training_curve_btn.clicked.connect(lambda: self.switch_view(0))
        switch_layout.addWidget(self.training_curve_btn)
        
        self.tb_btn = QPushButton("TensorBoard可视化")
        self.tb_btn.setCheckable(True)
        self.tb_btn.clicked.connect(lambda: self.switch_view(1))
        switch_layout.addWidget(self.tb_btn)
        
        self.eval_btn = QPushButton("模型评估")
        self.eval_btn.setCheckable(True)
        self.eval_btn.clicked.connect(lambda: self.switch_view(2))
        switch_layout.addWidget(self.eval_btn)
        
        main_layout.addLayout(switch_layout)
        
        # 创建堆叠小部件用于切换视图
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        # 创建实时训练曲线视图
        self.training_curve_widget = QWidget()
        self.setup_training_curve_ui()
        self.stacked_widget.addWidget(self.training_curve_widget)
        
        # 创建TensorBoard视图
        self.tb_widget = QWidget()
        self.setup_tb_ui()
        self.stacked_widget.addWidget(self.tb_widget)
        
        # 创建评估视图
        self.eval_widget = QWidget()
        self.setup_eval_ui()
        self.stacked_widget.addWidget(self.eval_widget)
        
        # 添加弹性空间
        main_layout.addStretch()
    
    def setup_eval_ui(self):
        """设置评估UI"""
        eval_layout = QVBoxLayout(self.eval_widget)
        
        # 创建模型目录选择组
        models_group = QGroupBox("模型目录")
        models_layout = QGridLayout()
        
        self.models_path_edit = QLineEdit()
        self.models_path_edit.setReadOnly(True)
        self.models_path_edit.setPlaceholderText("请选择包含模型的目录")
        
        models_btn = QPushButton("浏览...")
        models_btn.clicked.connect(self.select_models_dir)
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_model_list)
        
        models_layout.addWidget(QLabel("模型目录:"), 0, 0)
        models_layout.addWidget(self.models_path_edit, 0, 1)
        models_layout.addWidget(models_btn, 0, 2)
        models_layout.addWidget(refresh_btn, 0, 3)
        
        models_group.setLayout(models_layout)
        eval_layout.addWidget(models_group)
        
        # 创建模型列表组
        list_group = QGroupBox("可用模型")
        list_layout = QVBoxLayout()
        
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.NoSelection)  # 禁用选择模式，使用复选框代替
        self.model_list.setMinimumHeight(150)
        list_layout.addWidget(self.model_list)
        
        list_group.setLayout(list_layout)
        eval_layout.addWidget(list_group)
        
        # 创建比较按钮
        compare_btn = QPushButton("比较选中模型")
        compare_btn.clicked.connect(self.compare_models)
        eval_layout.addWidget(compare_btn)
        
        # 创建结果表格
        result_group = QGroupBox("比较结果")
        result_layout = QVBoxLayout()
        
        self.result_table = QTableWidget(0, 5)  # 行数将根据选择的模型数量动态调整
        self.result_table.setHorizontalHeaderLabels(["模型名称", "准确率", "损失", "参数量", "推理时间"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        result_layout.addWidget(self.result_table)
        
        result_group.setLayout(result_layout)
        eval_layout.addWidget(result_group)
    
    def setup_tb_ui(self):
        """设置TensorBoard UI"""
        tb_layout = QVBoxLayout(self.tb_widget)
        
        # 创建日志目录选择组
        log_group = QGroupBox("TensorBoard日志目录")
        log_layout = QGridLayout()
        
        self.log_path_edit = QLineEdit()
        self.log_path_edit.setReadOnly(True)
        self.log_path_edit.setPlaceholderText("请选择TensorBoard日志目录")
        
        log_btn = QPushButton("浏览...")
        log_btn.clicked.connect(self.select_log_dir)
        
        log_layout.addWidget(QLabel("日志目录:"), 0, 0)
        log_layout.addWidget(self.log_path_edit, 0, 1)
        log_layout.addWidget(log_btn, 0, 2)
        
        log_group.setLayout(log_layout)
        tb_layout.addWidget(log_group)
        
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
        
        tb_layout.addLayout(control_layout)
        
        # 创建TensorBoard嵌入视图
        self.tensorboard_widget = TensorBoardWidget()
        tb_layout.addWidget(self.tensorboard_widget)
        
        # 添加状态标签
        self.tb_status_label = QLabel("TensorBoard未启动")
        self.tb_status_label.setAlignment(Qt.AlignCenter)
        tb_layout.addWidget(self.tb_status_label)
    
    def setup_training_curve_ui(self):
        """设置实时训练曲线UI"""
        training_curve_layout = QVBoxLayout(self.training_curve_widget)
        
        # 添加说明标签
        info_label = QLabel("实时训练曲线可视化")
        info_label.setFont(QFont('微软雅黑', 12, QFont.Bold))
        info_label.setAlignment(Qt.AlignCenter)
        training_curve_layout.addWidget(info_label)
        
        # 添加训练参数说明 - 移除旧的静态参数说明
        # explanation_group = QGroupBox("曲线参数说明")
        # explanation_layout = QVBoxLayout()
        # params_explanation = """
        # <b>基础评估指标</b>：
        # <b>训练损失</b>：模型在训练集上的误差，值越小表示模型在训练数据上拟合得越好。
        # <b>验证损失</b>：模型在验证集上的误差，是评估模型泛化能力的重要指标。
        # <b>训练准确率</b>：模型在训练集上的准确率，表示模型对训练数据的拟合程度。
        # <b>验证准确率</b>：模型在验证集上的准确率，反映模型在未见过数据上的表现。
        # <b>精确率(Precision)</b>：正确预测为正例的数量占所有预测为正例的比例。
        # <b>召回率(Recall)</b>：正确预测为正例的数量占所有实际正例的比例。
        # <b>F1-Score</b>：精确率和召回率的调和平均值，平衡了这两个指标。
        # 
        # <b>分类特有指标</b>：
        # <b>ROC-AUC</b>：ROC曲线下的面积，衡量分类器的性能，越接近1越好。
        # <b>平均精度(Average Precision)</b>：PR曲线下的面积，在不平衡数据集上比AUC更有参考价值。
        # <b>Top-K准确率</b>：预测的前K个类别中包含正确类别的比例。
        # <b>平衡准确率</b>：考虑了类别不平衡问题的准确率指标。
        # 
        # <b>检测特有指标</b>：
        # <b>mAP</b>：平均精度均值，目标检测的主要评价指标。
        # <b>mAP50</b>：IOU阈值为0.5时的平均精度均值。
        # <b>mAP75</b>：IOU阈值为0.75时的平均精度均值，要求更加严格。
        # <b>类别损失</b>：分类分支的损失值。
        # <b>目标损失</b>：物体存在置信度的损失值。
        # <b>框损失</b>：边界框回归的损失值。
        # """
        # 
        # params_label = QLabel(params_explanation)
        # params_label.setWordWrap(True)
        # explanation_layout.addWidget(params_label)
        # explanation_group.setLayout(explanation_layout)
        # training_curve_layout.addWidget(explanation_group)
        
        # 添加训练开始/停止按键的提示
        control_tip = QLabel("训练停止条件：当验证损失在多个轮次后不再下降，或达到设定的最大轮次。")
        control_tip.setWordWrap(True)
        training_curve_layout.addWidget(control_tip)
        
        # 添加训练可视化组件
        self.training_visualization = TrainingVisualizationWidget()
        training_curve_layout.addWidget(self.training_visualization)
        
        # 添加当前训练状态标签
        self.training_status_label = QLabel("等待训练开始...")
        self.training_status_label.setAlignment(Qt.AlignCenter)
        training_curve_layout.addWidget(self.training_status_label)
    
    def switch_view(self, index):
        """切换视图"""
        self.stacked_widget.setCurrentIndex(index)
        
        # 更新按钮状态
        self.training_curve_btn.setChecked(index == 0)
        self.tb_btn.setChecked(index == 1)
        self.eval_btn.setChecked(index == 2)
    
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
    
    def select_log_dir(self):
        """选择TensorBoard日志目录"""
        folder = QFileDialog.getExistingDirectory(self, "选择TensorBoard日志目录")
        if folder:
            self.log_dir = folder
            self.log_path_edit.setText(folder)
            self.start_btn.setEnabled(True)
            self.tensorboard_widget.set_tensorboard_dir(folder)
    
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
        
        self.update_status("正在评估模型...")
        
        try:
            # 添加src目录到sys.path，确保能导入相关模块
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # 导入预测器和评估工具
            from predictor import Predictor
            
            # 从配置文件获取默认设置
            config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
            
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
                data_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
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
                if "resnet18" in model_filename.lower():
                    model_arch = "ResNet18"
                elif "resnet34" in model_filename.lower():
                    model_arch = "ResNet34"
                elif "resnet50" in model_filename.lower():
                    model_arch = "ResNet50"
                elif "mobilenet" in model_filename.lower():
                    model_arch = "MobileNetV2"
                elif "efficientnet" in model_filename.lower():
                    model_arch = "EfficientNetB0"
                elif "vgg16" in model_filename.lower():
                    model_arch = "VGG16"
                else:
                    # 默认使用ResNet18
                    model_arch = "ResNet18"
                
                # 更新状态
                self.update_status(f"正在评估模型 {model_filename}...")
                
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
                    accuracy = 0.0
                    loss = 0.0
                    params_count = 0
                    total_time = 0.0
                    total_samples = 0
                    
                    # 计算模型参数数量
                    params_count = sum(p.numel() for p in predictor.model.parameters())
                    
                    # 对分类模型进行评估
                    if task_type == "分类模型":
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
                                    transform = predictor.transform  # 从预测器获取转换
                                    
                                    # 如果预测器没有公开transform属性，则创建一个标准的测试转换
                                    if not hasattr(predictor, 'transform'):
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
                        if total_samples > 0:
                            loss = total_loss / total_samples
                            accuracy = correct / total_samples
                    
                    # 对检测模型进行评估
                    elif task_type == "检测模型":
                        # 检测模型的评估方法与分类模型不同
                        # 这里简化实现，只计算推理时间
                        test_img_dir = os.path.join(test_dir, 'images', 'val')
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
                    
                    # 计算平均推理时间
                    avg_inference_time = 0
                    if total_samples > 0:
                        avg_inference_time = (total_time / total_samples) * 1000  # 转换为毫秒
                    
                    # 添加新行到表格
                    row_position = self.result_table.rowCount()
                    self.result_table.insertRow(row_position)
                    
                    # 填充表格
                    self.result_table.setItem(row_position, 0, QTableWidgetItem(model_filename))
                    self.result_table.setItem(row_position, 1, QTableWidgetItem(f"{accuracy:.2%}"))
                    self.result_table.setItem(row_position, 2, QTableWidgetItem(f"{loss:.4f}"))
                    self.result_table.setItem(row_position, 3, QTableWidgetItem(f"{params_count:,}"))
                    self.result_table.setItem(row_position, 4, QTableWidgetItem(f"{avg_inference_time:.2f} ms"))
                    
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
            
            self.update_status(f"已完成 {len(selected_models)} 个模型的评估")
            
        except Exception as e:
            import traceback
            error_msg = f"模型评估过程出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            QMessageBox.critical(self, "错误", error_msg)
    
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
                self.update_status(f"已找到tensorboard_logs目录: {log_dir}")
                
            # 启动TensorBoard进程
            port = 6006  # 默认TensorBoard端口
            cmd = f"tensorboard --logdir={log_dir} --port={port}"
            
            self.update_status(f"启动TensorBoard，命令: {cmd}")
            
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
            import webbrowser
            webbrowser.open(f"http://localhost:{port}")
            
            self.tb_status_label.setText(f"TensorBoard已启动，端口: {port}，日志目录: {log_dir}")
            self.update_status(f"TensorBoard已启动，端口: {port}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动TensorBoard失败: {str(e)}")
    
    def stop_tensorboard(self):
        """停止TensorBoard"""
        if self.tensorboard_process:
            try:
                # 终止TensorBoard进程
                if os.name == 'nt':  # Windows
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.tensorboard_process.pid)])
                else:  # Linux/Mac
                    self.tensorboard_process.terminate()
                    self.tensorboard_process.wait()
                
                self.tensorboard_process = None
                
                # 更新UI状态
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                
                self.tb_status_label.setText("TensorBoard已停止")
                self.update_status("TensorBoard已停止")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"停止TensorBoard失败: {str(e)}")
    
    def update_training_visualization(self, data):
        """更新训练可视化"""
        try:
            if hasattr(self, 'training_visualization'):
                # 检查是否存储了上一次的训练和验证损失值
                if not hasattr(self, 'last_train_loss'):
                    self.last_train_loss = 0.0
                if not hasattr(self, 'last_val_loss'):
                    self.last_val_loss = 0.0
                
                # 初始化性能指标变量 - 区分分类和检测
                # 分类任务指标
                if not hasattr(self, 'last_val_accuracy'):
                    self.last_val_accuracy = 0.0
                if not hasattr(self, 'last_train_accuracy'):
                    self.last_train_accuracy = 0.0
                if not hasattr(self, 'last_roc_auc'):
                    self.last_roc_auc = 0.0
                if not hasattr(self, 'last_average_precision'):
                    self.last_average_precision = 0.0
                if not hasattr(self, 'last_top_k_accuracy'):
                    self.last_top_k_accuracy = 0.0
                if not hasattr(self, 'last_balanced_accuracy'):
                    self.last_balanced_accuracy = 0.0
                
                # 检测任务指标
                if not hasattr(self, 'last_val_map'):
                    self.last_val_map = 0.0
                if not hasattr(self, 'last_train_map'):
                    self.last_train_map = 0.0
                if not hasattr(self, 'last_map50'):
                    self.last_map50 = 0.0
                if not hasattr(self, 'last_map75'):
                    self.last_map75 = 0.0
                
                # 共用指标
                if not hasattr(self, 'last_precision'):
                    self.last_precision = 0.0
                if not hasattr(self, 'last_recall'):
                    self.last_recall = 0.0
                if not hasattr(self, 'last_f1_score'):
                    self.last_f1_score = 0.0
                if not hasattr(self, 'last_class_loss'):
                    self.last_class_loss = 0.0
                if not hasattr(self, 'last_obj_loss'):
                    self.last_obj_loss = 0.0
                if not hasattr(self, 'last_box_loss'):
                    self.last_box_loss = 0.0
                
                # 数据格式转换
                is_train = data.get('phase') == 'train'
                epoch = data.get('epoch', 0)
                loss = float(data.get('loss', 0))
                learning_rate = float(data.get('learning_rate', 0.001))
                
                # 根据任务类型获取性能指标
                is_classification = 'accuracy' in data
                
                # 更新对应的损失值和指标值
                if is_train:
                    self.last_train_loss = loss
                    
                    # 更新训练阶段指标
                    if is_classification:
                        self.last_train_accuracy = float(data.get('accuracy', 0.0))
                    else:
                        self.last_train_map = float(data.get('mAP', 0.0))
                else:
                    self.last_val_loss = loss
                    
                    # 根据任务类型更新相应指标
                    if is_classification:
                        # 分类任务指标
                        self.last_val_accuracy = float(data.get('accuracy', 0.0))
                        self.last_roc_auc = float(data.get('roc_auc', 0.0))
                        self.last_average_precision = float(data.get('average_precision', 0.0))
                        self.last_top_k_accuracy = float(data.get('top_k_accuracy', 0.0))
                        self.last_balanced_accuracy = float(data.get('balanced_accuracy', 0.0))
                    else:
                        # 检测任务指标
                        self.last_val_map = float(data.get('mAP', 0.0))
                        self.last_map50 = float(data.get('mAP50', 0.0))
                        self.last_map75 = float(data.get('mAP75', 0.0))
                        self.last_class_loss = float(data.get('class_loss', 0.0))
                        self.last_obj_loss = float(data.get('obj_loss', 0.0))
                        self.last_box_loss = float(data.get('box_loss', 0.0))
                    
                    # 共用指标
                    self.last_precision = float(data.get('precision', 0.0))
                    self.last_recall = float(data.get('recall', 0.0))
                    self.last_f1_score = float(data.get('f1_score', 0.0))
                
                # 构建TrainingVisualizationWidget期望的指标格式
                metrics = {
                    'epoch': epoch,
                    'train_loss': self.last_train_loss,
                    'val_loss': self.last_val_loss,
                    'learning_rate': learning_rate,
                    'precision': self.last_precision,
                    'recall': self.last_recall,
                    'f1_score': self.last_f1_score
                }
                
                # 根据任务类型添加相应的性能指标
                if is_classification:
                    # 添加分类特有指标
                    metrics['val_accuracy'] = self.last_val_accuracy
                    metrics['train_accuracy'] = self.last_train_accuracy
                    metrics['roc_auc'] = self.last_roc_auc
                    metrics['average_precision'] = self.last_average_precision
                    metrics['top_k_accuracy'] = self.last_top_k_accuracy
                    metrics['balanced_accuracy'] = self.last_balanced_accuracy
                else:
                    # 添加检测特有指标
                    metrics['val_map'] = self.last_val_map
                    metrics['train_map'] = self.last_train_map
                    metrics['mAP50'] = self.last_map50
                    metrics['mAP75'] = self.last_map75
                    metrics['class_loss'] = self.last_class_loss
                    metrics['obj_loss'] = self.last_obj_loss
                    metrics['box_loss'] = self.last_box_loss
                
                # 更新可视化
                self.training_visualization.update_metrics(metrics)
                
                # 更新状态标签
                phase_text = "训练" if is_train else "验证"
                
                # 根据任务类型显示不同的状态文本
                if is_classification:
                    accuracy_value = float(data.get('accuracy', 0.0))
                    status_text = f"轮次 {epoch}: {phase_text}损失 = {loss:.4f}, {phase_text}准确率 = {accuracy_value:.4f}"
                    
                    # 添加分类额外指标到状态
                    if not is_train and self.last_precision > 0:
                        status_text += f", 精确率 = {self.last_precision:.4f}, 召回率 = {self.last_recall:.4f}"
                else:
                    map_value = float(data.get('mAP', 0.0))
                    status_text = f"轮次 {epoch}: {phase_text}损失 = {loss:.4f}, {phase_text}mAP = {map_value:.4f}"
                    
                    # 添加检测额外指标到状态
                    if not is_train and self.last_map50 > 0:
                        status_text += f", mAP50 = {self.last_map50:.4f}, mAP75 = {self.last_map75:.4f}"
                
                self.training_status_label.setText(status_text)
                
        except Exception as e:
            import traceback
            print(f"更新训练可视化时出错: {str(e)}")
            print(traceback.format_exc())
    
    def reset_training_visualization(self):
        """重置训练可视化"""
        if hasattr(self, 'training_visualization'):
            # 重置数据存储
            if hasattr(self, 'last_train_loss'):
                self.last_train_loss = 0.0
            if hasattr(self, 'last_val_loss'):
                self.last_val_loss = 0.0
            
            # 重置分类指标
            if hasattr(self, 'last_val_accuracy'):
                self.last_val_accuracy = 0.0
            if hasattr(self, 'last_train_accuracy'):
                self.last_train_accuracy = 0.0
            
            # 重置检测指标
            if hasattr(self, 'last_val_map'):
                self.last_val_map = 0.0
            if hasattr(self, 'last_train_map'):
                self.last_train_map = 0.0
                
            # 重置其他指标
            if hasattr(self, 'last_precision'):
                self.last_precision = 0.0
            if hasattr(self, 'last_recall'):
                self.last_recall = 0.0
            if hasattr(self, 'last_f1_score'):
                self.last_f1_score = 0.0
            if hasattr(self, 'last_map50'):
                self.last_map50 = 0.0
            if hasattr(self, 'last_map75'):
                self.last_map75 = 0.0
            if hasattr(self, 'last_class_loss'):
                self.last_class_loss = 0.0
            if hasattr(self, 'last_obj_loss'):
                self.last_obj_loss = 0.0
            if hasattr(self, 'last_box_loss'):
                self.last_box_loss = 0.0
                
            # 重置分类特有指标
            if hasattr(self, 'last_roc_auc'):
                self.last_roc_auc = 0.0
            if hasattr(self, 'last_average_precision'):
                self.last_average_precision = 0.0
            if hasattr(self, 'last_top_k_accuracy'):
                self.last_top_k_accuracy = 0.0
            if hasattr(self, 'last_balanced_accuracy'):
                self.last_balanced_accuracy = 0.0
            
            # 调用TrainingVisualizationWidget的reset_plots方法
            self.training_visualization.reset_plots()
            self.training_status_label.setText("等待训练开始...")
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 确保在关闭窗口时停止TensorBoard进程
        self.stop_tensorboard()
        super().closeEvent(event) 

    def apply_config(self, config):
        """应用配置，从配置中加载默认设置"""
        if not config:
            print("EvaluationTab: 配置为空，无法应用")
            return
            
        print(f"EvaluationTab正在应用配置: {config}")
            
        # 加载默认模型评估文件夹
        default_model_eval_dir = config.get('default_model_eval_dir', '')
        if default_model_eval_dir and os.path.exists(default_model_eval_dir):
            self.models_dir = default_model_eval_dir
            self.models_path_edit.setText(default_model_eval_dir)
            self.refresh_model_list()
            print(f"EvaluationTab: 已应用默认模型评估文件夹: {default_model_eval_dir}")
        else:
            print(f"EvaluationTab: 默认模型评估文件夹无效或不存在: {default_model_eval_dir}")
            
        # 加载TensorBoard日志目录
        log_dir = config.get('default_tensorboard_log_dir', '')
        if log_dir and os.path.exists(log_dir):
            self.log_dir = log_dir
            self.log_path_edit.setText(log_dir)
            self.start_btn.setEnabled(True)
            self.tensorboard_widget.set_tensorboard_dir(log_dir)
            print(f"EvaluationTab: 已应用TensorBoard日志目录: {log_dir}")
        else:
            # 尝试使用旧的配置项
            old_log_dir = config.get('tensorboard_log_dir', '')
            if old_log_dir and os.path.exists(old_log_dir):
                self.log_dir = old_log_dir
                self.log_path_edit.setText(old_log_dir)
                self.start_btn.setEnabled(True)
                self.tensorboard_widget.set_tensorboard_dir(old_log_dir)
                print(f"EvaluationTab: 已应用旧的TensorBoard日志目录: {old_log_dir}")
            else:
                print(f"EvaluationTab: TensorBoard日志目录无效或不存在: {log_dir}")
    
    def setup_trainer(self, trainer):
        """设置训练器并连接信号"""
        try:
            if hasattr(self, 'training_visualization') and trainer is not None:
                # 直接连接TrainingVisualizationWidget和训练器
                self.training_visualization.connect_signals(trainer)
                
                # 记录设置成功的日志
                print(f"已成功设置训练器并连接信号到训练可视化组件")
                return True
        except Exception as e:
            import traceback
            print(f"设置训练器时出错: {str(e)}")
            print(traceback.format_exc())
            return False 