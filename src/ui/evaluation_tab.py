from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QComboBox, QGroupBox, QGridLayout, QListWidget,
                           QSizePolicy, QLineEdit, QMessageBox, QTableWidget, QTableWidgetItem,
                           QHeaderView, QStackedWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import os
import json
import subprocess
from .base_tab import BaseTab
from .training_visualization import TensorBoardWidget, TrainingVisualizationWidget

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
        
        # 添加训练参数说明
        explanation_group = QGroupBox("曲线参数说明")
        explanation_layout = QVBoxLayout()
        
        # 添加各指标说明
        params_explanation = """
        <b>训练损失</b>：模型在训练集上的误差，值越小表示模型在训练数据上拟合得越好。
        <b>验证损失</b>：模型在验证集上的误差，是评估模型泛化能力的重要指标。
        <b>训练准确率</b>：模型在训练集上的准确率，表示模型对训练数据的拟合程度。
        <b>验证准确率</b>：模型在验证集上的准确率，反映模型在未见过数据上的表现。
        """
        
        params_label = QLabel(params_explanation)
        params_label.setWordWrap(True)
        explanation_layout.addWidget(params_label)
        
        # 添加模型状态判断说明
        model_state_explanation = """
        <b>正常训练</b>：训练损失和验证损失都在下降，训练准确率和验证准确率都在上升。
        <b>过拟合</b>：训练损失继续下降但验证损失开始上升，或训练准确率继续上升但验证准确率开始下降，表明模型过度拟合了训练数据，失去泛化能力。
        <b>欠拟合</b>：训练损失和验证损失都较高且下降缓慢，训练准确率和验证准确率都较低且提升缓慢，表明模型能力不足或训练不充分。
        <b>何时停止训练</b>：当验证损失在多个轮次后不再下降或开始上升时，应考虑停止训练以避免过拟合。
        """
        
        state_label = QLabel(model_state_explanation)
        state_label.setWordWrap(True)
        explanation_layout.addWidget(state_label)
        
        explanation_group.setLayout(explanation_layout)
        training_curve_layout.addWidget(explanation_group)
        
        # 添加训练可视化组件
        self.training_visualization = TrainingVisualizationWidget()
        training_curve_layout.addWidget(self.training_visualization)
        
        # 添加状态标签
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
            
        # 清空结果表格
        self.result_table.setRowCount(0)
        
        # 这里应该调用实际的模型评估逻辑
        # 为了演示，我们使用模拟数据
        for i, model_name in enumerate(selected_models):
            # 添加新行
            row_position = self.result_table.rowCount()
            self.result_table.insertRow(row_position)
            
            # 模拟评估结果
            accuracy = 0.85 + (i * 0.02)  # 模拟不同的准确率
            loss = 0.3 - (i * 0.05)  # 模拟不同的损失
            params = 1000000 + (i * 500000)  # 模拟不同的参数量
            inference_time = 10 + (i * 2)  # 模拟不同的推理时间
            
            # 填充表格
            self.result_table.setItem(row_position, 0, QTableWidgetItem(model_name))
            self.result_table.setItem(row_position, 1, QTableWidgetItem(f"{accuracy:.2%}"))
            self.result_table.setItem(row_position, 2, QTableWidgetItem(f"{loss:.4f}"))
            self.result_table.setItem(row_position, 3, QTableWidgetItem(f"{params:,}"))
            self.result_table.setItem(row_position, 4, QTableWidgetItem(f"{inference_time} ms"))
        
        self.update_status(f"已比较 {len(selected_models)} 个模型")
    
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
                
            # 启动TensorBoard进程
            port = 6006  # 默认TensorBoard端口
            cmd = f"tensorboard --logdir={self.log_dir} --port={port}"
            
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
            self.tensorboard_widget.set_tensorboard_dir(self.log_dir)
            
            self.tb_status_label.setText(f"TensorBoard已启动，端口: {port}")
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
            print(f"收到训练数据更新: {data}")  # 添加调试信息
            if hasattr(self, 'training_visualization'):
                # 准备数据
                epoch_results = {
                    'phase': data['phase'],
                    'epoch': data['epoch'],
                    'loss': float(data['loss']),  # 确保转换为浮点数
                    'accuracy': float(data['accuracy'])  # 确保转换为浮点数
                }
                
                # 更新可视化
                self.training_visualization.update_plots(epoch_results)
                
                # 更新状态标签
                phase = "训练" if data['phase'] == 'train' else "验证"
                loss = data['loss']
                acc = data['accuracy']
                
                self.training_status_label.setText(f"轮次 {data['epoch']}: {phase}损失 = {loss:.4f}, {phase}准确率 = {acc:.4f}")
        except Exception as e:
            import traceback
            print(f"更新训练可视化时出错: {str(e)}")
            print(traceback.format_exc())
    
    def reset_training_visualization(self):
        """重置训练可视化"""
        if hasattr(self, 'training_visualization'):
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
        log_dir = config.get('tensorboard_log_dir', '')
        if log_dir and os.path.exists(log_dir):
            self.log_dir = log_dir
            self.log_path_edit.setText(log_dir)
            self.start_btn.setEnabled(True)
            self.tensorboard_widget.set_tensorboard_dir(log_dir)
            print(f"EvaluationTab: 已应用TensorBoard日志目录: {log_dir}")
        else:
            print(f"EvaluationTab: TensorBoard日志目录无效或不存在: {log_dir}") 