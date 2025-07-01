from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QGroupBox, QGridLayout, QMessageBox, QScrollArea,
                           QTextEdit, QTabWidget, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import os
import subprocess
import webbrowser
from .tensorboard_widget import TensorBoardWidget


class TensorBoardParameterGuideWidget(QWidget):
    """TensorBoard参数监控说明组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """初始化参数说明界面"""
        layout = QVBoxLayout(self)
        
        # 创建标签页
        tab_widget = QTabWidget()
        
        # 基础监控标签页
        basic_tab = self.create_basic_monitoring_tab()
        tab_widget.addTab(basic_tab, "基础监控")
        
        # 高级监控标签页
        advanced_tab = self.create_advanced_monitoring_tab()
        tab_widget.addTab(advanced_tab, "高级监控")
        
        # 性能监控标签页
        performance_tab = self.create_performance_monitoring_tab()
        tab_widget.addTab(performance_tab, "性能监控")
        
        # 使用指南标签页
        guide_tab = self.create_usage_guide_tab()
        tab_widget.addTab(guide_tab, "使用指南")
        
        layout.addWidget(tab_widget)
        
    def create_basic_monitoring_tab(self):
        """创建基础监控说明标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 标题
        title_label = QLabel("基础训练监控参数")
        title_label.setFont(QFont('微软雅黑', 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(title_label)
        
        # 损失监控
        loss_group = QGroupBox("损失函数监控 (SCALARS)")
        loss_layout = QVBoxLayout()
        loss_content = QTextEdit()
        loss_content.setReadOnly(True)
        loss_content.setMaximumHeight(150)
        loss_content.setHtml("""
        <b>监控参数:</b><br>
        • Loss/train - 训练损失<br>
        • Loss/val - 验证损失<br>
        • Loss_Components/* - 损失组件分解<br><br>
        
        <b>图表解读:</b><br>
        • <span style="color: red;">下降趋势</span>: 模型正在学习，性能提升<br>
        • <span style="color: orange;">震荡</span>: 学习率可能过高，考虑降低<br>
        • <span style="color: green;">平稳</span>: 模型收敛，可考虑早停<br>
        • <span style="color: red;">上升</span>: 可能过拟合，检查正则化<br><br>
        
        <b>实际建议:</b><br>
        • 训练损失持续下降但验证损失上升 → 过拟合，增加正则化<br>
        • 两者都不下降 → 学习率过低或模型容量不足<br>
        • 损失剧烈震荡 → 学习率过高，建议减半
        """)
        loss_layout.addWidget(loss_content)
        loss_group.setLayout(loss_layout)
        scroll_layout.addWidget(loss_group)
        
        # 准确率监控
        acc_group = QGroupBox("准确率监控 (SCALARS)")
        acc_layout = QVBoxLayout()
        acc_content = QTextEdit()
        acc_content.setReadOnly(True)
        acc_content.setMaximumHeight(150)
        acc_content.setHtml("""
        <b>监控参数:</b><br>
        • Accuracy/train - 训练准确率<br>
        • Accuracy/val - 验证准确率<br>
        • Advanced_Metrics/val_balanced_accuracy - 平衡准确率<br><br>
        
        <b>图表解读:</b><br>
        • <span style="color: green;">稳步上升</span>: 模型学习效果良好<br>
        • <span style="color: orange;">训练验证差距大</span>: 过拟合风险<br>
        • <span style="color: red;">验证准确率下降</span>: 过拟合已发生<br><br>
        
        <b>实际建议:</b><br>
        • 训练准确率95%+但验证准确率<85% → 严重过拟合，增加Dropout<br>
        • 准确率提升缓慢 → 增加模型复杂度或调整学习率<br>
        • 不平衡数据集优先看平衡准确率
        """)
        acc_layout.addWidget(acc_content)
        acc_group.setLayout(acc_layout)
        scroll_layout.addWidget(acc_group)
        
        # 学习率监控
        lr_group = QGroupBox("学习率监控 (SCALARS)")
        lr_layout = QVBoxLayout()
        lr_content = QTextEdit()
        lr_content.setReadOnly(True)
        lr_content.setMaximumHeight(120)
        lr_content.setHtml("""
        <b>监控参数:</b><br>
        • Learning_Rate/group_* - 各参数组学习率<br><br>
        
        <b>图表解读:</b><br>
        • <span style="color: blue;">学习率调度</span>: 观察衰减策略效果<br>
        • <span style="color: green;">适当衰减</span>: 有助于模型精细调优<br><br>
        
        <b>实际建议:</b><br>
        • 损失平台期时学习率应该衰减<br>
        • 过早衰减可能导致欠拟合<br>
        • 建议使用余弦退火或步长衰减
        """)
        lr_layout.addWidget(lr_content)
        lr_group.setLayout(lr_layout)
        scroll_layout.addWidget(lr_group)
        
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        return widget
        
    def create_advanced_monitoring_tab(self):
        """创建高级监控说明标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 标题
        title_label = QLabel("高级模型分析参数")
        title_label.setFont(QFont('微软雅黑', 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(title_label)
        
        # 权重和梯度监控
        weights_group = QGroupBox("权重和梯度监控 (HISTOGRAMS)")
        weights_layout = QVBoxLayout()
        weights_content = QTextEdit()
        weights_content.setReadOnly(True)
        weights_content.setMaximumHeight(180)
        weights_content.setHtml("""
        <b>监控参数:</b><br>
        • Weights/* - 各层权重分布直方图<br>
        • Gradients/* - 各层梯度分布直方图<br>
        • Gradient_Norms/* - 各层梯度范数<br>
        • Gradients/total_norm - 总梯度范数<br><br>
        
        <b>图表解读:</b><br>
        • <span style="color: green;">权重分布正常</span>: 钟形分布，标准差适中<br>
        • <span style="color: red;">权重过大/过小</span>: 可能梯度爆炸/消失<br>
        • <span style="color: orange;">梯度范数过大</span>: >1.0需要梯度裁剪<br>
        • <span style="color: red;">梯度范数接近0</span>: 梯度消失问题<br><br>
        
        <b>实际建议:</b><br>
        • 总梯度范数>10 → 使用梯度裁剪，clip_norm=1.0<br>
        • 后层梯度范数<0.001 → 梯度消失，考虑ResNet或BatchNorm<br>
        • 权重标准差>1 → 权重初始化问题，使用Xavier或He初始化
        """)
        weights_layout.addWidget(weights_content)
        weights_group.setLayout(weights_layout)
        scroll_layout.addWidget(weights_group)
        
        # 高级评估指标
        metrics_group = QGroupBox("高级评估指标 (SCALARS)")
        metrics_layout = QVBoxLayout()
        metrics_content = QTextEdit()
        metrics_content.setReadOnly(True)
        metrics_content.setMaximumHeight(160)
        metrics_content.setHtml("""
        <b>监控参数:</b><br>
        • Advanced_Metrics/*_precision - 精确率<br>
        • Advanced_Metrics/*_recall - 召回率<br>
        • Advanced_Metrics/*_f1_score - F1分数<br>
        • Advanced_Metrics/*_auc_roc - ROC曲线下面积<br><br>
        
        <b>图表解读:</b><br>
        • <span style="color: green;">精确率高</span>: 误报少，预测可信<br>
        • <span style="color: green;">召回率高</span>: 漏报少，覆盖全面<br>
        • <span style="color: blue;">F1分数</span>: 精确率和召回率的调和平均<br>
        • <span style="color: purple;">AUC>0.9</span>: 模型分类能力优秀<br><br>
        
        <b>实际建议:</b><br>
        • 精确率低 → 增加负样本或调整分类阈值<br>
        • 召回率低 → 增加正样本或降低分类阈值<br>
        • 不平衡数据集重点关注F1分数和AUC
        """)
        metrics_layout.addWidget(metrics_content)
        metrics_group.setLayout(metrics_layout)
        scroll_layout.addWidget(metrics_group)
        
        # 可视化监控
        viz_group = QGroupBox("可视化监控 (IMAGES)")
        viz_layout = QVBoxLayout()
        viz_content = QTextEdit()
        viz_content.setReadOnly(True)
        viz_content.setMaximumHeight(140)
        viz_content.setHtml("""
        <b>监控内容:</b><br>
        • Sample Images - 训练样本可视化<br>
        • Confusion Matrix - 混淆矩阵热力图<br>
        • Model Predictions - 模型预测结果<br>
        • Class Distribution - 类别分布图表<br><br>
        
        <b>分析要点:</b><br>
        • <span style="color: green;">混淆矩阵对角线亮</span>: 分类效果好<br>
        • <span style="color: red;">非对角线亮</span>: 类别混淆，需要更多数据<br>
        • <span style="color: blue;">预测置信度</span>: 观察模型确信程度<br><br>
        
        <b>实际建议:</b><br>
        • 特定类别经常被误分 → 增加该类别训练数据<br>
        • 预测置信度普遍低 → 模型不够确信，需要更多训练
        """)
        viz_layout.addWidget(viz_content)
        viz_group.setLayout(viz_layout)
        scroll_layout.addWidget(viz_group)
        
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        return widget
        
    def create_performance_monitoring_tab(self):
        """创建性能监控说明标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 标题
        title_label = QLabel("性能和资源监控参数")
        title_label.setFont(QFont('微软雅黑', 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(title_label)
        
        # 训练性能监控
        perf_group = QGroupBox("训练性能监控 (SCALARS)")
        perf_layout = QVBoxLayout()
        perf_content = QTextEdit()
        perf_content.setReadOnly(True)
        perf_content.setMaximumHeight(150)
        perf_content.setHtml("""
        <b>监控参数:</b><br>
        • Performance/samples_per_second - 每秒处理样本数<br>
        • Performance/total_training_time - 总训练时间<br>
        • Performance/gpu_utilization - GPU利用率<br><br>
        
        <b>图表解读:</b><br>
        • <span style="color: green;">样本处理速度稳定</span>: 训练效率良好<br>
        • <span style="color: orange;">处理速度下降</span>: 可能内存不足或瓶颈<br>
        • <span style="color: blue;">GPU利用率高</span>: 硬件充分利用<br><br>
        
        <b>实际建议:</b><br>
        • 样本处理速度<50/秒 → 增加batch_size或优化数据加载<br>
        • GPU利用率<80% → 可能CPU瓶颈，增加num_workers<br>
        • 训练时间过长 → 考虑混合精度训练或模型剪枝
        """)
        perf_layout.addWidget(perf_content)
        perf_group.setLayout(perf_layout)
        scroll_layout.addWidget(perf_group)
        
        # 内存监控
        memory_group = QGroupBox("内存监控 (SCALARS)")
        memory_layout = QVBoxLayout()
        memory_content = QTextEdit()
        memory_content.setReadOnly(True)
        memory_content.setMaximumHeight(140)
        memory_content.setHtml("""
        <b>监控参数:</b><br>
        • Memory/gpu_memory_allocated_gb - GPU已分配内存<br>
        • Memory/gpu_memory_reserved_gb - GPU保留内存<br>
        • System/memory_usage_percent - 系统内存使用率<br><br>
        
        <b>图表解读:</b><br>
        • <span style="color: green;">内存使用稳定</span>: 无内存泄漏<br>
        • <span style="color: red;">内存持续增长</span>: 可能内存泄漏<br>
        • <span style="color: orange;">内存使用率>90%</span>: 接近内存上限<br><br>
        
        <b>实际建议:</b><br>
        • GPU内存>95% → 减少batch_size或使用梯度累积<br>
        • 内存持续增长 → 检查数据加载器和变量引用<br>
        • 系统内存不足 → 减少num_workers或数据预加载
        """)
        memory_layout.addWidget(memory_content)
        memory_group.setLayout(memory_layout)
        scroll_layout.addWidget(memory_group)
        
        # 系统资源监控
        system_group = QGroupBox("系统资源监控 (SCALARS)")
        system_layout = QVBoxLayout()
        system_content = QTextEdit()
        system_content.setReadOnly(True)
        system_content.setMaximumHeight(120)
        system_content.setHtml("""
        <b>监控参数:</b><br>
        • System/cpu_usage_percent - CPU使用率<br>
        • System/memory_usage_percent - 内存使用率<br><br>
        
        <b>图表解读:</b><br>
        • <span style="color: green;">CPU使用合理</span>: 数据处理无瓶颈<br>
        • <span style="color: red;">CPU使用率100%</span>: 数据加载瓶颈<br><br>
        
        <b>实际建议:</b><br>
        • CPU瓶颈 → 优化数据预处理或增加num_workers<br>
        • 内存不足 → 使用更小的batch_size或数据流式加载
        """)
        system_layout.addWidget(system_content)
        system_group.setLayout(system_layout)
        scroll_layout.addWidget(system_group)
        
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        return widget
        
    def create_usage_guide_tab(self):
        """创建使用指南标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 标题
        title_label = QLabel("TensorBoard使用指南")
        title_label.setFont(QFont('微软雅黑', 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(title_label)
        
        # 快速开始
        start_group = QGroupBox("快速开始")
        start_layout = QVBoxLayout()
        start_content = QTextEdit()
        start_content.setReadOnly(True)
        start_content.setMaximumHeight(120)
        start_content.setHtml("""
        <b>开始使用TensorBoard:</b><br>
        1. 在训练配置中启用 "使用TensorBoard记录"<br>
        2. 开始训练模型，系统会自动记录各项指标<br>
        3. 在此页面选择日志目录并启动TensorBoard<br>
        4. 在浏览器中查看详细的训练分析<br><br>
        
        <b>推荐工作流:</b><br>
        训练开始 → 观察基础指标 → 检查高级分析 → 优化超参数 → 重新训练
        """)
        start_layout.addWidget(start_content)
        start_group.setLayout(start_layout)
        scroll_layout.addWidget(start_group)
        
        # 常见问题诊断
        diagnosis_group = QGroupBox("常见问题诊断")
        diagnosis_layout = QVBoxLayout()
        diagnosis_content = QTextEdit()
        diagnosis_content.setReadOnly(True)
        diagnosis_content.setMaximumHeight(200)
        diagnosis_content.setHtml("""
        <b>训练不收敛:</b><br>
        • 检查学习率是否过高/过低<br>
        • 观察梯度范数是否正常<br>
        • 查看权重分布是否合理<br><br>
        
        <b>过拟合问题:</b><br>
        • 训练验证损失差距大<br>
        • 训练准确率高但验证准确率低<br>
        • 建议增加正则化或Dropout<br><br>
        
        <b>性能问题:</b><br>
        • 训练速度慢 → 检查GPU利用率和内存使用<br>
        • 内存不足 → 减少batch_size<br>
        • CPU瓶颈 → 增加数据加载线程<br><br>
        
        <b>模型效果差:</b><br>
        • 查看混淆矩阵找出问题类别<br>
        • 检查类别分布是否平衡<br>
        • 观察预测置信度分布
        """)
        diagnosis_layout.addWidget(diagnosis_content)
        diagnosis_group.setLayout(diagnosis_layout)
        scroll_layout.addWidget(diagnosis_group)
        
        # 高级技巧
        tips_group = QGroupBox("高级使用技巧")
        tips_layout = QVBoxLayout()
        tips_content = QTextEdit()
        tips_content.setReadOnly(True)
        tips_content.setMaximumHeight(160)
        tips_content.setHtml("""
        <b>多实验对比:</b><br>
        • 使用HParams功能对比不同超参数配置<br>
        • 在同一图表中查看多次训练结果<br><br>
        
        <b>自定义标量:</b><br>
        • 可以添加自定义的评估指标<br>
        • 使用正则表达式过滤感兴趣的指标<br><br>
        
        <b>图像分析:</b><br>
        • 点击图像查看高分辨率版本<br>
        • 使用滑块查看不同epoch的结果<br><br>
        
        <b>性能优化:</b><br>
        • 定期清理旧的日志文件<br>
        • 可以设置采样频率减少记录开销<br>
        • 使用--reload_multifile参数实时更新
        """)
        tips_layout.addWidget(tips_content)
        tips_group.setLayout(tips_layout)
        scroll_layout.addWidget(tips_group)
        
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        return widget


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
        layout.addWidget(log_group)
        
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
        
        layout.addLayout(control_layout)
        
        # 添加参数监控说明组件
        self.parameter_guide = TensorBoardParameterGuideWidget()
        layout.addWidget(self.parameter_guide)
        
        # 创建TensorBoard嵌入视图
        self.tensorboard_widget = TensorBoardWidget()
        layout.addWidget(self.tensorboard_widget)
        
        # 添加状态标签
        self.tb_status_label = QLabel("TensorBoard未启动")
        self.tb_status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.tb_status_label)
    
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
        
    def __del__(self):
        """析构方法，确保在对象被销毁时停止TensorBoard进程"""
        try:
            self.stop_tensorboard()
        except:
            # 在析构时忽略异常
            pass 