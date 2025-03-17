from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QTextBrowser, 
                           QDialogButtonBox, QLabel, QScrollArea, QWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class TrainingHelpDialog(QDialog):
    """训练帮助对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("模型训练帮助文档")
        self.setMinimumSize(800, 600)
        self.resize(1000, 800)
        
        # 创建主布局
        layout = QVBoxLayout(self)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 添加标题
        title = QLabel("模型训练帮助文档")
        title.setFont(QFont('微软雅黑', 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(title)
        
        # 创建文本浏览器显示帮助内容
        help_text = QTextBrowser()
        help_text.setOpenExternalLinks(True)
        help_text.setHtml(self._get_help_text())
        scroll_layout.addWidget(help_text)
        
        # 设置滚动区域
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        
        # 添加确定按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)
        
    def _get_help_text(self):
        """获取帮助文档HTML内容"""
        return """
        <h2>1. 模型文件说明</h2>
        
        <h3>1.1 预训练模型</h3>
        <ul>
            <li><b>分类模型</b>：
                <ul>
                    <li>ResNet系列：经典的残差网络，适用于一般图像分类任务</li>
                    <li>DenseNet系列：密集连接网络，特征重用效率高</li>
                    <li>EfficientNet系列：高效网络，平衡计算量和精度</li>
                    <li>MobileNet系列：轻量级网络，适合移动端部署</li>
                </ul>
            </li>
            <li><b>检测模型</b>：
                <ul>
                    <li>YOLO系列：单阶段检测器，速度快精度高</li>
                    <li>SSD系列：单阶段检测器，多尺度特征融合</li>
                    <li>Faster R-CNN：两阶段检测器，精度高</li>
                    <li>RetinaNet：单阶段检测器，解决类别不平衡</li>
                </ul>
            </li>
        </ul>
        
        <h3>1.2 模型保存</h3>
        <ul>
            <li><b>保存内容</b>：
                <ul>
                    <li>模型权重文件(.pth)：包含训练后的网络参数</li>
                    <li>配置文件(.json)：包含模型结构和训练参数</li>
                    <li>训练日志：记录训练过程中的指标变化</li>
                </ul>
            </li>
            <li><b>保存策略</b>：
                <ul>
                    <li>定期保存：每N个epoch保存一次检查点</li>
                    <li>最佳模型：保存验证集性能最好的模型</li>
                    <li>最终模型：训练结束时的模型状态</li>
                </ul>
            </li>
        </ul>
        
        <h3>1.3 模型导出</h3>
        <ul>
            <li><b>导出格式</b>：
                <ul>
                    <li>ONNX：跨平台部署标准格式</li>
                    <li>TorchScript：PyTorch优化的部署格式</li>
                    <li>TensorRT：NVIDIA GPU加速格式</li>
                </ul>
            </li>
            <li><b>部署优化</b>：
                <ul>
                    <li>模型压缩：减小模型体积</li>
                    <li>计算优化：加快推理速度</li>
                    <li>量化：降低计算精度以提升性能</li>
                </ul>
            </li>
        </ul>
        
        <h2>2. 训练参数说明</h2>
        
        <h3>2.1 基本训练参数</h3>
        <ul>
            <li><b>批次大小(Batch Size)</b>：
                <ul>
                    <li>定义：每次迭代处理的样本数量</li>
                    <li>建议值：分类16-128，检测8-32</li>
                    <li>影响：内存占用、训练速度、模型稳定性</li>
                    <li>选择原则：根据GPU显存和数据特点选择</li>
                </ul>
            </li>
            <li><b>学习率(Learning Rate)</b>：
                <ul>
                    <li>定义：模型参数更新的步长</li>
                    <li>建议值：0.0001-0.01</li>
                    <li>调整策略：预热、衰减、动态调整</li>
                    <li>注意事项：过大导致不稳定，过小收敛慢</li>
                </ul>
            </li>
            <li><b>训练轮数(Epochs)</b>：
                <ul>
                    <li>定义：完整遍历训练集的次数</li>
                    <li>建议值：分类20-100，检测50-300</li>
                    <li>影响因素：数据量、模型复杂度</li>
                    <li>终止条件：早停机制、性能饱和</li>
                </ul>
            </li>
        </ul>
        
        <h3>2.2 高级训练参数</h3>
        <ul>
            <li><b>优化器(Optimizer)</b>：
                <ul>
                    <li>SGD：最基础的优化器，需要精细调参</li>
                    <li>Adam：自适应优化器，收敛快但可能过拟合</li>
                    <li>AdamW：改进版Adam，更好的权重衰减</li>
                </ul>
            </li>
            <li><b>学习率调度(LR Scheduler)</b>：
                <ul>
                    <li>Step：固定步长衰减</li>
                    <li>Cosine：余弦退火衰减</li>
                    <li>ReduceLROnPlateau：动态调整</li>
                </ul>
            </li>
            <li><b>正则化(Regularization)</b>：
                <ul>
                    <li>权重衰减：L2正则化</li>
                    <li>Dropout：随机失活</li>
                    <li>数据增强：增加训练样本多样性</li>
                </ul>
            </li>
        </ul>
        
        <h3>2.3 检测特有参数</h3>
        <ul>
            <li><b>锚框设置(Anchor)</b>：
                <ul>
                    <li>尺寸：根据目标大小分布设置</li>
                    <li>比例：考虑目标形状多样性</li>
                    <li>生成策略：K-means聚类或人工设计</li>
                </ul>
            </li>
            <li><b>NMS参数</b>：
                <ul>
                    <li>IoU阈值：0.3-0.7，控制框重叠度</li>
                    <li>置信度阈值：0.3-0.8，控制检测置信度</li>
                    <li>最大检测数：限制每张图片的检测框数量</li>
                </ul>
            </li>
        </ul>
        
        <h2>3. 评估方法说明</h2>
        
        <h3>3.1 分类评估指标</h3>
        <ul>
            <li><b>准确率(Accuracy)</b>：
                <ul>
                    <li>定义：正确预测数/总样本数</li>
                    <li>适用：类别平衡数据集</li>
                    <li>局限：对类别不平衡不敏感</li>
                </ul>
            </li>
            <li><b>精确率和召回率</b>：
                <ul>
                    <li>精确率：正确正例/预测正例总数</li>
                    <li>召回率：正确正例/实际正例总数</li>
                    <li>F1分数：精确率和召回率的调和平均</li>
                </ul>
            </li>
            <li><b>混淆矩阵</b>：
                <ul>
                    <li>功能：详细展示各类别预测情况</li>
                    <li>分析：查找易混淆类别</li>
                    <li>应用：模型优化方向指导</li>
                </ul>
            </li>
        </ul>
        
        <h3>3.2 检测评估指标</h3>
        <ul>
            <li><b>mAP(mean Average Precision)</b>：
                <ul>
                    <li>定义：各类别AP的平均值</li>
                    <li>计算：不同IoU阈值下的平均精度</li>
                    <li>常用标准：mAP@0.5，mAP@0.75</li>
                </ul>
            </li>
            <li><b>检测速度</b>：
                <ul>
                    <li>FPS：每秒处理图片数</li>
                    <li>推理时间：单张图片处理时间</li>
                    <li>影响因素：模型大小、硬件性能</li>
                </ul>
            </li>
        </ul>
        
        <h2>4. 训练过程监控</h2>
        <ul>
            <li><b>损失曲线</b>：
                <ul>
                    <li>训练损失：反映模型拟合程度</li>
                    <li>验证损失：反映模型泛化能力</li>
                    <li>分析：过拟合、欠拟合判断</li>
                </ul>
            </li>
            <li><b>评估指标曲线</b>：
                <ul>
                    <li>准确率/mAP变化</li>
                    <li>学习率变化</li>
                    <li>其他自定义指标</li>
                </ul>
            </li>
            <li><b>资源监控</b>：
                <ul>
                    <li>GPU利用率</li>
                    <li>内存占用</li>
                    <li>训练速度</li>
                </ul>
            </li>
        </ul>
        
        <h2>5. 常见问题与解决方案</h2>
        <ul>
            <li><b>训练不收敛</b>：
                <ul>
                    <li>检查学习率是否合适</li>
                    <li>验证数据预处理是否正确</li>
                    <li>确认损失函数计算是否正确</li>
                </ul>
            </li>
            <li><b>过拟合</b>：
                <ul>
                    <li>增加正则化强度</li>
                    <li>使用数据增强</li>
                    <li>减小模型复杂度</li>
                </ul>
            </li>
            <li><b>显存不足</b>：
                <ul>
                    <li>减小批次大小</li>
                    <li>使用混合精度训练</li>
                    <li>选择更小的模型</li>
                </ul>
            </li>
        </ul>
        """ 