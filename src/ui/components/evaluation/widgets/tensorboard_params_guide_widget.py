from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QGroupBox, QTextEdit, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
import os


class TensorBoardParamsGuideWidget(QWidget):
    """TensorBoard参数监控说明控件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # 标题
        title_label = QLabel("TensorBoard 参数监控说明")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title_label)
        
        # 创建各个监控参数组
        self.create_basic_metrics_group(layout)
        self.create_advanced_metrics_group(layout)
        self.create_model_analysis_group(layout)
        self.create_performance_group(layout)
        self.create_visualization_group(layout)
        self.create_usage_tips_group(layout)
        
        layout.addStretch()
    
    def create_basic_metrics_group(self, parent_layout):
        """创建基础训练指标组"""
        group = QGroupBox("📊 基础训练指标 (SCALARS 标签页)")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 8px;
                margin: 5px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                color: #3498db;
            }
        """)
        
        layout = QVBoxLayout()
        
        metrics_info = [
            ("Loss/train & Loss/val", "训练和验证损失",
             "🔍 如何判断：\n"
             "• 理想状态：两条线平稳下降，验证损失略高于训练损失\n"
             "• 过拟合：训练损失持续下降，验证损失上升或波动\n"
             "• 欠拟合：两条线都很高且下降缓慢\n"
             "\n⚡ 具体操作：\n"
             "• 如果验证损失不再下降超过5个epoch → 启用早停\n"
             "• 如果损失震荡剧烈 → 学习率降低至当前的1/10\n"
             "• 如果损失下降过慢 → 学习率提高2-5倍\n"
             "• 如果出现过拟合 → 增加Dropout(0.3-0.5)或L2正则化(1e-4)"),
            
            ("Accuracy/train & Accuracy/val", "训练和验证准确率",
             "🔍 如何判断：\n"
             "• 健康状态：验证准确率在训练准确率的80%-95%之间\n"
             "• 过拟合：训练准确率>95%，验证准确率<80%\n"
             "• 数据问题：准确率长期停滞在某个值(如50%)\n"
             "\n⚡ 具体操作：\n"
             "• 准确率差距>20% → 增加数据增强(旋转±15°，缩放0.8-1.2)\n"
             "• 验证准确率停滞 → 检查数据标签是否正确\n"
             "• 准确率波动大 → 减小学习率并增加batch_size\n"
             "• 准确率过低 → 检查模型复杂度是否足够"),
            
            ("Learning_Rate/lr", "学习率变化",
             "🔍 如何判断：\n"
             "• 正常衰减：学习率按设定策略平稳下降\n"
             "• 衰减过早：损失还在快速下降时学习率已很小\n"
             "• 衰减过晚：损失已收敛但学习率仍很大\n"
             "\n⚡ 具体操作：\n"
             "• ResNet类模型 → 初始lr=0.1，每30epochs降10倍\n"
             "• MobileNet类模型 → 初始lr=0.01，余弦衰减\n"
             "• 损失震荡 → 立即将学习率降为当前的1/10\n"
             "• 使用ReduceLROnPlateau → patience=5, factor=0.5"),
            
            ("Advanced_Metrics/*", "高级评估指标",
             "🔍 如何判断：\n"
             "• Precision高(>0.9)，Recall低(<0.7) → 模型保守，漏检多\n"
             "• Precision低(<0.7)，Recall高(>0.9) → 模型激进，误检多\n"
             "• F1-Score < 0.8 → 需要重点优化的类别\n"
             "\n⚡ 具体操作：\n"
             "• 类别不平衡 → 使用focal_loss或调整类别权重\n"
             "• 某类别F1低 → 增加该类别训练样本或使用SMOTE\n"
             "• AUC-ROC < 0.85 → 检查特征工程和模型架构\n"
             "• 多分类accuracy < 单类平均F1 → 存在类别混淆问题")
        ]
        
        for metric_name, description, guidance in metrics_info:
            self.add_metric_item(layout, metric_name, description, guidance)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def create_advanced_metrics_group(self, parent_layout):
        """创建高级分析指标组"""
        group = QGroupBox("🔬 高级分析指标 (SCALARS & HISTOGRAMS 标签页)")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #e74c3c;
                border-radius: 8px;
                margin: 5px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                color: #e74c3c;
            }
        """)
        
        layout = QVBoxLayout()
        
        metrics_info = [
            ("Weights/{layer_name}", "层权重分布直方图",
             "🔍 如何判断：\n"
             "• 健康权重：呈正态分布，均值接近0，标准差0.1-0.3\n"
             "• 权重爆炸：分布向两端偏移，绝对值>2.0的权重过多\n"
             "• 权重消失：分布过于集中在0附近，标准差<0.01\n"
             "• 死神经元：某层权重长期不变化\n"
             "\n⚡ 具体操作：\n"
             "• 权重爆炸 → 学习率减小10倍，添加梯度裁剪(clip_norm=1.0)\n"
             "• 权重消失 → 使用Xavier或He初始化，检查激活函数\n"
             "• 分布偏斜 → 添加BatchNorm层或权重正则化\n"
             "• 死神经元 → 降低学习率，使用LeakyReLU替代ReLU"),
            
            ("Gradients/{layer_name}", "层梯度分布直方图",
             "🔍 如何判断：\n"
             "• 健康梯度：分布均匀，数量级在1e-4到1e-1之间\n"
             "• 梯度消失：深层梯度接近0，浅层正常(典型深度网络问题)\n"
             "• 梯度爆炸：梯度值>1.0，分布尖锐且数值巨大\n"
             "• 梯度截断：直方图在某个值处突然截断\n"
             "\n⚡ 具体操作：\n"
             "• 梯度消失 → 使用ResNet/DenseNet，或LSTM/GRU\n"
             "• 梯度爆炸 → 立即设置梯度裁剪max_norm=0.5\n"
             "• 浅层梯度过大 → 使用不同学习率，浅层lr=深层lr/10\n"
             "• 添加残差连接或使用更好的激活函数(Swish/GELU)"),
            
            ("Gradient_Norms/*", "梯度范数监控",
             "🔍 如何判断：\n"
             "• 正常范围：总梯度范数在0.1-10之间，相对稳定\n"
             "• 范数爆炸：突然跳跃到>100，或持续增长\n"
             "• 范数消失：逐渐降至<0.01，训练停滞\n"
             "• 范数震荡：剧烈波动，无明显趋势\n"
             "\n⚡ 具体操作：\n"
             "• 范数>10 → 设置clip_grad_norm_(model.parameters(), 1.0)\n"
             "• 范数<0.01 → 学习率提高5-10倍，检查权重初始化\n"
             "• 震荡严重 → 使用AdamW优化器，减小学习率\n"
             "• 监控每层范数比例，调整不同层的学习率")
        ]
        
        for metric_name, description, guidance in metrics_info:
            self.add_metric_item(layout, metric_name, description, guidance)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def create_model_analysis_group(self, parent_layout):
        """创建模型分析指标组"""
        group = QGroupBox("🧠 模型分析指标 (多个标签页)")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #9b59b6;
                border-radius: 8px;
                margin: 5px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                color: #9b59b6;
            }
        """)
        
        layout = QVBoxLayout()
        
        metrics_info = [
            ("Model Structure (GRAPHS)", "模型结构图",
             "🔍 如何分析：\n"
             "• 检查连接：确认skip connections、attention机制正确连接\n"
             "• 参数统计：每层参数量、总参数量、FLOPs计算\n"
             "• 瓶颈识别：找出计算密集层和内存占用大的层\n"
             "\n⚡ 具体操作：\n"
             "• 总参数>50M → 考虑模型剪枝或使用MobileNet架构\n"
             "• 发现断连 → 检查forward函数实现\n"
             "• 某层参数过多 → 使用深度可分离卷积或分组卷积\n"
             "• 模型过深 → 添加残差连接避免梯度消失"),
            
            ("Confusion Matrix (IMAGES)", "混淆矩阵热力图",
             "🔍 如何分析：\n"
             "• 对角线亮：正确分类，对角线越亮越好\n"
             "• 非对角亮点：类别混淆，需要重点关注\n"
             "• 某行全暗：该类别召回率极低，样本不足或特征不明显\n"
             "• 某列全暗：该类别精确率极低，容易被误分类\n"
             "\n⚡ 具体操作：\n"
             "• A类常被误分为B类 → 增加A类难样本，或使用hard mining\n"
             "• 某类召回率<50% → 该类样本数量至少增加2倍\n"
             "• 多类混淆严重 → 使用更强的特征提取器(ResNet→EfficientNet)\n"
             "• 类间相似度高 → 使用triplet loss或center loss"),
            
            ("Sample Images (IMAGES)", "训练样本可视化",
             "🔍 如何检查：\n"
             "• 图像质量：分辨率、亮度、对比度是否合适\n"
             "• 标签正确性：图像内容与标签是否匹配\n"
             "• 数据增强效果：旋转、缩放、颜色变化是否合理\n"
             "• 数据分布：各类别样本是否均衡\n"
             "\n⚡ 具体操作：\n"
             "• 图像模糊 → 检查resize方法，使用双三次插值\n"
             "• 标签错误 → 清洗数据集，移除错误标签样本\n"
             "• 增强过度 → 减小变换参数，保持原始特征\n"
             "• 类别不平衡 → 使用WeightedRandomSampler平衡采样")
        ]
        
        for metric_name, description, guidance in metrics_info:
            self.add_metric_item(layout, metric_name, description, guidance)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def create_performance_group(self, parent_layout):
        """创建性能监控指标组"""
        group = QGroupBox("⚡ 性能监控指标 (SCALARS 标签页)")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #f39c12;
                border-radius: 8px;
                margin: 5px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                color: #f39c12;
            }
        """)
        
        layout = QVBoxLayout()
        
        metrics_info = [
            ("Performance/samples_per_second", "训练速度",
             "🔍 性能基准：\n"
             "• ResNet50+ImageNet: 100-300 samples/sec (V100)\n"
             "• MobileNetV2: 500-1000 samples/sec\n"
             "• 小模型(<10M参数): >1000 samples/sec\n"
             "• 速度持续下降: 可能有内存泄漏或数据加载瓶颈\n"
             "\n⚡ 具体操作：\n"
             "• 速度<50 samples/sec → 检查数据加载，增加num_workers\n"
             "• 速度波动大 → 检查数据预处理，避免复杂变换\n"
             "• GPU利用率低但速度慢 → 增加batch_size或减少数据加载开销\n"
             "• 使用mixed precision训练可提速1.5-2倍"),
            
            ("Memory/gpu_memory_*", "GPU内存使用",
             "🔍 内存标准：\n"
             "• 健康状态：allocated占总显存的60-80%\n"
             "• 内存泄漏：allocated持续增长，不会释放\n"
             "• 碎片化：reserved远大于allocated\n"
             "• OOM风险：allocated >90%总显存\n"
             "\n⚡ 具体操作：\n"
             "• 内存使用>90% → 立即减小batch_size至当前的1/2\n"
             "• 内存泄漏 → 检查是否有tensor.detach()遗漏\n"
             "• 碎片化严重 → 定期调用torch.cuda.empty_cache()\n"
             "• 8G显存：batch_size建议16-32；16G显存：batch_size可到64-128"),
            
            ("System/cpu_usage_percent", "CPU使用率",
             "🔍 CPU标准：\n"
             "• 理想状态：30-70%，为数据预处理和系统预留空间\n"
             "• CPU过载：>90%，影响数据加载和GPU调度\n"
             "• CPU闲置：<20%，数据加载可能是瓶颈\n"
             "• 不平衡：某些核心100%，其他核心闲置\n"
             "\n⚡ 具体操作：\n"
             "• CPU>90% → 减少num_workers或简化数据预处理\n"
             "• CPU<20% → 增加num_workers，通常设为cpu核心数\n"
             "• 数据加载慢 → 使用SSD存储，或预先缓存至内存\n"
             "• 复杂数据增强 → 考虑使用GPU进行数据增强"),
            
            ("Performance/gpu_utilization", "GPU利用率",
             "🔍 GPU标准：\n"
             "• 理想状态：80-95%，持续高利用率\n"
             "• 利用率低：<60%，GPU资源浪费\n"
             "• 波动大：0-100%间跳动，存在数据加载瓶颈\n"
             "• 利用率100%：可能过载，需要监控温度\n"
             "\n⚡ 具体操作：\n"
             "• 利用率<60% → 增加batch_size或模型复杂度\n"
             "• 波动严重 → 增加数据加载线程，使用pin_memory=True\n"
             "• 持续100% → 检查GPU温度，必要时降低batch_size\n"
             "• 多GPU训练：每个GPU利用率应该基本一致")
        ]
        
        for metric_name, description, guidance in metrics_info:
            self.add_metric_item(layout, metric_name, description, guidance)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def create_visualization_group(self, parent_layout):
        """创建可视化说明组"""
        group = QGroupBox("🎨 TensorBoard 标签页说明")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #1abc9c;
                border-radius: 8px;
                margin: 5px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                color: #1abc9c;
            }
        """)
        
        layout = QVBoxLayout()
        
        tabs_info = [
            ("SCALARS", "标量指标图表", 
             "显示随时间变化的标量值，如损失、准确率、学习率等"),
            ("HISTOGRAMS", "直方图分布",
             "显示权重、梯度等参数的分布变化，用于分析模型内部状态"),
            ("IMAGES", "图像可视化",
             "显示样本图像、混淆矩阵、预测结果等可视化内容"),
            ("GRAPHS", "计算图结构",
             "显示模型的网络结构和计算图，便于理解模型架构"),
            ("HPARAMS", "超参数分析",
             "比较不同超参数组合的训练效果，便于超参数调优")
        ]
        
        for tab_name, description, detail in tabs_info:
            tab_frame = QFrame()
            tab_frame.setFrameStyle(QFrame.Box)
            tab_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border-radius: 3px; }")
            tab_layout = QVBoxLayout()
            
            tab_title = QLabel(f"• {tab_name}: {description}")
            tab_title.setFont(QFont("", 10, QFont.Bold))
            tab_title.setStyleSheet("color: #2c3e50; padding: 5px;")
            
            tab_detail = QLabel(f"  {detail}")
            tab_detail.setWordWrap(True)
            tab_detail.setStyleSheet("color: #34495e; padding: 2px 5px 5px 15px;")
            
            tab_layout.addWidget(tab_title)
            tab_layout.addWidget(tab_detail)
            tab_frame.setLayout(tab_layout)
            
            layout.addWidget(tab_frame)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def create_usage_tips_group(self, parent_layout):
        """创建使用技巧组"""
        group = QGroupBox("💡 实用技巧与建议")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #27ae60;
                border-radius: 8px;
                margin: 5px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                color: #27ae60;
            }
        """)
        
        layout = QVBoxLayout()
        
        tips_text = QTextEdit()
        tips_text.setReadOnly(True)
        tips_text.setMaximumHeight(200)
        tips_text.setStyleSheet("""
            QTextEdit {
                background-color: #f1f2f6;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                font-family: Arial, sans-serif;
                font-size: 10pt;
                line-height: 1.4;
            }
        """)
        
        tips_content = """
🔍 TensorBoard操作技巧：
• 平滑参数设置：损失曲线建议0.6-0.8，准确率建议0.3-0.5
• 横轴切换：Step显示训练步数，Relative显示相对时间，Wall显示实际时间
• 多实验对比：在左侧勾选不同实验运行，可同时显示多条曲线
• 下载数据：点击左下角下载按钮，可导出CSV数据进行深度分析

📊 关键图表解读：
• Loss图：关注收敛速度和稳定性，理想曲线应平滑下降
• Accuracy图：验证准确率应在训练准确率的80-95%之间
• Learning Rate图：检查学习率调度是否符合预期
• Histogram图：权重分布应呈正态分布，梯度不应为0或过大

🚨 异常情况快速诊断：
• 损失不下降：检查学习率(可能过小)、标签正确性、模型架构
• 损失爆炸(NaN)：立即停止训练，降低学习率10倍，添加梯度裁剪
• 准确率震荡：减小学习率，增加batch_size，检查数据加载
• GPU利用率低：增加batch_size，检查数据加载瓶颈，优化预处理

⚡ 实战优化流程：
1. 训练前3个epoch：重点观察损失下降趋势和梯度健康状况
2. 训练中期：监控过拟合信号，调整正则化强度
3. 训练后期：关注收敛稳定性，考虑学习率fine-tuning
4. 模型选择：基于验证指标选择最佳checkpoint，而非训练指标

📋 问题排查清单：
□ 检查数据标签是否正确(查看Sample Images)
□ 确认模型架构合理(查看Model Graph)
□ 监控资源使用是否均衡(CPU/GPU/Memory)
□ 验证学习率调度策略(Learning Rate曲线)
□ 分析类别混淆模式(Confusion Matrix)
        """
        
        tips_text.setPlainText(tips_content)
        layout.addWidget(tips_text)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def add_metric_item(self, parent_layout, metric_name, description, guidance):
        """添加指标项"""
        item_frame = QFrame()
        item_frame.setFrameStyle(QFrame.Box)
        item_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                margin: 2px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        
        # 指标名称和描述
        header_layout = QHBoxLayout()
        
        name_label = QLabel(metric_name)
        name_label.setFont(QFont("Consolas", 9, QFont.Bold))
        name_label.setStyleSheet("color: #2980b9; padding: 2px;")
        
        desc_label = QLabel(f"- {description}")
        desc_label.setFont(QFont("", 9))
        desc_label.setStyleSheet("color: #7f8c8d; padding: 2px;")
        
        header_layout.addWidget(name_label)
        header_layout.addWidget(desc_label)
        header_layout.addStretch()
        
        # 使用指导
        guidance_label = QLabel(guidance)
        guidance_label.setWordWrap(True)
        guidance_label.setFont(QFont("", 8))
        guidance_label.setStyleSheet("""
            color: #2c3e50;
            padding: 5px;
            background-color: #f8f9fa;
            border-radius: 3px;
            border-left: 3px solid #3498db;
        """)
        
        layout.addLayout(header_layout)
        layout.addWidget(guidance_label)
        
        item_frame.setLayout(layout)
        parent_layout.addWidget(item_frame) 