from PyQt5.QtWidgets import QTextEdit, QGroupBox, QVBoxLayout

class MetricsExplanation:
    """指标说明组件，提供各种训练指标的解释"""
    
    def __init__(self):
        """初始化指标说明组件"""
        self.explanations = self._create_explanations()
    
    def _create_explanations(self):
        """创建所有指标的说明文本"""
        return {
            0: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">损失和准确率总览</h3>
                <p>同时显示训练损失、验证损失、训练准确率和验证准确率曲线，提供整体训练进度视图。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>训练损失</b>：模型在训练集上的误差，值越小表示模型在训练数据上拟合得越好。</li>
                <li><b>验证损失</b>：模型在验证集上的误差，是评估模型泛化能力的重要指标。</li>
                <li><b>训练准确率</b>：模型在训练集上的预测准确率。</li>
                <li><b>验证准确率</b>：模型在验证集上的预测准确率，反映模型泛化能力。</li>
                </ul>
                </div>""",
                
            1: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">仅损失曲线</h3>
                <p>仅显示训练损失和验证损失曲线，专注于损失变化趋势。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>训练损失</b>：模型在训练集上的误差，值越小表示模型在训练数据上拟合得越好。</li>
                <li><b>验证损失</b>：模型在验证集上的误差，是评估模型泛化能力的重要指标。</li>
                </ul>
                <p>两者的变化趋势可以帮助判断模型是否过拟合。验证损失上升而训练损失下降，表明模型可能过拟合了训练数据。</p>
                </div>""",
                
            2: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">仅分类准确率</h3>
                <p>仅显示分类模型的验证准确率曲线。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>准确率</b>：正确预测的样本数占总样本数的比例，是分类任务的主要评价指标。</li>
                </ul>
                <p>准确率越高表示模型性能越好，曲线上升表明模型不断改进，平稳则表明训练可能达到瓶颈。</p>
                </div>""",
                
            3: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">仅目标检测mAP</h3>
                <p>仅显示目标检测模型的验证mAP曲线。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>mAP</b>：平均精度均值，是目标检测任务的标准评价指标。</li>
                </ul>
                <p>mAP综合考虑了检测的精确性和完整性，值越高表示检测性能越好。</p>
                </div>""",
                
            4: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">Precision/Recall</h3>
                <p>显示精确率和召回率曲线。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>精确率(Precision)</b>：正确预测为正例的数量占所有预测为正例的比例。</li>
                <li><b>召回率(Recall)</b>：正确预测为正例的数量占所有实际正例的比例。</li>
                </ul>
                <p>这两个指标通常是此消彼长的关系，需要在应用中找到合适的平衡点。</p>
                </div>""",
                
            5: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">F1-Score</h3>
                <p>显示F1分数曲线。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>F1-Score</b>：精确率和召回率的调和平均值，平衡了这两个指标的重要性。</li>
                </ul>
                <p>F1-Score是0到1之间的值，越接近1表示模型的精确率和召回率都较高。</p>
                <p>对于类别不平衡的数据集，F1-Score通常比简单的准确率更能反映模型性能。</p>
                </div>""",
                
            6: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">mAP50/mAP75</h3>
                <p>显示在不同IOU阈值下的mAP曲线。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>mAP50</b>：IOU阈值为0.5时的平均精度均值。</li>
                <li><b>mAP75</b>：IOU阈值为0.75时的平均精度均值，要求更加严格。</li>
                </ul>
                <p>IOU阈值越高，对目标检测位置精度的要求越高。mAP75更能反映模型的精细定位能力。</p>
                </div>""",
                
            7: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">类别损失/目标损失</h3>
                <p>显示目标检测中的分类分支和目标存在性损失曲线。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>类别损失</b>：目标检测中负责将物体分类到正确类别的损失函数。</li>
                <li><b>目标损失</b>：检测是否存在物体的置信度损失函数。</li>
                </ul>
                <p>这些损失值的变化可以帮助诊断检测模型在分类和定位方面的问题。</p>
                </div>""",
                
            8: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">框损失/置信度损失</h3>
                <p>显示目标检测中的边界框回归和置信度损失曲线。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>框损失</b>：边界框位置和尺寸预测的损失函数。</li>
                <li><b>置信度损失</b>：模型对预测框置信度的损失函数。</li>
                </ul>
                <p>框损失下降表明模型定位能力提升，置信度损失下降表明模型对预测的确信度提高。</p>
                </div>""",
                
            9: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">ROC-AUC曲线</h3>
                <p>显示ROC曲线下的面积变化。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>ROC-AUC</b>：ROC曲线下的面积，衡量分类器的性能，越接近1越好。</li>
                </ul>
                <p>AUC值为0.5表示模型性能与随机猜测相当；接近1表示模型有很强的区分能力。</p>
                <p>AUC对不平衡数据集不敏感，即使在正负样本比例变化时也能保持稳定。</p>
                </div>""",
                
            10: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">Average Precision</h3>
                <p>显示PR曲线下的面积变化。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>平均精度</b>：PR曲线下的面积，在不平衡数据集上比AUC更有参考价值。</li>
                </ul>
                <p>该指标综合考虑了精确率和召回率，特别适用于正负样本不平衡的情况。</p>
                <p>在多类别任务中，可以计算每个类别的AP然后取平均值(mAP)。</p>
                </div>""",
                
            11: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">Top-K准确率</h3>
                <p>显示模型预测正确类别位于前K个预测中的比例。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>Top-K准确率</b>：预测的前K个类别中包含正确类别的比例。</li>
                </ul>
                <p>对于大量类别的分类任务，Top-K准确率通常比Top-1准确率更为宽松。</p>
                <p>常见的K值有1、3、5等，对于复杂分类任务，Top-5准确率是一个常用指标。</p>
                </div>""",
                
            12: """<div style="margin: 5px;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">平衡准确率</h3>
                <p>显示考虑了类别不平衡的准确率指标。</p>
                <ul style="margin-left: 15px; padding-left: 0;">
                <li><b>平衡准确率</b>：每个类别的准确率的平均值，而不是简单地计算正确样本比例。</li>
                </ul>
                <p>对于类别不平衡的数据集，这个指标比普通准确率更公平。</p>
                <p>它给予所有类别相同的权重，防止模型偏向主导类别。</p>
                </div>"""
        }
    
    def get_explanation(self, index):
        """获取指定索引的指标说明"""
        return self.explanations.get(index, "<div style='margin:5px;'><p>无相关说明</p></div>")
    
    def create_explanation_widget(self):
        """创建指标说明小部件"""
        explanation_group = QGroupBox("当前指标说明")
        explanation_text = QTextEdit()
        explanation_text.setReadOnly(True)
        explanation_text.setMaximumHeight(150)  # 限制最大高度，避免占用太多空间
        
        # 设置文本编辑器样式
        explanation_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                font-family: '微软雅黑', '宋体', sans-serif;
                font-size: 13px;
                line-height: 1.4;
            }
        """)
        
        explanation_inner_layout = QVBoxLayout(explanation_group)
        explanation_inner_layout.addWidget(explanation_text)
        
        return explanation_group, explanation_text 