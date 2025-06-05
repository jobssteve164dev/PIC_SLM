"""
结果显示管理器

负责管理数据集评估结果的表格显示和指标说明
"""

from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QLabel
from PyQt5.QtCore import QObject, pyqtSignal


class ResultDisplayManager(QObject):
    """结果显示管理器"""
    
    # 定义信号
    result_updated = pyqtSignal()
    
    def __init__(self, result_table=None, metrics_info_label=None):
        super().__init__()
        self.result_table = result_table
        self.metrics_info_label = metrics_info_label
        
    def setup_table(self, table, info_label):
        """设置表格和信息标签"""
        self.result_table = table
        self.metrics_info_label = info_label
        
    def clear_results(self):
        """清空结果显示"""
        if self.result_table:
            self.result_table.setRowCount(0)
        if self.metrics_info_label:
            self.metrics_info_label.setText("")
            
    def display_classification_distribution_results(self, analysis_result):
        """显示分类数据集分布分析结果"""
        if not self.result_table:
            return
            
        # 设置表格内容
        self.result_table.setRowCount(4)
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["指标", "值"])
        
        self.result_table.setItem(0, 0, QTableWidgetItem("总类别数"))
        self.result_table.setItem(0, 1, QTableWidgetItem(str(analysis_result.get('total_classes', 0))))
        
        self.result_table.setItem(1, 0, QTableWidgetItem("训练集总样本数"))
        self.result_table.setItem(1, 1, QTableWidgetItem(str(analysis_result.get('train_total', 0))))
        
        self.result_table.setItem(2, 0, QTableWidgetItem("验证集总样本数"))
        self.result_table.setItem(2, 1, QTableWidgetItem(str(analysis_result.get('val_total', 0))))
        
        self.result_table.setItem(3, 0, QTableWidgetItem("类别不平衡度"))
        imbalance = analysis_result.get('imbalance_ratio', 0)
        self.result_table.setItem(3, 1, QTableWidgetItem(f"{imbalance:.2f}" if imbalance != float('inf') else "∞"))
        
        self.result_table.resizeColumnsToContents()
        
        # 设置指标说明
        metrics_info = """
        <b>指标影响说明:</b><br>
        <b>总类别数</b>: 反映模型需要学习的概念数量，类别越多，模型通常越复杂<br>
        <b>样本数量</b>: 一般而言，更多的样本有助于模型学习更稳健的特征<br>
        <b>类别不平衡度</b>: 值越接近1越好，过高表示数据严重不平衡，模型可能偏向样本多的类别<br><br>
        <b>建议:</b><br>
        - 类别不平衡度 < 2: 非常均衡，无需特殊处理<br>
        - 类别不平衡度 2~10: 中度不平衡，可考虑数据增强或权重调整<br>
        - 类别不平衡度 > 10: 严重不平衡，建议采用过采样、欠采样或合成样本
        """
        if self.metrics_info_label:
            self.metrics_info_label.setText(metrics_info)
            
        self.result_updated.emit()
        
    def display_detection_distribution_results(self, analysis_result):
        """显示目标检测数据集分布分析结果"""
        if not self.result_table:
            return
            
        # 设置表格内容
        self.result_table.setRowCount(5)
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["指标", "值"])
        
        self.result_table.setItem(0, 0, QTableWidgetItem("总类别数"))
        self.result_table.setItem(0, 1, QTableWidgetItem(str(analysis_result.get('total_classes', 0))))
        
        self.result_table.setItem(1, 0, QTableWidgetItem("训练集总标注数"))
        self.result_table.setItem(1, 1, QTableWidgetItem(str(analysis_result.get('train_total', 0))))
        
        self.result_table.setItem(2, 0, QTableWidgetItem("验证集总标注数"))
        self.result_table.setItem(2, 1, QTableWidgetItem(str(analysis_result.get('val_total', 0))))
        
        imbalance = analysis_result.get('imbalance_ratio', 0)
        self.result_table.setItem(3, 0, QTableWidgetItem("类别不平衡度"))
        self.result_table.setItem(3, 1, QTableWidgetItem(f"{imbalance:.2f}" if imbalance != float('inf') else "∞"))
        
        similarity = analysis_result.get('similarity', 0)
        self.result_table.setItem(4, 0, QTableWidgetItem("训练/验证集相似度"))
        self.result_table.setItem(4, 1, QTableWidgetItem(f"{similarity:.4f}"))
        
        self.result_table.resizeColumnsToContents()
        
        # 设置指标说明
        metrics_info = """
        <b>指标影响说明:</b><br>
        <b>类别不平衡度</b>: 影响模型对少数类的检测能力<br>
        • < 2: 极佳的类别平衡<br>
        • 2-5: 良好平衡<br>
        • 5-20: 中度不平衡，可能需要特殊处理<br>
        • > 20: 严重不平衡，建议数据增强或采样调整<br>
        <b>训练/验证集相似度</b>: 衡量分布一致性<br>
        • > 0.9: 非常相似，验证集能很好地反映训练集<br>
        • 0.7-0.9: 良好相似度<br>
        • < 0.7: 分布差异大，验证结果可能不可靠<br>
        <b>建议</b>: 对于不平衡类别，考虑使用focal loss或过采样技术
        """
        if self.metrics_info_label:
            self.metrics_info_label.setText(metrics_info)
            
        self.result_updated.emit()
        
    def display_image_quality_results(self, analysis_result):
        """显示图像质量分析结果"""
        if not self.result_table:
            return
            
        # 设置表格内容
        self.result_table.setRowCount(4)
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["指标", "值"])
        
        self.result_table.setItem(0, 0, QTableWidgetItem("平均图像尺寸(像素)"))
        self.result_table.setItem(0, 1, QTableWidgetItem(f"{analysis_result.get('avg_size', 0):.0f}"))
        
        self.result_table.setItem(1, 0, QTableWidgetItem("图像尺寸标准差"))
        self.result_table.setItem(1, 1, QTableWidgetItem(f"{analysis_result.get('size_std', 0):.0f}"))
        
        self.result_table.setItem(2, 0, QTableWidgetItem("平均清晰度"))
        self.result_table.setItem(2, 1, QTableWidgetItem(f"{analysis_result.get('avg_quality', 0):.2f}"))
        
        self.result_table.setItem(3, 0, QTableWidgetItem("清晰度标准差"))
        self.result_table.setItem(3, 1, QTableWidgetItem(f"{analysis_result.get('quality_std', 0):.2f}"))
        
        self.result_table.resizeColumnsToContents()
        
        # 设置指标说明
        metrics_info = """
        <b>指标影响说明:</b><br>
        <b>图像尺寸</b>: 影响特征提取能力，太小可能细节不足，太大会增加计算量<br>
        <b>尺寸标准差</b>: 反映数据集的一致性，过高表示尺寸差异大<br>
        <b>图像清晰度</b>: 值越高表示图像越清晰，清晰图像通常含有更多可学习特征<br>
        <b>清晰度标准差</b>: 反映数据集质量的一致性<br><br>
        <b>建议:</b><br>
        - 尺寸标准差/平均尺寸 < 0.1: 尺寸高度一致，有利于模型学习<br>
        - 平均清晰度 < 100: 较为模糊，可能影响特征识别<br>
        - 平均清晰度 > 500: 较为清晰，有利于模型学习细节
        """
        if self.metrics_info_label:
            self.metrics_info_label.setText(metrics_info)
            
        self.result_updated.emit()
        
    def display_annotation_quality_results(self, analysis_result):
        """显示标注质量分析结果"""
        if not self.result_table:
            return
            
        # 设置表格内容
        self.result_table.setRowCount(5)
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["指标", "值"])
        
        self.result_table.setItem(0, 0, QTableWidgetItem("平均标注数量"))
        self.result_table.setItem(0, 1, QTableWidgetItem(f"{analysis_result.get('avg_annotations', 0):.2f}"))
        
        self.result_table.setItem(1, 0, QTableWidgetItem("最大标注数量"))
        self.result_table.setItem(1, 1, QTableWidgetItem(f"{analysis_result.get('max_annotations', 0)}"))
        
        self.result_table.setItem(2, 0, QTableWidgetItem("最小标注数量"))
        self.result_table.setItem(2, 1, QTableWidgetItem(f"{analysis_result.get('min_annotations', 0)}"))
        
        self.result_table.setItem(3, 0, QTableWidgetItem("平均文件大小"))
        avg_size_kb = analysis_result.get('avg_file_size', 0) / 1024
        self.result_table.setItem(3, 1, QTableWidgetItem(f"{avg_size_kb:.2f} KB"))
        
        self.result_table.setItem(4, 0, QTableWidgetItem("标注一致性(CV)"))
        self.result_table.setItem(4, 1, QTableWidgetItem(f"{analysis_result.get('consistency_cv', 0):.4f}"))
        
        self.result_table.resizeColumnsToContents()
        
        # 设置指标说明
        metrics_info = """
        <b>指标影响说明:</b><br>
        <b>平均标注数量</b>: 影响每个类别的学习充分度<br>
        • < 50: 样本可能不足，考虑数据增强<br>
        • > 1000: 通常足够训练稳定模型<br>
        <b>标注一致性(CV)</b>: 变异系数反映类别间数量平衡性<br>
        • < 0.2: 非常均衡<br>
        • 0.2-0.5: 适度均衡<br>
        • > 0.5: 不均衡，考虑均衡采样策略<br>
        <b>文件大小差异</b>: 反映图像复杂度和质量一致性<br>
        <b>建议</b>: 标注数量少的类别考虑增加样本或使用权重调整
        """
        if self.metrics_info_label:
            self.metrics_info_label.setText(metrics_info)
            
        self.result_updated.emit()
        
    def display_feature_distribution_results(self, analysis_result):
        """显示特征分布分析结果"""
        if not self.result_table:
            return
            
        # 设置表格内容
        self.result_table.setRowCount(4)
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["指标", "值"])
        
        self.result_table.setItem(0, 0, QTableWidgetItem("平均亮度"))
        self.result_table.setItem(0, 1, QTableWidgetItem(f"{analysis_result.get('avg_brightness', 0):.2f}"))
        
        self.result_table.setItem(1, 0, QTableWidgetItem("亮度变异系数"))
        self.result_table.setItem(1, 1, QTableWidgetItem(f"{analysis_result.get('brightness_cv', 0):.4f}"))
        
        self.result_table.setItem(2, 0, QTableWidgetItem("平均对比度"))
        self.result_table.setItem(2, 1, QTableWidgetItem(f"{analysis_result.get('avg_contrast', 0):.2f}"))
        
        self.result_table.setItem(3, 0, QTableWidgetItem("对比度变异系数"))
        self.result_table.setItem(3, 1, QTableWidgetItem(f"{analysis_result.get('contrast_cv', 0):.4f}"))
        
        self.result_table.resizeColumnsToContents()
        
        # 设置指标说明
        metrics_info = """
        <b>指标影响说明:</b><br>
        <b>平均亮度</b>: 影响模型对不同光照条件的适应性<br>
        • 过低 (<50) 表示图像较暗，可能丢失细节<br>
        • 过高 (>200) 表示图像过亮，可能导致过曝<br>
        • 理想范围: 80-180<br>
        <b>亮度变异系数</b>: 衡量亮度分布一致性，值越小越一致<br>
        <b>平均对比度</b>: 影响特征的可区分性<br>
        • 过低 (<30) 表示图像平淡，特征不明显<br>
        • 过高 (>80) 可能导致某些区域信息丢失<br>
        <b>建议</b>: 考虑添加数据增强，增加亮度和对比度的多样性
        """
        if self.metrics_info_label:
            self.metrics_info_label.setText(metrics_info)
            
        self.result_updated.emit()
        
    def display_detection_annotation_quality_results(self, analysis_result):
        """显示目标检测标注质量分析结果"""
        if not self.result_table:
            return
            
        # 设置表格内容
        self.result_table.setRowCount(5)
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["指标", "值"])
        
        self.result_table.setItem(0, 0, QTableWidgetItem("平均每图标注数"))
        self.result_table.setItem(0, 1, QTableWidgetItem(f"{analysis_result.get('avg_boxes_per_image', 0):.2f}"))
        
        self.result_table.setItem(1, 0, QTableWidgetItem("无标注图像比例"))
        self.result_table.setItem(1, 1, QTableWidgetItem(f"{analysis_result.get('empty_ratio', 0):.2%}"))
        
        self.result_table.setItem(2, 0, QTableWidgetItem("小目标比例"))
        self.result_table.setItem(2, 1, QTableWidgetItem(f"{analysis_result.get('small_box_ratio', 0):.2%}"))
        
        self.result_table.setItem(3, 0, QTableWidgetItem("平均边界框大小"))
        self.result_table.setItem(3, 1, QTableWidgetItem(f"{analysis_result.get('avg_box_size', 0):.4f}"))
        
        self.result_table.setItem(4, 0, QTableWidgetItem("边界框大小方差"))
        self.result_table.setItem(4, 1, QTableWidgetItem(f"{analysis_result.get('box_size_variance', 0):.6f}"))
        
        self.result_table.resizeColumnsToContents()
        
        # 设置指标说明
        metrics_info = """
        <b>指标影响说明:</b><br>
        <b>平均每图标注数</b>: 反映目标密度<br>
        • 过低(<1)可能导致检测器难以学习足够特征<br>
        • 过高(>10)可能使模型对拥挤场景过度适应<br>
        <b>无标注图像比例</b>: 应尽量避免，通常应<5%<br>
        <b>小目标比例</b>: 影响检测难度<br>
        • >50%: 小目标主导，可能需要专门的小目标检测策略<br>
        • <10%: 大目标主导，通常检测更容易<br>
        <b>边界框大小方差</b>: 目标尺寸一致性<br>
        <b>建议</b>: 小目标比例高时，考虑使用FPN、多尺度训练等策略
        """
        if self.metrics_info_label:
            self.metrics_info_label.setText(metrics_info)
            
        self.result_updated.emit()
        
    def display_weight_generation_results(self, class_counts, weight_strategies, stats):
        """显示权重生成结果"""
        if not self.result_table:
            return
            
        class_names = list(class_counts.keys())
        
        # 设置表格显示权重策略对比
        self.result_table.setRowCount(len(class_names) + 4)
        self.result_table.setColumnCount(6)
        self.result_table.setHorizontalHeaderLabels([
            "类别", "样本数", "Balanced", "Inverse", "Log_Inverse", "Normalized"
        ])
        
        # 添加统计信息
        self.result_table.setItem(0, 0, QTableWidgetItem("总类别数"))
        self.result_table.setItem(0, 1, QTableWidgetItem(str(len(class_names))))
        
        self.result_table.setItem(1, 0, QTableWidgetItem("总样本数"))
        self.result_table.setItem(1, 1, QTableWidgetItem(str(sum(class_counts.values()))))
        
        self.result_table.setItem(2, 0, QTableWidgetItem("类别不平衡度"))
        imbalance = stats.get('imbalance_ratio', 0)
        self.result_table.setItem(2, 1, QTableWidgetItem(f"{imbalance:.2f}"))
        
        self.result_table.setItem(3, 0, QTableWidgetItem("变异系数"))
        cv = stats.get('cv_coefficient', 0)
        self.result_table.setItem(3, 1, QTableWidgetItem(f"{cv:.3f}"))
        
        # 填充每个类别的权重信息
        for i, class_name in enumerate(class_names):
            row = i + 4
            self.result_table.setItem(row, 0, QTableWidgetItem(class_name))
            self.result_table.setItem(row, 1, QTableWidgetItem(str(class_counts[class_name])))
            
            balanced_weight = weight_strategies.get('balanced', {}).get(class_name, 0)
            self.result_table.setItem(row, 2, QTableWidgetItem(f"{balanced_weight:.3f}"))
            
            inverse_weight = weight_strategies.get('inverse', {}).get(class_name, 0)
            self.result_table.setItem(row, 3, QTableWidgetItem(f"{inverse_weight:.3f}"))
            
            log_inverse_weight = weight_strategies.get('log_inverse', {}).get(class_name, 0)
            self.result_table.setItem(row, 4, QTableWidgetItem(f"{log_inverse_weight:.3f}"))
            
            normalized_weight = weight_strategies.get('normalized', {}).get(class_name, 0)
            self.result_table.setItem(row, 5, QTableWidgetItem(f"{normalized_weight:.3f}"))
        
        self.result_table.resizeColumnsToContents()
        
        # 设置指标说明
        imbalance_level = "轻度" if imbalance < 3 else "中度" if imbalance < 10 else "严重"
        recommended_strategy = stats.get('recommended_strategy', 'balanced')
        
        metrics_info = f"""
        <b>类别权重生成结果:</b><br>
        <b>数据集不平衡程度:</b> {imbalance_level} (比率: {imbalance:.2f})<br>
        <b>推荐权重策略:</b> <span style="color: red; font-weight: bold;">{recommended_strategy}</span><br><br>
        
        <b>权重策略说明:</b><br>
        • <b>Balanced</b>: sklearn自动平衡权重，适合大多数情况<br>
        • <b>Inverse</b>: 逆频率权重，样本少的类别权重高<br>
        • <b>Log_Inverse</b>: 对数逆频率，适合极度不平衡的数据<br>
        • <b>Normalized</b>: 归一化权重，相对温和的权重调整<br><br>
        
        <b>使用建议:</b><br>
        • 不平衡度 < 3: 可以不使用权重或使用Normalized<br>
        • 不平衡度 3-10: 推荐使用Balanced或Inverse<br>
        • 不平衡度 > 10: 推荐使用Log_Inverse<br><br>
        
        <b>点击下方按钮导出权重配置到文件</b>
        """
        if self.metrics_info_label:
            self.metrics_info_label.setText(metrics_info)
            
        self.result_updated.emit() 