"""
图表管理器

负责数据集评估结果的可视化展示，包括各种统计图表的绘制和管理
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QObject, pyqtSignal

# 导入matplotlib配置和标准化函数
from src.utils.matplotlib_config import suppress_matplotlib_warnings

# 抑制matplotlib警告
suppress_matplotlib_warnings()


class ChartManager(QObject):
    """图表管理器"""
    
    # 定义信号
    chart_updated = pyqtSignal()
    
    def __init__(self, figure=None, ax1=None, ax2=None):
        super().__init__()
        self.figure = figure
        self.ax1 = ax1
        self.ax2 = ax2
        
        # 设置中文字体和样式
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        plt.rcParams.update({'font.size': 8})  # 全局字体大小
        
    def setup_figure(self, canvas):
        """设置图表画布"""
        if canvas and hasattr(canvas, 'figure'):
            self.figure = canvas.figure
            if len(self.figure.axes) >= 2:
                self.ax1, self.ax2 = self.figure.axes[0], self.figure.axes[1]
                
    def clear_charts(self):
        """清空图表"""
        if self.ax1:
            self.ax1.clear()
        if self.ax2:
            self.ax2.clear()
            
    def plot_with_adjusted_font(self, ax, title, xlabel, ylabel, x_rotation=45):
        """使用调整过的字体大小绘制图表"""
        if not ax:
            return
            
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.tick_params(axis='x', rotation=x_rotation)
        
        # 自动调整刻度数量，避免拥挤
        if x_rotation > 0:
            ax.locator_params(axis='x', nbins=6)
            
    def plot_classification_distribution(self, train_classes, val_classes):
        """绘制分类数据集的类别分布"""
        self.clear_charts()
        
        if not self.ax1 or not self.ax2:
            return
            
        # 绘制训练集分布
        if train_classes:
            self.ax1.bar(train_classes.keys(), train_classes.values(), color='skyblue', alpha=0.7)
            self.plot_with_adjusted_font(self.ax1, "训练集类别分布", "类别", "样本数量")
            
        # 绘制验证集分布
        if val_classes:
            self.ax2.bar(val_classes.keys(), val_classes.values(), color='lightcoral', alpha=0.7)
            self.plot_with_adjusted_font(self.ax2, "验证集类别分布", "类别", "样本数量")
            
        self._finalize_plot()
        
    def plot_detection_distribution(self, train_annotations, val_annotations):
        """绘制目标检测数据集的标注分布"""
        self.clear_charts()
        
        if not self.ax1 or not self.ax2:
            return
            
        # 绘制训练集标注分布
        if train_annotations:
            self.ax1.bar(train_annotations.keys(), train_annotations.values(), color='skyblue', alpha=0.7)
            self.plot_with_adjusted_font(self.ax1, "训练集类别分布", "类别", "标注数量")
            
        # 绘制验证集标注分布
        if val_annotations:
            self.ax2.bar(val_annotations.keys(), val_annotations.values(), color='lightcoral', alpha=0.7)
            self.plot_with_adjusted_font(self.ax2, "验证集类别分布", "类别", "标注数量")
            
        self._finalize_plot()
        
    def plot_image_quality(self, image_sizes, image_qualities):
        """绘制图像质量分布"""
        self.clear_charts()
        
        if not self.ax1 or not self.ax2:
            return
            
        # 绘制图像尺寸分布
        if image_sizes:
            self.ax1.hist(image_sizes, bins=30, color='skyblue', alpha=0.7)
            self.plot_with_adjusted_font(self.ax1, "图像尺寸分布", "像素数量", "频次", x_rotation=0)
            
        # 绘制图像质量分布
        if image_qualities:
            self.ax2.hist(image_qualities, bins=30, color='lightgreen', alpha=0.7)
            self.plot_with_adjusted_font(self.ax2, "图像质量分布（清晰度）", "清晰度值", "频次", x_rotation=0)
            
        self._finalize_plot()
        
    def plot_annotation_quality(self, class_names, annotation_counts, annotation_sizes):
        """绘制标注质量分布"""
        self.clear_charts()
        
        if not self.ax1 or not self.ax2:
            return
            
        # 绘制标注数量分布
        if class_names and annotation_counts:
            x = np.arange(len(class_names))
            self.ax1.bar(x, annotation_counts, color='orange', alpha=0.7)
            
            # 如果类别数量多，减少显示的类别名称
            if len(class_names) > 6:
                indices = np.linspace(0, len(class_names)-1, 6, dtype=int)
                self.ax1.set_xticks(indices)
                self.ax1.set_xticklabels([class_names[i] for i in indices])
            else:
                self.ax1.set_xticks(x)
                self.ax1.set_xticklabels(class_names)
                
            self.plot_with_adjusted_font(self.ax1, "各类别标注数量分布", "类别", "标注数量")
            
        # 绘制标注文件大小分布
        if annotation_sizes:
            sizes_kb = np.array(annotation_sizes) / 1024  # 转换为KB
            self.ax2.hist(sizes_kb, bins=30, color='purple', alpha=0.7)
            self.plot_with_adjusted_font(self.ax2, "标注文件大小分布", "文件大小 (KB)", "频次", x_rotation=0)
            
        self._finalize_plot()
        
    def plot_feature_distribution(self, brightness_values, contrast_values):
        """绘制特征分布"""
        self.clear_charts()
        
        if not self.ax1 or not self.ax2:
            return
            
        # 绘制亮度分布
        if brightness_values:
            self.ax1.hist(brightness_values, bins=30, color='gold', alpha=0.7)
            self.plot_with_adjusted_font(self.ax1, "图像亮度分布", "亮度值", "频次", x_rotation=0)
            
        # 绘制对比度分布
        if contrast_values:
            self.ax2.hist(contrast_values, bins=30, color='lightblue', alpha=0.7)
            self.plot_with_adjusted_font(self.ax2, "图像对比度分布", "对比度值", "频次", x_rotation=0)
            
        self._finalize_plot()
        
    def plot_detection_annotation_quality(self, box_counts, box_sizes):
        """绘制目标检测标注质量"""
        self.clear_charts()
        
        if not self.ax1 or not self.ax2:
            return
            
        # 绘制每张图像的标注数量分布
        if box_counts:
            bins = min(30, len(set(box_counts))) if set(box_counts) else 10
            self.ax1.hist(box_counts, bins=bins, color='coral', alpha=0.7)
            self.plot_with_adjusted_font(self.ax1, "每张图像的标注数量分布", "标注数量", "图像数量", x_rotation=0)
            
        # 绘制边界框大小分布
        if box_sizes:
            self.ax2.hist(box_sizes, bins=30, color='lightgreen', alpha=0.7)
            self.plot_with_adjusted_font(self.ax2, "边界框大小分布", "边界框大小(相对面积)", "频次", x_rotation=0)
            
        self._finalize_plot()
        
    def plot_weight_distribution(self, class_names, class_counts, weight_values):
        """绘制权重分布图"""
        self.clear_charts()
        
        if not self.ax1 or not self.ax2 or not class_names:
            return
            
        # 绘制类别分布柱状图
        if class_counts:
            bars = self.ax1.bar(range(len(class_names)), class_counts, color='skyblue', alpha=0.7)
            self.ax1.set_xlabel('类别')
            self.ax1.set_ylabel('样本数量')
            self.ax1.set_title('训练集类别分布')
            self.ax1.set_xticks(range(len(class_names)))
            self.ax1.set_xticklabels(class_names, rotation=45, ha='right')
            
            # 在柱状图上添加数值标签
            total_samples = sum(class_counts)
            for bar, count in zip(bars, class_counts):
                height = bar.get_height()
                self.ax1.text(bar.get_x() + bar.get_width()/2., height + total_samples*0.01,
                             f'{count}', ha='center', va='bottom', fontsize=8)
                             
        # 绘制权重分布
        if weight_values:
            bars2 = self.ax2.bar(range(len(class_names)), weight_values, color='lightcoral', alpha=0.7)
            self.ax2.set_xlabel('类别')
            self.ax2.set_ylabel('权重值')
            self.ax2.set_title('权重分布')
            self.ax2.set_xticks(range(len(class_names)))
            self.ax2.set_xticklabels(class_names, rotation=45, ha='right')
            
            # 在权重图上添加数值标签
            for bar, weight in zip(bars2, weight_values):
                height = bar.get_height()
                self.ax2.text(bar.get_x() + bar.get_width()/2., height + max(weight_values)*0.01,
                             f'{weight:.2f}', ha='center', va='bottom', fontsize=8)
                             
        self._finalize_plot()
        
    def _finalize_plot(self):
        """完成图表绘制"""
        if self.figure:
            self.figure.tight_layout(pad=2.0)
            
        self.chart_updated.emit()
        
    def save_chart(self, file_path, dpi=300):
        """保存图表到文件
        
        Args:
            file_path (str): 保存路径
            dpi (int): 图像分辨率
        """
        if self.figure:
            try:
                self.figure.savefig(file_path, dpi=dpi, bbox_inches='tight')
                return True
            except Exception as e:
                print(f"保存图表失败: {str(e)}")
                return False
        return False
        
    def get_chart_summary(self):
        """获取图表摘要信息"""
        summary = {
            'has_figure': self.figure is not None,
            'axes_count': len(self.figure.axes) if self.figure else 0,
            'figure_size': self.figure.get_size_inches() if self.figure else None
        }
        
        return summary
        
    def set_style(self, style='default'):
        """设置图表样式
        
        Args:
            style (str): 样式名称，如 'default', 'seaborn', 'ggplot'
        """
        try:
            plt.style.use(style)
        except Exception as e:
            print(f"设置样式失败: {str(e)}")
            plt.style.use('default')  # 回退到默认样式 