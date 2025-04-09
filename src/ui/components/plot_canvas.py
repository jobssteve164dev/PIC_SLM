import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QSizePolicy
import numpy as np
import logging

# 配置matplotlib使用中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['font.size'] = 12  # 设置字体大小

logger = logging.getLogger(__name__)

class PlotCanvas:
    """基础绘图组件，处理matplotlib图表的创建和更新"""
    
    def __init__(self, figsize=(12, 8), dpi=100):
        """
        初始化绘图组件
        
        Args:
            figsize: 图表尺寸元组 (宽, 高)
            dpi: 分辨率
        """
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.figure.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
        
        # 创建画布
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumWidth(600)
        
        # 创建子图
        self.loss_ax = self.figure.add_subplot(211)
        self.loss_ax.set_title('训练和验证损失')
        self.loss_ax.set_xlabel('训练轮次')
        self.loss_ax.set_ylabel('损失值')
        
        self.acc_ax = self.figure.add_subplot(212)
        self.acc_ax.set_title('训练和验证准确率/mAP')
        self.acc_ax.set_xlabel('训练轮次')
        self.acc_ax.set_ylabel('准确率/mAP')
        
        # 使用tight_layout但保留足够的边距
        self.figure.tight_layout(pad=2.0)
        
        # 历史极值记录，用于调整坐标轴范围
        self.loss_min_history = float('inf')
        self.loss_max_history = 0.0
        self.acc_min_history = float('inf')
        self.acc_max_history = 0.0
    
    def get_canvas(self):
        """获取FigureCanvas实例"""
        return self.canvas
    
    def clear_axes(self):
        """清除所有子图"""
        self.loss_ax.clear()
        self.acc_ax.clear()
        self.canvas.draw()
    
    def reset_axes_titles(self, loss_title='训练和验证损失', acc_title='训练和验证准确率/mAP'):
        """重设子图标题"""
        self.loss_ax.set_title(loss_title)
        self.loss_ax.set_xlabel('训练轮次')
        self.loss_ax.set_ylabel('损失值')
        
        self.acc_ax.set_title(acc_title)
        self.acc_ax.set_xlabel('训练轮次')
        self.acc_ax.set_ylabel('准确率/mAP')
    
    def resize_plot(self):
        """调整图表布局"""
        self.figure.tight_layout(pad=2.0)
        self.figure.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
        self.canvas.draw()
    
    def set_loss_axes_visible(self, visible=True):
        """设置损失子图的可见性"""
        self.loss_ax.set_visible(visible)
        self.canvas.draw()
    
    def set_acc_axes_visible(self, visible=True):
        """设置准确率/mAP子图的可见性"""
        self.acc_ax.set_visible(visible)
        self.canvas.draw()
    
    def generate_x_ticks(self, epochs):
        """生成合适的x轴刻度"""
        if not epochs:
            return np.arange(0, 2, 1)
            
        max_epoch = max(epochs)
        # 添加一点额外空间在右侧
        max_epoch = max_epoch + max(1, max_epoch * 0.1)
            
        # 确保x轴刻度合理
        if max_epoch <= 10:
            return np.arange(0, max_epoch + 1, 1)  # 少于10个epoch时每个都显示
        else:
            return np.arange(0, max_epoch + 1, max(1, int(max_epoch / 10)))  # 否则大约显示10个刻度

    def calculate_y_range(self, values, is_accuracy=False):
        """计算Y轴合适的显示范围"""
        if not values or all(v == 0 for v in values):
            return (0, 1.0) if is_accuracy else (0, 10.0)
            
        filtered_values = [v for v in values if v is not None and v > 0]
        if not filtered_values:
            return (0, 1.0) if is_accuracy else (0, 10.0)
            
        min_val = min(filtered_values)
        max_val = max(filtered_values)
            
        # 更新历史记录
        if is_accuracy:
            self.acc_min_history = min(self.acc_min_history, min_val)
            self.acc_max_history = max(self.acc_max_history, max_val)
            
            # 设置y轴范围，确保所有历史数据可见
            y_min = max(0, self.acc_min_history * 0.9)
            y_max = min(1.05, self.acc_max_history * 1.1)
            
            # 特殊情况处理
            if y_max < 0.1:  # 如果所有值都很小
                y_max = 0.2  # 设置更合理的上限
            elif y_min > 0.9:  # 如果所有值都很大
                y_min = 0.8  # 设置更合理的下限
            
            # 如果范围太小，扩大显示
            if y_max - y_min < 0.1:
                y_margin = (0.2 - (y_max - y_min)) / 2
                y_min = max(0, y_min - y_margin)
                y_max = min(1.05, y_max + y_margin)
        else:
            self.loss_min_history = min(self.loss_min_history, min_val)
            self.loss_max_history = max(self.loss_max_history, max_val)
            
            # 使用历史记录设置y轴范围
            y_min = max(0, self.loss_min_history * 0.9)
            y_max = self.loss_max_history * 1.1
            
            # 处理极端值情况
            if len(filtered_values) > 3:
                # 计算标准差用于检测异常值
                mean_val = sum(filtered_values) / len(filtered_values)
                std_val = (sum((x - mean_val) ** 2 for x in filtered_values) / len(filtered_values)) ** 0.5
                
                # 如果标准差很大，说明数据波动剧烈，需要特殊处理
                if std_val > mean_val * 0.5:
                    # 使用百分位数设置范围，排除极端值的影响
                    sorted_vals = sorted(filtered_values)
                    lower_bound = sorted_vals[int(len(sorted_vals) * 0.05)]  # 5%分位数
                    upper_bound = sorted_vals[int(len(sorted_vals) * 0.95)]  # 95%分位数
                    
                    # 但仍然确保能显示所有点，只是缩小主要显示区域
                    y_min = max(0, lower_bound * 0.9)
                    y_max = upper_bound * 1.1
            
        return (y_min, y_max)
    
    def reset_history(self):
        """重置历史极值记录"""
        self.loss_min_history = float('inf')
        self.loss_max_history = 0.0
        self.acc_min_history = float('inf')
        self.acc_max_history = 0.0 