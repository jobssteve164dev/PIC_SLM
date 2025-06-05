import numpy as np
import logging
import matplotlib


class ChartRenderer:
    """训练指标图表渲染器"""
    
    def __init__(self, figure, loss_ax, acc_ax):
        """
        初始化图表渲染器
        
        Args:
            figure: matplotlib图形对象
            loss_ax: 损失子图
            acc_ax: 准确率/mAP子图
        """
        self.figure = figure
        self.loss_ax = loss_ax
        self.acc_ax = acc_ax
        self.logger = logging.getLogger(__name__)
        
    def get_x_axis_config(self, epochs):
        """获取x轴配置"""
        if not epochs:
            return 1, np.array([0, 1])
        
        # 确保坐标轴足够宽以显示所有点
        max_epoch = max(epochs)
        # 添加一点额外空间在右侧
        max_epoch = max_epoch + max(1, max_epoch * 0.1)
        
        # 确保x轴刻度合理
        if max_epoch <= 10:
            x_ticks = np.arange(0, max_epoch + 1, 1)  # 少于10个epoch时每个都显示
        else:
            x_ticks = np.arange(0, max_epoch + 1, max(1, int(max_epoch / 10)))  # 否则大约显示10个刻度
        
        return max_epoch, x_ticks
        
    def calculate_loss_range(self, all_losses, historical_ranges):
        """计算损失的y轴范围"""
        if not all_losses:
            return 0, 1
            
        # 更新历史最大最小值
        current_min = min(all_losses)
        current_max = max(all_losses)
        
        # 使用历史记录设置y轴范围
        loss_min_history = min(historical_ranges['loss_min'], current_min)
        loss_max_history = max(historical_ranges['loss_max'], current_max)
        
        y_min = max(0, loss_min_history * 0.9)
        y_max = loss_max_history * 1.1
        
        # 处理极端值情况
        if len(all_losses) > 3:
            # 计算标准差用于检测异常值
            mean_loss = sum(all_losses) / len(all_losses)
            std_loss = (sum((x - mean_loss) ** 2 for x in all_losses) / len(all_losses)) ** 0.5
            
            # 如果标准差很大，说明数据波动剧烈，需要特殊处理
            if std_loss > mean_loss * 0.5:
                # 使用百分位数设置范围，排除极端值的影响
                sorted_losses = sorted(all_losses)
                lower_bound = sorted_losses[int(len(sorted_losses) * 0.05)]  # 5%分位数
                upper_bound = sorted_losses[int(len(sorted_losses) * 0.95)]  # 95%分位数
                
                # 但仍然确保能显示所有点，只是缩小主要显示区域
                y_min = max(0, lower_bound * 0.9)
                y_max = upper_bound * 1.1
                
                # 如果有更极端的值，记录但不影响主显示区域
                if current_max > upper_bound * 1.5:
                    return y_min, y_max, current_max  # 返回极值用于标注
        
        return y_min, y_max
        
    def calculate_accuracy_range(self, all_metrics, historical_ranges):
        """计算准确率/mAP的y轴范围"""
        if not all_metrics:
            return 0, 1.0
            
        # 获取当前数据的范围
        current_min = min(all_metrics)
        current_max = max(all_metrics)
        
        # 更新历史记录，确保包含所有历史数据
        acc_min_history = min(historical_ranges['acc_min'], current_min) if historical_ranges['acc_min'] != float('inf') else current_min
        acc_max_history = max(historical_ranges['acc_max'], current_max)
        
        # 设置y轴范围，确保所有历史数据可见
        y_min = max(0, acc_min_history * 0.9)
        y_max = min(1.05, acc_max_history * 1.1)
        
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
        
        return y_min, y_max
        
    def calculate_general_range(self, values, is_percentage=True):
        """计算一般指标的y轴范围"""
        if not values:
            return 0, 1.0 if is_percentage else 10
            
        # 过滤掉负值或极小值
        filtered_values = [v for v in values if v > 0.0001]
        if not filtered_values:
            return 0, 1.0 if is_percentage else 10
            
        y_min = min(filtered_values)
        y_max = max(values)
        
        # 确保y_min和y_max有足够的差距，防止曲线太平
        if abs(y_max - y_min) < 0.05:
            y_margin = 0.05
        else:
            # 使用百分比计算边距
            y_margin = max(y_max * 0.1, 0.02)
        
        # 确保上下界不会太接近
        y_min = max(0, y_min - y_margin)
        
        if is_percentage:
            # 如果y_max已经接近1或超过1，设置上限为1.05
            if y_max >= 0.95:
                y_max = 1.05
            else:
                y_max = min(1.0, y_max + y_margin)
            
            # 如果y_max太小，扩大范围使曲线可见
            if y_max < 0.2:
                y_max = min(1.0, y_max * 1.5)
        else:
            y_max += y_margin
            
        return y_min, y_max
        
    def render_overview_chart(self, data, task_type, historical_ranges):
        """渲染损失和准确率总览图表"""
        epochs = data['epochs']
        train_losses = data['train_losses']
        val_losses = data['val_losses']
        maps = data['maps']
        train_maps = data['train_maps']
        
        max_epoch, x_ticks = self.get_x_axis_config(epochs)
        
        # 上方图显示损失
        self.loss_ax.set_title('训练和验证损失')
        self.loss_ax.set_xlabel('训练轮次')
        self.loss_ax.set_ylabel('损失值')
        self.loss_ax.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制损失数据
        if train_losses:
            self.loss_ax.plot(epochs, train_losses, 'b-', label='训练损失', marker='o', markersize=4)
        
        if val_losses:
            self.loss_ax.plot(epochs, val_losses, 'r-', label='验证损失', marker='o', markersize=4)
            
            # 计算损失y轴范围
            all_losses = []
            if train_losses:
                all_losses.extend([l for l in train_losses if l is not None and l > 0])
            if val_losses:
                all_losses.extend([l for l in val_losses if l is not None and l > 0])
            
            if all_losses:
                range_result = self.calculate_loss_range(all_losses, historical_ranges)
                if len(range_result) == 3:  # 有极值需要标注
                    y_min, y_max, extreme_max = range_result
                    self.loss_ax.text(
                        max(epochs), y_max * 0.95, 
                        f"最大值: {extreme_max:.4f}", 
                        ha='right', va='bottom', fontsize=8, color='red'
                    )
                else:
                    y_min, y_max = range_result
                
                self.loss_ax.set_ylim([y_min, y_max])
        
        self.loss_ax.legend(loc='upper right')
        self.loss_ax.set_xticks(x_ticks)
        self.loss_ax.set_visible(True)
        
        # 下方图显示准确率/mAP
        if any(m > 0 for m in maps):
            # 根据任务类型设置图表
            if task_type == 'classification':
                self.acc_ax.set_title('训练和验证准确率')
                self.acc_ax.set_ylabel('准确率')
                val_metric_label = '验证准确率'
                train_metric_label = '训练准确率'
            else:  # 默认为detection或未设置
                self.acc_ax.set_title('训练和验证mAP')
                self.acc_ax.set_ylabel('mAP')
                val_metric_label = '验证mAP'
                train_metric_label = '训练mAP'
            
            self.acc_ax.set_xlabel('训练轮次')
            self.acc_ax.grid(True, linestyle='--', alpha=0.7)
            
            # 绘制验证性能指标数据
            if maps:
                self.acc_ax.plot(epochs, maps, 'g-', label=val_metric_label, marker='o', markersize=4)
            
            # 绘制训练性能指标数据
            if train_maps and any(m > 0 for m in train_maps):
                self.acc_ax.plot(epochs, train_maps, 'c-', label=train_metric_label, marker='o', markersize=4)
            
            # 计算准确率/mAP y轴范围
            all_metrics = []
            if maps:
                all_metrics.extend([m for m in maps if m is not None and m > 0])
            if train_maps:
                all_metrics.extend([m for m in train_maps if m is not None and m > 0])
            
            if all_metrics:
                y_min, y_max = self.calculate_accuracy_range(all_metrics, historical_ranges)
                self.acc_ax.set_ylim([y_min, y_max])
            else:
                self.acc_ax.set_ylim([0, 1.0])
            
            self.acc_ax.legend(loc='lower right')
            self.acc_ax.set_xticks(x_ticks)
            self.acc_ax.set_visible(True)
        else:
            self.acc_ax.set_visible(False)
    
    def render_loss_only_chart(self, data, historical_ranges):
        """渲染仅损失曲线图表"""
        epochs = data['epochs']
        train_losses = data['train_losses']
        val_losses = data['val_losses']
        
        max_epoch, x_ticks = self.get_x_axis_config(epochs)
        
        self.loss_ax.set_title('训练和验证损失')
        self.loss_ax.set_xlabel('训练轮次')
        self.loss_ax.set_ylabel('损失值')
        self.loss_ax.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制损失数据
        if train_losses:
            self.loss_ax.plot(epochs, train_losses, 'b-', label='训练损失', marker='o', markersize=4)
        
        if val_losses:
            self.loss_ax.plot(epochs, val_losses, 'r-', label='验证损失', marker='o', markersize=4)
            
            # 计算损失y轴范围
            all_losses = []
            if train_losses:
                all_losses.extend([l for l in train_losses if l is not None and l > 0])
            if val_losses:
                all_losses.extend([l for l in val_losses if l is not None and l > 0])
            
            if all_losses:
                range_result = self.calculate_loss_range(all_losses, historical_ranges)
                if len(range_result) == 3:  # 有极值需要标注
                    y_min, y_max, extreme_max = range_result
                    self.loss_ax.text(
                        max(epochs), y_max * 0.95, 
                        f"最大值: {extreme_max:.4f}", 
                        ha='right', va='bottom', fontsize=8, color='red'
                    )
                else:
                    y_min, y_max = range_result
                
                self.loss_ax.set_ylim([y_min, y_max])
        
        self.loss_ax.legend(loc='upper right')
        self.loss_ax.set_xticks(x_ticks)
        self.loss_ax.set_visible(True)
        self.acc_ax.set_visible(False)
        
    def render_accuracy_only_chart(self, data, task_type):
        """渲染仅准确率图表"""
        epochs = data['epochs']
        maps = data['maps']
        train_maps = data['train_maps']
        
        max_epoch, x_ticks = self.get_x_axis_config(epochs)
        
        self.loss_ax.set_visible(False)
        
        if task_type == 'classification':
            self.acc_ax.set_title('训练和验证准确率')
            self.acc_ax.set_xlabel('训练轮次')
            self.acc_ax.set_ylabel('准确率')
            self.acc_ax.grid(True, linestyle='--', alpha=0.7)
            
            # 绘制验证准确率数据
            if maps:
                self.acc_ax.plot(epochs, maps, 'g-', label='验证准确率', marker='o', markersize=4)
            
            # 绘制训练准确率数据
            if train_maps and any(m > 0 for m in train_maps):
                self.acc_ax.plot(epochs, train_maps, 'c-', label='训练准确率', marker='o', markersize=4)
            
            # 设置y轴范围
            all_accuracies = maps.copy()
            if train_maps:
                all_accuracies.extend(train_maps)
            
            if all_accuracies:
                y_min, y_max = self.calculate_general_range(all_accuracies)
                self.acc_ax.set_ylim([y_min, y_max])
            
            self.acc_ax.legend(loc='lower right')
            self.acc_ax.set_xticks(x_ticks)
            self.acc_ax.set_visible(True)
        else:
            # 如果不是分类任务，显示提示信息
            self.acc_ax.text(0.5, 0.5, '当前模型不是分类模型', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=self.acc_ax.transAxes, fontsize=14)
            self.acc_ax.set_visible(True)
            
    def render_map_only_chart(self, data, task_type):
        """渲染仅mAP图表"""
        epochs = data['epochs']
        maps = data['maps']
        train_maps = data['train_maps']
        
        max_epoch, x_ticks = self.get_x_axis_config(epochs)
        
        self.loss_ax.set_visible(False)
        
        if task_type == 'detection' or task_type is None:
            self.acc_ax.set_title('训练和验证mAP')
            self.acc_ax.set_xlabel('训练轮次')
            self.acc_ax.set_ylabel('mAP')
            self.acc_ax.grid(True, linestyle='--', alpha=0.7)
            
            # 绘制验证mAP数据
            if maps:
                self.acc_ax.plot(epochs, maps, 'g-', label='验证mAP', marker='o', markersize=4)
            
            # 绘制训练mAP数据
            if train_maps and any(m > 0 for m in train_maps):
                self.acc_ax.plot(epochs, train_maps, 'c-', label='训练mAP', marker='o', markersize=4)
            
            # 设置y轴范围
            all_maps = []
            if maps:
                all_maps.extend(maps)
            if train_maps:
                all_maps.extend(train_maps)
            
            if all_maps:
                y_min, y_max = self.calculate_general_range(all_maps)
                self.acc_ax.set_ylim([y_min, y_max])
            
            self.acc_ax.legend(loc='lower right')
            self.acc_ax.set_xticks(x_ticks)
            self.acc_ax.set_visible(True)
        else:
            # 如果不是检测任务，显示提示信息
            self.acc_ax.text(0.5, 0.5, '当前模型不是检测模型', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=self.acc_ax.transAxes, fontsize=14)
            self.acc_ax.set_visible(True)
            
    def render_precision_recall_chart(self, data):
        """渲染精确率/召回率图表"""
        epochs = data['epochs']
        precisions = data['precisions']
        recalls = data['recalls']
        
        max_epoch, x_ticks = self.get_x_axis_config(epochs)
        
        self.loss_ax.set_visible(False)
        
        self.acc_ax.set_title('精确率和召回率')
        self.acc_ax.set_xlabel('训练轮次')
        self.acc_ax.set_ylabel('值')
        self.acc_ax.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制Precision数据
        if precisions:
            self.acc_ax.plot(epochs, precisions, 'g-', label='精确率', marker='o', markersize=4)
        
        # 绘制Recall数据
        if recalls:
            self.acc_ax.plot(epochs, recalls, 'm-', label='召回率', marker='o', markersize=4)
        
        # 设置y轴范围
        all_values = []
        if precisions:
            all_values.extend(precisions)
        if recalls:
            all_values.extend(recalls)
        
        if all_values:
            y_min, y_max = self.calculate_general_range(all_values)
            self.acc_ax.set_ylim([y_min, y_max])
        
        self.acc_ax.legend(loc='lower right')
        self.acc_ax.set_xticks(x_ticks)
        self.acc_ax.set_visible(True)
        
    def render_single_metric_chart(self, data, metric_key, title, ylabel, label, color='c-'):
        """渲染单一指标图表"""
        epochs = data['epochs']
        values = data[metric_key]
        
        max_epoch, x_ticks = self.get_x_axis_config(epochs)
        
        self.loss_ax.set_visible(False)
        
        self.acc_ax.set_title(title)
        self.acc_ax.set_xlabel('训练轮次')
        self.acc_ax.set_ylabel(ylabel)
        self.acc_ax.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制数据
        if values:
            self.acc_ax.plot(epochs, values, color, label=label, marker='o', markersize=4)
        
        # 设置y轴范围
        if values:
            y_min, y_max = self.calculate_general_range(values)
            self.acc_ax.set_ylim([y_min, y_max])
        
        self.acc_ax.legend(loc='lower right')
        self.acc_ax.set_xticks(x_ticks)
        self.acc_ax.set_visible(True)
        
    def render_dual_metric_chart(self, data, metric1_key, metric2_key, title, ylabel, label1, label2, color1='r-', color2='b-'):
        """渲染双指标图表"""
        epochs = data['epochs']
        values1 = data[metric1_key]
        values2 = data[metric2_key]
        
        max_epoch, x_ticks = self.get_x_axis_config(epochs)
        
        self.loss_ax.set_visible(False)
        
        self.acc_ax.set_title(title)
        self.acc_ax.set_xlabel('训练轮次')
        self.acc_ax.set_ylabel(ylabel)
        self.acc_ax.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制数据
        if values1:
            self.acc_ax.plot(epochs, values1, color1, label=label1, marker='o', markersize=4)
        
        if values2:
            self.acc_ax.plot(epochs, values2, color2, label=label2, marker='o', markersize=4)
        
        # 设置y轴范围
        all_values = []
        if values1:
            all_values.extend(values1)
        if values2:
            all_values.extend(values2)
        
        if all_values:
            y_min, y_max = self.calculate_general_range(all_values)
            self.acc_ax.set_ylim([y_min, y_max])
        
        self.acc_ax.legend(loc='lower right')
        self.acc_ax.set_xticks(x_ticks)
        self.acc_ax.set_visible(True)
        
    def render_loss_metrics_chart(self, data, metric1_key, metric2_key, title, label1, label2, color1='m-', color2='c-'):
        """渲染损失类指标图表（显示在上方图）"""
        epochs = data['epochs']
        values1 = data[metric1_key]
        values2 = data[metric2_key]
        
        max_epoch, x_ticks = self.get_x_axis_config(epochs)
        
        self.loss_ax.set_title(title)
        self.loss_ax.set_xlabel('训练轮次')
        self.loss_ax.set_ylabel('损失值')
        self.loss_ax.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制数据
        if values1:
            self.loss_ax.plot(epochs, values1, color1, label=label1, marker='o', markersize=4)
        
        if values2:
            self.loss_ax.plot(epochs, values2, color2, label=label2, marker='o', markersize=4)
        
        # 设置y轴范围（损失类指标，不使用百分比范围）
        all_values = []
        if values1:
            all_values.extend(values1)
        if values2:
            all_values.extend(values2)
        
        if all_values:
            # 过滤异常值，但保留最大边界
            all_max = max(all_values)
            filtered_values = [v for v in all_values if v > 0 and v < min(10, all_max * 1.5)]
            if filtered_values:
                y_min = min(filtered_values)
                y_max = max(filtered_values)
                
                # 确保y轴范围合理
                y_margin = max(y_max * 0.15, 0.1)
                y_min = max(0, y_min - y_margin)
                
                # 如果最大值过大，但确保能完整显示
                if all_max > y_max * 1.5:
                    y_max = all_max * 1.1
                else:
                    y_max += y_margin
                
                self.loss_ax.set_ylim([y_min, y_max])
        
        self.loss_ax.legend(loc='upper right')
        self.loss_ax.set_xticks(x_ticks)
        self.loss_ax.set_visible(True)
        
        # 隐藏下方图表（如果需要）
        self.acc_ax.set_visible(False)
        
    def finalize_layout(self):
        """完成布局调整"""
        # 重新调整布局和绘制
        self.figure.tight_layout(pad=2.0)
        self.figure.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
        
    def clear_charts(self):
        """清空图表"""
        self.loss_ax.clear()
        self.acc_ax.clear() 