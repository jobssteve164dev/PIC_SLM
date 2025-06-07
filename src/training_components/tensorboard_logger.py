"""
TensorBoard日志记录器 - 负责记录训练过程中的各种指标和可视化信息

主要功能：
- 记录训练和验证的损失、准确率
- 记录类别分布和权重信息
- 记录模型结构图
- 记录样本图像
- 记录混淆矩阵
"""

import os
import io
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from PyQt5.QtCore import QObject, pyqtSignal


class TensorBoardLogger(QObject):
    """TensorBoard日志记录器"""
    
    # 信号定义
    status_updated = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.writer = None
        self.log_dir = None
    
    def initialize(self, config, model_name):
        """
        初始化TensorBoard记录器
        
        Args:
            config: 训练配置
            model_name: 模型名称
            
        Returns:
            str: TensorBoard日志目录路径
        """
        if not config.get('use_tensorboard', True):
            return None
        
        # 获取TensorBoard日志目录
        model_save_dir = config.get('model_save_dir', 'models/saved_models')
        tensorboard_dir = config.get('tensorboard_log_dir', 
                                   os.path.join(model_save_dir, 'tensorboard_logs'))
        
        # 创建带有时间戳的唯一运行目录
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_run_name = f"{model_name}_{timestamp}"
        self.log_dir = os.path.join(tensorboard_dir, model_run_name)
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        return self.log_dir
    
    def log_model_graph(self, model, dataloader, device):
        """
        记录模型结构图
        
        Args:
            model: PyTorch模型
            dataloader: 数据加载器
            device: 计算设备
        """
        if not self.writer:
            return
        
        try:
            # 获取一个批次的样本数据用于记录模型图
            sample_inputs, _ = next(iter(dataloader))
            sample_inputs = sample_inputs.to(device)
            self.writer.add_graph(model, sample_inputs)
        except Exception as e:
            print(f"记录模型图时出错: {str(e)}")
    
    def log_class_info(self, class_names, class_distribution, class_weights, epoch=0):
        """
        记录类别分布和权重信息
        
        Args:
            class_names: 类别名称列表
            class_distribution: 类别分布字典
            class_weights: 类别权重张量
            epoch: 当前epoch
        """
        if not self.writer or not class_distribution or class_weights is None:
            return
        
        try:
            # 创建类别信息图表
            self._create_class_info_chart(class_names, class_distribution, class_weights, epoch)
            
            # 记录权重和分布的数值到标量
            weights = class_weights.cpu().numpy()
            for i, (name, weight) in enumerate(zip(class_names, weights)):
                self.writer.add_scalar(f'Class_Weights/{name}', weight, epoch)
                self.writer.add_scalar(f'Class_Distribution/{name}', 
                                     class_distribution[name], epoch)
        except Exception as e:
            print(f"记录类别信息到TensorBoard时出错: {str(e)}")
    
    def log_epoch_metrics(self, epoch, phase, loss, accuracy):
        """
        记录每个epoch的指标
        
        Args:
            epoch: epoch编号
            phase: 训练阶段（train/val）
            loss: 损失值
            accuracy: 准确率
        """
        if not self.writer:
            return
        
        self.writer.add_scalar(f'Loss/{phase}', loss, epoch)
        self.writer.add_scalar(f'Accuracy/{phase}', accuracy, epoch)
    
    def log_sample_images(self, dataloader, epoch, max_images=8):
        """
        记录样本图像
        
        Args:
            dataloader: 数据加载器
            epoch: 当前epoch
            max_images: 最大图像数量
        """
        if not self.writer:
            return
        
        try:
            # 获取一个批次的样本数据
            sample_inputs, sample_labels = next(iter(dataloader))
            
            # 记录样本图像
            grid = torchvision.utils.make_grid(sample_inputs[:max_images])
            self.writer.add_image('Sample Images', grid, epoch)
        except Exception as e:
            print(f"记录样本图像时出错: {str(e)}")
    
    def log_confusion_matrix(self, all_labels, all_preds, class_names, epoch):
        """
        记录混淆矩阵
        
        Args:
            all_labels: 真实标签列表
            all_preds: 预测标签列表
            class_names: 类别名称列表
            epoch: 当前epoch
        """
        if not self.writer:
            return
        
        try:
            from sklearn.metrics import confusion_matrix
            
            # 确保使用非交互式后端
            plt.switch_backend('Agg')
            plt.ioff()  # 关闭交互模式
            
            # 计算混淆矩阵
            cm = confusion_matrix(all_labels, all_preds)
            
            # 创建混淆矩阵图表
            confusion_matrix_image = self._create_confusion_matrix_chart(cm, class_names)
            
            # 添加到TensorBoard
            self.writer.add_image('Confusion Matrix', confusion_matrix_image, epoch)
            
        except Exception as e:
            print(f"记录混淆矩阵时出错: {str(e)}")
    
    def flush(self):
        """刷新数据到磁盘"""
        if self.writer:
            self.writer.flush()
    
    def close(self):
        """关闭记录器"""
        if self.writer:
            self.writer.flush()
            self.writer.close()
            self.writer = None
    
    def _create_class_info_chart(self, class_names, class_distribution, class_weights, epoch):
        """创建类别信息图表"""
        plt.figure(figsize=(12, 6))
        
        # 子图1: 类别分布
        plt.subplot(1, 2, 1)
        counts = [class_distribution[name] for name in class_names]
        plt.bar(range(len(class_names)), counts)
        plt.title('类别样本分布')
        plt.xlabel('类别')
        plt.ylabel('样本数量')
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        
        # 在柱状图上添加数值
        for i, count in enumerate(counts):
            plt.text(i, count + max(counts) * 0.01, str(count), 
                    ha='center', va='bottom')
        
        # 子图2: 类别权重
        plt.subplot(1, 2, 2)
        weights = class_weights.cpu().numpy()
        plt.bar(range(len(class_names)), weights, color='orange')
        plt.title('类别权重分布')
        plt.xlabel('类别')
        plt.ylabel('权重值')
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        
        # 在柱状图上添加数值
        for i, weight in enumerate(weights):
            plt.text(i, weight + max(weights) * 0.01, f'{weight:.3f}', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 转换为张量并添加到TensorBoard
        img_tensor = self._plt_to_tensor()
        self.writer.add_image('Class Distribution and Weights', img_tensor, epoch)
        
        plt.close()
    
    def _create_confusion_matrix_chart(self, cm, class_names):
        """创建混淆矩阵图表"""
        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # 在格子中添加数值
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # 转换为张量
        img_tensor = self._plt_to_tensor()
        plt.close()
        
        return img_tensor
    
    def _plt_to_tensor(self):
        """将matplotlib图形转换为张量"""
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = transforms.ToTensor()(img)
        return img_tensor
    
    def get_log_dir(self):
        """获取日志目录"""
        return self.log_dir 