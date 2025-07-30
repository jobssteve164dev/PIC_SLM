"""
TensorBoard日志记录器 - 负责记录训练过程中的各种指标和可视化信息

主要功能：
- 记录训练和验证的损失、准确率
- 记录类别分布和权重信息
- 记录模型结构图
- 记录样本图像
- 记录混淆矩阵
- 记录模型权重和梯度统计
- 记录性能和资源使用情况
- 记录高级评估指标
- 支持实时数据流输出到API服务器
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
import psutil  # 用于系统资源监控
from .real_time_metrics_collector import get_global_metrics_collector

# 导入matplotlib配置和标准化函数
from src.utils.matplotlib_config import (
    normalize_image_for_matplotlib, 
    normalize_feature_map, 
    safe_imshow,
    suppress_matplotlib_warnings
)

# 抑制matplotlib警告
suppress_matplotlib_warnings()


class TensorBoardLogger(QObject):
    """增强版TensorBoard日志记录器，支持实时数据流"""
    
    # 信号定义
    status_updated = pyqtSignal(str)
    # 新增：实时指标数据流信号
    metrics_stream = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.writer = None
        self.log_dir = None
        self.start_time = None
        self.last_log_time = None
        
        # 数据流相关
        self.stream_server = None
        self.enable_streaming = False
        self.current_metrics = {}
        
        # 实时指标采集器
        self.metrics_collector = get_global_metrics_collector()
    
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
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # 启动实时指标采集
        training_session_id = f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.metrics_collector.start_collection(training_session_id)
        
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
    
    def log_hyperparameters(self, config, final_metrics):
        """
        记录超参数和最终指标
        
        Args:
            config: 训练配置字典
            final_metrics: 最终训练指标字典
        """
        if not self.writer:
            return
            
        try:
            # 提取关键超参数（包含第二阶段新参数）
            hparams = {
                # 基础训练参数
                'learning_rate': config.get('learning_rate', 0.001),
                'batch_size': config.get('batch_size', 32),
                'optimizer': config.get('optimizer', 'Adam'),
                'weight_decay': config.get('weight_decay', 0.0001),
                'dropout_rate': config.get('dropout_rate', 0.0),
                'model_name': config.get('model_name', 'Unknown'),
                'use_pretrained': config.get('use_pretrained', False),
                'early_stopping_patience': config.get('early_stopping_patience', 10),
                
                # 第一阶段高级参数
                'beta1': config.get('beta1', 0.9),
                'beta2': config.get('beta2', 0.999),
                'momentum': config.get('momentum', 0.9),
                'nesterov': config.get('nesterov', False),
                'warmup_steps': config.get('warmup_steps', 0),
                'warmup_ratio': config.get('warmup_ratio', 0.0),
                'warmup_method': config.get('warmup_method', 'linear'),
                'min_lr': config.get('min_lr', 1e-6),
                'label_smoothing': config.get('label_smoothing', 0.0),
                
                # 第二阶段高级参数
                'model_ema': config.get('model_ema', False),
                'model_ema_decay': config.get('model_ema_decay', 0.9999),
                'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 1),
                'cutmix_prob': config.get('cutmix_prob', 0.0),
                'mixup_alpha': config.get('mixup_alpha', 0.0),
                'loss_scale': config.get('loss_scale', 'dynamic'),
                'static_loss_scale': config.get('static_loss_scale', 128.0),
            }
            
            # 记录超参数和指标
            self.writer.add_hparams(hparams, final_metrics)
            
        except Exception as e:
            print(f"记录超参数时出错: {str(e)}")
    
    def log_model_weights_and_gradients(self, model, epoch):
        """
        记录模型权重和梯度的直方图
        
        Args:
            model: PyTorch模型
            epoch: 当前epoch
        """
        if not self.writer:
            return
            
        try:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # 记录权重分布
                    self.writer.add_histogram(f'Weights/{name}', param.data, epoch)
                    
                    # 记录梯度分布（如果存在）
                    if param.grad is not None:
                        self.writer.add_histogram(f'Gradients/{name}', param.grad.data, epoch)
                        
                        # 记录梯度范数
                        grad_norm = param.grad.data.norm(2).item()
                        self.writer.add_scalar(f'Gradient_Norms/{name}', grad_norm, epoch)
            
            # 记录总梯度范数
            total_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.writer.add_scalar('Gradients/total_norm', total_norm, epoch)
            
        except Exception as e:
            print(f"记录权重和梯度时出错: {str(e)}")
    
    def log_performance_metrics(self, epoch, batch_idx=None, num_samples=None):
        """
        记录性能指标
        
        Args:
            epoch: 当前epoch
            batch_idx: 当前batch索引
            num_samples: 处理的样本数量
        """
        if not self.writer:
            return
            
        try:
            current_time = time.time()
            
            # 计算训练速度
            if self.last_log_time and num_samples:
                time_elapsed = current_time - self.last_log_time
                samples_per_second = num_samples / time_elapsed if time_elapsed > 0 else 0
                self.writer.add_scalar('Performance/samples_per_second', samples_per_second, epoch)
            
            # 记录总训练时间
            total_time = current_time - self.start_time
            self.writer.add_scalar('Performance/total_training_time', total_time, epoch)
            
            # 记录GPU内存使用（如果可用）
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                self.writer.add_scalar('Memory/gpu_memory_allocated_gb', memory_allocated, epoch)
                self.writer.add_scalar('Memory/gpu_memory_reserved_gb', memory_reserved, epoch)
                
                # GPU利用率（如果可以获取）
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.writer.add_scalar('Performance/gpu_utilization', util.gpu, epoch)
                except:
                    pass  # pynvml可能未安装
            
            # 记录系统资源使用
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            self.writer.add_scalar('System/cpu_usage_percent', cpu_percent, epoch)
            self.writer.add_scalar('System/memory_usage_percent', memory_percent, epoch)
            
            self.last_log_time = current_time
            
        except Exception as e:
            print(f"记录性能指标时出错: {str(e)}")
    
    def log_advanced_metrics(self, y_true, y_pred, y_scores=None, epoch=0, phase='val'):
        """
        记录高级评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_scores: 预测概率/分数（可选）
            epoch: 当前epoch
            phase: 训练阶段
        """
        if not self.writer:
            return
            
        try:
            from sklearn.metrics import (precision_score, recall_score, f1_score, 
                                       roc_auc_score, average_precision_score,
                                       balanced_accuracy_score)
            
            # 基础指标
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            balanced_acc = balanced_accuracy_score(y_true, y_pred)
            
            self.writer.add_scalar(f'Advanced_Metrics/{phase}_precision', precision, epoch)
            self.writer.add_scalar(f'Advanced_Metrics/{phase}_recall', recall, epoch)
            self.writer.add_scalar(f'Advanced_Metrics/{phase}_f1_score', f1, epoch)
            self.writer.add_scalar(f'Advanced_Metrics/{phase}_balanced_accuracy', balanced_acc, epoch)
            
            # 如果提供了概率分数，计算AUC相关指标
            if y_scores is not None:
                try:
                    # 多类别情况下的AUC
                    auc_roc = roc_auc_score(y_true, y_scores, multi_class='ovr', average='weighted')
                    self.writer.add_scalar(f'Advanced_Metrics/{phase}_auc_roc', auc_roc, epoch)
                    
                    # 平均精确率
                    avg_precision = average_precision_score(y_true, y_scores, average='weighted')
                    self.writer.add_scalar(f'Advanced_Metrics/{phase}_avg_precision', avg_precision, epoch)
                except:
                    pass  # 可能是二分类问题或其他情况
            
        except Exception as e:
            print(f"记录高级指标时出错: {str(e)}")
    
    def log_learning_rate_schedule(self, optimizer, epoch):
        """
        记录学习率调度
        
        Args:
            optimizer: 优化器
            epoch: 当前epoch
        """
        if not self.writer:
            return
            
        try:
            # 记录每个参数组的学习率
            for i, param_group in enumerate(optimizer.param_groups):
                lr = param_group['lr']
                self.writer.add_scalar(f'Learning_Rate/group_{i}', lr, epoch)
                
                # 如果有其他参数（如momentum, weight_decay），也记录
                if 'momentum' in param_group:
                    self.writer.add_scalar(f'Optimizer/momentum_group_{i}', 
                                         param_group['momentum'], epoch)
                if 'weight_decay' in param_group:
                    self.writer.add_scalar(f'Optimizer/weight_decay_group_{i}', 
                                         param_group['weight_decay'], epoch)
        except Exception as e:
            print(f"记录学习率时出错: {str(e)}")
    
    def log_loss_components(self, loss_dict, epoch, phase='train'):
        """
        记录损失函数的各个组成部分
        
        Args:
            loss_dict: 损失组件字典
            epoch: 当前epoch
            phase: 训练阶段
        """
        if not self.writer:
            return
            
        try:
            for loss_name, loss_value in loss_dict.items():
                if isinstance(loss_value, torch.Tensor):
                    loss_value = loss_value.item()
                self.writer.add_scalar(f'Loss_Components/{phase}_{loss_name}', loss_value, epoch)
        except Exception as e:
            print(f"记录损失组件时出错: {str(e)}")
    
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
        
        # 构建指标数据
        metrics_data = {
            'epoch': epoch,
            'phase': phase,
            'loss': float(loss) if hasattr(loss, 'item') else float(loss),
            'accuracy': float(accuracy) if hasattr(accuracy, 'item') else float(accuracy),
            'timestamp': time.time()
        }
        
        # 更新当前指标并发送到数据流
        self._update_current_metrics(metrics_data)
        
        # 非侵入式地复制数据到实时采集器
        self.metrics_collector.collect_tensorboard_metrics(epoch, phase, {
            'loss': metrics_data['loss'],
            'accuracy': metrics_data['accuracy']
        })
    
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
            
            # 确保图像数据在0-1范围内
            sample_inputs = torch.clamp(sample_inputs, 0, 1)
            
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
    
    def log_model_predictions(self, model, dataloader, class_names, epoch, device, max_images=4):
        """
        记录模型预测结果的可视化
        
        Args:
            model: 训练的模型
            dataloader: 数据加载器
            class_names: 类别名称列表
            epoch: 当前epoch
            device: 计算设备
            max_images: 最大图像数量
        """
        if not self.writer:
            return
            
        try:
            model.eval()
            with torch.no_grad():
                images, labels = next(iter(dataloader))
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                # 创建预测结果图表
                fig = self._create_predictions_chart(
                    images[:max_images], labels[:max_images], 
                    predicted[:max_images], outputs[:max_images], class_names
                )
                
                # 转换为张量并添加到TensorBoard
                img_tensor = self._plt_to_tensor()
                self.writer.add_image('Model Predictions', img_tensor, epoch)
                
                plt.close(fig)
            model.train()
            
        except Exception as e:
            print(f"记录模型预测时出错: {str(e)}")
    
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
            
        # 停止实时指标采集
        self.metrics_collector.stop_collection()
    
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
        
        # 创建混淆矩阵图像并保存mappable对象
        im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        
        # 添加colorbar（使用保存的mappable对象）
        plt.colorbar(im)
        
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
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        # 转换为张量
        img_tensor = self._plt_to_tensor()
        plt.close()
        
        return img_tensor
    
    def _create_predictions_chart(self, images, true_labels, predicted_labels, outputs, class_names):
        """创建预测结果图表"""
        fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
        if len(images) == 1:
            axes = [axes]
            
        for i, (img, true_label, pred_label, output) in enumerate(
            zip(images, true_labels, predicted_labels, outputs)):
            
            # 将图像转换为可显示格式
            img = img.cpu()
            if img.shape[0] == 1:  # 灰度图像
                img = img.squeeze(0)
                # 标准化灰度图像数据到0-1范围
                img = torch.clamp(img, 0, 1)
                axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
            else:  # 彩色图像
                img = img.permute(1, 2, 0)
                # 标准化彩色图像数据到0-1范围
                img = torch.clamp(img, 0, 1)
                axes[i].imshow(img, vmin=0, vmax=1)
            
            # 获取预测概率
            probs = torch.nn.functional.softmax(output, dim=0)
            confidence = probs[pred_label].item()
            
            # 设置标题
            true_class = class_names[true_label.item()]
            pred_class = class_names[pred_label.item()]
            color = 'green' if true_label == pred_label else 'red'
            
            axes[i].set_title(f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}', 
                            color=color, fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def _plt_to_tensor(self):
        """将matplotlib图表转换为张量"""
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = transforms.ToTensor()(img)
        buf.close()
        return img_tensor
    
    def get_log_dir(self):
        """获取日志目录"""
        return self.log_dir 
    
    def set_stream_server(self, stream_server):
        """设置数据流服务器"""
        self.stream_server = stream_server
        self.enable_streaming = True
        
    def _update_current_metrics(self, metrics_update):
        """更新当前指标并发送到数据流"""
        # 更新当前指标缓存
        self.current_metrics.update(metrics_update)
        
        # 发送到数据流服务器
        if self.enable_streaming and self.stream_server:
            try:
                self.stream_server.broadcast_metrics(self.current_metrics.copy())
            except Exception as e:
                print(f"发送指标到数据流时出错: {str(e)}")
        
        # 发送信号
        self.metrics_stream.emit(self.current_metrics.copy())
    
    def _get_gpu_memory_info(self):
        """获取GPU内存信息"""
        if not torch.cuda.is_available():
            return {'gpu_memory_used': 0, 'gpu_memory_total': 0}
        
        try:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            return {
                'gpu_memory_used': round(memory_allocated, 2),
                'gpu_memory_total': round(memory_reserved, 2)
            }
        except:
            return {'gpu_memory_used': 0, 'gpu_memory_total': 0}
    
    def _get_system_info(self):
        """获取系统资源信息"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            return {
                'cpu_usage_percent': round(cpu_percent, 1),
                'memory_usage_percent': round(memory_info.percent, 1),
                'memory_available_gb': round(memory_info.available / 1024**3, 2)
            }
        except:
            return {
                'cpu_usage_percent': 0,
                'memory_usage_percent': 0,
                'memory_available_gb': 0
            }
    
    def log_comprehensive_metrics(self, epoch, phase, metrics_dict):
        """记录综合指标并发送到数据流"""
        if not self.writer:
            return
        
        # 记录到TensorBoard
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'{key}', value, epoch)
        
        # 准备数据流格式的指标
        stream_metrics = {
            'epoch': epoch,
            'phase': phase,
            'timestamp': time.time(),
            **metrics_dict,
            **self._get_gpu_memory_info(),
            **self._get_system_info()
        }
        
        # 发送到数据流
        self._update_current_metrics(stream_metrics) 