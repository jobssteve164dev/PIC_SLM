import os
import time
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免线程问题
import matplotlib.pyplot as plt
import io
from PIL import Image
from torchvision import transforms
import sklearn.metrics as metrics

def create_tensorboard_writer(config):
    """创建TensorBoard的SummaryWriter实例
    
    Args:
        config: 配置字典，包含模型名称和保存目录等信息
        
    Returns:
        SummaryWriter实例
    """
    tensorboard_dir = config.get('tensorboard_log_dir')
    model_name = config.get('model_name', 'Unknown')
    
    if not tensorboard_dir:
        model_save_dir = config.get('model_save_dir', 'models/saved_models')
        tensorboard_dir = os.path.join(model_save_dir, 'tensorboard_logs')
    
    # 创建带有时间戳的唯一运行目录
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_run_name = f"{model_name}_{timestamp}"
    tensorboard_run_dir = os.path.join(tensorboard_dir, model_run_name)
    os.makedirs(tensorboard_run_dir, exist_ok=True)
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(tensorboard_run_dir)
    
    return writer, tensorboard_run_dir

def log_model_graph(writer, model, sample_input_size=(1, 3, 224, 224), device='cuda'):
    """记录模型结构图到TensorBoard
    
    Args:
        writer: TensorBoard的SummaryWriter实例
        model: PyTorch模型
        sample_input_size: 样本输入尺寸
        device: 运行设备
    """
    try:
        # 创建样本输入
        sample_input = torch.randn(sample_input_size).to(device)
        
        # 添加模型图到TensorBoard
        writer.add_graph(model, sample_input)
        writer.flush()  # 确保数据写入
    except Exception as e:
        print(f"记录模型图出错: {str(e)}")

def log_batch_stats(writer, phase, epoch, loss, accuracy, batch, total_batches):
    """记录批次训练统计数据到TensorBoard
    
    Args:
        writer: TensorBoard的SummaryWriter实例
        phase: 阶段（'train'或'val'）
        epoch: 当前轮次
        loss: 损失值
        accuracy: 准确率
        batch: 当前批次
        total_batches: 总批次数
    """
    # 计算全局步数
    global_step = epoch * total_batches + batch
    
    # 添加标量数据
    writer.add_scalar(f'BatchStats/{phase}_loss', loss, global_step)
    writer.add_scalar(f'BatchStats/{phase}_accuracy', accuracy, global_step)
    
    # 确保数据写入
    if batch % 10 == 0:
        writer.flush()

def log_epoch_stats(writer, phase, epoch, loss, accuracy):
    """记录每轮训练统计数据到TensorBoard
    
    Args:
        writer: TensorBoard的SummaryWriter实例
        phase: 阶段（'train'或'val'）
        epoch: 当前轮次
        loss: 损失值
        accuracy: 准确率
    """
    # 添加标量数据
    writer.add_scalar(f'Loss/{phase}', loss, epoch)
    writer.add_scalar(f'Accuracy/{phase}', accuracy, epoch)
    writer.flush()  # 确保数据写入

def log_sample_images(writer, dataloader, epoch, num_samples=8, device='cuda'):
    """记录样本图像到TensorBoard
    
    Args:
        writer: TensorBoard的SummaryWriter实例
        dataloader: 数据加载器
        epoch: 当前轮次
        num_samples: 记录的样本数量
        device: 运行设备
    """
    try:
        # 获取一个批次的样本数据
        sample_inputs, sample_labels = next(iter(dataloader))
        sample_inputs = sample_inputs.to(device)
        
        # 限制样本数量
        if sample_inputs.size(0) > num_samples:
            sample_inputs = sample_inputs[:num_samples]
        
        # 创建图像网格
        grid = torchvision.utils.make_grid(sample_inputs)
        
        # 添加图像到TensorBoard
        writer.add_image('Sample Images', grid, epoch)
        writer.flush()  # 确保数据写入
    except Exception as e:
        print(f"记录样本图像出错: {str(e)}")

def plot_confusion_matrix(all_labels, all_preds, class_names):
    """绘制混淆矩阵
    
    Args:
        all_labels: 所有真实标签
        all_preds: 所有预测标签
        class_names: 类名列表
        
    Returns:
        PIL图像对象
    """
    try:
        # 确保使用非交互式后端
        plt.switch_backend('Agg')
        plt.ioff()  # 关闭交互模式
        
        # 计算混淆矩阵
        cm = metrics.confusion_matrix(all_labels, all_preds)
        
        # 绘制混淆矩阵
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
        
        # 将图像转换为PIL图像
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        
        # 关闭图像以释放内存
        plt.close()
        
        return img
    except Exception as e:
        print(f"绘制混淆矩阵出错: {str(e)}")
        return None

def log_confusion_matrix(writer, all_labels, all_preds, class_names, epoch):
    """记录混淆矩阵到TensorBoard
    
    Args:
        writer: TensorBoard的SummaryWriter实例
        all_labels: 所有真实标签
        all_preds: 所有预测标签
        class_names: 类名列表
        epoch: 当前轮次
    """
    try:
        # 绘制混淆矩阵
        img = plot_confusion_matrix(all_labels, all_preds, class_names)
        
        if img:
            # 转换为tensor
            img_tensor = transforms.ToTensor()(img)
            
            # 添加到TensorBoard
            writer.add_image('Confusion Matrix', img_tensor, epoch)
            writer.flush()  # 确保数据写入
    except Exception as e:
        print(f"记录混淆矩阵出错: {str(e)}")

def log_learning_rate(writer, optimizer, epoch):
    """记录学习率到TensorBoard
    
    Args:
        writer: TensorBoard的SummaryWriter实例
        optimizer: 优化器
        epoch: 当前轮次
    """
    for i, param_group in enumerate(optimizer.param_groups):
        writer.add_scalar(f'Learning_rate/group_{i}', param_group['lr'], epoch)

def log_model_parameters(writer, model, epoch):
    """记录模型参数统计信息到TensorBoard
    
    Args:
        writer: TensorBoard的SummaryWriter实例
        model: PyTorch模型
        epoch: 当前轮次
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(f"Params/{name}", param.data, epoch)
            if param.grad is not None:
                writer.add_histogram(f"Grads/{name}", param.grad.data, epoch)

def close_tensorboard_writer(writer):
    """关闭TensorBoard写入器
    
    Args:
        writer: TensorBoard的SummaryWriter实例
    """
    if writer:
        writer.flush()  # 确保所有数据都写入
        writer.close()  # 关闭写入器 