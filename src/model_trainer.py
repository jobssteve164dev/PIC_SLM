import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import torchvision
from PyQt5.QtCore import QObject, pyqtSignal, QThread
import numpy as np
from typing import Dict, Any, Optional
import json
import subprocess
import sys
from torch.utils.tensorboard import SummaryWriter
from detection_trainer import DetectionTrainer
import time
from utils.model_utils import create_model, configure_model_layers  # 修改为绝对导入
from sklearn.utils.class_weight import compute_class_weight  # 添加sklearn的类别权重计算
from collections import Counter  # 添加Counter用于统计类别分布

# 设置matplotlib后端为Agg，解决线程安全问题
import matplotlib
matplotlib.use('Agg')  # 必须在导入pyplot之前设置
import matplotlib.pyplot as plt

class TrainingThread(QThread):
    """负责在单独线程中执行训练过程的类"""
    
    # 定义与ModelTrainer相同的信号，用于在线程中发送训练状态
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    training_finished = pyqtSignal()
    training_error = pyqtSignal(str)
    epoch_finished = pyqtSignal(dict)
    # 添加预训练模型下载失败信号
    model_download_failed = pyqtSignal(str, str) # 模型名称，下载链接
    # 添加训练停止信号
    training_stopped = pyqtSignal()
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.stop_training = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        # 添加training_info字典，用于存储训练过程中的信息
        self.training_info = {}
        # 添加类别权重相关属性
        self.class_weights = None
        self.class_distribution = None
    
    def _validate_weight_config(self, class_names):
        """
        验证权重配置是否与数据集类别匹配
        
        Args:
            class_names: 数据集中的类别名称列表
        """
        self.status_updated.emit("验证权重配置...")
        
        # 收集所有可能的权重源
        weight_sources = []
        
        if 'class_weights' in self.config:
            weight_sources.append(('配置中的class_weights', self.config.get('class_weights', {})))
        
        if 'custom_class_weights' in self.config:
            weight_sources.append(('配置中的custom_class_weights', self.config.get('custom_class_weights', {})))
        
        if 'weight_config_file' in self.config:
            weight_config_file = self.config.get('weight_config_file')
            if weight_config_file and os.path.exists(weight_config_file):
                try:
                    with open(weight_config_file, 'r', encoding='utf-8') as f:
                        weight_data = json.load(f)
                    
                    if 'weight_config' in weight_data:
                        weight_sources.append(('权重文件(weight_config)', weight_data['weight_config'].get('class_weights', {})))
                    elif 'class_weights' in weight_data:
                        weight_sources.append(('权重文件(class_weights)', weight_data.get('class_weights', {})))
                        
                except Exception as e:
                    self.status_updated.emit(f"无法读取权重配置文件: {str(e)}")
        
        if 'all_strategies' in self.config:
            strategies = self.config.get('all_strategies', {})
            if 'custom' in strategies:
                weight_sources.append(('all_strategies中的custom', strategies['custom']))
            elif strategies:
                first_strategy = list(strategies.keys())[0]
                weight_sources.append((f'all_strategies中的{first_strategy}', strategies[first_strategy]))
        
        # 分析权重配置
        if not weight_sources:
            self.status_updated.emit("警告: 未找到任何权重配置源，将使用默认权重1.0")
            return
        
        # 检查每个权重源
        for source_name, weights_dict in weight_sources:
            self.status_updated.emit(f"检查权重源: {source_name}")
            
            config_classes = set(weights_dict.keys())
            dataset_classes = set(class_names)
            
            # 检查类别匹配情况
            missing_in_config = dataset_classes - config_classes
            extra_in_config = config_classes - dataset_classes
            matching_classes = dataset_classes & config_classes
            
            self.status_updated.emit(f"  数据集类别数: {len(dataset_classes)}")
            self.status_updated.emit(f"  配置中类别数: {len(config_classes)}")
            self.status_updated.emit(f"  匹配类别数: {len(matching_classes)}")
            
            if missing_in_config:
                self.status_updated.emit(f"  数据集中有但配置中缺失的类别: {list(missing_in_config)[:3]}{'...' if len(missing_in_config) > 3 else ''}")
            
            if extra_in_config:
                self.status_updated.emit(f"  配置中有但数据集中不存在的类别: {list(extra_in_config)[:3]}{'...' if len(extra_in_config) > 3 else ''}")
            
            # 权重统计
            if weights_dict:
                weight_values = list(weights_dict.values())
                self.status_updated.emit(f"  权重范围: {min(weight_values):.3f} - {max(weight_values):.3f}")
                self.status_updated.emit(f"  权重均值: {sum(weight_values)/len(weight_values):.3f}")
                
                # 检查是否所有权重都相同
                if len(set(weight_values)) == 1:
                    self.status_updated.emit(f"  注意: 所有权重都相同 ({weight_values[0]:.3f})")
            
            # 如果有完全匹配的权重源，推荐使用
            if len(matching_classes) == len(dataset_classes) and not extra_in_config:
                self.status_updated.emit(f"  ✓ 推荐权重源: {source_name} (完全匹配)")
                break
            elif len(matching_classes) > 0:
                self.status_updated.emit(f"  ○ 可用权重源: {source_name} (部分匹配)")
            else:
                self.status_updated.emit(f"  ✗ 不可用权重源: {source_name} (无匹配)")
    
    def _calculate_class_weights(self, dataset, class_names, weight_strategy='balanced'):
        """
        计算类别权重
        
        Args:
            dataset: 训练数据集
            class_names: 类别名称列表
            weight_strategy: 权重策略 ('balanced', 'inverse', 'log_inverse', 'custom')
            
        Returns:
            torch.Tensor: 类别权重张量
        """
        self.status_updated.emit("正在计算类别权重...")
        
        # 验证和诊断权重配置
        if weight_strategy == 'custom':
            self._validate_weight_config(class_names)
        
        # 获取所有标签
        all_labels = []
        if hasattr(dataset, 'targets'):
            # ImageFolder dataset
            all_labels = dataset.targets
        else:
            # 手动遍历数据集获取标签
            for _, label in dataset:
                all_labels.append(label)
        
        # 统计类别分布
        label_counts = Counter(all_labels)
        self.class_distribution = {class_names[i]: label_counts.get(i, 0) for i in range(len(class_names))}
        
        # 打印类别分布信息
        self.status_updated.emit("类别分布统计:")
        for class_name, count in self.class_distribution.items():
            self.status_updated.emit(f"  {class_name}: {count} 个样本")
        
        # 计算权重
        if weight_strategy == 'balanced':
            # 使用sklearn的balanced权重计算
            class_weights = compute_class_weight(
                'balanced',
                classes=np.arange(len(class_names)),
                y=all_labels
            )
        elif weight_strategy == 'inverse':
            # 逆频率权重
            total_samples = len(all_labels)
            class_weights = []
            for i in range(len(class_names)):
                count = label_counts.get(i, 1)  # 避免除零
                weight = total_samples / (len(class_names) * count)
                class_weights.append(weight)
            class_weights = np.array(class_weights)
        elif weight_strategy == 'log_inverse':
            # 对数逆频率权重（减少权重差异）
            total_samples = len(all_labels)
            class_weights = []
            for i in range(len(class_names)):
                count = label_counts.get(i, 1)
                weight = np.log(total_samples / count)
                class_weights.append(weight)
            class_weights = np.array(class_weights)
        elif weight_strategy == 'custom':
            # 自定义权重（从配置中读取）
            # 支持多种配置格式的权重读取
            custom_weights = {}
            
            # 1. 首先尝试从 class_weights 字段读取 (设置界面格式)
            if 'class_weights' in self.config:
                custom_weights = self.config.get('class_weights', {})
                self.status_updated.emit("从配置中的class_weights字段读取权重")
            
            # 2. 如果为空，尝试从 custom_class_weights 字段读取 (旧版格式)
            elif 'custom_class_weights' in self.config:
                custom_weights = self.config.get('custom_class_weights', {})
                self.status_updated.emit("从配置中的custom_class_weights字段读取权重")
            
            # 3. 如果还为空，尝试从外部权重配置文件读取
            elif 'weight_config_file' in self.config:
                weight_config_file = self.config.get('weight_config_file')
                if weight_config_file and os.path.exists(weight_config_file):
                    try:
                        with open(weight_config_file, 'r', encoding='utf-8') as f:
                            weight_data = json.load(f)
                        
                        # 支持多种权重文件格式
                        if 'weight_config' in weight_data:
                            # 数据集评估导出格式
                            custom_weights = weight_data['weight_config'].get('class_weights', {})
                            self.status_updated.emit(f"从权重配置文件读取权重: {weight_config_file}")
                        elif 'class_weights' in weight_data:
                            # 直接包含class_weights的格式
                            custom_weights = weight_data.get('class_weights', {})
                            self.status_updated.emit(f"从权重配置文件读取权重: {weight_config_file}")
                        else:
                            self.status_updated.emit(f"权重配置文件格式不支持: {weight_config_file}")
                            
                    except Exception as e:
                        self.status_updated.emit(f"读取权重配置文件失败: {str(e)}")
            
            # 4. 如果仍然为空，检查是否有其他策略的权重可以使用
            if not custom_weights and 'all_strategies' in self.config:
                strategies = self.config.get('all_strategies', {})
                if 'custom' in strategies:
                    custom_weights = strategies['custom']
                    self.status_updated.emit("从all_strategies中的custom策略读取权重")
                elif strategies:
                    # 使用第一个可用的策略
                    first_strategy = list(strategies.keys())[0]
                    custom_weights = strategies[first_strategy]
                    self.status_updated.emit(f"使用{first_strategy}策略权重作为自定义权重")
            
            # 构建权重数组
            class_weights = []
            for class_name in class_names:
                weight = custom_weights.get(class_name, 1.0)
                class_weights.append(weight)
            class_weights = np.array(class_weights)
            
            # 如果所有权重都是1.0，给出警告
            if all(w == 1.0 for w in class_weights):
                self.status_updated.emit("警告: 所有类别权重都是1.0，相当于未使用权重")
            else:
                self.status_updated.emit(f"成功加载自定义权重，权重范围: {min(class_weights):.3f} - {max(class_weights):.3f}")
        else:
            # 默认均等权重
            class_weights = np.ones(len(class_names))
        
        # 打印权重信息
        self.status_updated.emit(f"使用权重策略: {weight_strategy}")
        for i, (class_name, weight) in enumerate(zip(class_names, class_weights)):
            self.status_updated.emit(f"  {class_name}: 权重 = {weight:.4f}")
        
        # 转换为torch张量
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        
        return class_weights_tensor
    
    def _log_class_info_to_tensorboard(self, writer, class_names, epoch=0):
        """
        将类别分布和权重信息记录到TensorBoard
        
        Args:
            writer: TensorBoard writer
            class_names: 类别名称列表
            epoch: 当前epoch
        """
        if not writer or not self.class_distribution or self.class_weights is None:
            return
            
        try:
            # 记录类别分布柱状图
            plt.figure(figsize=(12, 6))
            
            # 子图1: 类别分布
            plt.subplot(1, 2, 1)
            counts = [self.class_distribution[name] for name in class_names]
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
            weights = self.class_weights.cpu().numpy()
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
            
            # 保存为图像并添加到TensorBoard
            import io
            from PIL import Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            img = Image.open(buf)
            img_tensor = transforms.ToTensor()(img)
            writer.add_image('Class Distribution and Weights', img_tensor, epoch)
            
            plt.close()
            
            # 记录权重数值到标量
            for i, (name, weight) in enumerate(zip(class_names, weights)):
                writer.add_scalar(f'Class_Weights/{name}', weight, epoch)
                writer.add_scalar(f'Class_Distribution/{name}', 
                                self.class_distribution[name], epoch)
                
        except Exception as e:
            print(f"记录类别信息到TensorBoard时出错: {str(e)}")
    
    def run(self):
        """线程运行入口，执行模型训练"""
        try:
            # 重置停止标志
            self.stop_training = False
            
            # 提取基本参数
            data_dir = self.config.get('data_dir', '')
            model_name = self.config.get('model_name', 'ResNet50')
            num_epochs = self.config.get('num_epochs', 20)
            batch_size = self.config.get('batch_size', 32)
            learning_rate = self.config.get('learning_rate', 0.001)
            model_save_dir = self.config.get('model_save_dir', 'models/saved_models')
            task_type = self.config.get('task_type', 'classification')
            use_tensorboard = self.config.get('use_tensorboard', True)
            
            # 调用训练流程
            self.train_model(
                data_dir=data_dir,
                model_name=model_name,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                model_save_dir=model_save_dir,
                task_type=task_type,
                use_tensorboard=use_tensorboard
            )
            
            # 训练完成
            self.training_finished.emit()
            
        except Exception as e:
            self.training_error.emit(f"训练过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def stop(self):
        """停止训练过程"""
        # 仅设置停止标志，不发送任何信号
        self.stop_training = True
        self.status_updated.emit("训练线程正在停止...")
        # 不发送训练停止信号，由ModelTrainer类统一管理
    
    def _apply_activation_function(self, model, activation_name):
        """将指定的激活函数应用到模型的所有合适层中
        
        Args:
            model: PyTorch模型
            activation_name: 激活函数名称
            
        Returns:
            修改后的模型
        """
        self.status_updated.emit(f"正在应用激活函数: {activation_name}")
        
        # 如果选择无激活函数，则保持模型原样
        if activation_name == "None":
            self.status_updated.emit("保持模型原有的激活函数不变")
            return model
            
        # 创建激活函数实例
        if activation_name == "ReLU":
            activation = nn.ReLU(inplace=True)
        elif activation_name == "LeakyReLU":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_name == "PReLU":
            activation = nn.PReLU()
        elif activation_name == "ELU":
            activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_name == "SELU":
            activation = nn.SELU(inplace=True)
        elif activation_name == "GELU":
            activation = nn.GELU()
        elif activation_name == "Mish":
            try:
                activation = nn.Mish(inplace=True)
            except AttributeError:
                # 如果PyTorch版本不支持Mish，则手动实现
                class Mish(nn.Module):
                    def forward(self, x):
                        return x * torch.tanh(nn.functional.softplus(x))
                activation = Mish()
        elif activation_name == "Swish" or activation_name == "SiLU":
            try:
                activation = nn.SiLU(inplace=True)
            except AttributeError:
                # 如果PyTorch版本不支持SiLU，则手动实现
                class Swish(nn.Module):
                    def forward(self, x):
                        return x * torch.sigmoid(x)
                activation = Swish()
        else:
            self.status_updated.emit(f"未知的激活函数 {activation_name}，使用默认的ReLU")
            activation = nn.ReLU(inplace=True)
        
        # 递归替换模型中的激活函数
        def replace_activations(module):
            for name, child in module.named_children():
                if isinstance(child, (nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ELU, nn.SELU, nn.GELU)):
                    # 替换为新的激活函数
                    setattr(module, name, activation)
                else:
                    # 递归处理子模块
                    replace_activations(child)
        
        # 应用激活函数替换
        replace_activations(model)
        
        return model
    
    def _apply_dropout(self, model, dropout_rate):
        """将dropout应用到模型的所有全连接层
        
        Args:
            model: PyTorch模型
            dropout_rate: Dropout概率（0-1之间）
            
        Returns:
            修改后的模型
        """
        self.status_updated.emit(f"正在应用Dropout: {dropout_rate}")
        
        # 递归替换模型中的dropout层或在全连接层后添加dropout
        def add_dropout(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # 如果是线性层，检查下一层是否为dropout
                    next_is_dropout = False
                    for next_name in module._modules:
                        if next_name > name and isinstance(module._modules[next_name], nn.Dropout):
                            next_is_dropout = True
                            # 更新已有的dropout
                            module._modules[next_name].p = dropout_rate
                            break
                    
                    # 如果线性层后没有dropout，则添加
                    if not next_is_dropout and dropout_rate > 0:
                        # 创建新的子模块序列，包含原有的线性层和新的dropout
                        new_sequential = nn.Sequential(
                            child,
                            nn.Dropout(p=dropout_rate)
                        )
                        setattr(module, name, new_sequential)
                elif isinstance(child, nn.Dropout):
                    # 直接更新已有的dropout层
                    child.p = dropout_rate
                elif len(list(child.children())) > 0:
                    # 递归处理子模块
                    add_dropout(child)
        
        # 应用dropout
        add_dropout(model)
        
        return model
    
    def train_model(self, data_dir, model_name, num_epochs, batch_size, learning_rate, 
                   model_save_dir, task_type='classification', use_tensorboard=True):
        """与ModelTrainer.train_model相同的实现，但发送信号到主线程"""
        try:
            # 标准化路径格式，确保使用正斜杠
            data_dir = os.path.normpath(data_dir).replace('\\', '/')
            model_save_dir = os.path.normpath(model_save_dir).replace('\\', '/')
            
            # 检查训练和验证数据集目录是否存在
            train_dir = os.path.join(data_dir, 'train')
            val_dir = os.path.join(data_dir, 'val')
            
            if not os.path.exists(train_dir):
                self.training_error.emit(f"训练数据集目录不存在: {train_dir}")
                return
                
            if not os.path.exists(val_dir):
                self.training_error.emit(f"验证数据集目录不存在: {val_dir}")
                return
            
            # 数据转换
            data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }

            # 根据任务类型加载不同的数据集
            if task_type == 'classification':
                # 分类任务：使用ImageFolder加载数据
                self.status_updated.emit("加载分类数据集...")
                image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                        data_transforms[x])
                                for x in ['train', 'val']}
                
                dataloaders = {x: DataLoader(image_datasets[x],
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4)
                              for x in ['train', 'val']}
                
                dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
                class_names = image_datasets['train'].classes
                num_classes = len(class_names)

                # 保存类别信息
                class_info = {
                    'class_names': class_names,
                    'class_to_idx': image_datasets['train'].class_to_idx
                }
                
            elif task_type == 'detection':
                # 目标检测任务：这里需要实现自定义数据集加载逻辑
                # 由于目标检测数据集加载比较复杂，这里只是一个示例框架
                self.status_updated.emit("加载目标检测数据集...")
                self.training_error.emit("目标检测训练功能尚未完全实现")
                return
                
            else:
                self.training_error.emit(f"不支持的任务类型: {task_type}")
                return
                
            os.makedirs(model_save_dir, exist_ok=True)
            with open(os.path.join(model_save_dir, 'class_info.json'), 'w') as f:
                json.dump(class_info, f)

            # 创建模型
            self.model = self._create_model(model_name, num_classes, task_type)
            
            # 应用用户选择的激活函数
            activation_function = self.config.get('activation_function', 'ReLU')
            if activation_function:
                self.status_updated.emit(f"应用自定义激活函数: {activation_function}")
                self.model = self._apply_activation_function(self.model, activation_function)
            
            # 应用用户设置的dropout
            dropout_rate = self.config.get('dropout_rate', 0.0)
            if dropout_rate > 0:
                self.status_updated.emit(f"应用Dropout，丢弃率: {dropout_rate}")
                self.model = self._apply_dropout(self.model, dropout_rate)
            
            self.model = self.model.to(self.device)

            # 计算类别权重（如果启用）
            use_class_weights = self.config.get('use_class_weights', True)  # 默认启用类别权重
            weight_strategy = self.config.get('weight_strategy', 'balanced')  # 默认使用balanced策略
            
            if use_class_weights:
                self.class_weights = self._calculate_class_weights(
                    image_datasets['train'], class_names, weight_strategy
                )
                # 定义加权损失函数
                criterion = nn.CrossEntropyLoss(weight=self.class_weights)
                self.status_updated.emit(f"使用加权损失函数，权重策略: {weight_strategy}")
            else:
                self.class_weights = None
                # 定义标准损失函数
                criterion = nn.CrossEntropyLoss()
                self.status_updated.emit("使用标准损失函数（无类别权重）")
                
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # 初始化TensorBoard
            writer = None
            if use_tensorboard:
                # 获取TensorBoard日志目录
                tensorboard_dir = self.config.get('tensorboard_log_dir', os.path.join(model_save_dir, 'tensorboard_logs'))
                
                # 创建带有时间戳的唯一运行目录
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                model_run_name = f"{model_name}_{timestamp}"
                tensorboard_run_dir = os.path.join(tensorboard_dir, model_run_name)
                os.makedirs(tensorboard_run_dir, exist_ok=True)
                writer = SummaryWriter(tensorboard_run_dir)
                
                # 在训练信息中记录TensorBoard日志目录路径
                self.training_info['tensorboard_log_dir'] = tensorboard_run_dir
                
                # 记录类别信息到TensorBoard
                if use_class_weights and self.class_weights is not None:
                    self._log_class_info_to_tensorboard(writer, class_names, epoch=0)
                
                # 记录模型图
                try:
                    # 获取一个批次的样本数据用于记录模型图
                    sample_inputs, _ = next(iter(dataloaders['train']))
                    sample_inputs = sample_inputs.to(self.device)
                    writer.add_graph(self.model, sample_inputs)
                except Exception as e:
                    print(f"记录模型图时出错: {str(e)}")

            # 训练模型
            best_acc = 0.0
            for epoch in range(num_epochs):
                if self.stop_training:
                    self.status_updated.emit("训练已停止")
                    break

                self.status_updated.emit(f'Epoch {epoch+1}/{num_epochs}')

                # 训练和验证阶段
                for phase in ['train', 'val']:
                    if self.stop_training:
                        break

                    if phase == 'train':
                        self.model.train()
                    else:
                        self.model.eval()

                    running_loss = 0.0
                    running_corrects = 0
                    all_preds = []
                    all_labels = []

                    # 遍历数据
                    for i, (inputs, labels) in enumerate(dataloaders[phase]):
                        if self.stop_training:
                            self.status_updated.emit("训练已在batch处理时停止")
                            break

                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        optimizer.zero_grad()

                        # 前向传播
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            # 反向传播
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                                
                        # 再次检查停止状态
                        if self.stop_training:
                            self.status_updated.emit("训练已在反向传播后停止")
                            break

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        
                        # 收集预测和标签用于计算混淆矩阵
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                        # 更新进度
                        progress = int(((epoch * len(dataloaders[phase]) + i + 1) /
                                     (num_epochs * len(dataloaders[phase]))) * 100)
                        self.progress_updated.emit(progress)

                        # 每10个batch发送一次训练状态
                        if i % 10 == 0:
                            current_loss = running_loss / ((i + 1) * inputs.size(0))
                            current_acc = running_corrects.double() / ((i + 1) * inputs.size(0))
                            epoch_data = {
                                'epoch': epoch + 1,
                                'phase': phase,
                                'loss': float(current_loss),
                                'accuracy': float(current_acc.item()),
                                'batch': i + 1,
                                'total_batches': len(dataloaders[phase])
                            }
                            print(f"发送训练状态更新: {epoch_data}")  # 添加调试信息
                            self.epoch_finished.emit(epoch_data)

                    if self.stop_training:
                        break

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    # 发送每个epoch的结果
                    epoch_data = {
                        'epoch': epoch + 1,
                        'phase': phase,
                        'loss': float(epoch_loss),
                        'accuracy': float(epoch_acc.item()) if isinstance(epoch_acc, torch.Tensor) else float(epoch_acc),
                        'batch': len(dataloaders[phase]),
                        'total_batches': len(dataloaders[phase])
                    }
                    print(f"发送epoch结果: {epoch_data}")  # 添加调试信息
                    self.epoch_finished.emit(epoch_data)
                    
                    # 通知外部训练一个epoch已完成，并携带相关信息
                    epoch_info = {
                        'epoch': epoch + 1,
                        'train_loss': epoch_loss / dataset_sizes['train'],
                        'train_acc': epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc,
                        'val_loss': epoch_loss / dataset_sizes['val'] if phase == 'val' else None,
                        'val_acc': epoch_acc.item() if phase == 'val' and isinstance(epoch_acc, torch.Tensor) else (epoch_acc if phase == 'val' else None)
                    }
                    self.epoch_finished.emit(epoch_info)
                    
                    # 记录到TensorBoard
                    if writer:
                        writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
                        writer.add_scalar(f'Accuracy/{phase}', epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc, epoch)
                        
                        # 每个epoch结束时记录一些样本图像
                        if phase == 'val' and epoch % 5 == 0:  # 每5个epoch记录一次
                            try:
                                # 获取一个批次的样本数据
                                sample_inputs, sample_labels = next(iter(dataloaders[phase]))
                                sample_inputs = sample_inputs.to(self.device)
                                
                                # 记录样本图像
                                grid = torchvision.utils.make_grid(sample_inputs[:8])
                                writer.add_image('Sample Images', grid, epoch)
                            except Exception as e:
                                print(f"记录样本图像时出错: {str(e)}")
                        
                        # 如果是验证阶段，记录混淆矩阵
                        if phase == 'val':
                            try:
                                from sklearn.metrics import confusion_matrix
                                import io
                                from PIL import Image
                                
                                # 确保使用非交互式后端
                                plt.switch_backend('Agg')
                                plt.ioff()  # 关闭交互模式
                                
                                # 计算混淆矩阵
                                cm = confusion_matrix(all_labels, all_preds)
                                
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
                                
                                # 将图像转换为tensor
                                buf = io.BytesIO()
                                plt.savefig(buf, format='png')
                                buf.seek(0)
                                img = Image.open(buf)
                                img_tensor = transforms.ToTensor()(img)
                                
                                # 添加到TensorBoard
                                writer.add_image('Confusion Matrix', img_tensor, epoch)
                                
                                plt.close()
                            except Exception as e:
                                print(f"记录混淆矩阵时出错: {str(e)}")
                        
                        # 确保记录到此轮次的数据
                        writer.flush()

                    # 每个epoch结束时刷新TensorBoard数据，确保及时写入
                    if writer:
                        writer.flush()

                if self.stop_training:
                    break

                # 如果是验证阶段，检查是否为最佳模型
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    # 保存最佳模型，使用统一的命名格式
                    model_note = self.config.get('model_note', '')
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    model_save_path = os.path.join(model_save_dir, f'{model_name}_{timestamp}_{model_note}_best.pth')
                    torch.save(self.model.state_dict(), model_save_path)
                    self.status_updated.emit(f'保存最佳模型，Epoch {epoch+1}, Acc: {best_acc:.4f}')
                    
                    # 导出ONNX模型，也使用统一的命名格式
                    try:
                        onnx_save_path = os.path.join(model_save_dir, f'{model_name}_{timestamp}_{model_note}_best.onnx')
                        sample_input = torch.randn(1, 3, 224, 224).to(self.device)
                        torch.onnx.export(
                            self.model, 
                            sample_input, 
                            onnx_save_path,
                            export_params=True,
                            opset_version=11,
                            input_names=['input'],
                            output_names=['output'],
                            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                        )
                        self.status_updated.emit(f'导出ONNX模型: {onnx_save_path}')
                    except Exception as e:
                        self.status_updated.emit(f'导出ONNX模型时出错: {str(e)}')

            # 保存最终模型，使用统一的命名格式
            model_note = self.config.get('model_note', '')
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            final_model_path = os.path.join(model_save_dir, f'{model_name}_{timestamp}_{model_note}_final.pth')
            torch.save(self.model.state_dict(), final_model_path)
            self.status_updated.emit(f'保存最终模型: {final_model_path}')
            
            # 记录训练完成信息
            training_info = {
                'model_name': model_name,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'best_accuracy': best_acc.item() if isinstance(best_acc, torch.Tensor) else best_acc,
                'class_names': class_names,
                'model_path': final_model_path,
                'timestamp': timestamp  # 添加时间戳到训练信息中
            }
            
            with open(os.path.join(model_save_dir, 'training_info.json'), 'w') as f:
                json.dump(training_info, f, indent=4)
                
            self.status_updated.emit(f'训练完成，最佳准确率: {best_acc:.4f}')
            
            # 关闭TensorBoard写入器
            if writer:
                # 添加显式刷新操作，确保最后一轮数据被记录
                writer.flush()
                writer.close()
                
        except Exception as e:
            self.training_error.emit(f"训练过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _create_model(self, model_name, num_classes, task_type='classification'):
        """与ModelTrainer._create_model相同的实现"""
        try:
            if task_type == 'classification':
                try:
                    if model_name == 'ResNet18':
                        try:
                            # 使用新的权重参数方式加载模型，避免警告
                            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        except Exception as e:
                            # 如果新API不可用，使用旧的方式
                            self.status_updated.emit(f"使用新API加载ResNet18模型失败: {str(e)}")
                            model = models.resnet18(pretrained=True)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                    elif model_name == 'ResNet34':
                        try:
                            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载ResNet34模型失败: {str(e)}")
                            model = models.resnet34(pretrained=True)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                    elif model_name == 'ResNet50':
                        try:
                            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载ResNet50模型失败: {str(e)}")
                            model = models.resnet50(pretrained=True)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                    elif model_name == 'ResNet101':
                        # 使用新的权重参数方式加载模型，避免警告和下载问题
                        try:
                            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        except Exception as e:
                            # 如果新API不可用，尝试旧的方式
                            self.status_updated.emit(f"使用新API加载模型失败: {str(e)}")
                            model = models.resnet101(pretrained=True)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                    elif model_name == 'ResNet152':
                        try:
                            model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载ResNet152模型失败: {str(e)}")
                            model = models.resnet152(pretrained=True)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                    elif model_name == 'MobileNetV2':
                        try:
                            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
                            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载MobileNetV2模型失败: {str(e)}")
                            model = models.mobilenet_v2(pretrained=True)
                            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                            return model
                    elif model_name == 'MobileNetV3':
                        try:
                            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
                            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载MobileNetV3模型失败: {str(e)}")
                            model = models.mobilenet_v3_large(pretrained=True)
                            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
                            return model
                    elif model_name == 'VGG16':
                        try:
                            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
                            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载VGG16模型失败: {str(e)}")
                            model = models.vgg16(pretrained=True)
                            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                            return model
                    elif model_name == 'VGG19':
                        try:
                            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
                            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载VGG19模型失败: {str(e)}")
                            model = models.vgg19(pretrained=True)
                            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                            return model
                    elif model_name == 'DenseNet121':
                        try:
                            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载DenseNet121模型失败: {str(e)}")
                            model = models.densenet121(pretrained=True)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                    elif model_name == 'DenseNet169':
                        try:
                            model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载DenseNet169模型失败: {str(e)}")
                            model = models.densenet169(pretrained=True)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                    elif model_name == 'DenseNet201':
                        try:
                            model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载DenseNet201模型失败: {str(e)}")
                            model = models.densenet201(pretrained=True)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                    elif model_name == 'InceptionV3':
                        try:
                            model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载InceptionV3模型失败: {str(e)}")
                            model = models.inception_v3(pretrained=True, aux_logits=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                    elif model_name in ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4']:
                        # 检查是否安装了efficientnet_pytorch
                        try:
                            from efficientnet_pytorch import EfficientNet
                            
                            # 根据模型名称选择对应版本
                            if model_name == 'EfficientNetB0':
                                model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
                            elif model_name == 'EfficientNetB1':
                                model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
                            elif model_name == 'EfficientNetB2':
                                model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
                            elif model_name == 'EfficientNetB3':
                                model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
                            elif model_name == 'EfficientNetB4':
                                model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
                            
                            return model
                        except ImportError:
                            # 如果没有安装efficientnet_pytorch，提示用户安装
                            self.status_updated.emit(f"未安装EfficientNet库，尝试自动安装...")
                            try:
                                subprocess.check_call([sys.executable, "-m", "pip", "install", "efficientnet_pytorch"])
                                from efficientnet_pytorch import EfficientNet
                                
                                # 根据模型名称选择对应版本
                                if model_name == 'EfficientNetB0':
                                    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
                                elif model_name == 'EfficientNetB1':
                                    model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
                                elif model_name == 'EfficientNetB2':
                                    model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
                                elif model_name == 'EfficientNetB3':
                                    model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
                                elif model_name == 'EfficientNetB4':
                                    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
                                
                                return model
                            except Exception as install_err:
                                raise Exception(f"无法安装EfficientNet库: {str(install_err)}")
                    elif model_name == 'Xception':
                        try:
                            # 使用torchvision替代timm
                            model = models.resnet50(pretrained=True)  # 临时使用ResNet50作为替代
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            self.status_updated.emit("注意：Xception模型暂未实现，使用ResNet50替代")
                            return model
                        except Exception as e:
                            raise Exception(f"创建Xception模型失败: {str(e)}")
                    else:
                        raise ValueError(f"不支持的模型名称: {model_name}")
                except Exception as e:
                    # 处理模型下载失败的情况
                    import urllib.error
                    download_failed = False
                    
                    # 检查是否为下载失败的错误
                    if isinstance(e, urllib.error.URLError) or isinstance(e, ConnectionError) or isinstance(e, TimeoutError):
                        download_failed = True
                    elif any(keyword in str(e).lower() for keyword in ["download", "connection", "timeout", "url", "connect", "network", "internet"]):
                        download_failed = True
                    
                    if download_failed:
                        # 根据模型名称构建下载链接和说明
                        model_links = {
                            'ResNet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
                            'ResNet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
                            'ResNet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
                            'ResNet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
                            'ResNet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
                            'MobileNetV2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
                            'MobileNetV3Small': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
                            'MobileNetV3Large': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth',
                            'VGG16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
                            'VGG19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
                            'DenseNet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
                            'DenseNet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
                            'DenseNet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
                            'InceptionV3': 'https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth',
                            'EfficientNetB0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
                            'EfficientNetB1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
                            'EfficientNetB2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
                            'EfficientNetB3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
                            'EfficientNetB4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth'
                        }
                        
                        # 尝试查找更精确的模型名称匹配
                        exact_model_name = model_name
                        if model_name not in model_links:
                            for key in model_links.keys():
                                if model_name.lower() in key.lower() or key.lower() in model_name.lower():
                                    exact_model_name = key
                                    break
                        
                        model_link = model_links.get(exact_model_name, "")
                        if model_link:
                            self.model_download_failed.emit(exact_model_name, model_link)
                            self.status_updated.emit(f"预训练模型 {model_name} 下载失败，已提供手动下载链接")
                        else:
                            self.training_error.emit(f"预训练模型 {model_name} 下载失败，无法找到下载链接: {str(e)}")
                        
                        # 尝试使用不带预训练权重的模型来继续
                        self.status_updated.emit("尝试使用未预训练的模型继续训练...")
                        
                        # 根据模型名称创建不带预训练权重的模型
                        if model_name == 'ResNet101':
                            self.status_updated.emit("使用不带预训练权重的ResNet101模型")
                            try:
                                # 尝试新API
                                model = models.resnet101(weights=None)
                            except:
                                # 如果新API不可用，使用旧API
                                model = models.resnet101(pretrained=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        elif model_name == 'ResNet50':
                            model = models.resnet50(pretrained=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        elif model_name == 'ResNet18':
                            model = models.resnet18(pretrained=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        elif model_name == 'ResNet34':
                            model = models.resnet34(pretrained=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        elif model_name == 'ResNet152':
                            model = models.resnet152(pretrained=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        elif model_name == 'MobileNetV2':
                            model = models.mobilenet_v2(pretrained=False)
                            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                            return model
                        elif model_name == 'MobileNetV3':
                            model = models.mobilenet_v3_large(pretrained=False)
                            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
                            return model
                        elif model_name == 'VGG16':
                            model = models.vgg16(pretrained=False)
                            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                            return model
                        elif model_name == 'VGG19':
                            model = models.vgg19(pretrained=False)
                            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                            return model
                        elif model_name == 'DenseNet121':
                            model = models.densenet121(pretrained=False)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                        elif model_name == 'DenseNet169':
                            model = models.densenet169(pretrained=False)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                        elif model_name == 'DenseNet201':
                            model = models.densenet201(pretrained=False)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                        elif model_name == 'InceptionV3':
                            model = models.inception_v3(pretrained=False, aux_logits=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        elif model_name in ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4']:
                            try:
                                from efficientnet_pytorch import EfficientNet
                                # 创建一个不带预训练权重的模型
                                if model_name == 'EfficientNetB0':
                                    model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
                                elif model_name == 'EfficientNetB1':
                                    model = EfficientNet.from_name('efficientnet-b1', num_classes=num_classes)
                                elif model_name == 'EfficientNetB2':
                                    model = EfficientNet.from_name('efficientnet-b2', num_classes=num_classes)
                                elif model_name == 'EfficientNetB3':
                                    model = EfficientNet.from_name('efficientnet-b3', num_classes=num_classes)
                                elif model_name == 'EfficientNetB4':
                                    model = EfficientNet.from_name('efficientnet-b4', num_classes=num_classes)
                                return model
                            except ImportError:
                                # 如果没有安装，尝试安装并再次创建
                                try:
                                    subprocess.check_call([sys.executable, "-m", "pip", "install", "efficientnet_pytorch"])
                                    from efficientnet_pytorch import EfficientNet
                                    if model_name == 'EfficientNetB0':
                                        model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
                                    elif model_name == 'EfficientNetB1':
                                        model = EfficientNet.from_name('efficientnet-b1', num_classes=num_classes)
                                    elif model_name == 'EfficientNetB2':
                                        model = EfficientNet.from_name('efficientnet-b2', num_classes=num_classes)
                                    elif model_name == 'EfficientNetB3':
                                        model = EfficientNet.from_name('efficientnet-b3', num_classes=num_classes)
                                    elif model_name == 'EfficientNetB4':
                                        model = EfficientNet.from_name('efficientnet-b4', num_classes=num_classes)
                                    return model
                                except Exception:
                                    # 如果安装失败，回退到ResNet50作为替代
                                    self.status_updated.emit("无法安装或创建EfficientNet模型，使用ResNet50替代")
                                    model = models.resnet50(pretrained=False)
                                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                                    return model
                        else:
                            # 对于所有其他情况，使用ResNet50作为后备方案
                            self.status_updated.emit(f"未知模型 {model_name}，使用ResNet50替代")
                            model = models.resnet50(pretrained=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                    
                    # 重新抛出其他类型的错误
                    raise
            else:
                raise ValueError(f"不支持的任务类型: {task_type}")
        except Exception as e:
            # 确保即使发生错误也返回一个有效的模型
            self.status_updated.emit(f"创建模型时出错: {str(e)}，使用默认的ResNet50模型")
            try:
                model = models.resnet50(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                return model
            except:
                # 如果连ResNet50都创建失败，尝试创建最简单的模型
                self.status_updated.emit("创建ResNet50模型失败，使用简单自定义模型")
                model = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, num_classes)
                )
                return model

class ModelTrainer(QObject):
    # 定义信号
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    training_finished = pyqtSignal()
    training_error = pyqtSignal(str)
    epoch_finished = pyqtSignal(dict)
    # 添加预训练模型下载失败信号
    model_download_failed = pyqtSignal(str, str)  # 模型名称，下载链接
    # 添加训练停止信号
    training_stopped = pyqtSignal()
    # 添加对应DetectionTrainer的信号
    metrics_updated = pyqtSignal(dict)
    tensorboard_updated = pyqtSignal(str, float, int)

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.stop_training = False
        self.training_thread = None
        self.detection_trainer = None
        # 添加类别权重相关属性
        self.class_weights = None
        self.class_distribution = None

    def configure_model(self, model, layer_config):
        """根据层配置调整模型结构"""
        if not layer_config or not layer_config.get('enabled', False):
            return model
            
        return configure_model_layers(model, layer_config)
        
    def train_model_with_config(self, config):
        """使用配置训练模型"""
        try:
            # 创建基础模型
            self.model = create_model(config)
            
            # 如果有层配置，应用层配置
            if 'layer_config' in config:
                self.model = self.configure_model(self.model, config['layer_config'])
            
            # 提取核心训练参数
            task_type = config.get('task_type', 'classification')
            
            # 获取当前时间戳，用于保存模型和配置文件名
            import time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_name = config.get('model_name', 'Unknown')
            model_note = config.get('model_note', '')
            
            # 构建统一的文件名基础部分
            model_filename = f"{model_name}_{timestamp}"
            if model_note:
                model_filename += f"_{model_note}"
            
            # 获取模型保存目录和参数保存目录
            model_save_dir = config.get('model_save_dir', 'models/saved_models')
            param_save_dir = config.get('default_param_save_dir', model_save_dir)  # 如果未指定参数保存目录，则使用模型保存目录
            
            # 标准化路径格式，确保所有路径使用相同的格式
            model_save_dir = os.path.normpath(model_save_dir)
            param_save_dir = os.path.normpath(param_save_dir)
            
            # 确保目录存在
            os.makedirs(model_save_dir, exist_ok=True)
            os.makedirs(param_save_dir, exist_ok=True)
            
            # 在配置中添加保存的文件名信息，供后续使用
            config['model_filename'] = model_filename
            config['timestamp'] = timestamp
            
            # 保存配置文件，使用统一的文件名格式，保存到参数保存目录
            config_file_path = os.path.join(param_save_dir, f"{model_filename}_config.json")
            try:
                with open(config_file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=4)
                self.status_updated.emit(f"训练配置已保存到: {config_file_path}")
            except Exception as e:
                self.training_error.emit(f"保存训练配置文件时发生错误: {str(e)}")
            
            # 根据任务类型选择不同的训练方法
            if task_type == 'classification':
                # 调用分类模型训练线程
                self.status_updated.emit("启动分类模型训练...")
                self.training_thread = TrainingThread(config)
                
                # 连接训练线程信号
                self.training_thread.progress_updated.connect(self.progress_updated)
                self.training_thread.status_updated.connect(self.status_updated)
                self.training_thread.training_finished.connect(self.training_finished)
                self.training_thread.training_error.connect(self.training_error)
                self.training_thread.epoch_finished.connect(self.epoch_finished)
                self.training_thread.model_download_failed.connect(self.model_download_failed)
                self.training_thread.training_stopped.connect(self.training_stopped)
                
                # 启动训练线程
                self.training_thread.start()
                
            elif task_type == 'detection':
                # 使用YOLOv5等目标检测模型的训练逻辑
                self.status_updated.emit("启动目标检测模型训练...")
                
                # 创建DetectionTrainer实例
                from detection_trainer import DetectionTrainer
                
                try:
                    # 初始化检测训练器
                    self.detection_trainer = DetectionTrainer(config)
                    
                    # 连接信号
                    self.detection_trainer.progress_updated.connect(self.progress_updated)
                    self.detection_trainer.status_updated.connect(self.status_updated)
                    self.detection_trainer.training_finished.connect(self.training_finished)
                    self.detection_trainer.training_error.connect(self.training_error)
                    self.detection_trainer.metrics_updated.connect(self.metrics_updated)
                    self.detection_trainer.tensorboard_updated.connect(self.tensorboard_updated)
                    self.detection_trainer.model_download_failed.connect(self.model_download_failed)
                    self.detection_trainer.training_stopped.connect(self.training_stopped)
                    
                    # 启动训练 - 确保使用包含模型文件名的配置
                    self.detection_trainer.start_training(config)
                    
                except Exception as e:
                    self.training_error.emit(f"创建目标检测训练器时出错: {str(e)}")
            else:
                self.training_error.emit(f"不支持的任务类型: {task_type}")
                
        except Exception as e:
            self.training_error.emit(f"训练初始化时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def stop(self):
        """停止训练过程"""
        try:
            if self.training_thread and self.training_thread.isRunning():
                self.training_thread.stop()
                self.training_thread.wait()
            
            if self.detection_trainer:
                self.detection_trainer.stop()
                
            self.stop_training = True
            self.status_updated.emit("训练已停止")
            
        except Exception as e:
            print(f"停止训练时出错: {str(e)}")
        
        # 无论线程是否正常结束，都发射一次训练停止信号
        self.training_stopped.emit()

    def _apply_activation_function(self, model, activation_name):
        """将指定的激活函数应用到模型的所有合适层中
        
        Args:
            model: PyTorch模型
            activation_name: 激活函数名称
            
        Returns:
            修改后的模型
        """
        self.status_updated.emit(f"正在应用激活函数: {activation_name}")
        
        # 如果选择无激活函数，则保持模型原样
        if activation_name == "None":
            self.status_updated.emit("保持模型原有的激活函数不变")
            return model
            
        # 创建激活函数实例
        if activation_name == "ReLU":
            activation = nn.ReLU(inplace=True)
        elif activation_name == "LeakyReLU":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_name == "PReLU":
            activation = nn.PReLU()
        elif activation_name == "ELU":
            activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_name == "SELU":
            activation = nn.SELU(inplace=True)
        elif activation_name == "GELU":
            activation = nn.GELU()
        elif activation_name == "Mish":
            try:
                activation = nn.Mish(inplace=True)
            except AttributeError:
                # 如果PyTorch版本不支持Mish，则手动实现
                class Mish(nn.Module):
                    def forward(self, x):
                        return x * torch.tanh(nn.functional.softplus(x))
                activation = Mish()
        elif activation_name == "Swish" or activation_name == "SiLU":
            try:
                activation = nn.SiLU(inplace=True)
            except AttributeError:
                # 如果PyTorch版本不支持SiLU，则手动实现
                class Swish(nn.Module):
                    def forward(self, x):
                        return x * torch.sigmoid(x)
                activation = Swish()
        else:
            self.status_updated.emit(f"未知的激活函数 {activation_name}，使用默认的ReLU")
            activation = nn.ReLU(inplace=True)
        
        # 递归替换模型中的激活函数
        def replace_activations(module):
            for name, child in module.named_children():
                if isinstance(child, (nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ELU, nn.SELU, nn.GELU)):
                    # 替换为新的激活函数
                    setattr(module, name, activation)
                else:
                    # 递归处理子模块
                    replace_activations(child)
        
        # 应用激活函数替换
        replace_activations(model)
        
        return model
    
    def _apply_dropout(self, model, dropout_rate):
        """将dropout应用到模型的所有全连接层
        
        Args:
            model: PyTorch模型
            dropout_rate: Dropout概率（0-1之间）
            
        Returns:
            修改后的模型
        """
        self.status_updated.emit(f"正在应用Dropout: {dropout_rate}")
        
        # 递归替换模型中的dropout层或在全连接层后添加dropout
        def add_dropout(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # 如果是线性层，检查下一层是否为dropout
                    next_is_dropout = False
                    for next_name in module._modules:
                        if next_name > name and isinstance(module._modules[next_name], nn.Dropout):
                            next_is_dropout = True
                            # 更新已有的dropout
                            module._modules[next_name].p = dropout_rate
                            break
                    
                    # 如果线性层后没有dropout，则添加
                    if not next_is_dropout and dropout_rate > 0:
                        # 创建新的子模块序列，包含原有的线性层和新的dropout
                        new_sequential = nn.Sequential(
                            child,
                            nn.Dropout(p=dropout_rate)
                        )
                        setattr(module, name, new_sequential)
                elif isinstance(child, nn.Dropout):
                    # 直接更新已有的dropout层
                    child.p = dropout_rate
                elif len(list(child.children())) > 0:
                    # 递归处理子模块
                    add_dropout(child)
        
        # 应用dropout
        add_dropout(model)
        
        return model
    
    def _create_model(self, model_name, num_classes, task_type='classification'):
        """与ModelTrainer._create_model相同的实现"""
        try:
            if task_type == 'classification':
                try:
                    if model_name == 'ResNet18':
                        try:
                            # 使用新的权重参数方式加载模型，避免警告
                            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        except Exception as e:
                            # 如果新API不可用，使用旧的方式
                            self.status_updated.emit(f"使用新API加载ResNet18模型失败: {str(e)}")
                            model = models.resnet18(pretrained=True)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                    elif model_name == 'ResNet34':
                        try:
                            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载ResNet34模型失败: {str(e)}")
                            model = models.resnet34(pretrained=True)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                    elif model_name == 'ResNet50':
                        try:
                            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载ResNet50模型失败: {str(e)}")
                            model = models.resnet50(pretrained=True)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                    elif model_name == 'ResNet101':
                        # 使用新的权重参数方式加载模型，避免警告和下载问题
                        try:
                            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        except Exception as e:
                            # 如果新API不可用，尝试旧的方式
                            self.status_updated.emit(f"使用新API加载模型失败: {str(e)}")
                            model = models.resnet101(pretrained=True)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                    elif model_name == 'ResNet152':
                        try:
                            model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载ResNet152模型失败: {str(e)}")
                            model = models.resnet152(pretrained=True)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                    elif model_name == 'MobileNetV2':
                        try:
                            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
                            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载MobileNetV2模型失败: {str(e)}")
                            model = models.mobilenet_v2(pretrained=True)
                            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                            return model
                    elif model_name == 'MobileNetV3':
                        try:
                            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
                            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载MobileNetV3模型失败: {str(e)}")
                            model = models.mobilenet_v3_large(pretrained=True)
                            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
                            return model
                    elif model_name == 'VGG16':
                        try:
                            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
                            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载VGG16模型失败: {str(e)}")
                            model = models.vgg16(pretrained=True)
                            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                            return model
                    elif model_name == 'VGG19':
                        try:
                            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
                            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载VGG19模型失败: {str(e)}")
                            model = models.vgg19(pretrained=True)
                            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                            return model
                    elif model_name == 'DenseNet121':
                        try:
                            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载DenseNet121模型失败: {str(e)}")
                            model = models.densenet121(pretrained=True)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                    elif model_name == 'DenseNet169':
                        try:
                            model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载DenseNet169模型失败: {str(e)}")
                            model = models.densenet169(pretrained=True)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                    elif model_name == 'DenseNet201':
                        try:
                            model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载DenseNet201模型失败: {str(e)}")
                            model = models.densenet201(pretrained=True)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                    elif model_name == 'InceptionV3':
                        try:
                            model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        except Exception as e:
                            self.status_updated.emit(f"使用新API加载InceptionV3模型失败: {str(e)}")
                            model = models.inception_v3(pretrained=True, aux_logits=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                    elif model_name in ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4']:
                        # 检查是否安装了efficientnet_pytorch
                        try:
                            from efficientnet_pytorch import EfficientNet
                            
                            # 根据模型名称选择对应版本
                            if model_name == 'EfficientNetB0':
                                model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
                            elif model_name == 'EfficientNetB1':
                                model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
                            elif model_name == 'EfficientNetB2':
                                model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
                            elif model_name == 'EfficientNetB3':
                                model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
                            elif model_name == 'EfficientNetB4':
                                model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
                            
                            return model
                        except ImportError:
                            # 如果没有安装efficientnet_pytorch，提示用户安装
                            self.status_updated.emit(f"未安装EfficientNet库，尝试自动安装...")
                            try:
                                subprocess.check_call([sys.executable, "-m", "pip", "install", "efficientnet_pytorch"])
                                from efficientnet_pytorch import EfficientNet
                                
                                # 根据模型名称选择对应版本
                                if model_name == 'EfficientNetB0':
                                    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
                                elif model_name == 'EfficientNetB1':
                                    model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
                                elif model_name == 'EfficientNetB2':
                                    model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
                                elif model_name == 'EfficientNetB3':
                                    model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
                                elif model_name == 'EfficientNetB4':
                                    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
                                
                                return model
                            except Exception as install_err:
                                raise Exception(f"无法安装EfficientNet库: {str(install_err)}")
                    elif model_name == 'Xception':
                        try:
                            # 使用torchvision替代timm
                            model = models.resnet50(pretrained=True)  # 临时使用ResNet50作为替代
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            self.status_updated.emit("注意：Xception模型暂未实现，使用ResNet50替代")
                            return model
                        except Exception as e:
                            raise Exception(f"创建Xception模型失败: {str(e)}")
                    else:
                        raise ValueError(f"不支持的模型名称: {model_name}")
                except Exception as e:
                    # 处理模型下载失败的情况
                    import urllib.error
                    download_failed = False
                    
                    # 检查是否为下载失败的错误
                    if isinstance(e, urllib.error.URLError) or isinstance(e, ConnectionError) or isinstance(e, TimeoutError):
                        download_failed = True
                    elif any(keyword in str(e).lower() for keyword in ["download", "connection", "timeout", "url", "connect", "network", "internet"]):
                        download_failed = True
                    
                    if download_failed:
                        # 根据模型名称构建下载链接和说明
                        model_links = {
                            'ResNet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
                            'ResNet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
                            'ResNet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
                            'ResNet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
                            'ResNet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
                            'MobileNetV2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
                            'MobileNetV3Small': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
                            'MobileNetV3Large': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth',
                            'VGG16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
                            'VGG19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
                            'DenseNet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
                            'DenseNet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
                            'DenseNet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
                            'InceptionV3': 'https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth',
                            'EfficientNetB0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
                            'EfficientNetB1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
                            'EfficientNetB2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
                            'EfficientNetB3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
                            'EfficientNetB4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth'
                        }
                        
                        # 尝试查找更精确的模型名称匹配
                        exact_model_name = model_name
                        if model_name not in model_links:
                            for key in model_links.keys():
                                if model_name.lower() in key.lower() or key.lower() in model_name.lower():
                                    exact_model_name = key
                                    break
                        
                        model_link = model_links.get(exact_model_name, "")
                        if model_link:
                            self.model_download_failed.emit(exact_model_name, model_link)
                            self.status_updated.emit(f"预训练模型 {model_name} 下载失败，已提供手动下载链接")
                        else:
                            self.training_error.emit(f"预训练模型 {model_name} 下载失败，无法找到下载链接: {str(e)}")
                        
                        # 尝试使用不带预训练权重的模型来继续
                        self.status_updated.emit("尝试使用未预训练的模型继续训练...")
                        
                        # 根据模型名称创建不带预训练权重的模型
                        if model_name == 'ResNet101':
                            self.status_updated.emit("使用不带预训练权重的ResNet101模型")
                            try:
                                # 尝试新API
                                model = models.resnet101(weights=None)
                            except:
                                # 如果新API不可用，使用旧API
                                model = models.resnet101(pretrained=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        elif model_name == 'ResNet50':
                            model = models.resnet50(pretrained=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        elif model_name == 'ResNet18':
                            model = models.resnet18(pretrained=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        elif model_name == 'ResNet34':
                            model = models.resnet34(pretrained=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        elif model_name == 'ResNet152':
                            model = models.resnet152(pretrained=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        elif model_name == 'MobileNetV2':
                            model = models.mobilenet_v2(pretrained=False)
                            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                            return model
                        elif model_name == 'MobileNetV3':
                            model = models.mobilenet_v3_large(pretrained=False)
                            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
                            return model
                        elif model_name == 'VGG16':
                            model = models.vgg16(pretrained=False)
                            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                            return model
                        elif model_name == 'VGG19':
                            model = models.vgg19(pretrained=False)
                            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                            return model
                        elif model_name == 'DenseNet121':
                            model = models.densenet121(pretrained=False)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                        elif model_name == 'DenseNet169':
                            model = models.densenet169(pretrained=False)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                        elif model_name == 'DenseNet201':
                            model = models.densenet201(pretrained=False)
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                            return model
                        elif model_name == 'InceptionV3':
                            model = models.inception_v3(pretrained=False, aux_logits=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                        elif model_name in ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4']:
                            try:
                                from efficientnet_pytorch import EfficientNet
                                # 创建一个不带预训练权重的模型
                                if model_name == 'EfficientNetB0':
                                    model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
                                elif model_name == 'EfficientNetB1':
                                    model = EfficientNet.from_name('efficientnet-b1', num_classes=num_classes)
                                elif model_name == 'EfficientNetB2':
                                    model = EfficientNet.from_name('efficientnet-b2', num_classes=num_classes)
                                elif model_name == 'EfficientNetB3':
                                    model = EfficientNet.from_name('efficientnet-b3', num_classes=num_classes)
                                elif model_name == 'EfficientNetB4':
                                    model = EfficientNet.from_name('efficientnet-b4', num_classes=num_classes)
                                return model
                            except ImportError:
                                # 如果没有安装，尝试安装并再次创建
                                try:
                                    subprocess.check_call([sys.executable, "-m", "pip", "install", "efficientnet_pytorch"])
                                    from efficientnet_pytorch import EfficientNet
                                    if model_name == 'EfficientNetB0':
                                        model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
                                    elif model_name == 'EfficientNetB1':
                                        model = EfficientNet.from_name('efficientnet-b1', num_classes=num_classes)
                                    elif model_name == 'EfficientNetB2':
                                        model = EfficientNet.from_name('efficientnet-b2', num_classes=num_classes)
                                    elif model_name == 'EfficientNetB3':
                                        model = EfficientNet.from_name('efficientnet-b3', num_classes=num_classes)
                                    elif model_name == 'EfficientNetB4':
                                        model = EfficientNet.from_name('efficientnet-b4', num_classes=num_classes)
                                    return model
                                except Exception:
                                    # 如果安装失败，回退到ResNet50作为替代
                                    self.status_updated.emit("无法安装或创建EfficientNet模型，使用ResNet50替代")
                                    model = models.resnet50(pretrained=False)
                                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                                    return model
                        else:
                            # 对于所有其他情况，使用ResNet50作为后备方案
                            self.status_updated.emit(f"未知模型 {model_name}，使用ResNet50替代")
                            model = models.resnet50(pretrained=False)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                            return model
                    
                    # 重新抛出其他类型的错误
                    raise
            else:
                raise ValueError(f"不支持的任务类型: {task_type}")
        except Exception as e:
            # 确保即使发生错误也返回一个有效的模型
            self.status_updated.emit(f"创建模型时出错: {str(e)}，使用默认的ResNet50模型")
            try:
                model = models.resnet50(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                return model
            except:
                # 如果连ResNet50都创建失败，尝试创建最简单的模型
                self.status_updated.emit("创建ResNet50模型失败，使用简单自定义模型")
                model = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, num_classes)
                )
                return model 