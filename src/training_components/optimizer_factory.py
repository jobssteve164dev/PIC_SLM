"""
高级优化器工厂

实现报告中阶段一的超参数增强：
- 学习率预热 (Learning Rate Warmup)
- 标签平滑 (Label Smoothing) 
- 优化器高级参数 (Adam/SGD的beta1, beta2, momentum等)
- 高级学习率调度 (最小学习率、预热轮数等)

保证向后兼容性，所有新参数都有合理默认值。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math


class WarmupLRScheduler(_LRScheduler):
    """学习率预热调度器"""
    
    def __init__(self, optimizer, warmup_steps, warmup_method='linear', 
                 base_scheduler=None, last_epoch=-1):
        """
        Args:
            optimizer: 优化器
            warmup_steps: 预热步数
            warmup_method: 预热方法 ('linear' or 'cosine')
            base_scheduler: 预热结束后使用的基础调度器
            last_epoch: 最后的epoch编号
        """
        self.warmup_steps = warmup_steps
        self.warmup_method = warmup_method
        self.base_scheduler = base_scheduler
        self.warmup_factor = 1.0 / warmup_steps if warmup_steps > 0 else 1.0
        super(WarmupLRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # 预热阶段
            if self.warmup_method == 'linear':
                warmup_ratio = self.last_epoch / self.warmup_steps
            elif self.warmup_method == 'cosine':
                warmup_ratio = 0.5 * (1 + math.cos(math.pi * (1 - self.last_epoch / self.warmup_steps)))
            else:
                warmup_ratio = 1.0
            
            return [base_lr * warmup_ratio for base_lr in self.base_lrs]
        else:
            # 预热结束后使用基础调度器
            if self.base_scheduler:
                return self.base_scheduler.get_lr()
            else:
                return self.base_lrs


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失函数"""
    
    def __init__(self, smoothing=0.1, weight=None):
        """
        Args:
            smoothing: 标签平滑系数 (0.0-0.3)
            weight: 类别权重
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测logits [N, C]
            target: 真实标签 [N]
        """
        num_classes = pred.size(1)
        confidence = 1.0 - self.smoothing
        
        # 创建平滑标签
        smooth_target = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), confidence)
        smooth_target += self.smoothing / num_classes
        
        # 计算损失
        log_pred = torch.log_softmax(pred, dim=1)
        
        if self.weight is not None:
            # 应用类别权重
            weight_expanded = self.weight.unsqueeze(0).expand(pred.size(0), -1)
            loss = -(smooth_target * log_pred * weight_expanded).sum(dim=1).mean()
        else:
            loss = -(smooth_target * log_pred).sum(dim=1).mean()
        
        return loss


class OptimizerFactory:
    """高级优化器工厂"""
    
    @staticmethod
    def create_optimizer(model, config):
        """
        创建优化器，支持高级参数配置
        
        Args:
            model: 模型
            config: 配置字典
            
        Returns:
            optimizer: 配置好的优化器
        """
        # 基础参数（向后兼容）
        optimizer_name = config.get('optimizer', 'Adam')
        learning_rate = config.get('learning_rate', 0.001)
        weight_decay = config.get('weight_decay', 0.0001)
        
        # 高级参数（新增，带默认值）
        # Adam参数
        beta1 = config.get('beta1', 0.9)
        beta2 = config.get('beta2', 0.999)
        eps = config.get('eps', 1e-8)
        
        # SGD参数
        momentum = config.get('momentum', 0.9)
        nesterov = config.get('nesterov', False)
        
        # 创建优化器
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov
            )
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum
            )
        elif optimizer_name == 'RAdam':
            try:
                from torch.optim import RAdam
                optimizer = RAdam(
                    model.parameters(),
                    lr=learning_rate,
                    betas=(beta1, beta2),
                    eps=eps,
                    weight_decay=weight_decay
                )
            except ImportError:
                # 如果RAdam不可用，回退到Adam
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=learning_rate,
                    betas=(beta1, beta2),
                    eps=eps,
                    weight_decay=weight_decay
                )
        elif optimizer_name == 'AdaBelief':
            try:
                from adabelief_pytorch import AdaBelief
                optimizer = AdaBelief(
                    model.parameters(),
                    lr=learning_rate,
                    betas=(beta1, beta2),
                    eps=eps,
                    weight_decay=weight_decay
                )
            except ImportError:
                # 如果AdaBelief不可用，回退到Adam
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=learning_rate,
                    betas=(beta1, beta2),
                    eps=eps,
                    weight_decay=weight_decay
                )
        else:
            # 默认回退到Adam（向后兼容）
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay
            )
        
        return optimizer
    
    @staticmethod
    def create_scheduler(optimizer, config, total_steps=None):
        """
        创建学习率调度器，支持预热功能
        
        Args:
            optimizer: 优化器
            config: 配置字典
            total_steps: 总训练步数（用于某些调度器）
            
        Returns:
            scheduler: 配置好的调度器
        """
        # 基础调度器配置（向后兼容）
        scheduler_name = config.get('lr_scheduler', 'StepLR')
        
        # 高级调度器参数（新增，带默认值）
        warmup_steps = config.get('warmup_steps', 0)
        warmup_ratio = config.get('warmup_ratio', 0.0)
        warmup_method = config.get('warmup_method', 'linear')
        min_lr = config.get('min_lr', 1e-6)
        
        # 如果设置了warmup_ratio但没有设置warmup_steps，计算warmup_steps
        if warmup_ratio > 0 and warmup_steps == 0 and total_steps:
            warmup_steps = int(total_steps * warmup_ratio)
        
        # 创建基础调度器
        base_scheduler = None
        if scheduler_name == 'StepLR':
            step_size = config.get('step_size', 30)
            gamma = config.get('gamma', 0.1)
            base_scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_name == 'CosineAnnealingLR':
            T_max = config.get('T_max', 50)
            base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=min_lr
            )
        elif scheduler_name == 'ReduceLROnPlateau':
            mode = config.get('scheduler_mode', 'min')
            patience = config.get('scheduler_patience', 10)
            factor = config.get('scheduler_factor', 0.1)
            base_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, patience=patience, factor=factor, min_lr=min_lr
            )
        elif scheduler_name == 'OneCycleLR' and total_steps:
            max_lr = config.get('max_lr', config.get('learning_rate', 0.001) * 10)
            base_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=max_lr, total_steps=total_steps
            )
        elif scheduler_name == 'CyclicLR':
            base_lr = config.get('learning_rate', 0.001)
            max_lr = config.get('max_lr', base_lr * 10)
            step_size_up = config.get('step_size_up', 2000)
            base_scheduler = optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up
            )
        
        # 如果需要预热，创建预热调度器
        if warmup_steps > 0 and base_scheduler:
            scheduler = WarmupLRScheduler(
                optimizer, warmup_steps, warmup_method, base_scheduler
            )
        elif warmup_steps > 0:
            # 只有预热，没有基础调度器
            scheduler = WarmupLRScheduler(
                optimizer, warmup_steps, warmup_method
            )
        else:
            # 没有预热，使用基础调度器
            scheduler = base_scheduler
        
        return scheduler
    
    @staticmethod
    def create_criterion(config, class_weights=None):
        """
        创建损失函数，支持标签平滑
        
        Args:
            config: 配置字典
            class_weights: 类别权重
            
        Returns:
            criterion: 配置好的损失函数
        """
        # 标签平滑参数（新增，带默认值）
        label_smoothing = config.get('label_smoothing', 0.0)
        
        if label_smoothing > 0:
            # 使用标签平滑损失函数
            criterion = LabelSmoothingCrossEntropy(
                smoothing=label_smoothing, 
                weight=class_weights
            )
        else:
            # 使用标准交叉熵损失函数（向后兼容）
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        return criterion
    
    @staticmethod
    def get_supported_optimizers():
        """获取支持的优化器列表"""
        return [
            'Adam', 'SGD', 'AdamW', 'RMSprop', 'RAdam', 'AdaBelief'
        ]
    
    @staticmethod
    def get_supported_schedulers():
        """获取支持的学习率调度器列表"""
        return [
            'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', 
            'OneCycleLR', 'CyclicLR'
        ]
    
    @staticmethod
    def get_default_config():
        """获取默认配置（用于向后兼容）"""
        return {
            # 基础参数
            'optimizer': 'Adam',
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'lr_scheduler': 'StepLR',
            
            # 优化器高级参数
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-8,
            'momentum': 0.9,
            'nesterov': False,
            
            # 学习率预热
            'warmup_steps': 0,
            'warmup_ratio': 0.0,
            'warmup_method': 'linear',
            
            # 高级学习率调度
            'min_lr': 1e-6,
            'step_size': 30,
            'gamma': 0.1,
            'T_max': 50,
            
            # 标签平滑
            'label_smoothing': 0.0,
        } 