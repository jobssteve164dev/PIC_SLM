"""
高级数据增强 - 第二阶段高级特性

实现CutMix和MixUp等高级数据增强技术，提升模型泛化能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, Union


class MixUpAugmentation:
    """
    MixUp数据增强
    
    MixUp通过线性插值混合两个样本及其标签，生成新的训练样本。
    这种方法可以有效提升模型的泛化能力和鲁棒性。
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        初始化MixUp增强
        
        Args:
            alpha: Beta分布的参数，控制混合强度
        """
        self.alpha = alpha
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        执行MixUp增强
        
        Args:
            x: 输入图像张量 [batch_size, channels, height, width]
            y: 标签张量 [batch_size]
            
        Returns:
            mixed_x: 混合后的图像
            y_a: 第一个标签
            y_b: 第二个标签
            lam: 混合比例
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


class CutMixAugmentation:
    """
    CutMix数据增强
    
    CutMix通过剪切一个图像的矩形区域并粘贴到另一个图像上，
    同时按比例混合标签。这种方法保持了图像的局部特征。
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        初始化CutMix增强
        
        Args:
            alpha: Beta分布的参数，控制剪切区域大小
        """
        self.alpha = alpha
    
    def rand_bbox(self, size: Tuple[int, int, int, int], lam: float) -> Tuple[int, int, int, int]:
        """
        生成随机边界框
        
        Args:
            size: 图像尺寸 (batch_size, channels, height, width)
            lam: 混合比例
            
        Returns:
            边界框坐标 (bbx1, bby1, bbx2, bby2)
        """
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)
        
        # 随机选择中心点
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        执行CutMix增强
        
        Args:
            x: 输入图像张量 [batch_size, channels, height, width]
            y: 标签张量 [batch_size]
            
        Returns:
            mixed_x: 混合后的图像
            y_a: 第一个标签
            y_b: 第二个标签
            lam: 混合比例（基于实际剪切面积调整）
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        y_a, y_b = y, y[index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # 根据实际剪切面积调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        
        return x, y_a, y_b, lam


class MixCriterion:
    """
    混合损失函数
    
    用于计算MixUp和CutMix增强后的损失
    """
    
    def __init__(self, criterion):
        """
        初始化混合损失函数
        
        Args:
            criterion: 基础损失函数
        """
        self.criterion = criterion
    
    def __call__(self, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        """
        计算混合损失
        
        Args:
            pred: 模型预测输出
            y_a: 第一个标签
            y_b: 第二个标签
            lam: 混合比例
            
        Returns:
            混合损失值
        """
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)


class AdvancedAugmentationManager:
    """
    高级数据增强管理器
    
    统一管理MixUp和CutMix增强，根据配置自动选择增强策略
    """
    
    def __init__(self, config: dict):
        """
        初始化增强管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 检查是否启用高级数据增强
        self.advanced_augmentation_enabled = config.get('advanced_augmentation_enabled', False)
        
        # 只有在启用时才读取参数值
        if self.advanced_augmentation_enabled:
            self.mixup_prob = config.get('mixup_alpha', 0.0)
            self.cutmix_prob = config.get('cutmix_prob', 0.0)
            
            # 参数验证
            self.mixup_prob = max(0.0, min(2.0, self.mixup_prob))  # 限制在合理范围内
            self.cutmix_prob = max(0.0, min(1.0, self.cutmix_prob))  # 限制在合理范围内
        else:
            # 如果禁用，强制设置为0
            self.mixup_prob = 0.0
            self.cutmix_prob = 0.0
        
        # 初始化增强器（只有在启用且参数值大于0时）
        self.mixup = MixUpAugmentation(alpha=self.mixup_prob) if self.mixup_prob > 0 else None
        self.cutmix = CutMixAugmentation(alpha=1.0) if self.cutmix_prob > 0 else None
        
        # 启用状态：必须同时满足启用开关和参数值大于0
        self.enabled = self.advanced_augmentation_enabled and (self.mixup_prob > 0 or self.cutmix_prob > 0)
        
        # 记录配置信息
        self._log_configuration()
    
    def _log_configuration(self):
        """记录配置信息（用于调试）"""
        if self.enabled:
            methods = []
            if self.mixup_prob > 0:
                methods.append(f"MixUp(α={self.mixup_prob})")
            if self.cutmix_prob > 0:
                methods.append(f"CutMix(p={self.cutmix_prob})")
            print(f"🚀 高级数据增强已启用: {', '.join(methods)}")
        else:
            if self.advanced_augmentation_enabled:
                print("⚪ 高级数据增强已启用但参数值为0，实际未生效")
            else:
                print("⚪ 高级数据增强已禁用")
    
    def get_augmentation_info(self) -> dict:
        """获取增强配置信息"""
        return {
            'enabled': self.enabled,
            'advanced_augmentation_enabled': self.advanced_augmentation_enabled,
            'mixup_prob': self.mixup_prob,
            'cutmix_prob': self.cutmix_prob,
            'mixup_available': self.mixup is not None,
            'cutmix_available': self.cutmix is not None
        }
    
    def is_enabled(self) -> bool:
        """检查是否启用了高级增强"""
        return self.enabled
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, str]:
        """
        执行高级数据增强
        
        Args:
            x: 输入图像张量
            y: 标签张量
            
        Returns:
            mixed_x: 增强后的图像
            y_a: 第一个标签
            y_b: 第二个标签
            lam: 混合比例
            method: 使用的增强方法
        """
        if not self.enabled:
            return x, y, y, 1.0, 'none'
        
        # 随机选择增强方法
        r = np.random.rand(1)
        
        if self.cutmix_prob > 0 and r < self.cutmix_prob:
            # 使用CutMix
            mixed_x, y_a, y_b, lam = self.cutmix(x, y)
            return mixed_x, y_a, y_b, lam, 'cutmix'
        elif self.mixup_prob > 0:
            # 使用MixUp
            mixed_x, y_a, y_b, lam = self.mixup(x, y)
            return mixed_x, y_a, y_b, lam, 'mixup'
        else:
            # 不使用增强
            return x, y, y, 1.0, 'none'
    
    def get_criterion(self, base_criterion) -> MixCriterion:
        """
        获取混合损失函数
        
        Args:
            base_criterion: 基础损失函数
            
        Returns:
            混合损失函数
        """
        return MixCriterion(base_criterion)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失
    
    结合标签平滑和交叉熵，提升模型泛化能力
    """
    
    def __init__(self, smoothing: float = 0.1):
        """
        初始化标签平滑交叉熵
        
        Args:
            smoothing: 标签平滑参数
        """
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            pred: 模型预测 [batch_size, num_classes]
            target: 真实标签 [batch_size]
            
        Returns:
            损失值
        """
        num_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        
        # 创建平滑标签
        targets = torch.zeros_like(log_preds).scatter_(1, target.unsqueeze(1), 1)
        targets = (1 - self.smoothing) * targets + self.smoothing / num_classes
        
        return (-targets * log_preds).sum(dim=1).mean()


def create_advanced_criterion(config: dict, base_criterion=None):
    """
    创建高级损失函数
    
    Args:
        config: 配置字典
        base_criterion: 基础损失函数
        
    Returns:
        增强的损失函数
    """
    # 检查标签平滑启用状态
    label_smoothing_enabled = config.get('label_smoothing_enabled', False)
    label_smoothing = config.get('label_smoothing', 0.0) if label_smoothing_enabled else 0.0
    
    if label_smoothing_enabled and label_smoothing > 0:
        # 使用标签平滑交叉熵
        criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    else:
        # 使用基础损失函数
        criterion = base_criterion if base_criterion is not None else nn.CrossEntropyLoss()
    
    # 如果启用了高级数据增强，返回混合损失函数
    aug_manager = AdvancedAugmentationManager(config)
    if aug_manager.is_enabled():
        return aug_manager.get_criterion(criterion), aug_manager
    else:
        return criterion, None 