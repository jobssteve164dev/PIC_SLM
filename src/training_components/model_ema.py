"""
模型EMA (指数移动平均) - 第二阶段高级特性

实现模型权重的指数移动平均，提升模型稳定性和泛化能力
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional, Union


class ModelEMA:
    """
    模型指数移动平均 (Exponential Moving Average)
    
    EMA模型通过维护训练模型权重的指数移动平均来提高模型的稳定性和泛化能力。
    在训练过程中，EMA模型的权重会根据当前训练模型的权重进行平滑更新。
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: int = 2000):
        """
        初始化EMA模型
        
        Args:
            model: 要进行EMA的源模型
            decay: EMA衰减率，越接近1.0更新越平滑
            tau: EMA更新的时间常数，用于动态调整衰减率
        """
        self.model = deepcopy(model).eval()  # 创建模型的深度拷贝
        self.decay = decay
        self.tau = tau
        self.updates = 0  # 更新次数计数器
        
        # 冻结EMA模型的参数
        for param in self.model.parameters():
            param.requires_grad_(False)
    
    def update(self, model: nn.Module):
        """
        更新EMA模型权重
        
        Args:
            model: 当前训练的模型
        """
        with torch.no_grad():
            self.updates += 1
            
            # 计算动态衰减率
            d = self.decay
            if self.tau > 0:
                # 在训练初期使用较小的衰减率，随着训练进行逐渐增大
                d = min(self.decay, (1 + self.updates) / (10 + self.updates))
            
            # 更新EMA模型的每个参数
            for ema_param, model_param in zip(self.model.parameters(), model.parameters()):
                if model_param.dtype.is_floating_point:
                    ema_param.data.mul_(d).add_(model_param.data, alpha=1 - d)
    
    def update_attr(self, model: nn.Module, include: tuple = (), exclude: tuple = ('process_group', 'reducer')):
        """
        更新EMA模型的属性
        
        Args:
            model: 源模型
            include: 需要包含的属性名称
            exclude: 需要排除的属性名称
        """
        # 复制模型的其他属性（如buffers）
        for k, v in model.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(self.model, k, v)
    
    def copy_attr(self, model: nn.Module, include: tuple = (), exclude: tuple = ('process_group', 'reducer')):
        """
        从源模型复制属性到EMA模型
        
        Args:
            model: 源模型
            include: 需要包含的属性名称
            exclude: 需要排除的属性名称
        """
        for k, v in model.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(self.model, k, v)
    
    def state_dict(self):
        """返回EMA模型的状态字典"""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict: dict):
        """加载EMA模型的状态字典"""
        self.model.load_state_dict(state_dict)
    
    def to(self, device):
        """将EMA模型移动到指定设备"""
        self.model.to(device)
        return self
    
    def __call__(self, *args, **kwargs):
        """使EMA对象可调用，直接调用内部模型"""
        return self.model(*args, **kwargs)
    
    def eval(self):
        """将EMA模型设置为评估模式"""
        self.model.eval()
        return self
    
    def train(self, mode: bool = True):
        """设置EMA模型的训练模式"""
        self.model.train(mode)
        return self
    
    def parameters(self):
        """返回EMA模型的参数"""
        return self.model.parameters()
    
    def named_parameters(self):
        """返回EMA模型的命名参数"""
        return self.model.named_parameters()
    
    def modules(self):
        """返回EMA模型的所有模块"""
        return self.model.modules()
    
    def named_modules(self):
        """返回EMA模型的命名模块"""
        return self.model.named_modules()


class ModelEMAManager:
    """
    模型EMA管理器
    
    负责管理EMA模型的创建、更新和使用
    """
    
    def __init__(self, model: nn.Module, config: dict):
        """
        初始化EMA管理器
        
        Args:
            model: 源模型
            config: 配置字典
        """
        self.enabled = config.get('model_ema', False)
        self.ema_model = None
        
        if self.enabled:
            decay = config.get('model_ema_decay', 0.9999)
            tau = config.get('model_ema_tau', 2000)
            self.ema_model = ModelEMA(model, decay=decay, tau=tau)
    
    def update(self, model: nn.Module):
        """更新EMA模型"""
        if self.enabled and self.ema_model is not None:
            self.ema_model.update(model)
    
    def get_model(self) -> Optional[nn.Module]:
        """获取EMA模型"""
        if self.enabled and self.ema_model is not None:
            return self.ema_model.model
        return None
    
    def get_ema_model(self) -> Optional[ModelEMA]:
        """获取EMA对象"""
        return self.ema_model if self.enabled else None
    
    def is_enabled(self) -> bool:
        """检查EMA是否启用"""
        return self.enabled and self.ema_model is not None
    
    def save_ema_model(self, path: str):
        """保存EMA模型"""
        if self.enabled and self.ema_model is not None:
            torch.save(self.ema_model.state_dict(), path)
    
    def load_ema_model(self, path: str):
        """加载EMA模型"""
        if self.enabled and self.ema_model is not None:
            self.ema_model.load_state_dict(torch.load(path))
    
    def to(self, device):
        """将EMA模型移动到指定设备"""
        if self.enabled and self.ema_model is not None:
            self.ema_model.to(device)
        return self 