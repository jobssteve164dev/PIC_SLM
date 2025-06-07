"""
模型配置器 - 负责配置模型的各种参数

主要功能：
- 应用激活函数（ReLU, LeakyReLU, PReLU, ELU, SELU, GELU, Mish, Swish等）
- 应用Dropout层
- 配置模型层（来自utils.model_utils）
"""

import torch
import torch.nn as nn
from PyQt5.QtCore import QObject, pyqtSignal


class ModelConfigurator(QObject):
    """模型配置器"""
    
    # 信号定义
    status_updated = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
    
    def configure_model(self, model, config):
        """
        配置模型
        
        Args:
            model: PyTorch模型
            config: 配置字典
            
        Returns:
            配置后的模型
        """
        # 应用激活函数
        activation_function = config.get('activation_function')
        if activation_function:
            self.status_updated.emit(f"应用自定义激活函数: {activation_function}")
            model = self.apply_activation_function(model, activation_function)
        
        # 应用Dropout
        dropout_rate = config.get('dropout_rate', 0.0)
        if dropout_rate > 0:
            self.status_updated.emit(f"应用Dropout，丢弃率: {dropout_rate}")
            model = self.apply_dropout(model, dropout_rate)
        
        # 应用层配置（如果有）
        if 'layer_config' in config:
            layer_config = config['layer_config']
            if layer_config and layer_config.get('enabled', False):
                self.status_updated.emit("应用自定义层配置")
                model = self._configure_model_layers(model, layer_config)
        
        return model
    
    def apply_activation_function(self, model, activation_name):
        """
        将指定的激活函数应用到模型的所有合适层中
        
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
        activation = self._create_activation_function(activation_name)
        
        # 递归替换模型中的激活函数
        self._replace_activations(model, activation)
        
        return model
    
    def apply_dropout(self, model, dropout_rate):
        """
        将dropout应用到模型的所有全连接层
        
        Args:
            model: PyTorch模型
            dropout_rate: Dropout概率（0-1之间）
            
        Returns:
            修改后的模型
        """
        self.status_updated.emit(f"正在应用Dropout: {dropout_rate}")
        
        # 递归应用dropout
        self._add_dropout_to_model(model, dropout_rate)
        
        return model
    
    def _create_activation_function(self, activation_name):
        """创建激活函数实例"""
        if activation_name == "ReLU":
            return nn.ReLU(inplace=True)
        elif activation_name == "LeakyReLU":
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_name == "PReLU":
            return nn.PReLU()
        elif activation_name == "ELU":
            return nn.ELU(alpha=1.0, inplace=True)
        elif activation_name == "SELU":
            return nn.SELU(inplace=True)
        elif activation_name == "GELU":
            return nn.GELU()
        elif activation_name == "Mish":
            return self._create_mish_activation()
        elif activation_name == "Swish" or activation_name == "SiLU":
            return self._create_swish_activation()
        else:
            self.status_updated.emit(f"未知的激活函数 {activation_name}，使用默认的ReLU")
            return nn.ReLU(inplace=True)
    
    def _create_mish_activation(self):
        """创建Mish激活函数"""
        try:
            return nn.Mish(inplace=True)
        except AttributeError:
            # 如果PyTorch版本不支持Mish，则手动实现
            class Mish(nn.Module):
                def forward(self, x):
                    return x * torch.tanh(nn.functional.softplus(x))
            return Mish()
    
    def _create_swish_activation(self):
        """创建Swish/SiLU激活函数"""
        try:
            return nn.SiLU(inplace=True)
        except AttributeError:
            # 如果PyTorch版本不支持SiLU，则手动实现
            class Swish(nn.Module):
                def forward(self, x):
                    return x * torch.sigmoid(x)
            return Swish()
    
    def _replace_activations(self, module, new_activation):
        """递归替换模型中的激活函数"""
        for name, child in module.named_children():
            if isinstance(child, (nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ELU, nn.SELU, nn.GELU)):
                # 替换为新的激活函数
                setattr(module, name, new_activation)
            else:
                # 递归处理子模块
                self._replace_activations(child, new_activation)
    
    def _add_dropout_to_model(self, module, dropout_rate):
        """递归在模型中添加或更新dropout层"""
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
                self._add_dropout_to_model(child, dropout_rate)
    
    def _configure_model_layers(self, model, layer_config):
        """
        根据层配置调整模型结构
        
        Args:
            model: PyTorch模型
            layer_config: 层配置字典
            
        Returns:
            配置后的模型
        """
        try:
            # 尝试导入utils.model_utils中的函数
            from utils.model_utils import configure_model_layers
            return configure_model_layers(model, layer_config)
        except ImportError:
            self.status_updated.emit("警告: 无法导入utils.model_utils，跳过层配置")
            return model
        except Exception as e:
            self.status_updated.emit(f"配置模型层时出错: {str(e)}")
            return model 