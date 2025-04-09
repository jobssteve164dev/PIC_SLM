import torch
import torch.nn as nn
from typing import Dict, Any

def apply_activation_function(model, activation_name):
    """将指定的激活函数应用到模型的所有合适层中
    
    Args:
        model: PyTorch模型
        activation_name: 激活函数名称
        
    Returns:
        修改后的模型
    """
    # 如果选择无激活函数，则保持模型原样
    if activation_name == "None":
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
        # 未知的激活函数，使用默认的ReLU
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

def apply_dropout(model, dropout_rate):
    """将dropout应用到模型的所有全连接层
    
    Args:
        model: PyTorch模型
        dropout_rate: Dropout概率（0-1之间）
        
    Returns:
        修改后的模型
    """
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