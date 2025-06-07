"""
权重策略枚举类
"""

from enum import Enum


class WeightStrategy(Enum):
    """权重策略枚举"""
    
    BALANCED = ("balanced", "balanced (平衡权重)")
    INVERSE = ("inverse", "inverse (逆频率权重)")
    LOG_INVERSE = ("log_inverse", "log_inverse (对数逆频率权重)")
    CUSTOM = ("custom", "custom (自定义权重)")
    NONE = ("none", "none (无权重)")
    
    def __init__(self, value, display_name):
        # 不要重新设置self.value，因为它已经被Enum自动设置了
        self.display_name = display_name
    
    @classmethod
    def get_all_display_names(cls):
        """获取所有显示名称列表"""
        return [strategy.display_name for strategy in cls]
    
    @classmethod
    def from_display_name(cls, display_name):
        """根据显示名称获取策略"""
        for strategy in cls:
            if strategy.display_name == display_name:
                return strategy
        return cls.BALANCED  # 默认策略
    
    @classmethod
    def from_value(cls, value):
        """根据值获取策略"""
        for strategy in cls:
            if strategy.value == value:
                return strategy
        return cls.BALANCED  # 默认策略
    
    def is_custom(self):
        """是否为自定义权重策略"""
        return self == self.CUSTOM 