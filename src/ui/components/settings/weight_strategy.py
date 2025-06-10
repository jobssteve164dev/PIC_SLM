"""
权重策略枚举类
"""

from enum import Enum


class WeightStrategy(Enum):
    """权重策略枚举"""
    
    BALANCED = "balanced"
    INVERSE = "inverse"
    LOG_INVERSE = "log_inverse"
    CUSTOM = "custom"
    NONE = "none"
    
    @property
    def display_name(self):
        """获取显示名称"""
        display_names = {
            "balanced": "balanced (平衡权重)",
            "inverse": "inverse (逆频率权重)",
            "log_inverse": "log_inverse (对数逆频率权重)",
            "custom": "custom (自定义权重)",
            "none": "none (无权重)"
        }
        return display_names.get(self.value, self.value)
    
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
        # 处理不同格式的值
        if isinstance(value, (list, tuple)):
            # 如果是数组或元组格式（旧版本兼容）
            search_value = value[0] if value else "balanced"
        elif isinstance(value, str):
            search_value = value
        else:
            search_value = "balanced"
            
        for strategy in cls:
            if strategy.value == search_value:
                return strategy
        return cls.BALANCED  # 默认策略
    
    def is_custom(self):
        """是否为自定义权重策略"""
        return self == self.CUSTOM 