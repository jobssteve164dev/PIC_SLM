"""
集中化配置管理器
解决重复配置加载问题
"""
import os
import json
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime


class ConfigManager:
    """集中化配置管理器，提供单例模式和缓存机制"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._config_cache = {}
        self._config_path = None
        self._config_modified_time = None
        self._load_count = 0  # 统计加载次数
        self._observers = []  # 配置变更观察者
        
        print("ConfigManager: 初始化单例配置管理器")
    
    def set_config_path(self, config_path: str):
        """设置配置文件路径"""
        if self._config_path != config_path:
            self._config_path = config_path
            self._config_cache.clear()  # 清除缓存
            self._config_modified_time = None
            print(f"ConfigManager: 设置配置文件路径: {config_path}")
    
    def add_observer(self, callback: Callable[[Dict[str, Any]], None]):
        """添加配置变更观察者"""
        if callback not in self._observers:
            self._observers.append(callback)
    
    def remove_observer(self, callback: Callable[[Dict[str, Any]], None]):
        """移除配置变更观察者"""
        if callback in self._observers:
            self._observers.remove(callback)
    
    def _notify_observers(self, config: Dict[str, Any]):
        """通知所有观察者配置已变更"""
        for callback in self._observers:
            try:
                callback(config)
            except Exception as e:
                print(f"ConfigManager: 通知观察者时出错: {str(e)}")
    
    def get_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        获取配置，支持缓存和自动检测文件修改
        
        Args:
            force_reload: 是否强制重新加载
            
        Returns:
            配置字典
        """
        if not self._config_path:
            print("ConfigManager: 错误 - 配置文件路径未设置")
            return {}
        
        # 检查文件是否存在
        if not os.path.exists(self._config_path):
            print(f"ConfigManager: 警告 - 配置文件不存在: {self._config_path}")
            return {}
        
        # 获取文件修改时间
        current_modified_time = os.path.getmtime(self._config_path)
        
        # 检查是否需要重新加载
        need_reload = (
            force_reload or 
            not self._config_cache or 
            self._config_modified_time is None or 
            current_modified_time > self._config_modified_time
        )
        
        if need_reload:
            self._load_count += 1
            print(f"ConfigManager: 重新加载配置文件 (第{self._load_count}次): {self._config_path}")
            
            try:
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    self._config_cache = json.load(f)
                self._config_modified_time = current_modified_time
                
                print(f"ConfigManager: 成功加载配置，包含 {len(self._config_cache)} 个配置项")
                
                # 通知观察者
                self._notify_observers(self._config_cache.copy())
                
            except Exception as e:
                print(f"ConfigManager: 加载配置文件失败: {str(e)}")
                return {}
        else:
            print(f"ConfigManager: 使用缓存的配置 (已加载{self._load_count}次)")
        
        return self._config_cache.copy()
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        保存配置到文件
        
        Args:
            config: 配置字典
            
        Returns:
            是否保存成功
        """
        if not self._config_path:
            print("ConfigManager: 错误 - 配置文件路径未设置")
            return False
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            
            with open(self._config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            
            # 更新缓存和修改时间
            self._config_cache = config.copy()
            self._config_modified_time = os.path.getmtime(self._config_path)
            
            print(f"ConfigManager: 成功保存配置到: {self._config_path}")
            
            # 通知观察者
            self._notify_observers(self._config_cache.copy())
            
            return True
            
        except Exception as e:
            print(f"ConfigManager: 保存配置文件失败: {str(e)}")
            return False
    
    def get_config_item(self, key: str, default: Any = None) -> Any:
        """
        获取单个配置项
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值
        """
        config = self.get_config()
        return config.get(key, default)
    
    def set_config_item(self, key: str, value: Any, save_immediately: bool = True) -> bool:
        """
        设置单个配置项
        
        Args:
            key: 配置键
            value: 配置值
            save_immediately: 是否立即保存到文件
            
        Returns:
            是否成功
        """
        config = self.get_config()
        config[key] = value
        
        if save_immediately:
            return self.save_config(config)
        else:
            self._config_cache = config
            return True
    
    def clear_cache(self):
        """清除配置缓存"""
        self._config_cache.clear()
        self._config_modified_time = None
        print("ConfigManager: 配置缓存已清除")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取配置管理器统计信息"""
        return {
            'config_path': self._config_path,
            'load_count': self._load_count,
            'cache_size': len(self._config_cache),
            'has_cache': bool(self._config_cache),
            'observers_count': len(self._observers),
            'last_modified': datetime.fromtimestamp(self._config_modified_time) if self._config_modified_time else None
        }


# 全局配置管理器实例
config_manager = ConfigManager() 