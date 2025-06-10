#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的配置文件路径工具
确保所有组件使用相同的配置文件路径
"""

import os


class ConfigPath:
    """配置文件路径管理器"""
    
    _config_file_path = None
    
    @classmethod
    def get_config_file_path(cls) -> str:
        """获取统一的配置文件路径"""
        if cls._config_file_path is None:
            # 获取项目根目录：从当前文件向上找到包含src目录的目录
            current_file = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_file)
            
            # 向上查找直到找到项目根目录（包含src目录的目录）
            while current_dir != os.path.dirname(current_dir):  # 避免到达根目录
                parent_dir = os.path.dirname(current_dir)
                if os.path.exists(os.path.join(parent_dir, 'src')) and os.path.exists(os.path.join(parent_dir, 'config.json')):
                    cls._config_file_path = os.path.join(parent_dir, 'config.json')
                    break
                current_dir = parent_dir
            
            # 如果没找到，使用当前目录的上级目录
            if cls._config_file_path is None:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
                cls._config_file_path = os.path.join(project_root, 'config.json')
        
        return cls._config_file_path
    
    @classmethod
    def get_project_root(cls) -> str:
        """获取项目根目录"""
        config_path = cls.get_config_file_path()
        return os.path.dirname(config_path)
    
    @classmethod
    def reset_cache(cls):
        """重置路径缓存（用于测试）"""
        cls._config_file_path = None


def get_config_file_path() -> str:
    """获取配置文件路径的便捷函数"""
    return ConfigPath.get_config_file_path()


def get_project_root() -> str:
    """获取项目根目录的便捷函数"""
    return ConfigPath.get_project_root()


# 向后兼容的函数
def get_config_path() -> str:
    """获取配置文件路径（向后兼容）"""
    return get_config_file_path()


if __name__ == '__main__':
    # 测试代码
    print("配置文件路径测试")
    print("=" * 30)
    print(f"配置文件路径: {get_config_file_path()}")
    print(f"项目根目录: {get_project_root()}")
    print(f"配置文件存在: {os.path.exists(get_config_file_path())}") 