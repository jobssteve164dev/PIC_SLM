"""
配置管理器 - 负责应用配置的保存、加载和验证
"""

import os
import json
import time
import sys
from typing import Dict, List, Any, Optional, Tuple
from PyQt5.QtWidgets import QMessageBox
from .weight_strategy import WeightStrategy

# 导入统一的配置路径工具
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'src'))
from utils.config_path import get_config_file_path


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        # 使用统一的配置路径工具，确保和其他组件一致
        self.config_file_path = get_config_file_path()
    
    def _get_config_file_path(self) -> str:
        """获取配置文件路径（向后兼容）"""
        return self.config_file_path
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            print(f"ConfigManager: 尝试从以下路径加载配置: {self.config_file_path}")
            print(f"ConfigManager: 配置文件路径存在: {os.path.exists(self.config_file_path)}")
            if os.path.exists(self.config_file_path):
                with open(self.config_file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    print(f"ConfigManager: 已加载配置:")
                    print(f"  default_source_folder: {config.get('default_source_folder', 'NOT_SET')}")
                    print(f"  default_output_folder: {config.get('default_output_folder', 'NOT_SET')}")
                    return config
            else:
                print(f"ConfigManager: 配置文件不存在: {self.config_file_path}")
                return {}
        except Exception as e:
            print(f"ConfigManager: 加载配置失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """保存配置文件"""
        try:
            print(f"ConfigManager: 尝试保存配置到: {self.config_file_path}")
            os.makedirs(os.path.dirname(self.config_file_path), exist_ok=True)
            
            with open(self.config_file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            
            print(f"ConfigManager: 已保存配置: {config}")
            return True
        except Exception as e:
            print(f"ConfigManager: 保存配置失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_config_dict(self, 
                          default_source_folder: str = "",
                          default_output_folder: str = "",
                          default_model_file: str = "",
                          default_class_info_file: str = "",
                          default_model_eval_dir: str = "",
                          default_model_save_dir: str = "",
                          default_tensorboard_log_dir: str = "",
                          default_dataset_dir: str = "",
                          default_param_save_dir: str = "",
                          default_classes: List[str] = None,
                          class_weights: Dict[str, float] = None,
                          weight_strategy: WeightStrategy = WeightStrategy.BALANCED) -> Dict[str, Any]:
        """创建配置字典"""
        if default_classes is None:
            default_classes = []
        if class_weights is None:
            class_weights = {}
            
        return {
            'default_source_folder': default_source_folder,
            'default_output_folder': default_output_folder,
            'default_model_file': default_model_file,
            'default_class_info_file': default_class_info_file,
            'default_model_eval_dir': default_model_eval_dir,
            'default_model_save_dir': default_model_save_dir,
            'default_tensorboard_log_dir': default_tensorboard_log_dir,
            'default_dataset_dir': default_dataset_dir,
            'default_param_save_dir': default_param_save_dir,
            'default_classes': default_classes,
            'class_weights': class_weights,
            'weight_strategy': weight_strategy.value,
            'use_class_weights': weight_strategy != WeightStrategy.NONE
        }
    
    def save_config_to_file(self, config: Dict[str, Any], file_path: str) -> bool:
        """保存配置到指定文件"""
        try:
            # 添加导出信息
            export_config = config.copy()
            export_config['version'] = '2.0'
            export_config['export_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_config, f, ensure_ascii=False, indent=4)
            
            return True
        except Exception as e:
            print(f"ConfigManager: 保存配置到文件失败: {str(e)}")
            return False
    
    def load_config_from_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """从指定文件加载配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config
        except Exception as e:
            print(f"ConfigManager: 从文件加载配置失败: {str(e)}")
            return None
    
    def save_classes_config_to_file(self, 
                                   classes: List[str], 
                                   class_weights: Dict[str, float],
                                   weight_strategy: WeightStrategy,
                                   file_path: str) -> bool:
        """保存类别配置到文件"""
        try:
            classes_config = {
                "classes": classes,
                "class_weights": class_weights,
                "weight_strategy": weight_strategy.value,
                "use_class_weights": weight_strategy != WeightStrategy.NONE,
                "description": "缺陷类别配置文件，包含类别名称、权重信息和权重策略",
                "version": "2.0"
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(classes_config, f, ensure_ascii=False, indent=4)
            
            return True
        except Exception as e:
            print(f"ConfigManager: 保存类别配置失败: {str(e)}")
            return False
    
    def load_classes_from_file(self, file_path: str) -> Tuple[List[str], Dict[str, float], WeightStrategy]:
        """从文件加载类别配置"""
        try:
            loaded_classes = []
            loaded_weights = {}
            loaded_strategy = WeightStrategy.BALANCED
            
            # 根据文件扩展名处理不同格式
            if file_path.lower().endswith('.json'):
                # JSON文件格式
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    print(f"加载的JSON数据类型: {type(data)}")
                    if isinstance(data, dict):
                        print(f"JSON字典的主要键: {list(data.keys())}")
                    
                    # 检查数据集评估导出的格式（包含weight_config节点）- 优先检查
                    if isinstance(data, dict) and 'weight_config' in data:
                        print("检测到数据集评估导出格式")
                        weight_config = data.get('weight_config', {})
                        loaded_classes = weight_config.get('classes', [])
                        loaded_weights = weight_config.get('class_weights', {})
                        strategy_value = weight_config.get('weight_strategy', 'balanced')
                        loaded_strategy = WeightStrategy.from_value(strategy_value)
                    
                    # 检查新版本格式（包含权重信息但不是数据集评估格式）
                    elif isinstance(data, dict) and 'classes' in data:
                        print("检测到包含权重信息的配置文件格式")
                        loaded_classes = data.get('classes', [])
                        loaded_weights = data.get('class_weights', {})
                        strategy_value = data.get('weight_strategy', 'balanced')
                        loaded_strategy = WeightStrategy.from_value(strategy_value)
                    
                    # 检查旧版本格式或简单列表格式
                    elif isinstance(data, list):
                        print("检测到简单列表格式")
                        loaded_classes = data
                        # 为所有类别设置默认权重
                        loaded_weights = {class_name: 1.0 for class_name in loaded_classes}
                    
                    # 检查是否是主配置文件格式（包含default_classes）
                    elif isinstance(data, dict) and 'default_classes' in data:
                        print("检测到主配置文件格式")
                        loaded_classes = data.get('default_classes', [])
                        loaded_weights = data.get('class_weights', {})
                        strategy_value = data.get('weight_strategy', 'balanced')
                        loaded_strategy = WeightStrategy.from_value(strategy_value)
                    
                    else:
                        # 提供更详细的错误信息
                        if isinstance(data, dict):
                            available_keys = list(data.keys())
                            error_msg = f"不支持的JSON文件格式。\n\n检测到的键: {available_keys}\n\n支持的格式:\n1. 数据集评估导出: 包含'weight_config'键\n2. 类别配置: 包含'classes'键\n3. 主配置文件: 包含'default_classes'键\n4. 简单列表: 直接的类别名称列表"
                        else:
                            error_msg = f"不支持的JSON文件格式。文件内容类型: {type(data)}\n\n期望的格式: 字典或列表"
                        
                        raise ValueError(error_msg)
                        
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON文件格式错误: {str(e)}")
                except UnicodeDecodeError as e:
                    raise ValueError(f"文件编码错误，请确保文件是UTF-8编码: {str(e)}")
                    
            else:
                # 文本文件格式（每行一个类别）
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        loaded_classes = [line.strip() for line in f if line.strip()]
                    # 为所有类别设置默认权重
                    loaded_weights = {class_name: 1.0 for class_name in loaded_classes}
                except UnicodeDecodeError:
                    # 尝试其他编码
                    try:
                        with open(file_path, 'r', encoding='gbk') as f:
                            loaded_classes = [line.strip() for line in f if line.strip()]
                        loaded_weights = {class_name: 1.0 for class_name in loaded_classes}
                    except UnicodeDecodeError as e:
                        raise ValueError(f"无法读取文件，请检查文件编码: {str(e)}")
            
            if not loaded_classes:
                raise ValueError("文件中没有找到有效的类别信息")
            
            print(f"成功加载类别: {loaded_classes}")
            print(f"加载的权重: {loaded_weights}")
            print(f"权重策略: {loaded_strategy}")
            
            return loaded_classes, loaded_weights, loaded_strategy
            
        except Exception as e:
            print(f"ConfigManager: 加载类别配置时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """验证配置有效性，返回警告信息列表"""
        warnings = []
        
        # 验证文件夹路径
        folders = [
            ('default_source_folder', '默认源文件夹'),
            ('default_output_folder', '默认输出文件夹'),
            ('default_model_eval_dir', '默认模型评估文件夹'),
            ('default_model_save_dir', '默认模型保存文件夹'),
            ('default_tensorboard_log_dir', '默认TensorBoard日志文件夹'),
            ('default_dataset_dir', '默认数据集评估文件夹'),
            ('default_param_save_dir', '默认训练参数保存文件夹')
        ]
        
        for key, name in folders:
            path = config.get(key, '')
            if path and not os.path.exists(path):
                warnings.append(f"{name} 路径不存在: {path}")
        
        # 验证文件路径
        files = [
            ('default_model_file', '默认模型文件'),
            ('default_class_info_file', '默认类别信息文件')
        ]
        
        for key, name in files:
            path = config.get(key, '')
            if path and not os.path.exists(path):
                warnings.append(f"{name} 不存在: {path}")
        
        # 验证类别配置
        classes = config.get('default_classes', [])
        class_weights = config.get('class_weights', {})
        
        if classes and class_weights:
            # 检查是否所有类别都有权重
            for class_name in classes:
                if class_name not in class_weights:
                    warnings.append(f"类别 '{class_name}' 缺少权重配置")
            
            # 检查是否有多余的权重配置
            for class_name in class_weights:
                if class_name not in classes:
                    warnings.append(f"权重配置中存在未知类别: '{class_name}'")
        
        return warnings 