import os
import yaml
from typing import Dict, Any, Optional

class ConfigLoader:
    """配置加载器，用于读取和管理配置文件"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器
        
        参数:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        if config_path is None:
            # 默认配置文件路径
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, 'config', 'config.yaml')
            
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"加载配置文件时出错: {str(e)}")
            # 返回默认配置
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'preprocessing': {
                'default_size': '224x224',
                'default_format': 'jpg',
                'default_train_ratio': 0.8,
                'default_augmentation': '基础'
            },
            'training': {
                'default_model': 'ResNet50',
                'default_batch_size': 32,
                'default_learning_rate': 0.001,
                'default_epochs': 20,
                'use_gpu': True,
                'save_best_only': True
            },
            'defect_classes': [
                '划痕', '污点', '缺失', '变形', '异物'
            ],
            'ui': {
                'window_width': 1200,
                'window_height': 900,
                'style': 'Fusion',
                'language': 'zh_CN'
            },
            'paths': {
                'default_data_dir': 'data/raw',
                'default_output_dir': 'data/processed',
                'model_save_dir': 'models/saved_models'
            }
        }
        
    def get_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self.config
        
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """获取图像预处理配置"""
        return self.config.get('preprocessing', {})
        
    def get_training_config(self) -> Dict[str, Any]:
        """获取模型训练配置"""
        return self.config.get('training', {})
        
    def get_defect_classes(self) -> list:
        """获取预定义缺陷类别"""
        return self.config.get('defect_classes', [])
        
    def get_ui_config(self) -> Dict[str, Any]:
        """获取界面配置"""
        return self.config.get('ui', {})
        
    def get_paths_config(self) -> Dict[str, Any]:
        """获取路径配置"""
        return self.config.get('paths', {})
        
    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        保存配置到文件
        
        参数:
            config: 要保存的配置
            
        返回:
            是否保存成功
        """
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            self.config = config
            return True
        except Exception as e:
            print(f"保存配置文件时出错: {str(e)}")
            return False 