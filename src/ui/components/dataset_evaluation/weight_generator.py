"""
类别权重生成器

负责根据数据集的类别分布生成不同策略的类别权重，用于解决类别不平衡问题
"""

import os
import json
import numpy as np
import datetime
from sklearn.utils.class_weight import compute_class_weight
from PyQt5.QtCore import QObject, pyqtSignal


class WeightGenerator(QObject):
    """类别权重生成器"""
    
    # 定义信号
    status_updated = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
    def generate_weights(self, class_counts):
        """生成多种权重策略
        
        Args:
            class_counts (dict): 类别名称到样本数量的映射
            
        Returns:
            dict: 包含多种权重策略的字典
        """
        if not class_counts:
            return {}
            
        class_names = list(class_counts.keys())
        class_count_values = list(class_counts.values())
        total_samples = sum(class_count_values)
        
        weight_strategies = {}
        
        # 1. Balanced权重 (使用sklearn)
        try:
            # 为sklearn准备标签列表
            all_labels = []
            for class_name, count in class_counts.items():
                all_labels.extend([class_name] * count)
                
            balanced_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(class_names), 
                y=all_labels
            )
            weight_strategies['balanced'] = dict(zip(class_names, balanced_weights))
        except Exception as e:
            print(f"计算balanced权重失败: {e}")
            weight_strategies['balanced'] = {name: 1.0 for name in class_names}
        
        # 2. Inverse权重
        inverse_weights = {}
        for class_name, count in class_counts.items():
            inverse_weights[class_name] = total_samples / (len(class_names) * count)
        weight_strategies['inverse'] = inverse_weights
        
        # 3. Log inverse权重
        log_inverse_weights = {}
        for class_name, count in class_counts.items():
            log_inverse_weights[class_name] = np.log(total_samples / count)
        weight_strategies['log_inverse'] = log_inverse_weights
        
        # 4. 归一化权重 (将权重调整到合理范围)
        normalized_weights = {}
        max_count = max(class_count_values)
        for class_name, count in class_counts.items():
            normalized_weights[class_name] = max_count / count
        weight_strategies['normalized'] = normalized_weights
        
        return weight_strategies
        
    def get_recommended_strategy(self, imbalance_ratio, cv_coefficient=None):
        """根据数据集特征推荐权重策略
        
        Args:
            imbalance_ratio (float): 类别不平衡度
            cv_coefficient (float, optional): 变异系数
            
        Returns:
            str: 推荐的权重策略名称
        """
        if imbalance_ratio < 2:
            return "none (数据相对平衡)"
        elif imbalance_ratio < 5:
            return "balanced (推荐)"
        elif imbalance_ratio < 15:
            return "inverse (中度不平衡)"
        else:
            return "log_inverse (严重不平衡)"
            
    def export_weights_config(self, dataset_path, class_counts, weight_strategies, 
                            selected_strategy, file_path):
        """导出权重配置到文件
        
        Args:
            dataset_path (str): 数据集路径
            class_counts (dict): 类别计数
            weight_strategies (dict): 所有权重策略
            selected_strategy (str): 选择的策略
            file_path (str): 保存文件路径
        """
        try:
            class_names = list(class_counts.keys())
            imbalance_ratio = max(class_counts.values()) / min(class_counts.values()) if class_counts else 1.0
            
            # 策略显示名称映射
            strategy_names = {
                'balanced': 'Balanced (推荐)',
                'inverse': 'Inverse (逆频率)', 
                'log_inverse': 'Log Inverse (对数逆频率)',
                'normalized': 'Normalized (归一化)'
            }
            
            # 准备导出数据
            export_data = {
                "dataset_info": {
                    "dataset_path": dataset_path,
                    "total_classes": len(class_names),
                    "total_samples": sum(class_counts.values()),
                    "class_distribution": class_counts,
                    "imbalance_ratio": imbalance_ratio,
                    "analysis_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "weight_config": {
                    "classes": class_names,
                    "class_weights": weight_strategies.get(selected_strategy, {}),
                    "weight_strategy": selected_strategy,
                    "use_class_weights": True,
                    "description": f"使用{strategy_names.get(selected_strategy, selected_strategy)}策略生成的类别权重配置"
                },
                "all_strategies": weight_strategies,
                "usage_instructions": {
                    "设置界面": "在设置-默认缺陷类别与权重配置中，选择'custom'策略并手动设置权重值",
                    "训练配置": "在训练配置文件中设置use_class_weights=true和weight_strategy='custom'",
                    "权重含义": "较高的权重值会让模型更关注该类别的样本，用于平衡类别不均衡问题"
                },
                "version": "2.0"
            }
            
            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=4)
                
            self.status_updated.emit(f"权重配置已成功导出到: {file_path}")
            return True
            
        except Exception as e:
            self.status_updated.emit(f"导出权重配置失败: {str(e)}")
            return False
            
    def calculate_weight_stats(self, weight_strategies, class_counts):
        """计算权重统计信息
        
        Args:
            weight_strategies (dict): 权重策略字典
            class_counts (dict): 类别计数
            
        Returns:
            dict: 权重统计信息
        """
        if not weight_strategies or not class_counts:
            return {}
            
        stats = {}
        imbalance_ratio = max(class_counts.values()) / min(class_counts.values()) if class_counts else 1.0
        cv_coefficient = np.std(list(class_counts.values())) / np.mean(list(class_counts.values())) if class_counts else 0
        
        stats['imbalance_ratio'] = imbalance_ratio
        stats['cv_coefficient'] = cv_coefficient
        stats['recommended_strategy'] = self.get_recommended_strategy(imbalance_ratio, cv_coefficient)
        
        # 计算每种策略的统计信息
        for strategy_name, weights in weight_strategies.items():
            if weights:
                weight_values = list(weights.values())
                stats[f'{strategy_name}_min'] = min(weight_values)
                stats[f'{strategy_name}_max'] = max(weight_values)
                stats[f'{strategy_name}_mean'] = np.mean(weight_values)
                stats[f'{strategy_name}_std'] = np.std(weight_values)
                
        return stats
        
    def validate_weights(self, weights):
        """验证权重配置的有效性
        
        Args:
            weights (dict): 权重配置
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not weights:
            return False, "权重配置为空"
            
        try:
            # 检查所有权重是否为正数
            for class_name, weight in weights.items():
                if not isinstance(weight, (int, float)) or weight <= 0:
                    return False, f"类别 {class_name} 的权重必须为正数"
                    
            # 检查权重范围是否合理（通常在0.1到10之间）
            weight_values = list(weights.values())
            if max(weight_values) > 50:
                return False, "权重值过大，可能导致训练不稳定"
                
            if min(weight_values) < 0.01:
                return False, "权重值过小，可能影响训练效果"
                
            return True, "权重配置有效"
            
        except Exception as e:
            return False, f"权重验证失败: {str(e)}"
            
    def load_weights_from_file(self, file_path):
        """从文件加载权重配置
        
        Args:
            file_path (str): 权重配置文件路径
            
        Returns:
            dict: 权重配置数据，如果失败返回None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                
            # 验证配置文件格式
            if 'weight_config' not in config_data:
                raise ValueError("无效的权重配置文件格式")
                
            return config_data
            
        except Exception as e:
            self.status_updated.emit(f"加载权重配置失败: {str(e)}")
            return None 