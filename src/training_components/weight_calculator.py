"""
类别权重计算器 - 负责计算各种类别权重策略

支持的权重策略：
- balanced: 使用sklearn的balanced权重计算
- inverse: 逆频率权重
- log_inverse: 对数逆频率权重（减少权重差异）
- custom: 自定义权重（从配置中读取）
"""

import torch
import numpy as np
import json
import os
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from PyQt5.QtCore import QObject, pyqtSignal


class WeightCalculator(QObject):
    """类别权重计算器"""
    
    # 信号定义
    status_updated = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.class_distribution = None
    
    def calculate_class_weights(self, dataset, class_names, config, device):
        """
        计算类别权重
        
        Args:
            dataset: 训练数据集
            class_names: 类别名称列表
            config: 训练配置
            device: 计算设备
            
        Returns:
            torch.Tensor: 类别权重张量
        """
        weight_strategy = config.get('weight_strategy', 'balanced')
        
        self.status_updated.emit("正在计算类别权重...")
        
        # 验证和诊断权重配置
        if weight_strategy == 'custom':
            self._validate_weight_config(class_names, config)
        
        # 获取所有标签
        all_labels = self._extract_labels_from_dataset(dataset)
        
        # 统计类别分布
        self._calculate_class_distribution(all_labels, class_names)
        
        # 计算权重
        class_weights = self._calculate_weights_by_strategy(
            all_labels, class_names, weight_strategy, config
        )
        
        # 打印权重信息
        self._print_weight_info(class_names, class_weights, weight_strategy)
        
        # 转换为torch张量
        return torch.FloatTensor(class_weights).to(device)
    
    def _extract_labels_from_dataset(self, dataset):
        """从数据集中提取标签"""
        all_labels = []
        if hasattr(dataset, 'targets'):
            # ImageFolder dataset
            all_labels = dataset.targets
        else:
            # 手动遍历数据集获取标签
            for _, label in dataset:
                all_labels.append(label)
        return all_labels
    
    def _calculate_class_distribution(self, all_labels, class_names):
        """计算类别分布"""
        label_counts = Counter(all_labels)
        self.class_distribution = {
            class_names[i]: label_counts.get(i, 0) 
            for i in range(len(class_names))
        }
        
        # 打印类别分布信息
        self.status_updated.emit("类别分布统计:")
        for class_name, count in self.class_distribution.items():
            self.status_updated.emit(f"  {class_name}: {count} 个样本")
    
    def _calculate_weights_by_strategy(self, all_labels, class_names, weight_strategy, config):
        """根据策略计算权重"""
        if weight_strategy == 'balanced':
            return self._calculate_balanced_weights(all_labels, class_names)
        elif weight_strategy == 'inverse':
            return self._calculate_inverse_weights(all_labels, class_names)
        elif weight_strategy == 'log_inverse':
            return self._calculate_log_inverse_weights(all_labels, class_names)
        elif weight_strategy == 'custom':
            return self._calculate_custom_weights(class_names, config)
        else:
            # 默认均等权重
            return np.ones(len(class_names))
    
    def _calculate_balanced_weights(self, all_labels, class_names):
        """计算balanced权重"""
        return compute_class_weight(
            'balanced',
            classes=np.arange(len(class_names)),
            y=all_labels
        )
    
    def _calculate_inverse_weights(self, all_labels, class_names):
        """计算逆频率权重"""
        total_samples = len(all_labels)
        label_counts = Counter(all_labels)
        
        class_weights = []
        for i in range(len(class_names)):
            count = label_counts.get(i, 1)  # 避免除零
            weight = total_samples / (len(class_names) * count)
            class_weights.append(weight)
        
        return np.array(class_weights)
    
    def _calculate_log_inverse_weights(self, all_labels, class_names):
        """计算对数逆频率权重"""
        total_samples = len(all_labels)
        label_counts = Counter(all_labels)
        
        class_weights = []
        for i in range(len(class_names)):
            count = label_counts.get(i, 1)
            weight = np.log(total_samples / count)
            class_weights.append(weight)
        
        return np.array(class_weights)
    
    def _calculate_custom_weights(self, class_names, config):
        """计算自定义权重"""
        custom_weights = self._load_custom_weights(config)
        
        # 构建权重数组
        class_weights = []
        for class_name in class_names:
            weight = custom_weights.get(class_name, 1.0)
            class_weights.append(weight)
        
        class_weights = np.array(class_weights)
        
        # 权重验证
        if all(w == 1.0 for w in class_weights):
            self.status_updated.emit("警告: 所有类别权重都是1.0，相当于未使用权重")
        else:
            self.status_updated.emit(
                f"成功加载自定义权重，权重范围: {min(class_weights):.3f} - {max(class_weights):.3f}"
            )
        
        return class_weights
    
    def _load_custom_weights(self, config):
        """从配置中加载自定义权重"""
        custom_weights = {}
        
        # 1. 从 class_weights 字段读取
        if 'class_weights' in config:
            custom_weights = config.get('class_weights', {})
            self.status_updated.emit("从配置中的class_weights字段读取权重")
        
        # 2. 从 custom_class_weights 字段读取
        elif 'custom_class_weights' in config:
            custom_weights = config.get('custom_class_weights', {})
            self.status_updated.emit("从配置中的custom_class_weights字段读取权重")
        
        # 3. 从外部权重配置文件读取
        elif 'weight_config_file' in config:
            custom_weights = self._load_weights_from_file(config.get('weight_config_file'))
        
        # 4. 从其他策略的权重中读取
        elif 'all_strategies' in config:
            custom_weights = self._load_weights_from_strategies(config.get('all_strategies', {}))
        
        return custom_weights
    
    def _load_weights_from_file(self, weight_config_file):
        """从文件中加载权重"""
        custom_weights = {}
        
        if weight_config_file and os.path.exists(weight_config_file):
            try:
                with open(weight_config_file, 'r', encoding='utf-8') as f:
                    weight_data = json.load(f)
                
                if 'weight_config' in weight_data:
                    custom_weights = weight_data['weight_config'].get('class_weights', {})
                    self.status_updated.emit(f"从权重配置文件读取权重: {weight_config_file}")
                elif 'class_weights' in weight_data:
                    custom_weights = weight_data.get('class_weights', {})
                    self.status_updated.emit(f"从权重配置文件读取权重: {weight_config_file}")
                else:
                    self.status_updated.emit(f"权重配置文件格式不支持: {weight_config_file}")
                    
            except Exception as e:
                self.status_updated.emit(f"读取权重配置文件失败: {str(e)}")
        
        return custom_weights
    
    def _load_weights_from_strategies(self, strategies):
        """从策略配置中加载权重"""
        custom_weights = {}
        
        if 'custom' in strategies:
            custom_weights = strategies['custom']
            self.status_updated.emit("从all_strategies中的custom策略读取权重")
        elif strategies:
            first_strategy = list(strategies.keys())[0]
            custom_weights = strategies[first_strategy]
            self.status_updated.emit(f"使用{first_strategy}策略权重作为自定义权重")
        
        return custom_weights
    
    def _print_weight_info(self, class_names, class_weights, weight_strategy):
        """打印权重信息"""
        self.status_updated.emit(f"使用权重策略: {weight_strategy}")
        for i, (class_name, weight) in enumerate(zip(class_names, class_weights)):
            self.status_updated.emit(f"  {class_name}: 权重 = {weight:.4f}")
    
    def _validate_weight_config(self, class_names, config):
        """验证权重配置是否与数据集类别匹配"""
        self.status_updated.emit("验证权重配置...")
        
        # 收集所有可能的权重源
        weight_sources = self._collect_weight_sources(config)
        
        if not weight_sources:
            self.status_updated.emit("警告: 未找到任何权重配置源，将使用默认权重1.0")
            return
        
        # 检查每个权重源
        self._analyze_weight_sources(weight_sources, class_names)
    
    def _collect_weight_sources(self, config):
        """收集权重配置源"""
        weight_sources = []
        
        if 'class_weights' in config:
            weight_sources.append(('配置中的class_weights', config.get('class_weights', {})))
        
        if 'custom_class_weights' in config:
            weight_sources.append(('配置中的custom_class_weights', config.get('custom_class_weights', {})))
        
        if 'weight_config_file' in config:
            weight_config_file = config.get('weight_config_file')
            if weight_config_file and os.path.exists(weight_config_file):
                try:
                    with open(weight_config_file, 'r', encoding='utf-8') as f:
                        weight_data = json.load(f)
                    
                    if 'weight_config' in weight_data:
                        weight_sources.append(('权重文件(weight_config)', weight_data['weight_config'].get('class_weights', {})))
                    elif 'class_weights' in weight_data:
                        weight_sources.append(('权重文件(class_weights)', weight_data.get('class_weights', {})))
                        
                except Exception as e:
                    self.status_updated.emit(f"无法读取权重配置文件: {str(e)}")
        
        if 'all_strategies' in config:
            strategies = config.get('all_strategies', {})
            if 'custom' in strategies:
                weight_sources.append(('all_strategies中的custom', strategies['custom']))
            elif strategies:
                first_strategy = list(strategies.keys())[0]
                weight_sources.append((f'all_strategies中的{first_strategy}', strategies[first_strategy]))
        
        return weight_sources
    
    def _analyze_weight_sources(self, weight_sources, class_names):
        """分析权重配置源"""
        dataset_classes = set(class_names)
        
        for source_name, weights_dict in weight_sources:
            self.status_updated.emit(f"检查权重源: {source_name}")
            
            config_classes = set(weights_dict.keys())
            missing_in_config = dataset_classes - config_classes
            extra_in_config = config_classes - dataset_classes
            matching_classes = dataset_classes & config_classes
            
            self.status_updated.emit(f"  数据集类别数: {len(dataset_classes)}")
            self.status_updated.emit(f"  配置中类别数: {len(config_classes)}")
            self.status_updated.emit(f"  匹配类别数: {len(matching_classes)}")
            
            if missing_in_config:
                missing_list = list(missing_in_config)[:3]
                suffix = '...' if len(missing_in_config) > 3 else ''
                self.status_updated.emit(f"  数据集中有但配置中缺失的类别: {missing_list}{suffix}")
            
            if extra_in_config:
                extra_list = list(extra_in_config)[:3]
                suffix = '...' if len(extra_in_config) > 3 else ''
                self.status_updated.emit(f"  配置中有但数据集中不存在的类别: {extra_list}{suffix}")
            
            # 权重统计
            if weights_dict:
                weight_values = list(weights_dict.values())
                self.status_updated.emit(f"  权重范围: {min(weight_values):.3f} - {max(weight_values):.3f}")
                self.status_updated.emit(f"  权重均值: {sum(weight_values)/len(weight_values):.3f}")
                
                if len(set(weight_values)) == 1:
                    self.status_updated.emit(f"  注意: 所有权重都相同 ({weight_values[0]:.3f})")
            
            # 推荐权重源
            if len(matching_classes) == len(dataset_classes) and not extra_in_config:
                self.status_updated.emit(f"  ✓ 推荐权重源: {source_name} (完全匹配)")
                break
            elif len(matching_classes) > 0:
                self.status_updated.emit(f"  ○ 可用权重源: {source_name} (部分匹配)")
            else:
                self.status_updated.emit(f"  ✗ 不可用权重源: {source_name} (无匹配)")
    
    def get_class_distribution(self):
        """获取类别分布"""
        return self.class_distribution 