"""
智能训练参数冲突自动解决器

专门为智能训练设计的参数冲突自动修复机制，避免弹窗中断自动化流程
主要功能：
- 自动检测LLM建议参数与现有配置的冲突
- 智能选择最佳修复策略
- 无需用户交互的自动修复
- 记录修复过程和决策依据
"""

import copy
import json
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from PyQt5.QtCore import QObject, pyqtSignal


@dataclass
class ConflictResolution:
    """冲突解决记录"""
    conflict_type: str
    original_values: Dict[str, Any]
    resolved_values: Dict[str, Any]
    resolution_strategy: str
    confidence_level: float
    reason: str
    timestamp: float


class IntelligentConflictResolver(QObject):
    """智能参数冲突自动解决器"""
    
    # 信号定义
    conflict_resolved = pyqtSignal(dict)  # 冲突解决信号
    resolution_applied = pyqtSignal(dict)  # 修复应用信号
    status_updated = pyqtSignal(str)  # 状态更新信号
    
    def __init__(self):
        super().__init__()
        
        # 解决策略优先级
        self.resolution_strategies = {
            'llm_priority': 0.9,      # 优先采用LLM建议
            'compatibility': 0.8,     # 兼容性修复
            'conservative': 0.7,      # 保守修复
            'disable_conflict': 0.6   # 禁用冲突参数
        }
        
        # 冲突解决历史
        self.resolution_history: List[ConflictResolution] = []
        
        # 参数重要性权重
        self.parameter_importance = {
            'learning_rate': 1.0,
            'batch_size': 0.9,
            'optimizer': 0.8,
            'weight_decay': 0.7,
            'warmup_steps': 0.6,
            'warmup_ratio': 0.6,
            'beta1': 0.5,
            'beta2': 0.5,
            'momentum': 0.5,
            'nesterov': 0.4,
            'mixed_precision': 0.8,
            'loss_scaling_enabled': 0.7,
            'gradient_accumulation_enabled': 0.6,
            'use_augmentation': 0.7,
            'cutmix_prob': 0.5,
            'mixup_alpha': 0.5,
            'label_smoothing_enabled': 0.6,
            'model_ema': 0.6
        }
    
    def resolve_conflicts_automatically(self, 
                                      config: Dict[str, Any],
                                      llm_suggested_changes: Dict[str, Any] = None) -> Tuple[Dict[str, Any], List[ConflictResolution]]:
        """
        自动解决参数冲突
        
        Args:
            config: 当前配置
            llm_suggested_changes: LLM建议的参数变更
            
        Returns:
            tuple: (修复后的配置, 解决记录列表)
        """
        try:
            self.status_updated.emit("🔧 开始智能参数冲突自动修复...")
            
            # 导入验证器进行冲突检测
            from .training_validator import TrainingValidator
            validator = TrainingValidator()
            
            # 检测冲突
            conflicts, suggestions = validator.detect_hyperparameter_conflicts(config)
            
            if not conflicts:
                self.status_updated.emit("✅ 未检测到参数冲突")
                return config, []
            
            self.status_updated.emit(f"🔍 检测到 {len(conflicts)} 个参数冲突，开始自动修复...")
            
            # 自动解决冲突
            resolved_config = copy.deepcopy(config)
            resolutions = []
            
            for i, (conflict, suggestion) in enumerate(zip(conflicts, suggestions)):
                resolution = self._resolve_single_conflict(
                    resolved_config, conflict, suggestion, llm_suggested_changes
                )
                
                if resolution:
                    # 应用修复
                    resolved_config = self._apply_resolution(resolved_config, resolution)
                    resolutions.append(resolution)
                    
                    self.status_updated.emit(
                        f"✅ 已修复冲突 {i+1}/{len(conflicts)}: {resolution.conflict_type}"
                    )
            
            # 记录解决历史
            self.resolution_history.extend(resolutions)
            
            # 发射信号
            self.conflict_resolved.emit({
                'original_config': config,
                'resolved_config': resolved_config,
                'resolutions': [self._resolution_to_dict(r) for r in resolutions],
                'timestamp': time.time()
            })
            
            self.status_updated.emit(f"🎉 智能冲突修复完成，共修复 {len(resolutions)} 个冲突")
            
            return resolved_config, resolutions
            
        except Exception as e:
            self.status_updated.emit(f"❌ 智能冲突修复失败: {str(e)}")
            return config, []
    
    def _resolve_single_conflict(self, 
                               config: Dict[str, Any],
                               conflict: Dict[str, Any],
                               suggestion: Dict[str, Any],
                               llm_changes: Dict[str, Any] = None) -> Optional[ConflictResolution]:
        """解决单个冲突"""
        try:
            conflict_type = conflict.get('type', 'unknown')
            parameter = suggestion.get('parameter', '')
            
            # 确定解决策略
            strategy, confidence = self._determine_resolution_strategy(
                conflict, suggestion, llm_changes
            )
            
            # 根据策略生成解决方案
            resolution_values = self._generate_resolution_values(
                config, conflict, suggestion, strategy, llm_changes
            )
            
            if not resolution_values:
                return None
            
            # 创建解决记录
            resolution = ConflictResolution(
                conflict_type=conflict_type,
                original_values=self._extract_original_values(config, parameter),
                resolved_values=resolution_values,
                resolution_strategy=strategy,
                confidence_level=confidence,
                reason=self._generate_resolution_reason(conflict, suggestion, strategy),
                timestamp=time.time()
            )
            
            return resolution
            
        except Exception as e:
            print(f"[ERROR] 解决单个冲突失败: {str(e)}")
            return None
    
    def _determine_resolution_strategy(self, 
                                     conflict: Dict[str, Any],
                                     suggestion: Dict[str, Any],
                                     llm_changes: Dict[str, Any] = None) -> Tuple[str, float]:
        """确定解决策略"""
        parameter = suggestion.get('parameter', '')
        conflict_type = conflict.get('type', '')
        
        # 如果LLM有相关建议，优先考虑
        if llm_changes and any(param in parameter for param in llm_changes.keys()):
            return 'llm_priority', self.resolution_strategies['llm_priority']
        
        # 根据冲突类型选择策略
        if '不匹配' in conflict_type or '冲突' in conflict_type:
            return 'compatibility', self.resolution_strategies['compatibility']
        elif '无效' in conflict_type or '错误' in conflict_type:
            return 'conservative', self.resolution_strategies['conservative']
        else:
            return 'disable_conflict', self.resolution_strategies['disable_conflict']
    
    def _generate_resolution_values(self, 
                                  config: Dict[str, Any],
                                  conflict: Dict[str, Any],
                                  suggestion: Dict[str, Any],
                                  strategy: str,
                                  llm_changes: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成解决方案的具体参数值"""
        parameter = suggestion.get('parameter', '')
        action = suggestion.get('action', '')
        
        resolution_values = {}
        
        try:
            if strategy == 'llm_priority' and llm_changes:
                # 优先使用LLM建议的值
                for param_name, param_value in llm_changes.items():
                    if param_name in parameter:
                        resolution_values[param_name] = param_value
            
            # 如果LLM建议不足，使用默认修复逻辑
            if not resolution_values:
                resolution_values = self._apply_default_fixes(parameter, action, config)
            
            return resolution_values
            
        except Exception as e:
            print(f"[ERROR] 生成解决方案值失败: {str(e)}")
            return {}
    
    def _apply_default_fixes(self, parameter: str, action: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用默认修复逻辑"""
        fixes = {}
        
        if parameter == 'beta1/beta2':
            fixes['beta1'] = 0.9
            fixes['beta2'] = 0.999
        elif parameter == 'momentum/nesterov':
            fixes['momentum'] = 0.9
            fixes['nesterov'] = False
        elif parameter == 'warmup_ratio':
            if '设置为0.05' in action:
                fixes['warmup_ratio'] = 0.05
            else:
                fixes['warmup_ratio'] = 0.0
        elif parameter == 'warmup_enabled':
            fixes['warmup_enabled'] = False
        elif parameter == 'warmup_steps/warmup_ratio':
            # 智能选择：如果warmup_steps > 0，禁用warmup_ratio
            if config.get('warmup_steps', 0) > 0:
                fixes['warmup_ratio'] = 0.0
            else:
                fixes['warmup_steps'] = 0
        elif parameter == 'advanced_augmentation_enabled':
            fixes['advanced_augmentation_enabled'] = False
        elif parameter == 'cutmix_prob/mixup_alpha':
            fixes['cutmix_prob'] = 0.0
            fixes['mixup_alpha'] = 0.0
        elif parameter == 'cutmix_prob或mixup_alpha':
            # 保留其中一个，优先保留CutMix
            if config.get('cutmix_prob', 0) > 0:
                fixes['mixup_alpha'] = 0.0
            else:
                fixes['cutmix_prob'] = 0.0
        elif parameter == 'gradient_accumulation_steps':
            batch_size = config.get('batch_size', 32)
            fixes['gradient_accumulation_steps'] = max(1, 512 // batch_size)
        elif parameter == 'beta1':
            fixes['beta1'] = 0.9
        elif parameter == 'label_smoothing':
            fixes['label_smoothing'] = 0.0
        elif parameter == 'label_smoothing_enabled':
            fixes['label_smoothing_enabled'] = False
        elif parameter == 'mixed_precision':
            fixes['mixed_precision'] = True
        elif parameter == 'loss_scaling_enabled':
            fixes['loss_scaling_enabled'] = False
        
        return fixes
    
    def _apply_resolution(self, config: Dict[str, Any], resolution: ConflictResolution) -> Dict[str, Any]:
        """应用解决方案到配置"""
        modified_config = copy.deepcopy(config)
        
        for param_name, param_value in resolution.resolved_values.items():
            modified_config[param_name] = param_value
        
        return modified_config
    
    def _extract_original_values(self, config: Dict[str, Any], parameter: str) -> Dict[str, Any]:
        """提取原始参数值"""
        original_values = {}
        
        # 根据参数名提取相关的原始值
        if '/' in parameter:
            # 处理复合参数
            param_names = parameter.replace('或', '/').split('/')
            for param_name in param_names:
                param_name = param_name.strip()
                if param_name in config:
                    original_values[param_name] = config[param_name]
        else:
            if parameter in config:
                original_values[parameter] = config[parameter]
        
        return original_values
    
    def _generate_resolution_reason(self, 
                                  conflict: Dict[str, Any],
                                  suggestion: Dict[str, Any],
                                  strategy: str) -> str:
        """生成解决原因说明"""
        conflict_type = conflict.get('type', 'unknown')
        parameter = suggestion.get('parameter', '')
        
        strategy_reasons = {
            'llm_priority': f"采用LLM建议的{parameter}参数值，以保持智能优化的连续性",
            'compatibility': f"修复{parameter}参数兼容性问题，确保{conflict_type}得到解决",
            'conservative': f"采用保守策略修复{parameter}参数，避免{conflict_type}影响训练",
            'disable_conflict': f"禁用冲突的{parameter}参数，确保训练能够正常进行"
        }
        
        return strategy_reasons.get(strategy, f"自动修复{parameter}参数冲突")
    
    def _resolution_to_dict(self, resolution: ConflictResolution) -> Dict[str, Any]:
        """将解决记录转换为字典"""
        return {
            'conflict_type': resolution.conflict_type,
            'original_values': resolution.original_values,
            'resolved_values': resolution.resolved_values,
            'resolution_strategy': resolution.resolution_strategy,
            'confidence_level': resolution.confidence_level,
            'reason': resolution.reason,
            'timestamp': resolution.timestamp
        }
    
    def get_resolution_history(self) -> List[Dict[str, Any]]:
        """获取解决历史"""
        return [self._resolution_to_dict(r) for r in self.resolution_history]
    
    def clear_history(self):
        """清除解决历史"""
        self.resolution_history.clear()
        self.status_updated.emit("📝 冲突解决历史已清除")
