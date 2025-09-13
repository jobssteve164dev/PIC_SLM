"""
参数调整可解释性引擎

为智能训练系统的参数调整提供详细的可解释性分析
主要功能：
- 为每个参数调整提供详细的理由说明
- 基于训练指标和配置状态生成解释
- 提供参数调整的预期效果和风险评估
- 生成结构化的可解释性报告
"""

import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ParameterType(Enum):
    """参数类型枚举"""
    LEARNING_RATE = "learning_rate"
    BATCH_SIZE = "batch_size"
    OPTIMIZER = "optimizer"
    REGULARIZATION = "regularization"
    DATA_AUGMENTATION = "data_augmentation"
    SCHEDULER = "scheduler"
    LOSS_FUNCTION = "loss_function"
    MODEL_ARCHITECTURE = "model_architecture"


class AdjustmentReason(Enum):
    """调整原因枚举"""
    CONVERGENCE_ISSUE = "convergence_issue"
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    TRAINING_STABILITY = "training_stability"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    CONFLICT_RESOLUTION = "conflict_resolution"


@dataclass
class ParameterExplanation:
    """参数调整解释数据结构"""
    parameter_name: str
    parameter_type: ParameterType
    original_value: Any
    new_value: Any
    adjustment_reason: AdjustmentReason
    detailed_reason: str
    expected_impact: str
    risk_assessment: str
    confidence_level: float  # 0.0 - 1.0
    supporting_evidence: List[str]
    alternative_options: List[Dict[str, Any]]
    implementation_notes: str


class ParameterExplanationEngine:
    """参数调整可解释性引擎"""
    
    def __init__(self):
        self.explanation_templates = self._load_explanation_templates()
        self.parameter_constraints = self._load_parameter_constraints()
        self.performance_benchmarks = self._load_performance_benchmarks()
    
    def generate_parameter_explanations(self, 
                                      original_config: Dict[str, Any],
                                      adjusted_config: Dict[str, Any],
                                      training_metrics: Dict[str, Any],
                                      llm_analysis: Dict[str, Any]) -> List[ParameterExplanation]:
        """生成参数调整的可解释性分析"""
        try:
            explanations = []
            
            # 识别所有变更的参数
            changed_parameters = self._identify_changed_parameters(original_config, adjusted_config)
            
            for param_name, (old_value, new_value) in changed_parameters.items():
                explanation = self._generate_single_parameter_explanation(
                    param_name, old_value, new_value, 
                    original_config, training_metrics, llm_analysis
                )
                if explanation:
                    explanations.append(explanation)
            
            # 按优先级排序
            explanations.sort(key=lambda x: x.confidence_level, reverse=True)
            
            return explanations
            
        except Exception as e:
            print(f"生成参数解释失败: {str(e)}")
            return []
    
    def _identify_changed_parameters(self, 
                                   original_config: Dict[str, Any], 
                                   adjusted_config: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        """识别变更的参数"""
        changed_params = {}
        
        for key in adjusted_config:
            if key in original_config:
                old_value = original_config[key]
                new_value = adjusted_config[key]
                if old_value != new_value:
                    changed_params[key] = (old_value, new_value)
            else:
                # 新增的参数
                changed_params[key] = (None, adjusted_config[key])
        
        return changed_params
    
    def _generate_single_parameter_explanation(self,
                                             param_name: str,
                                             old_value: Any,
                                             new_value: Any,
                                             original_config: Dict[str, Any],
                                             training_metrics: Dict[str, Any],
                                             llm_analysis: Dict[str, Any]) -> Optional[ParameterExplanation]:
        """为单个参数生成解释"""
        try:
            # 确定参数类型
            param_type = self._determine_parameter_type(param_name)
            
            # 确定调整原因
            adjustment_reason = self._determine_adjustment_reason(
                param_name, old_value, new_value, training_metrics, llm_analysis
            )
            
            # 生成详细解释
            detailed_reason = self._generate_detailed_reason(
                param_name, param_type, old_value, new_value, 
                adjustment_reason, training_metrics, original_config
            )
            
            # 评估预期影响
            expected_impact = self._assess_expected_impact(
                param_name, param_type, old_value, new_value, training_metrics
            )
            
            # 风险评估
            risk_assessment = self._assess_risks(
                param_name, param_type, old_value, new_value, original_config
            )
            
            # 计算置信度
            confidence_level = self._calculate_confidence(
                param_name, param_type, adjustment_reason, training_metrics
            )
            
            # 收集支持证据
            supporting_evidence = self._collect_supporting_evidence(
                param_name, training_metrics, llm_analysis
            )
            
            # 生成替代选项
            alternative_options = self._generate_alternative_options(
                param_name, param_type, old_value, new_value
            )
            
            # 实现说明
            implementation_notes = self._generate_implementation_notes(
                param_name, param_type, new_value
            )
            
            return ParameterExplanation(
                parameter_name=param_name,
                parameter_type=param_type,
                original_value=old_value,
                new_value=new_value,
                adjustment_reason=adjustment_reason,
                detailed_reason=detailed_reason,
                expected_impact=expected_impact,
                risk_assessment=risk_assessment,
                confidence_level=confidence_level,
                supporting_evidence=supporting_evidence,
                alternative_options=alternative_options,
                implementation_notes=implementation_notes
            )
            
        except Exception as e:
            print(f"生成参数 {param_name} 的解释失败: {str(e)}")
            return None
    
    def _determine_parameter_type(self, param_name: str) -> ParameterType:
        """确定参数类型"""
        type_mapping = {
            'learning_rate': ParameterType.LEARNING_RATE,
            'batch_size': ParameterType.BATCH_SIZE,
            'optimizer': ParameterType.OPTIMIZER,
            'weight_decay': ParameterType.REGULARIZATION,
            'dropout_rate': ParameterType.REGULARIZATION,
            'advanced_augmentation_enabled': ParameterType.DATA_AUGMENTATION,
            'cutmix_prob': ParameterType.DATA_AUGMENTATION,
            'mixup_alpha': ParameterType.DATA_AUGMENTATION,
            'lr_scheduler': ParameterType.SCHEDULER,
            'gradient_accumulation_steps': ParameterType.REGULARIZATION,
            'class_weights': ParameterType.LOSS_FUNCTION,
        }
        
        return type_mapping.get(param_name, ParameterType.REGULARIZATION)
    
    def _determine_adjustment_reason(self,
                                   param_name: str,
                                   old_value: Any,
                                   new_value: Any,
                                   training_metrics: Dict[str, Any],
                                   llm_analysis: Dict[str, Any]) -> AdjustmentReason:
        """确定调整原因"""
        # 基于训练指标判断
        train_loss = training_metrics.get('loss', 0)
        accuracy = training_metrics.get('accuracy', 0)
        epoch = training_metrics.get('epoch', 0)
        
        # 学习率调整
        if param_name == 'learning_rate':
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                if new_value < old_value:
                    if accuracy < 0.2:  # 准确率极低
                        return AdjustmentReason.CONVERGENCE_ISSUE
                    elif train_loss > 1.5:  # 损失过高
                        return AdjustmentReason.TRAINING_STABILITY
                    else:
                        return AdjustmentReason.PERFORMANCE_OPTIMIZATION
        
        # 数据增强调整
        if param_name in ['advanced_augmentation_enabled', 'cutmix_prob', 'mixup_alpha']:
            if old_value and not new_value:
                return AdjustmentReason.CONFLICT_RESOLUTION
            elif not old_value and new_value:
                return AdjustmentReason.PERFORMANCE_OPTIMIZATION
        
        # 梯度累积调整
        if param_name == 'gradient_accumulation_steps':
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                if new_value < old_value:
                    return AdjustmentReason.RESOURCE_OPTIMIZATION
        
        # 类别权重调整
        if param_name == 'class_weights':
            return AdjustmentReason.PERFORMANCE_OPTIMIZATION
        
        # 默认情况
        return AdjustmentReason.PERFORMANCE_OPTIMIZATION
    
    def _generate_detailed_reason(self,
                                param_name: str,
                                param_type: ParameterType,
                                old_value: Any,
                                new_value: Any,
                                adjustment_reason: AdjustmentReason,
                                training_metrics: Dict[str, Any],
                                original_config: Dict[str, Any]) -> str:
        """生成详细的调整理由"""
        
        # 获取训练状态信息
        accuracy = training_metrics.get('accuracy', 0)
        loss = training_metrics.get('loss', 0)
        epoch = training_metrics.get('epoch', 0)
        
        # 基于参数类型和调整原因生成解释
        if param_name == 'learning_rate':
            if adjustment_reason == AdjustmentReason.CONVERGENCE_ISSUE:
                return f"""学习率从 {old_value} 降低到 {new_value}，主要原因是：
1. 当前验证准确率仅为 {accuracy:.1%}，远低于预期水平
2. 训练损失 {loss:.3f} 表明模型收敛困难
3. 降低学习率有助于模型更稳定地收敛到更好的局部最优解
4. 对于预训练模型微调，较小的学习率通常能获得更好的泛化性能"""
            
            elif adjustment_reason == AdjustmentReason.TRAINING_STABILITY:
                return f"""学习率从 {old_value} 降低到 {new_value}，主要原因是：
1. 当前训练损失 {loss:.3f} 较高，可能存在训练不稳定问题
2. 较小的学习率能减少梯度震荡，提高训练稳定性
3. 有助于模型在训练过程中保持稳定的收敛轨迹"""
        
        elif param_name == 'advanced_augmentation_enabled':
            if adjustment_reason == AdjustmentReason.CONFLICT_RESOLUTION:
                return f"""数据增强从启用改为禁用，主要原因是：
1. 检测到CutMix和MixUp同时启用可能导致过度增强
2. 当前准确率 {accuracy:.1%} 极低，复杂的数据增强可能干扰学习过程
3. 在训练初期，简单的数据增强策略通常更有效
4. 避免过度增强导致的训练不稳定问题"""
        
        elif param_name == 'gradient_accumulation_steps':
            if adjustment_reason == AdjustmentReason.RESOURCE_OPTIMIZATION:
                batch_size = original_config.get('batch_size', 32)
                effective_batch_size_old = batch_size * old_value
                effective_batch_size_new = batch_size * new_value
                return f"""梯度累积步数从 {old_value} 减少到 {new_value}，主要原因是：
1. 原始有效批次大小 {effective_batch_size_old} 可能过大，影响训练稳定性
2. 当前准确率 {accuracy:.1%} 表明需要更频繁的参数更新
3. 较小的有效批次大小有助于模型更快地适应数据分布
4. 减少内存使用，提高训练效率"""
        
        elif param_name == 'class_weights':
            if adjustment_reason == AdjustmentReason.PERFORMANCE_OPTIMIZATION:
                return f"""类别权重从手动设置改为自动计算，主要原因是：
1. 手动设置的权重均为1.0，与启用类别权重功能冲突
2. 自动计算能根据数据集中各类别的实际分布调整权重
3. 有助于解决类别不平衡问题，提高模型在少数类别上的性能
4. 基于数据驱动的权重设置更科学合理"""
        
        elif param_name == 'weight_decay':
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                return f"""权重衰减从 {old_value} 调整到 {new_value}，主要原因是：
1. 与学习率调整保持一致的优化策略
2. 适当的权重衰减有助于防止过拟合
3. 在降低学习率的同时调整正则化强度，保持训练平衡"""
        
        # 默认解释
        return f"""参数 {param_name} 从 {old_value} 调整为 {new_value}，主要原因是：
1. 基于当前训练指标（准确率: {accuracy:.1%}, 损失: {loss:.3f}）的优化
2. 旨在改善模型的训练效果和收敛性能
3. 根据深度学习最佳实践进行的参数调优"""
    
    def _assess_expected_impact(self,
                              param_name: str,
                              param_type: ParameterType,
                              old_value: Any,
                              new_value: Any,
                              training_metrics: Dict[str, Any]) -> str:
        """评估预期影响"""
        
        if param_name == 'learning_rate':
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                if new_value < old_value:
                    return """预期影响：
1. 训练收敛更稳定，减少损失震荡
2. 可能需要更多epoch才能收敛，但最终性能可能更好
3. 降低过拟合风险，提高模型泛化能力
4. 训练时间可能略微增加"""
        
        elif param_name == 'advanced_augmentation_enabled':
            if old_value and not new_value:
                return """预期影响：
1. 训练过程更稳定，减少数据增强带来的噪声
2. 模型能更专注于学习基本的特征表示
3. 训练速度可能略有提升
4. 在数据量充足的情况下，性能影响较小"""
        
        elif param_name == 'gradient_accumulation_steps':
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                if new_value < old_value:
                    return """预期影响：
1. 参数更新更频繁，模型适应更快
2. 训练稳定性可能略有下降，但收敛速度提升
3. 内存使用减少，训练效率提高
4. 在数据量较小时效果更明显"""
        
        elif param_name == 'class_weights':
            return """预期影响：
1. 自动平衡各类别的学习权重
2. 提高模型在少数类别上的识别能力
3. 整体准确率可能提升，特别是类别不平衡的情况下
4. 训练过程更加公平，避免偏向多数类别"""
        
        return "预期将改善模型的训练效果和性能表现"
    
    def _assess_risks(self,
                     param_name: str,
                     param_type: ParameterType,
                     old_value: Any,
                     new_value: Any,
                     original_config: Dict[str, Any]) -> str:
        """评估调整风险"""
        
        risks = []
        
        if param_name == 'learning_rate':
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                if new_value < old_value:
                    risks.append("学习率过低可能导致训练过慢")
                    risks.append("需要更多epoch才能收敛")
        
        elif param_name == 'advanced_augmentation_enabled':
            if old_value and not new_value:
                risks.append("数据增强禁用可能降低模型泛化能力")
                risks.append("在数据量不足时可能影响性能")
        
        elif param_name == 'gradient_accumulation_steps':
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                if new_value < old_value:
                    risks.append("有效批次大小减小可能影响训练稳定性")
                    risks.append("梯度估计的方差可能增加")
        
        if not risks:
            risks.append("调整风险较低，属于常规优化策略")
        
        return "风险评估：\n" + "\n".join(f"- {risk}" for risk in risks)
    
    def _calculate_confidence(self,
                            param_name: str,
                            param_type: ParameterType,
                            adjustment_reason: AdjustmentReason,
                            training_metrics: Dict[str, Any]) -> float:
        """计算调整的置信度"""
        base_confidence = 0.7
        
        # 基于训练指标调整置信度
        accuracy = training_metrics.get('accuracy', 0)
        loss = training_metrics.get('loss', 0)
        
        # 如果指标异常，提高置信度
        if accuracy < 0.2 or loss > 1.5:
            base_confidence += 0.2
        
        # 基于调整原因调整置信度
        if adjustment_reason == AdjustmentReason.CONVERGENCE_ISSUE:
            base_confidence += 0.1
        elif adjustment_reason == AdjustmentReason.CONFLICT_RESOLUTION:
            base_confidence += 0.15
        
        # 基于参数类型调整置信度
        if param_type in [ParameterType.LEARNING_RATE, ParameterType.REGULARIZATION]:
            base_confidence += 0.05
        
        return min(1.0, base_confidence)
    
    def _collect_supporting_evidence(self,
                                   param_name: str,
                                   training_metrics: Dict[str, Any],
                                   llm_analysis: Dict[str, Any]) -> List[str]:
        """收集支持证据"""
        evidence = []
        
        # 从训练指标收集证据
        accuracy = training_metrics.get('accuracy', 0)
        loss = training_metrics.get('loss', 0)
        epoch = training_metrics.get('epoch', 0)
        
        if accuracy < 0.2:
            evidence.append(f"验证准确率 {accuracy:.1%} 远低于预期")
        if loss > 1.5:
            evidence.append(f"训练损失 {loss:.3f} 过高")
        if epoch > 0:
            evidence.append(f"当前训练轮数 {epoch}")
        
        # 从LLM分析收集证据
        if isinstance(llm_analysis, dict):
            if 'analysis' in llm_analysis:
                evidence.append("LLM分析确认需要参数调整")
            if 'suggestions' in llm_analysis:
                evidence.append("LLM提供了具体的优化建议")
        
        return evidence
    
    def _generate_alternative_options(self,
                                    param_name: str,
                                    param_type: ParameterType,
                                    old_value: Any,
                                    new_value: Any) -> List[Dict[str, Any]]:
        """生成替代选项"""
        alternatives = []
        
        if param_name == 'learning_rate':
            if isinstance(new_value, (int, float)):
                alternatives.extend([
                    {
                        'value': new_value * 0.5,
                        'description': '更保守的学习率调整',
                        'pros': '更稳定的训练',
                        'cons': '收敛可能更慢'
                    },
                    {
                        'value': new_value * 2,
                        'description': '更激进的学习率调整',
                        'pros': '可能更快收敛',
                        'cons': '训练可能不稳定'
                    }
                ])
        
        elif param_name == 'advanced_augmentation_enabled':
            alternatives.extend([
                {
                    'value': True,
                    'description': '保持数据增强，但只启用一种',
                    'pros': '保持数据增强的好处',
                    'cons': '需要进一步配置'
                }
            ])
        
        return alternatives
    
    def _generate_implementation_notes(self,
                                     param_name: str,
                                     param_type: ParameterType,
                                     new_value: Any) -> str:
        """生成实现说明"""
        if param_name == 'learning_rate':
            return "学习率调整将立即生效，建议监控前几个epoch的收敛情况"
        elif param_name == 'advanced_augmentation_enabled':
            return "数据增强设置将在下次训练开始时生效"
        elif param_name == 'gradient_accumulation_steps':
            return "梯度累积步数调整将影响内存使用和训练速度"
        elif param_name == 'class_weights':
            return "类别权重将根据数据集自动计算，无需手动设置"
        
        return "参数调整将在下次训练迭代中生效"
    
    def _load_explanation_templates(self) -> Dict[str, Any]:
        """加载解释模板"""
        return {
            'learning_rate': {
                'convergence_issue': "学习率过高导致收敛困难",
                'stability_issue': "学习率过高导致训练不稳定",
                'optimization': "学习率优化以提升性能"
            },
            'data_augmentation': {
                'conflict_resolution': "解决数据增强冲突",
                'optimization': "优化数据增强策略"
            }
        }
    
    def _load_parameter_constraints(self) -> Dict[str, Dict[str, Any]]:
        """加载参数约束"""
        return {
            'learning_rate': {'min': 1e-6, 'max': 0.1, 'recommended': [1e-4, 1e-3]},
            'batch_size': {'min': 1, 'max': 256, 'recommended': [16, 32, 64]},
            'weight_decay': {'min': 0, 'max': 0.01, 'recommended': [1e-4, 1e-3]}
        }
    
    def _load_performance_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """加载性能基准"""
        return {
            'accuracy': {'excellent': 0.9, 'good': 0.8, 'fair': 0.6, 'poor': 0.4},
            'loss': {'excellent': 0.1, 'good': 0.3, 'fair': 0.6, 'poor': 1.0}
        }
    
    def format_explanations_for_report(self, explanations: List[ParameterExplanation]) -> str:
        """格式化解释用于报告"""
        if not explanations:
            return "无参数调整解释"
        
        content = []
        content.append("## 🔍 参数调整详细解释")
        content.append("")
        
        for i, explanation in enumerate(explanations, 1):
            content.append(f"### {i}. {explanation.parameter_name}")
            content.append("")
            
            # 基本信息
            content.append(f"**参数类型**: {explanation.parameter_type.value}")
            content.append(f"**调整原因**: {explanation.adjustment_reason.value}")
            content.append(f"**置信度**: {explanation.confidence_level:.1%}")
            content.append("")
            
            # 详细理由
            content.append("**调整理由**:")
            content.append(explanation.detailed_reason)
            content.append("")
            
            # 预期影响
            content.append("**预期影响**:")
            content.append(explanation.expected_impact)
            content.append("")
            
            # 风险评估
            content.append("**风险评估**:")
            content.append(explanation.risk_assessment)
            content.append("")
            
            # 支持证据
            if explanation.supporting_evidence:
                content.append("**支持证据**:")
                for evidence in explanation.supporting_evidence:
                    content.append(f"- {evidence}")
                content.append("")
            
            # 替代选项
            if explanation.alternative_options:
                content.append("**替代选项**:")
                for alt in explanation.alternative_options:
                    content.append(f"- **{alt['value']}**: {alt['description']}")
                    content.append(f"  - 优点: {alt['pros']}")
                    content.append(f"  - 缺点: {alt['cons']}")
                content.append("")
            
            # 实现说明
            content.append("**实现说明**:")
            content.append(explanation.implementation_notes)
            content.append("")
        
        return "\n".join(content)


# 使用示例
if __name__ == "__main__":
    # 创建解释引擎
    engine = ParameterExplanationEngine()
    
    # 测试数据
    original_config = {
        'learning_rate': 0.001,
        'advanced_augmentation_enabled': True,
        'gradient_accumulation_steps': 4,
        'class_weights': {'划痕': 1.0, '污点': 1.0}
    }
    
    adjusted_config = {
        'learning_rate': 0.0001,
        'advanced_augmentation_enabled': False,
        'gradient_accumulation_steps': 1,
        'class_weights': 'auto'
    }
    
    training_metrics = {
        'accuracy': 0.167,
        'loss': 1.84,
        'epoch': 2
    }
    
    llm_analysis = {
        'reason': 'LLM智能分析建议',
        'analysis': '当前配置存在多个问题需要调整'
    }
    
    # 生成解释
    explanations = engine.generate_parameter_explanations(
        original_config, adjusted_config, training_metrics, llm_analysis
    )
    
    # 格式化输出
    report_content = engine.format_explanations_for_report(explanations)
    print(report_content)

