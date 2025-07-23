"""
Prompt Templates for LLM Interactions

This module provides standardized prompt templates for different
types of training analysis and AI assistance scenarios.
"""

from typing import Dict, Any, List
import json


class PromptTemplates:
    """提示词模板管理器"""
    
    # 系统提示词模板
    SYSTEM_PROMPTS = {
        "training_analyst": """
你是一个专业的深度学习训练分析师。你的任务是:
1. 分析训练指标数据，识别训练状态
2. 提供具体的优化建议
3. 诊断常见的训练问题
4. 使用专业但易懂的语言解释

请始终基于提供的数据进行分析，避免过度推测。
回答要简洁明了，重点突出，用中文回答。
        """,
        
        "hyperparameter_optimizer": """
你是一个超参数优化专家。基于训练指标:
1. 分析当前超参数的效果
2. 建议具体的参数调整方案
3. 预测调整后的预期效果
4. 提供调整的优先级排序

请提供可执行的、具体的建议，用中文回答。
        """,
        
        "training_troubleshooter": """
你是一个训练问题诊断专家。你需要:
1. 识别训练中的异常模式
2. 诊断可能的根本原因
3. 提供分步骤的解决方案
4. 建议预防措施

请优先解决最严重的问题，用中文回答。
        """,
        
        "general_assistant": """
你是一个深度学习训练助手。你可以:
1. 回答关于训练过程的问题
2. 解释模型架构和参数
3. 提供最佳实践建议
4. 协助解决技术问题

请保持专业性，用中文回答，并尽量提供具体的建议。
        """
    }
    
    @classmethod
    def get_system_prompt(cls, prompt_type: str) -> str:
        """获取系统提示词"""
        return cls.SYSTEM_PROMPTS.get(prompt_type, cls.SYSTEM_PROMPTS["general_assistant"])
    
    @classmethod
    def build_metrics_analysis_prompt(cls, metrics_data: Dict[str, Any]) -> str:
        """构建指标分析提示词"""
        prompt = f"""
请分析以下CV模型训练指标并提供专业建议:

训练指标:
{json.dumps(metrics_data, ensure_ascii=False, indent=2)}

请分析:
1. 当前训练状态 (收敛情况、过拟合/欠拟合)
2. 关键指标趋势
3. 具体的优化建议
4. 潜在问题诊断

请用中文回答，并保持专业性。
"""
        return prompt
    
    @classmethod
    def build_hyperparameter_tuning_prompt(cls, current_metrics: Dict, history: List[Dict], 
                                         current_params: Dict = None) -> str:
        """构建超参数调优提示词"""
        
        history_summary = cls._summarize_trends(history) if history else "暂无历史数据"
        params_info = f"\n当前超参数:\n{json.dumps(current_params, ensure_ascii=False, indent=2)}" if current_params else ""
        
        prompt = f"""
基于以下训练指标，分析模型训练状态并提供超参数调优建议:

当前指标:
- 训练损失: {current_metrics.get('train_loss', 'N/A')}
- 验证损失: {current_metrics.get('val_loss', 'N/A')}
- 训练准确率: {current_metrics.get('train_accuracy', 'N/A')}
- 验证准确率: {current_metrics.get('val_accuracy', 'N/A')}
- 学习率: {current_metrics.get('learning_rate', 'N/A')}
- GPU内存使用: {current_metrics.get('gpu_memory_used', 'N/A')}GB

历史趋势: {history_summary}{params_info}

请分析:
1. 当前训练状态 (过拟合/欠拟合/正常)
2. 具体的参数调整建议
3. 预期的改进效果
4. 调整的优先级排序

请提供具体的数值建议，用中文回答。
"""
        return prompt
    
    @classmethod
    def build_problem_diagnosis_prompt(cls, metrics_data: Dict, error_info: str = None) -> str:
        """构建问题诊断提示词"""
        error_section = f"\n错误信息:\n{error_info}" if error_info else ""
        
        prompt = f"""
请诊断以下训练过程中的问题:

训练指标:
{json.dumps(metrics_data, ensure_ascii=False, indent=2)}{error_section}

请分析:
1. 识别异常模式和潜在问题
2. 诊断可能的根本原因
3. 提供具体的解决方案
4. 建议预防措施

请按问题严重程度排序，用中文回答。
"""
        return prompt
    
    @classmethod
    def build_model_comparison_prompt(cls, model_results: List[Dict]) -> str:
        """构建模型对比提示词"""
        models_info = "\n".join([
            f"模型 {i+1}: {json.dumps(model, ensure_ascii=False, indent=2)}"
            for i, model in enumerate(model_results)
        ])
        
        prompt = f"""
请对比分析以下多个模型的训练结果:

{models_info}

请分析:
1. 各模型的性能对比
2. 优劣势分析
3. 选择建议
4. 改进方向

请提供客观的对比分析，用中文回答。
"""
        return prompt
    
    @classmethod
    def build_training_strategy_prompt(cls, dataset_info: Dict, model_info: Dict, 
                                     target_metrics: Dict = None) -> str:
        """构建训练策略提示词"""
        target_section = f"\n目标指标:\n{json.dumps(target_metrics, ensure_ascii=False, indent=2)}" if target_metrics else ""
        
        prompt = f"""
基于以下信息，请制定训练策略:

数据集信息:
{json.dumps(dataset_info, ensure_ascii=False, indent=2)}

模型信息:
{json.dumps(model_info, ensure_ascii=False, indent=2)}{target_section}

请提供:
1. 推荐的训练策略
2. 超参数初始设置建议
3. 数据预处理建议
4. 训练监控要点
5. 预期训练时间和资源需求

请提供具体可执行的建议，用中文回答。
"""
        return prompt
    
    @classmethod
    def build_custom_question_prompt(cls, question: str, context: Dict = None) -> str:
        """构建自定义问题提示词"""
        context_section = f"\n相关上下文:\n{json.dumps(context, ensure_ascii=False, indent=2)}" if context else ""
        
        prompt = f"""
用户问题: {question}{context_section}

请基于上下文信息回答用户的问题，如果没有足够的上下文信息，请说明需要哪些额外信息。
用中文回答，保持专业性和准确性。
"""
        return prompt
    
    @staticmethod
    def _summarize_trends(history: List[Dict]) -> str:
        """总结历史趋势"""
        if not history or len(history) < 2:
            return "历史数据不足"
        
        recent = history[-5:]  # 最近5个数据点
        
        # 分析损失趋势
        train_losses = [h.get('train_loss', 0) for h in recent if h.get('train_loss')]
        val_losses = [h.get('val_loss', 0) for h in recent if h.get('val_loss')]
        
        train_trend = "稳定"
        val_trend = "稳定"
        
        if len(train_losses) >= 2:
            if train_losses[-1] < train_losses[0]:
                train_trend = "下降"
            elif train_losses[-1] > train_losses[0]:
                train_trend = "上升"
        
        if len(val_losses) >= 2:
            if val_losses[-1] < val_losses[0]:
                val_trend = "下降"
            elif val_losses[-1] > val_losses[0]:
                val_trend = "上升"
        
        # 分析准确率趋势
        train_accs = [h.get('train_accuracy', 0) for h in recent if h.get('train_accuracy')]
        val_accs = [h.get('val_accuracy', 0) for h in recent if h.get('val_accuracy')]
        
        acc_trend = "稳定"
        if len(val_accs) >= 2:
            if val_accs[-1] > val_accs[0]:
                acc_trend = "提升"
            elif val_accs[-1] < val_accs[0]:
                acc_trend = "下降"
        
        return f"训练损失{train_trend}，验证损失{val_trend}，准确率{acc_trend}"


class PromptBuilder:
    """动态提示词构建器"""
    
    def __init__(self, template_manager: PromptTemplates = None):
        self.template_manager = template_manager or PromptTemplates()
        self.context_history = []
    
    def add_context(self, context: Dict):
        """添加上下文信息"""
        self.context_history.append(context)
        # 保持最近10个上下文
        if len(self.context_history) > 10:
            self.context_history.pop(0)
    
    def build_contextual_prompt(self, base_prompt: str, include_history: bool = True) -> str:
        """构建带上下文的提示词"""
        if not include_history or not self.context_history:
            return base_prompt
        
        context_summary = self._build_context_summary()
        
        enhanced_prompt = f"""
{base_prompt}

相关上下文信息:
{context_summary}

请结合上下文信息提供更准确的分析和建议。
"""
        return enhanced_prompt
    
    def _build_context_summary(self) -> str:
        """构建上下文摘要"""
        if not self.context_history:
            return "无相关上下文"
        
        recent_context = self.context_history[-3:]  # 最近3个上下文
        
        summary_parts = []
        for i, ctx in enumerate(recent_context, 1):
            ctx_type = ctx.get('type', '未知')
            ctx_data = {k: v for k, v in ctx.items() if k != 'type'}
            summary_parts.append(f"{i}. {ctx_type}: {json.dumps(ctx_data, ensure_ascii=False)}")
        
        return "\n".join(summary_parts)
    
    def clear_context(self):
        """清空上下文历史"""
        self.context_history.clear()


# 使用示例
if __name__ == "__main__":
    # 测试提示词模板
    templates = PromptTemplates()
    
    # 测试指标分析提示词
    test_metrics = {
        "epoch": 15,
        "train_loss": 0.234,
        "val_loss": 0.287,
        "train_accuracy": 0.894,
        "val_accuracy": 0.856,
        "learning_rate": 0.001
    }
    
    print("=== 指标分析提示词 ===")
    analysis_prompt = templates.build_metrics_analysis_prompt(test_metrics)
    print(analysis_prompt)
    
    print("\n=== 超参数调优提示词 ===")
    tuning_prompt = templates.build_hyperparameter_tuning_prompt(
        test_metrics, 
        [test_metrics], 
        {"batch_size": 32, "learning_rate": 0.001}
    )
    print(tuning_prompt)
    
    print("\n=== 动态提示词构建器测试 ===")
    builder = PromptBuilder()
    builder.add_context({"type": "训练开始", "model": "ResNet50", "dataset": "CIFAR-10"})
    builder.add_context({"type": "指标更新", "epoch": 10, "accuracy": 0.85})
    
    contextual_prompt = builder.build_contextual_prompt("请分析当前训练状态")
    print(contextual_prompt) 