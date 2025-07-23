"""
Training Analysis Engine

This module provides intelligent analysis capabilities for training metrics,
including performance analysis, optimization suggestions, and problem diagnosis.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from .model_adapters import LLMAdapter, MockLLMAdapter
from .prompt_templates import PromptTemplates, PromptBuilder


class TrainingAnalysisEngine:
    """训练分析引擎"""
    
    def __init__(self, llm_adapter: LLMAdapter = None):
        self.llm = llm_adapter or MockLLMAdapter()
        self.prompt_templates = PromptTemplates()
        self.prompt_builder = PromptBuilder()
        self.analysis_history = []
        self.metrics_buffer = []
        
    def analyze_training_progress(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析训练进度"""
        try:
            # 添加到指标缓冲区
            self.metrics_buffer.append({
                'timestamp': time.time(),
                'metrics': metrics_data.copy()
            })
            
            # 保持最近100个指标
            if len(self.metrics_buffer) > 100:
                self.metrics_buffer.pop(0)
            
            # 构建分析提示词
            prompt = self.prompt_templates.build_metrics_analysis_prompt(metrics_data)
            
            # 获取LLM分析
            llm_analysis = self.llm.analyze_metrics(metrics_data)
            
            # 结合规则分析
            rule_analysis = self._rule_based_analysis(metrics_data)
            
            # 生成综合分析结果
            analysis_result = {
                'timestamp': time.time(),
                'metrics': metrics_data,
                'llm_analysis': llm_analysis,
                'rule_analysis': rule_analysis,
                'combined_insights': self._combine_analyses(llm_analysis, rule_analysis),
                'recommendations': self._generate_recommendations(metrics_data, llm_analysis, rule_analysis),
                'alerts': self._check_alerts(metrics_data)
            }
            
            # 添加到分析历史
            self.analysis_history.append(analysis_result)
            if len(self.analysis_history) > 50:
                self.analysis_history.pop(0)
            
            return analysis_result
            
        except Exception as e:
            return {
                'error': f"分析过程中出现错误: {str(e)}",
                'timestamp': time.time(),
                'metrics': metrics_data
            }
    
    def suggest_hyperparameter_tuning(self, current_metrics: Dict, 
                                    current_params: Dict = None) -> Dict[str, Any]:
        """建议超参数调优"""
        try:
            # 获取历史指标
            history = [item['metrics'] for item in self.metrics_buffer[-10:]]
            
            # 构建调优提示词
            prompt = self.prompt_templates.build_hyperparameter_tuning_prompt(
                current_metrics, history, current_params
            )
            
            # 获取LLM建议
            llm_suggestions = self.llm.generate_response(
                prompt, 
                context={'type': 'hyperparameter_tuning', 'metrics': current_metrics}
            )
            
            # 生成规则建议
            rule_suggestions = self._rule_based_hyperparameter_suggestions(
                current_metrics, history, current_params
            )
            
            return {
                'timestamp': time.time(),
                'current_metrics': current_metrics,
                'current_params': current_params,
                'llm_suggestions': llm_suggestions,
                'rule_suggestions': rule_suggestions,
                'priority_actions': self._prioritize_suggestions(rule_suggestions),
                'expected_improvements': self._estimate_improvements(rule_suggestions)
            }
            
        except Exception as e:
            return {
                'error': f"超参数调优建议生成失败: {str(e)}",
                'timestamp': time.time()
            }
    
    def diagnose_training_issues(self, metrics_data: Dict, error_info: str = None) -> Dict[str, Any]:
        """诊断训练问题"""
        try:
            # 检测异常模式
            anomalies = self._detect_anomalies(metrics_data)
            
            # 构建诊断提示词
            prompt = self.prompt_templates.build_problem_diagnosis_prompt(metrics_data, error_info)
            
            # 获取LLM诊断
            llm_diagnosis = self.llm.generate_response(
                prompt,
                context={'type': 'problem_diagnosis', 'anomalies': anomalies}
            )
            
            # 规则诊断
            rule_diagnosis = self._rule_based_diagnosis(metrics_data, anomalies)
            
            return {
                'timestamp': time.time(),
                'metrics': metrics_data,
                'detected_anomalies': anomalies,
                'llm_diagnosis': llm_diagnosis,
                'rule_diagnosis': rule_diagnosis,
                'severity_assessment': self._assess_severity(anomalies),
                'recommended_actions': self._recommend_actions(anomalies, rule_diagnosis),
                'prevention_tips': self._generate_prevention_tips(anomalies)
            }
            
        except Exception as e:
            return {
                'error': f"问题诊断失败: {str(e)}",
                'timestamp': time.time()
            }
    
    def compare_models(self, model_results: List[Dict]) -> Dict[str, Any]:
        """对比多个模型"""
        try:
            # 构建对比提示词
            prompt = self.prompt_templates.build_model_comparison_prompt(model_results)
            
            # 获取LLM对比分析
            llm_comparison = self.llm.generate_response(
                prompt,
                context={'type': 'model_comparison', 'models_count': len(model_results)}
            )
            
            # 规则对比
            rule_comparison = self._rule_based_model_comparison(model_results)
            
            return {
                'timestamp': time.time(),
                'models': model_results,
                'llm_comparison': llm_comparison,
                'rule_comparison': rule_comparison,
                'best_model': self._select_best_model(model_results),
                'performance_ranking': self._rank_models(model_results)
            }
            
        except Exception as e:
            return {
                'error': f"模型对比失败: {str(e)}",
                'timestamp': time.time()
            }
    
    def chat_with_context(self, user_question: str) -> Dict[str, Any]:
        """基于训练上下文的对话"""
        try:
            # 构建上下文
            context = self._build_chat_context()
            
            # 构建问题提示词
            prompt = self.prompt_templates.build_custom_question_prompt(user_question, context)
            
            # 获取LLM回答
            response = self.llm.generate_response(prompt, context)
            
            # 更新上下文
            self.prompt_builder.add_context({
                'type': 'user_question',
                'question': user_question,
                'response': response,
                'timestamp': time.time()
            })
            
            return {
                'timestamp': time.time(),
                'question': user_question,
                'response': response,
                'context_used': context,
                'llm_stats': self.llm.get_stats()
            }
            
        except Exception as e:
            return {
                'error': f"对话处理失败: {str(e)}",
                'timestamp': time.time(),
                'question': user_question
            }
    
    def _rule_based_analysis(self, metrics: Dict) -> Dict[str, Any]:
        """基于规则的分析"""
        analysis = {
            'training_state': 'unknown',
            'convergence_status': 'unknown',
            'overfitting_risk': 'low',
            'performance_assessment': 'unknown'
        }
        
        train_loss = metrics.get('train_loss', 0)
        val_loss = metrics.get('val_loss', 0)
        train_acc = metrics.get('train_accuracy', 0)
        val_acc = metrics.get('val_accuracy', 0)
        epoch = metrics.get('epoch', 0)
        
        # 判断训练状态
        if train_loss > 0 and val_loss > 0:
            if val_loss > train_loss * 1.5:
                analysis['training_state'] = 'overfitting'
                analysis['overfitting_risk'] = 'high'
            elif val_loss < train_loss * 0.8:
                analysis['training_state'] = 'underfitting'
            else:
                analysis['training_state'] = 'normal'
        
        # 判断收敛状态
        if len(self.metrics_buffer) >= 5:
            recent_losses = [item['metrics'].get('train_loss', 0) 
                           for item in self.metrics_buffer[-5:]]
            if len(recent_losses) >= 2:
                loss_change = abs(recent_losses[-1] - recent_losses[0])
                if loss_change < 0.001:
                    analysis['convergence_status'] = 'converged'
                elif recent_losses[-1] < recent_losses[0]:
                    analysis['convergence_status'] = 'converging'
                else:
                    analysis['convergence_status'] = 'diverging'
        
        # 性能评估
        if val_acc > 0.9:
            analysis['performance_assessment'] = 'excellent'
        elif val_acc > 0.8:
            analysis['performance_assessment'] = 'good'
        elif val_acc > 0.7:
            analysis['performance_assessment'] = 'fair'
        else:
            analysis['performance_assessment'] = 'poor'
        
        return analysis
    
    def _rule_based_hyperparameter_suggestions(self, current_metrics: Dict, 
                                             history: List[Dict], 
                                             current_params: Dict = None) -> List[Dict]:
        """基于规则的超参数建议"""
        suggestions = []
        
        train_loss = current_metrics.get('train_loss', 0)
        val_loss = current_metrics.get('val_loss', 0)
        learning_rate = current_metrics.get('learning_rate', 0.001)
        
        # 学习率建议
        if len(history) >= 3:
            recent_losses = [h.get('train_loss', 0) for h in history[-3:]]
            if all(l1 <= l2 for l1, l2 in zip(recent_losses[:-1], recent_losses[1:])):
                suggestions.append({
                    'parameter': 'learning_rate',
                    'current_value': learning_rate,
                    'suggested_value': learning_rate * 0.1,
                    'reason': '训练损失停止下降，建议降低学习率',
                    'priority': 'high'
                })
        
        # 过拟合建议
        if val_loss > train_loss * 1.3:
            suggestions.append({
                'parameter': 'regularization',
                'current_value': 'unknown',
                'suggested_value': 'increase_dropout',
                'reason': '检测到过拟合，建议增加正则化',
                'priority': 'high'
            })
        
        # 批量大小建议
        gpu_memory = current_metrics.get('gpu_memory_used', 0)
        if gpu_memory < 4.0:  # GPU内存使用较低
            suggestions.append({
                'parameter': 'batch_size',
                'current_value': current_params.get('batch_size', 32) if current_params else 32,
                'suggested_value': 64,
                'reason': 'GPU内存使用率较低，可以增加批量大小',
                'priority': 'medium'
            })
        
        return suggestions
    
    def _detect_anomalies(self, metrics: Dict) -> List[Dict]:
        """检测异常模式"""
        anomalies = []
        
        train_loss = metrics.get('train_loss', 0)
        val_loss = metrics.get('val_loss', 0)
        learning_rate = metrics.get('learning_rate', 0)
        
        # 检测NaN或无穷大
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if value != value:  # NaN检测
                    anomalies.append({
                        'type': 'nan_value',
                        'parameter': key,
                        'severity': 'critical',
                        'description': f'{key}出现NaN值'
                    })
                elif abs(value) == float('inf'):
                    anomalies.append({
                        'type': 'infinite_value',
                        'parameter': key,
                        'severity': 'critical',
                        'description': f'{key}出现无穷大值'
                    })
        
        # 检测损失爆炸
        if train_loss > 100:
            anomalies.append({
                'type': 'loss_explosion',
                'parameter': 'train_loss',
                'severity': 'critical',
                'description': '训练损失过大，可能发生梯度爆炸'
            })
        
        # 检测学习率过大
        if learning_rate > 1.0:
            anomalies.append({
                'type': 'high_learning_rate',
                'parameter': 'learning_rate',
                'severity': 'high',
                'description': '学习率过大，可能导致训练不稳定'
            })
        
        return anomalies
    
    def _combine_analyses(self, llm_analysis: str, rule_analysis: Dict) -> str:
        """结合LLM和规则分析"""
        combined = f"""
## 综合分析结果

### 规则分析摘要
- 训练状态: {rule_analysis.get('training_state', '未知')}
- 收敛状态: {rule_analysis.get('convergence_status', '未知')}
- 过拟合风险: {rule_analysis.get('overfitting_risk', '未知')}
- 性能评估: {rule_analysis.get('performance_assessment', '未知')}

### AI分析洞察
{llm_analysis}

### 结论
基于规则分析和AI洞察的综合判断，当前训练状态为{rule_analysis.get('training_state', '未知')}，
建议关注{rule_analysis.get('convergence_status', '收敛')}情况。
"""
        return combined
    
    def _generate_recommendations(self, metrics: Dict, llm_analysis: str, rule_analysis: Dict) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于规则分析的建议
        if rule_analysis.get('training_state') == 'overfitting':
            recommendations.append("增加正则化强度或减少模型复杂度")
        elif rule_analysis.get('training_state') == 'underfitting':
            recommendations.append("增加模型容量或减少正则化")
        
        if rule_analysis.get('convergence_status') == 'diverging':
            recommendations.append("降低学习率或检查梯度裁剪设置")
        
        # 从LLM分析中提取建议（简化版）
        if "学习率" in llm_analysis and "降低" in llm_analysis:
            recommendations.append("考虑调整学习率")
        
        return recommendations
    
    def _check_alerts(self, metrics: Dict) -> List[Dict]:
        """检查警报"""
        alerts = []
        
        # GPU内存警报
        gpu_memory = metrics.get('gpu_memory_used', 0)
        gpu_total = metrics.get('gpu_memory_total', 8)
        if gpu_memory / gpu_total > 0.95:
            alerts.append({
                'type': 'gpu_memory_high',
                'severity': 'warning',
                'message': 'GPU内存使用率超过95%，可能导致内存溢出'
            })
        
        # 训练速度警报
        training_speed = metrics.get('training_speed', 0)
        if training_speed < 0.1:
            alerts.append({
                'type': 'slow_training',
                'severity': 'info',
                'message': '训练速度较慢，可能需要优化'
            })
        
        return alerts
    
    def _build_chat_context(self) -> Dict:
        """构建聊天上下文"""
        context = {
            'current_metrics': self.metrics_buffer[-1]['metrics'] if self.metrics_buffer else {},
            'recent_analysis': self.analysis_history[-1] if self.analysis_history else {},
            'model_info': {
                'llm_type': type(self.llm).__name__,
                'requests_made': self.llm.request_count,
                'total_tokens': getattr(self.llm, 'total_tokens', 0)
            }
        }
        return context
    
    def get_engine_stats(self) -> Dict:
        """获取引擎统计信息"""
        return {
            'analyses_performed': len(self.analysis_history),
            'metrics_processed': len(self.metrics_buffer),
            'llm_stats': self.llm.get_stats(),
            'last_analysis_time': self.analysis_history[-1]['timestamp'] if self.analysis_history else None
        }
    
    def _prioritize_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        """对建议进行优先级排序"""
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        return sorted(suggestions, key=lambda x: priority_order.get(x.get('priority', 'low'), 3))
    
    def _estimate_improvements(self, suggestions: List[Dict]) -> Dict[str, str]:
        """估算改进效果"""
        improvements = {}
        for suggestion in suggestions:
            param = suggestion.get('parameter', '')
            if param == 'learning_rate':
                improvements[param] = "预期训练收敛速度提升20-30%"
            elif param == 'regularization':
                improvements[param] = "预期过拟合风险降低，验证准确率提升5-10%"
            elif param == 'batch_size':
                improvements[param] = "预期训练稳定性提升，GPU利用率提高"
            else:
                improvements[param] = "预期训练效果有所改善"
        return improvements
    
    def _rule_based_diagnosis(self, metrics: Dict, anomalies: List[Dict]) -> Dict[str, Any]:
        """基于规则的问题诊断"""
        diagnosis = {
            'primary_issues': [],
            'secondary_issues': [],
            'root_causes': [],
            'confidence_level': 'medium'
        }
        
        # 分析异常严重程度
        critical_anomalies = [a for a in anomalies if a.get('severity') == 'critical']
        high_anomalies = [a for a in anomalies if a.get('severity') == 'high']
        
        if critical_anomalies:
            diagnosis['primary_issues'].extend([a['description'] for a in critical_anomalies])
            diagnosis['confidence_level'] = 'high'
        
        if high_anomalies:
            diagnosis['secondary_issues'].extend([a['description'] for a in high_anomalies])
        
        # 分析根本原因
        train_loss = metrics.get('train_loss', 0)
        val_loss = metrics.get('val_loss', 0)
        learning_rate = metrics.get('learning_rate', 0)
        
        if train_loss > 10:
            diagnosis['root_causes'].append("学习率可能过高，导致训练不稳定")
        
        if val_loss > train_loss * 2:
            diagnosis['root_causes'].append("模型过拟合，需要增加正则化")
        
        return diagnosis
    
    def _assess_severity(self, anomalies: List[Dict]) -> str:
        """评估问题严重程度"""
        if any(a.get('severity') == 'critical' for a in anomalies):
            return 'critical'
        elif any(a.get('severity') == 'high' for a in anomalies):
            return 'high'
        elif any(a.get('severity') == 'medium' for a in anomalies):
            return 'medium'
        else:
            return 'low'
    
    def _recommend_actions(self, anomalies: List[Dict], diagnosis: Dict) -> List[Dict]:
        """推荐具体行动"""
        actions = []
        
        for anomaly in anomalies:
            if anomaly.get('type') == 'loss_explosion':
                actions.append({
                    'action': 'reduce_learning_rate',
                    'description': '立即将学习率降低到当前值的1/10',
                    'priority': 'immediate',
                    'expected_time': '1分钟'
                })
            elif anomaly.get('type') == 'nan_value':
                actions.append({
                    'action': 'restart_training',
                    'description': '重启训练并检查数据预处理',
                    'priority': 'immediate',
                    'expected_time': '5分钟'
                })
            elif anomaly.get('type') == 'high_learning_rate':
                actions.append({
                    'action': 'adjust_learning_rate',
                    'description': '调整学习率到0.001-0.01范围',
                    'priority': 'high',
                    'expected_time': '2分钟'
                })
        
        return actions
    
    def _generate_prevention_tips(self, anomalies: List[Dict]) -> List[str]:
        """生成预防建议"""
        tips = []
        
        anomaly_types = {a.get('type') for a in anomalies}
        
        if 'loss_explosion' in anomaly_types:
            tips.append("使用梯度裁剪防止梯度爆炸")
            tips.append("采用学习率预热策略")
        
        if 'nan_value' in anomaly_types:
            tips.append("在训练前验证数据集完整性")
            tips.append("使用数值稳定的损失函数")
        
        if 'high_learning_rate' in anomaly_types:
            tips.append("使用学习率调度器自动调整")
            tips.append("进行学习率范围测试找到最优值")
        
        return tips
    
    def _rule_based_model_comparison(self, model_results: List[Dict]) -> Dict[str, Any]:
        """基于规则的模型对比"""
        if not model_results:
            return {'error': '没有模型结果可供对比'}
        
        comparison = {
            'best_accuracy': None,
            'best_loss': None,
            'most_stable': None,
            'fastest_convergence': None,
            'summary': {}
        }
        
        # 找到最佳准确率
        best_acc_model = max(model_results, key=lambda x: x.get('val_accuracy', 0))
        comparison['best_accuracy'] = best_acc_model
        
        # 找到最低损失
        best_loss_model = min(model_results, key=lambda x: x.get('val_loss', float('inf')))
        comparison['best_loss'] = best_loss_model
        
        # 生成摘要
        comparison['summary'] = {
            'total_models': len(model_results),
            'accuracy_range': [
                min(m.get('val_accuracy', 0) for m in model_results),
                max(m.get('val_accuracy', 0) for m in model_results)
            ],
            'loss_range': [
                min(m.get('val_loss', float('inf')) for m in model_results),
                max(m.get('val_loss', 0) for m in model_results)
            ]
        }
        
        return comparison
    
    def _select_best_model(self, model_results: List[Dict]) -> Dict:
        """选择最佳模型"""
        if not model_results:
            return {}
        
        # 综合评分：准确率权重0.7，损失权重0.3
        def calculate_score(model):
            acc = model.get('val_accuracy', 0)
            loss = model.get('val_loss', float('inf'))
            # 损失越小越好，所以取倒数
            loss_score = 1 / (1 + loss) if loss != float('inf') else 0
            return acc * 0.7 + loss_score * 0.3
        
        return max(model_results, key=calculate_score)
    
    def _rank_models(self, model_results: List[Dict]) -> List[Dict]:
        """对模型进行排名"""
        if not model_results:
            return []
        
        def calculate_score(model):
            acc = model.get('val_accuracy', 0)
            loss = model.get('val_loss', float('inf'))
            loss_score = 1 / (1 + loss) if loss != float('inf') else 0
            return acc * 0.7 + loss_score * 0.3
        
        ranked = sorted(model_results, key=calculate_score, reverse=True)
        
        # 添加排名信息
        for i, model in enumerate(ranked, 1):
            model['rank'] = i
            model['score'] = calculate_score(model)
        
        return ranked
    
    def clear_history(self):
        """清空历史记录"""
        self.analysis_history.clear()
        self.metrics_buffer.clear()
        self.prompt_builder.clear_context()


# 使用示例
if __name__ == "__main__":
    # 创建分析引擎
    engine = TrainingAnalysisEngine()
    
    # 测试指标分析
    test_metrics = {
        "epoch": 15,
        "train_loss": 0.234,
        "val_loss": 0.287,
        "train_accuracy": 0.894,
        "val_accuracy": 0.856,
        "learning_rate": 0.001,
        "gpu_memory_used": 6.2,
        "gpu_memory_total": 8.0,
        "training_speed": 1.23
    }
    
    print("=== 训练进度分析测试 ===")
    analysis_result = engine.analyze_training_progress(test_metrics)
    print(f"分析完成，状态: {analysis_result.get('rule_analysis', {}).get('training_state', '未知')}")
    
    print("\n=== 超参数调优建议测试 ===")
    tuning_result = engine.suggest_hyperparameter_tuning(
        test_metrics, 
        {"batch_size": 32, "learning_rate": 0.001}
    )
    print(f"生成了 {len(tuning_result.get('rule_suggestions', []))} 条规则建议")
    
    print("\n=== 对话测试 ===")
    chat_result = engine.chat_with_context("当前训练效果如何？")
    print(f"对话回复长度: {len(chat_result.get('response', ''))}")
    
    print(f"\n=== 引擎统计 ===")
    stats = engine.get_engine_stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2)) 