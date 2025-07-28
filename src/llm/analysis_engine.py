"""
Training Analysis Engine

This module provides intelligent analysis capabilities for training metrics,
including performance analysis, optimization suggestions, and problem diagnosis.
"""

import json
import time
import os
import glob
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
        
        # 实时指标采集器 - 延迟导入避免循环依赖
        self.metrics_collector = None
        
        # 训练配置文件缓存
        self.training_config_cache = {}
        self.config_cache_timestamp = 0
        
    def _find_latest_training_config(self) -> Optional[Dict[str, Any]]:
        """查找最新的训练配置文件"""
        try:
            # 查找训练配置文件目录
            config_dirs = [
                "models/params/classification",
                "models/params/detection", 
                "train_config",
                "models/saved_models"
            ]
            
            latest_config = None
            latest_timestamp = 0
            
            for config_dir in config_dirs:
                if not os.path.exists(config_dir):
                    continue
                    
                # 查找配置文件
                patterns = [
                    os.path.join(config_dir, "*_config.json"),
                    os.path.join(config_dir, "*.json")
                ]
                
                for pattern in patterns:
                    config_files = glob.glob(pattern)
                    for config_file in config_files:
                        try:
                            # 获取文件修改时间
                            file_timestamp = os.path.getmtime(config_file)
                            
                            # 读取配置文件
                            with open(config_file, 'r', encoding='utf-8') as f:
                                config_data = json.load(f)
                                
                            # 检查是否是训练配置文件（包含关键训练参数）
                            if self._is_training_config(config_data):
                                if file_timestamp > latest_timestamp:
                                    latest_timestamp = file_timestamp
                                    latest_config = {
                                        'file_path': config_file,
                                        'timestamp': file_timestamp,
                                        'config': config_data
                                    }
                                    
                        except Exception as e:
                            print(f"读取配置文件失败 {config_file}: {str(e)}")
                            continue
            
            return latest_config
            
        except Exception as e:
            print(f"查找训练配置文件时出错: {str(e)}")
            return None
    
    def _is_training_config(self, config_data: Dict[str, Any]) -> bool:
        """判断是否为训练配置文件"""
        # 检查是否包含关键训练参数
        training_keys = [
            'model_name', 'num_epochs', 'batch_size', 'learning_rate',
            'task_type', 'optimizer', 'data_dir'
        ]
        
        return any(key in config_data for key in training_keys)
    
    def _get_training_config_context(self) -> str:
        """获取训练配置上下文"""
        try:
            # 查找最新配置
            latest_config = self._find_latest_training_config()
            
            if not latest_config:
                return "⚠️ **训练配置**: 未找到训练配置文件"
            
            config_data = latest_config['config']
            file_path = latest_config['file_path']
            
            # 构建配置上下文
            context = f"""
## 📋 训练配置信息
**配置文件**: {os.path.basename(file_path)}
**配置时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest_config['timestamp']))}

### 🏗️ 模型配置
- **模型架构**: {config_data.get('model_name', 'N/A')}
- **任务类型**: {config_data.get('task_type', 'N/A')}
- **激活函数**: {config_data.get('activation_function', 'N/A')}
- **Dropout率**: {config_data.get('dropout_rate', 'N/A')}

### ⚙️ 训练参数
- **训练轮数**: {config_data.get('num_epochs', 'N/A')}
- **批次大小**: {config_data.get('batch_size', 'N/A')}
- **学习率**: {config_data.get('learning_rate', 'N/A')}
- **优化器**: {config_data.get('optimizer', 'N/A')}
- **权重衰减**: {config_data.get('weight_decay', 'N/A')}
- **学习率调度**: {config_data.get('lr_scheduler', 'N/A')}

### 🔧 高级配置
- **数据增强**: {'启用' if config_data.get('use_augmentation') else '禁用'}
- **早停机制**: {'启用' if config_data.get('early_stopping') else '禁用'}
- **早停耐心值**: {config_data.get('early_stopping_patience', 'N/A')}
- **梯度裁剪**: {'启用' if config_data.get('gradient_clipping') else '禁用'}
- **混合精度**: {'启用' if config_data.get('mixed_precision') else '禁用'}

### ⚖️ 类别权重配置
- **使用类别权重**: {'是' if config_data.get('use_class_weights') else '否'}
- **权重策略**: {config_data.get('weight_strategy', 'N/A')}
"""
            
            # 添加类别权重详情
            if config_data.get('class_weights'):
                context += "\n**类别权重详情**:\n"
                for class_name, weight in config_data['class_weights'].items():
                    context += f"- {class_name}: {weight}\n"
            
            # 添加高级超参数（如果存在）
            advanced_params = []
            if config_data.get('warmup_enabled'):
                advanced_params.append(f"预热步数: {config_data.get('warmup_steps', 'N/A')}")
            if config_data.get('min_lr_enabled'):
                advanced_params.append(f"最小学习率: {config_data.get('min_lr', 'N/A')}")
            if config_data.get('label_smoothing_enabled'):
                advanced_params.append(f"标签平滑: {config_data.get('label_smoothing', 'N/A')}")
            if config_data.get('model_ema'):
                advanced_params.append(f"模型EMA: {config_data.get('model_ema_decay', 'N/A')}")
            
            if advanced_params:
                context += "\n**高级超参数**:\n"
                for param in advanced_params:
                    context += f"- {param}\n"
            
            # 添加目标检测特有参数（如果存在）
            if config_data.get('task_type') == 'detection':
                context += f"""
### 🎯 目标检测特有参数
- **IOU阈值**: {config_data.get('iou_threshold', 'N/A')}
- **置信度阈值**: {config_data.get('conf_threshold', 'N/A')}
- **分辨率**: {config_data.get('resolution', 'N/A')}
- **马赛克增强**: {'启用' if config_data.get('use_mosaic') else '禁用'}
- **多尺度训练**: {'启用' if config_data.get('use_multiscale') else '禁用'}
- **EMA**: {'启用' if config_data.get('use_ema') else '禁用'}
"""
            
            return context.strip()
            
        except Exception as e:
            return f"⚠️ **训练配置**: 读取配置文件时出错: {str(e)}"
    
    def _build_enhanced_analysis_prompt(self, metrics_data: Dict, trends: Dict, real_data: Dict) -> str:
        """构建增强的分析提示词（包含训练配置）"""
        # 获取训练配置上下文
        config_context = self._get_training_config_context()
        
        return f"""
请基于以下完整信息进行专业的深度学习训练分析：

{config_context}

## 📊 实时训练数据
- 训练会话: {real_data.get('session_id', 'Unknown')}
- 数据采集时长: {real_data.get('collection_duration', 0):.1f}秒
- 数据点数量: {real_data.get('total_data_points', 0)}个
- 训练状态: {real_data.get('training_status', 'unknown')}

## 📈 当前训练指标
- 当前Epoch: {metrics_data.get('epoch', 'N/A')}
- 训练损失: {metrics_data.get('train_loss', 'N/A')}
- 验证损失: {metrics_data.get('val_loss', 'N/A')}
- 训练准确率: {metrics_data.get('train_accuracy', 'N/A')}
- 验证准确率: {metrics_data.get('val_accuracy', 'N/A')}

## 📉 训练趋势分析
- 训练损失趋势: {trends.get('train_losses', [])}
- 验证损失趋势: {trends.get('val_losses', [])}
- 训练准确率趋势: {trends.get('train_accuracies', [])}
- 验证准确率趋势: {trends.get('val_accuracies', [])}

## 🎯 分析要求

请基于以上训练配置和实时数据，提供以下分析：

### 1. 参数配置评估
- 当前参数配置是否合理？
- 是否存在明显的参数冲突或不当设置？
- 针对当前数据集和任务，参数是否需要调整？

### 2. 训练状态诊断
- 当前训练状态（收敛情况、过拟合/欠拟合风险）
- 基于趋势数据的训练进展评估
- 与配置参数的关联分析

### 3. 针对性优化建议
- 基于具体参数配置的优化建议
- 参数调整的优先级和预期效果
- 针对当前配置的具体改进方案

### 4. 潜在问题预警
- 需要关注的潜在问题
- 配置参数可能带来的风险
- 预防措施和建议

请用中文回答，保持专业性和实用性，重点关注参数配置与训练表现的关联性。
"""

    def analyze_real_training_progress(self) -> Dict[str, Any]:
        """分析真实的训练进度（使用实时采集的数据）"""
        try:
            # 延迟导入并获取指标采集器
            if self.metrics_collector is None:
                try:
                    import sys
                    import os
                    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                    from training_components.real_time_metrics_collector import get_global_metrics_collector
                    self.metrics_collector = get_global_metrics_collector()
                except ImportError:
                    # 如果导入失败，直接读取文件
                    return self._analyze_from_file()
            
            # 获取真实的训练数据
            real_data = self.metrics_collector.get_current_training_data_for_ai()
            
            if "error" in real_data:
                return {
                    'error': f"无法获取真实训练数据: {real_data['error']}",
                    'timestamp': time.time(),
                    'data_source': 'real_time_collector'
                }
            
            # 从真实数据中提取关键指标
            current_metrics = real_data.get('current_metrics', {})
            trends = real_data.get('training_trends', {})
            
            # 构建分析用的指标数据
            if current_metrics:
                metrics_data = {
                    'epoch': current_metrics.get('epoch', 0),
                    'train_loss': trends.get('train_losses', [])[-1] if trends.get('train_losses') else current_metrics.get('loss', 0),
                    'val_loss': trends.get('val_losses', [])[-1] if trends.get('val_losses') else current_metrics.get('loss', 0),
                    'train_accuracy': trends.get('train_accuracies', [])[-1] if trends.get('train_accuracies') else current_metrics.get('accuracy', 0),
                    'val_accuracy': trends.get('val_accuracies', [])[-1] if trends.get('val_accuracies') else current_metrics.get('accuracy', 0),
                    'training_duration': real_data.get('collection_duration', 0),
                    'data_points': real_data.get('total_data_points', 0),
                    'training_status': real_data.get('training_status', 'unknown')
                }
            else:
                return {
                    'error': '当前没有可用的训练指标数据',
                    'timestamp': time.time(),
                    'data_source': 'real_time_collector'
                }
            
            # 构建增强的分析提示词，包含训练配置和真实数据上下文
            prompt = self._build_enhanced_analysis_prompt(metrics_data, trends, real_data)
            
            # 获取LLM分析（使用增强的提示词）
            llm_analysis = self.llm.analyze_metrics(metrics_data, prompt)
            
            # 结合规则分析
            rule_analysis = self._rule_based_analysis(metrics_data)
            
            # 生成综合分析结果
            analysis_result = {
                'timestamp': time.time(),
                'data_source': 'real_time_collector',
                'session_id': real_data.get('session_id', 'unknown'),
                'metrics': metrics_data,
                'trends': trends,
                'raw_data_summary': {
                    'total_data_points': real_data.get('total_data_points', 0),
                    'collection_duration': real_data.get('collection_duration', 0),
                    'training_status': real_data.get('training_status', 'unknown')
                },
                'llm_analysis': llm_analysis,
                'rule_analysis': rule_analysis,
                'combined_insights': self._combine_real_data_analyses(llm_analysis, rule_analysis, real_data),
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
                'error': f"真实数据分析过程中出现错误: {str(e)}",
                'timestamp': time.time(),
                'data_source': 'real_time_collector'
            }
        
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
            
            # 构建增强的分析提示词，包含训练配置
            config_context = self._get_training_config_context()
            prompt = f"""
请基于以下完整信息进行专业的深度学习训练分析：

{config_context}

## 📊 训练指标数据
{json.dumps(metrics_data, ensure_ascii=False, indent=2)}

## 🎯 分析要求

请基于以上训练配置和指标数据，提供以下分析：

### 1. 参数配置评估
- 当前参数配置是否合理？
- 是否存在明显的参数冲突或不当设置？
- 针对当前数据集和任务，参数是否需要调整？

### 2. 训练状态诊断
- 当前训练状态（收敛情况、过拟合/欠拟合风险）
- 基于指标数据的训练进展评估
- 与配置参数的关联分析

### 3. 针对性优化建议
- 基于具体参数配置的优化建议
- 参数调整的优先级和预期效果
- 针对当前配置的具体改进方案

### 4. 潜在问题预警
- 需要关注的潜在问题
- 配置参数可能带来的风险
- 预防措施和建议

请用中文回答，保持专业性和实用性，重点关注参数配置与训练表现的关联性。
"""
            
            # 获取LLM分析（使用增强的提示词）
            llm_analysis = self.llm.analyze_metrics(metrics_data, prompt)
            
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
    
    def _build_real_data_analysis_prompt(self, metrics_data: Dict, trends: Dict, real_data: Dict) -> str:
        """构建基于真实数据的分析提示词"""
        return f"""
请基于以下真实训练数据进行专业分析：

## 训练会话信息
- 会话ID: {real_data.get('session_id', 'Unknown')}
- 训练时长: {real_data.get('collection_duration', 0):.1f}秒
- 数据点数量: {real_data.get('total_data_points', 0)}个
- 训练状态: {real_data.get('training_status', 'unknown')}

## 当前训练指标
- 当前Epoch: {metrics_data.get('epoch', 'N/A')}
- 训练损失: {metrics_data.get('train_loss', 'N/A')}
- 验证损失: {metrics_data.get('val_loss', 'N/A')}
- 训练准确率: {metrics_data.get('train_accuracy', 'N/A')}
- 验证准确率: {metrics_data.get('val_accuracy', 'N/A')}

## 训练趋势分析
- 训练损失趋势: {trends.get('train_losses', [])}
- 验证损失趋势: {trends.get('val_losses', [])}
- 训练准确率趋势: {trends.get('train_accuracies', [])}
- 验证准确率趋势: {trends.get('val_accuracies', [])}

请分析：
1. 当前训练状态（收敛情况、过拟合/欠拟合风险）
2. 基于趋势数据的训练进展评估
3. 具体的优化建议
4. 需要关注的潜在问题

请用中文回答，保持专业性和实用性。
"""

    def _combine_real_data_analyses(self, llm_analysis: str, rule_analysis: Dict, real_data: Dict) -> str:
        """结合LLM和规则分析（针对真实数据）"""
        # 获取训练配置信息
        config_context = self._get_training_config_context()
        
        combined = f"""
## 综合分析结果（基于真实训练数据和训练配置）

{config_context}

### 📊 数据来源信息
- 训练会话: {real_data.get('session_id', 'Unknown')}
- 数据采集时长: {real_data.get('collection_duration', 0):.1f}秒
- 总数据点: {real_data.get('total_data_points', 0)}个
- 训练状态: {real_data.get('training_status', 'unknown')}

### 📈 规则分析摘要
- 训练状态: {rule_analysis.get('training_state', '未知')}
- 收敛状态: {rule_analysis.get('convergence_status', '未知')}
- 过拟合风险: {rule_analysis.get('overfitting_risk', '未知')}
- 性能评估: {rule_analysis.get('performance_assessment', '未知')}

### 🤖 AI分析洞察
{llm_analysis}

### 🎯 结论
基于真实训练数据和训练配置的综合判断，当前训练状态为{rule_analysis.get('training_state', '未知')}，
建议关注{rule_analysis.get('convergence_status', '收敛')}情况。

**注意**: 此分析基于实时采集的真实训练数据和训练配置文件，具有较高的准确性和针对性。
"""
        return combined
    
    def _analyze_from_file(self) -> Dict[str, Any]:
        """直接从文件读取数据进行分析（备用方案）"""
        try:
            import glob
            import os
            
            # 查找最新的训练数据文件
            data_dir = "logs/real_time_metrics"
            if not os.path.exists(data_dir):
                return {'error': '训练数据目录不存在'}
                
            pattern = os.path.join(data_dir, "*_metrics.json")
            files = glob.glob(pattern)
            
            if not files:
                return {'error': '没有找到训练数据文件'}
                
            # 获取最新文件
            latest_file = max(files, key=os.path.getmtime)
            
            # 读取文件内容
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 提取最新的关键指标
            metrics_history = data.get("metrics_history", [])
            if not metrics_history:
                return {"error": "训练数据为空"}
                
            # 获取最新的训练指标
            current_metrics = data.get('current_metrics', {})
            
            # 计算训练趋势
            train_losses = []
            val_losses = []
            train_accs = []
            val_accs = []
            epochs = []
            
            for metric in metrics_history[-10:]:  # 最近10个数据点
                if metric.get("phase") == "train":
                    if "loss" in metric:
                        train_losses.append(metric["loss"])
                    if "accuracy" in metric:
                        train_accs.append(metric["accuracy"])
                elif metric.get("phase") == "val":
                    if "loss" in metric:
                        val_losses.append(metric["loss"])
                    if "accuracy" in metric:
                        val_accs.append(metric["accuracy"])
                        
                if "epoch" in metric:
                    epochs.append(metric["epoch"])
            
            # 构建AI分析用的数据结构
            real_data = {
                "session_id": data.get('session_id', 'unknown'),
                "current_metrics": current_metrics,
                "training_trends": {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_accuracies": train_accs,
                    "val_accuracies": val_accs,
                    "epochs": list(set(epochs))[-10:] if epochs else []
                },
                "training_status": data.get("training_status", "unknown"),
                "total_data_points": len(metrics_history),
                "collection_duration": time.time() - (data.get("start_time") or time.time())
            }
            
            # 构建分析用的指标数据
            if current_metrics:
                metrics_data = {
                    'epoch': current_metrics.get('epoch', 0),
                    'train_loss': train_losses[-1] if train_losses else current_metrics.get('loss', 0),
                    'val_loss': val_losses[-1] if val_losses else current_metrics.get('loss', 0),
                    'train_accuracy': train_accs[-1] if train_accs else current_metrics.get('accuracy', 0),
                    'val_accuracy': val_accs[-1] if val_accs else current_metrics.get('accuracy', 0),
                    'training_duration': real_data.get('collection_duration', 0),
                    'data_points': real_data.get('total_data_points', 0),
                    'training_status': real_data.get('training_status', 'unknown')
                }
            else:
                return {'error': '当前没有可用的训练指标数据'}
            
            # 获取LLM分析
            llm_analysis = self.llm.analyze_metrics(metrics_data)
            
            # 结合规则分析
            rule_analysis = self._rule_based_analysis(metrics_data)
            
            # 生成综合分析结果
            analysis_result = {
                'timestamp': time.time(),
                'data_source': 'file_direct_read',
                'session_id': real_data.get('session_id', 'unknown'),
                'metrics': metrics_data,
                'trends': real_data.get('training_trends', {}),
                'raw_data_summary': {
                    'total_data_points': real_data.get('total_data_points', 0),
                    'collection_duration': real_data.get('collection_duration', 0),
                    'training_status': real_data.get('training_status', 'unknown'),
                    'data_file': latest_file
                },
                'llm_analysis': llm_analysis,
                'rule_analysis': rule_analysis,
                'combined_insights': self._combine_real_data_analyses(llm_analysis, rule_analysis, real_data),
                'recommendations': self._generate_recommendations(metrics_data, llm_analysis, rule_analysis),
                'alerts': self._check_alerts(metrics_data)
            }
            
            return analysis_result
            
        except Exception as e:
            return {
                'error': f"直接文件读取分析失败: {str(e)}",
                'timestamp': time.time(),
                'data_source': 'file_direct_read'
            }
    
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