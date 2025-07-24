"""
LLM Framework Main Class

This module provides the main LLM framework class that integrates all components
and provides a unified interface for AI-assisted training analysis.
"""

import json
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from .model_adapters import LLMAdapter, create_llm_adapter, MockLLMAdapter
from .analysis_engine import TrainingAnalysisEngine
from .prompt_templates import PromptTemplates


class LLMFramework:
    """大语言模型集成框架"""
    
    def __init__(self, 
                 adapter_type: str = 'mock',
                 adapter_config: Dict = None,
                 enable_streaming: bool = True):
        """
        初始化LLM框架
        
        Args:
            adapter_type: 适配器类型 ('openai', 'local', 'mock')
            adapter_config: 适配器配置参数
            enable_streaming: 是否启用流式处理
        """
        self.adapter_type = adapter_type
        self.adapter_config = adapter_config or {}
        self.enable_streaming = enable_streaming
        
        # 初始化组件
        self.llm_adapter = self._create_adapter()
        self.analysis_engine = TrainingAnalysisEngine(self.llm_adapter)
        self.prompt_templates = PromptTemplates()
        
        # 状态管理
        self.is_active = False
        self.processing_queue = []
        self.processing_lock = threading.Lock()
        self.callback_functions = {}
        
        # 性能统计
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0,
            'last_request_time': None
        }
        
        print(f"LLM框架初始化完成，使用适配器: {type(self.llm_adapter).__name__}")
    
    def _create_adapter(self) -> LLMAdapter:
        """创建LLM适配器"""
        try:
            if self.adapter_type == 'mock':
                return MockLLMAdapter()
            else:
                return create_llm_adapter(self.adapter_type, **self.adapter_config)
        except Exception as e:
            print(f"创建适配器失败，使用模拟适配器: {str(e)}")
            return MockLLMAdapter()
    
    def start(self):
        """启动LLM框架"""
        self.is_active = True
        print("LLM框架已启动")
    
    def stop(self):
        """停止LLM框架"""
        self.is_active = False
        print("LLM框架已停止")
    
    def analyze_training_metrics(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析训练指标"""
        if not self.is_active:
            return {'error': 'LLM框架未启动'}
        
        start_time = time.time()
        
        try:
            self.stats['total_requests'] += 1
            
            # 使用分析引擎进行分析
            result = self.analysis_engine.analyze_training_progress(metrics_data)
            
            # 添加框架元信息
            result['framework_info'] = {
                'adapter_type': self.adapter_type,
                'processing_time': time.time() - start_time,
                'timestamp': time.time()
            }
            
            self.stats['successful_requests'] += 1
            self._update_response_time(time.time() - start_time)
            
            # 触发回调
            self._trigger_callback('metrics_analyzed', result)
            
            return result
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            error_result = {
                'error': f'分析失败: {str(e)}',
                'timestamp': time.time(),
                'metrics': metrics_data
            }
            self._trigger_callback('analysis_error', error_result)
            return error_result
    
    def analyze_real_training_metrics(self) -> Dict[str, Any]:
        """分析真实的训练指标（使用实时采集的数据）"""
        if not self.is_active:
            return {'error': 'LLM框架未启动'}
        
        start_time = time.time()
        
        try:
            self.stats['total_requests'] += 1
            
            # 使用分析引擎进行真实数据分析
            result = self.analysis_engine.analyze_real_training_progress()
            
            # 添加框架元信息
            result['framework_info'] = {
                'adapter_type': self.adapter_type,
                'processing_time': time.time() - start_time,
                'timestamp': time.time(),
                'data_source': 'real_time_collector'
            }
            
            self.stats['successful_requests'] += 1
            self._update_response_time(time.time() - start_time)
            
            # 触发回调
            self._trigger_callback('real_metrics_analyzed', result)
            
            return result
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            error_result = {
                'error': f'真实数据分析失败: {str(e)}',
                'framework_info': {
                    'adapter_type': self.adapter_type,
                    'processing_time': time.time() - start_time,
                    'timestamp': time.time(),
                    'data_source': 'real_time_collector'
                }
            }
            self._trigger_callback('real_metrics_analysis_failed', error_result)
            return error_result
    
    def get_real_hyperparameter_suggestions(self) -> Dict[str, Any]:
        """获取基于真实数据的超参数优化建议"""
        if not self.is_active:
            return {'error': 'LLM框架未启动'}
        
        start_time = time.time()
        
        try:
            self.stats['total_requests'] += 1
            
            # 获取真实训练数据进行分析
            real_analysis = self.analysis_engine.analyze_real_training_progress()
            
            if 'error' in real_analysis:
                # 如果无法获取真实数据，返回错误
                return {
                    'error': f'无法获取真实训练数据进行建议生成: {real_analysis["error"]}',
                    'framework_info': {
                        'adapter_type': self.adapter_type,
                        'processing_time': time.time() - start_time,
                        'timestamp': time.time(),
                        'data_source': 'real_time_collector'
                    }
                }
            
            # 从真实分析结果中提取指标和参数
            metrics = real_analysis.get('metrics', {})
            current_params = {
                'epoch': metrics.get('epoch', 0),
                'batch_size': 32,  # 默认值，实际应从配置中获取
                'learning_rate': 0.001  # 默认值，实际应从配置中获取
            }
            
            # 使用分析引擎生成建议
            result = self.analysis_engine.suggest_hyperparameter_tuning(metrics, current_params)
            
            # 添加真实数据上下文
            result['real_data_context'] = {
                'session_id': real_analysis.get('session_id', 'unknown'),
                'data_points': real_analysis.get('raw_data_summary', {}).get('total_data_points', 0),
                'training_duration': real_analysis.get('raw_data_summary', {}).get('collection_duration', 0),
                'training_status': real_analysis.get('raw_data_summary', {}).get('training_status', 'unknown')
            }
            
            # 添加框架元信息
            result['framework_info'] = {
                'adapter_type': self.adapter_type,
                'processing_time': time.time() - start_time,
                'timestamp': time.time(),
                'data_source': 'real_time_collector'
            }
            
            self.stats['successful_requests'] += 1
            self._update_response_time(time.time() - start_time)
            
            # 触发回调
            self._trigger_callback('real_suggestions_generated', result)
            
            return result
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            error_result = {
                'error': f'真实数据建议生成失败: {str(e)}',
                'framework_info': {
                    'adapter_type': self.adapter_type,
                    'processing_time': time.time() - start_time,
                    'timestamp': time.time(),
                    'data_source': 'real_time_collector'
                }
            }
            self._trigger_callback('real_suggestions_failed', error_result)
            return error_result
    
    def diagnose_real_training_problems(self) -> Dict[str, Any]:
        """基于真实数据诊断训练问题"""
        if not self.is_active:
            return {'error': 'LLM框架未启动'}
        
        start_time = time.time()
        
        try:
            self.stats['total_requests'] += 1
            
            # 获取真实训练数据进行分析
            real_analysis = self.analysis_engine.analyze_real_training_progress()
            
            if 'error' in real_analysis:
                # 如果无法获取真实数据，返回错误
                return {
                    'error': f'无法获取真实训练数据进行问题诊断: {real_analysis["error"]}',
                    'framework_info': {
                        'adapter_type': self.adapter_type,
                        'processing_time': time.time() - start_time,
                        'timestamp': time.time(),
                        'data_source': 'real_time_collector'
                    }
                }
            
            # 从真实分析结果中提取指标
            metrics = real_analysis.get('metrics', {})
            trends = real_analysis.get('trends', {})
            
            # 检查是否存在问题指标
            problem_indicators = self._detect_training_problems(metrics, trends)
            
            # 使用分析引擎进行问题诊断
            result = self.analysis_engine.diagnose_training_issues(metrics, problem_indicators)
            
            # 添加真实数据上下文
            result['real_data_context'] = {
                'session_id': real_analysis.get('session_id', 'unknown'),
                'data_points': real_analysis.get('raw_data_summary', {}).get('total_data_points', 0),
                'training_duration': real_analysis.get('raw_data_summary', {}).get('collection_duration', 0),
                'training_status': real_analysis.get('raw_data_summary', {}).get('training_status', 'unknown'),
                'detected_problems': problem_indicators
            }
            
            # 添加框架元信息
            result['framework_info'] = {
                'adapter_type': self.adapter_type,
                'processing_time': time.time() - start_time,
                'timestamp': time.time(),
                'data_source': 'real_time_collector'
            }
            
            self.stats['successful_requests'] += 1
            self._update_response_time(time.time() - start_time)
            
            # 触发回调
            self._trigger_callback('real_diagnosis_completed', result)
            
            return result
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            error_result = {
                'error': f'真实数据问题诊断失败: {str(e)}',
                'framework_info': {
                    'adapter_type': self.adapter_type,
                    'processing_time': time.time() - start_time,
                    'timestamp': time.time(),
                    'data_source': 'real_time_collector'
                }
            }
            self._trigger_callback('real_diagnosis_failed', error_result)
            return error_result
    
    def get_hyperparameter_suggestions(self, 
                                     current_metrics: Dict,
                                     current_params: Dict = None) -> Dict[str, Any]:
        """获取超参数优化建议"""
        if not self.is_active:
            return {'error': 'LLM框架未启动'}
        
        start_time = time.time()
        
        try:
            self.stats['total_requests'] += 1
            
            result = self.analysis_engine.suggest_hyperparameter_tuning(
                current_metrics, current_params
            )
            
            result['framework_info'] = {
                'adapter_type': self.adapter_type,
                'processing_time': time.time() - start_time,
                'timestamp': time.time()
            }
            
            self.stats['successful_requests'] += 1
            self._update_response_time(time.time() - start_time)
            
            self._trigger_callback('suggestions_generated', result)
            
            return result
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            error_result = {
                'error': f'建议生成失败: {str(e)}',
                'timestamp': time.time()
            }
            self._trigger_callback('suggestion_error', error_result)
            return error_result
    
    def diagnose_training_problems(self, 
                                 metrics_data: Dict,
                                 error_info: str = None) -> Dict[str, Any]:
        """诊断训练问题"""
        if not self.is_active:
            return {'error': 'LLM框架未启动'}
        
        start_time = time.time()
        
        try:
            self.stats['total_requests'] += 1
            
            result = self.analysis_engine.diagnose_training_issues(metrics_data, error_info)
            
            result['framework_info'] = {
                'adapter_type': self.adapter_type,
                'processing_time': time.time() - start_time,
                'timestamp': time.time()
            }
            
            self.stats['successful_requests'] += 1
            self._update_response_time(time.time() - start_time)
            
            self._trigger_callback('diagnosis_completed', result)
            
            return result
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            error_result = {
                'error': f'诊断失败: {str(e)}',
                'timestamp': time.time()
            }
            self._trigger_callback('diagnosis_error', error_result)
            return error_result
    
    def chat_with_training_context(self, user_question: str) -> Dict[str, Any]:
        """基于训练上下文的对话"""
        if not self.is_active:
            return {'error': 'LLM框架未启动'}
        
        start_time = time.time()
        
        try:
            self.stats['total_requests'] += 1
            
            result = self.analysis_engine.chat_with_context(user_question)
            
            result['framework_info'] = {
                'adapter_type': self.adapter_type,
                'processing_time': time.time() - start_time,
                'timestamp': time.time()
            }
            
            self.stats['successful_requests'] += 1
            self._update_response_time(time.time() - start_time)
            
            self._trigger_callback('chat_completed', result)
            
            return result
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            error_result = {
                'error': f'对话失败: {str(e)}',
                'timestamp': time.time(),
                'question': user_question
            }
            self._trigger_callback('chat_error', error_result)
            return error_result
    
    def compare_model_results(self, model_results: List[Dict]) -> Dict[str, Any]:
        """对比多个模型结果"""
        if not self.is_active:
            return {'error': 'LLM框架未启动'}
        
        start_time = time.time()
        
        try:
            self.stats['total_requests'] += 1
            
            result = self.analysis_engine.compare_models(model_results)
            
            result['framework_info'] = {
                'adapter_type': self.adapter_type,
                'processing_time': time.time() - start_time,
                'timestamp': time.time()
            }
            
            self.stats['successful_requests'] += 1
            self._update_response_time(time.time() - start_time)
            
            self._trigger_callback('comparison_completed', result)
            
            return result
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            error_result = {
                'error': f'模型对比失败: {str(e)}',
                'timestamp': time.time()
            }
            self._trigger_callback('comparison_error', error_result)
            return error_result
    
    def register_callback(self, event_type: str, callback_func: Callable):
        """注册回调函数"""
        if event_type not in self.callback_functions:
            self.callback_functions[event_type] = []
        self.callback_functions[event_type].append(callback_func)
        print(f"已注册回调函数: {event_type}")
    
    def unregister_callback(self, event_type: str, callback_func: Callable):
        """取消注册回调函数"""
        if event_type in self.callback_functions:
            try:
                self.callback_functions[event_type].remove(callback_func)
                print(f"已取消注册回调函数: {event_type}")
            except ValueError:
                pass
    
    def _trigger_callback(self, event_type: str, data: Dict):
        """触发回调函数"""
        if event_type in self.callback_functions:
            for callback in self.callback_functions[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"回调函数执行失败: {str(e)}")
    
    def _detect_training_problems(self, metrics: Dict, trends: Dict) -> str:
        """检测训练问题指标"""
        problems = []
        
        # 检查过拟合
        train_loss = metrics.get('train_loss', 0)
        val_loss = metrics.get('val_loss', 0)
        if val_loss > train_loss * 1.5:
            problems.append("过拟合: 验证损失明显高于训练损失")
        
        # 检查欠拟合
        train_acc = metrics.get('train_accuracy', 0)
        val_acc = metrics.get('val_accuracy', 0)
        if train_acc < 0.6 and val_acc < 0.6:
            problems.append("欠拟合: 训练和验证准确率都较低")
        
        # 检查收敛问题
        train_losses = trends.get('train_losses', [])
        if len(train_losses) >= 3:
            recent_losses = train_losses[-3:]
            if all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses))):
                problems.append("收敛问题: 训练损失持续上升")
        
        # 检查梯度问题（基于损失变化）
        if len(train_losses) >= 2:
            loss_change = abs(train_losses[-1] - train_losses[-2])
            if loss_change < 1e-6:
                problems.append("可能的梯度消失: 损失变化极小")
            elif loss_change > 1.0:
                problems.append("可能的梯度爆炸: 损失变化过大")
        
        return "; ".join(problems) if problems else "未检测到明显问题"
    
    def _update_response_time(self, response_time: float):
        """更新平均响应时间"""
        if self.stats['average_response_time'] == 0:
            self.stats['average_response_time'] = response_time
        else:
            # 使用指数移动平均
            alpha = 0.1
            self.stats['average_response_time'] = (
                alpha * response_time + 
                (1 - alpha) * self.stats['average_response_time']
            )
        self.stats['last_request_time'] = time.time()
    
    def get_framework_stats(self) -> Dict[str, Any]:
        """获取框架统计信息"""
        return {
            'framework_status': 'active' if self.is_active else 'inactive',
            'adapter_info': {
                'type': self.adapter_type,
                'available': getattr(self.llm_adapter, 'available', True),
                'model_name': getattr(self.llm_adapter, 'model_name', 'unknown')
            },
            'performance_stats': self.stats.copy(),
            'engine_stats': self.analysis_engine.get_engine_stats(),
            'llm_adapter_stats': self.llm_adapter.get_stats()
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        health = {
            'overall_status': 'healthy',
            'components': {},
            'issues': [],
            'recommendations': []
        }
        
        # 检查LLM适配器状态
        adapter_available = getattr(self.llm_adapter, 'available', True)
        health['components']['llm_adapter'] = {
            'status': 'healthy' if adapter_available else 'unhealthy',
            'type': type(self.llm_adapter).__name__
        }
        
        if not adapter_available:
            health['issues'].append('LLM适配器不可用')
            health['recommendations'].append('检查LLM服务连接或API密钥')
        
        # 检查框架状态
        last_request_time = self.stats.get('last_request_time')
        current_time = time.time()
        uptime = current_time - last_request_time if last_request_time is not None else 0
        
        health['components']['framework'] = {
            'status': 'healthy' if self.is_active else 'inactive',
            'uptime': uptime
        }
        
        # 检查错误率
        total_requests = self.stats['total_requests']
        if total_requests > 0:
            error_rate = self.stats['failed_requests'] / total_requests
            if error_rate > 0.1:  # 错误率超过10%
                health['issues'].append(f'错误率过高: {error_rate:.1%}')
                health['recommendations'].append('检查网络连接和LLM服务状态')
        
        # 检查响应时间
        avg_time = self.stats['average_response_time']
        if avg_time > 10:  # 平均响应时间超过10秒
            health['issues'].append(f'响应时间过长: {avg_time:.1f}秒')
            health['recommendations'].append('考虑使用更快的LLM模型或本地部署')
        
        # 更新整体状态
        if health['issues']:
            health['overall_status'] = 'degraded' if len(health['issues']) < 3 else 'unhealthy'
        
        return health
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0,
            'last_request_time': None
        }
        print("统计信息已重置")
    
    def clear_history(self):
        """清空所有历史记录"""
        self.analysis_engine.clear_history()
        print("历史记录已清空")
    
    def switch_adapter(self, adapter_type: str, adapter_config: Dict = None):
        """切换LLM适配器"""
        try:
            old_adapter = type(self.llm_adapter).__name__
            
            self.adapter_type = adapter_type
            self.adapter_config = adapter_config or {}
            
            new_adapter = self._create_adapter()
            self.llm_adapter = new_adapter
            self.analysis_engine.llm = new_adapter
            
            print(f"适配器已从 {old_adapter} 切换到 {type(new_adapter).__name__}")
            
        except Exception as e:
            print(f"切换适配器失败: {str(e)}")
    
    def export_analysis_report(self, include_history: bool = True) -> Dict[str, Any]:
        """导出分析报告"""
        report = {
            'export_time': time.time(),
            'framework_info': {
                'adapter_type': self.adapter_type,
                'status': 'active' if self.is_active else 'inactive'
            },
            'performance_summary': self.get_framework_stats(),
            'system_health': self.get_system_health()
        }
        
        if include_history:
            report['analysis_history'] = self.analysis_engine.analysis_history
            report['metrics_history'] = self.analysis_engine.metrics_buffer
        
        return report


# 工厂函数
def create_llm_framework(config: Dict[str, Any]) -> LLMFramework:
    """创建LLM框架实例"""
    adapter_type = config.get('adapter_type', 'mock')
    adapter_config = config.get('adapter_config', {})
    enable_streaming = config.get('enable_streaming', True)
    
    framework = LLMFramework(
        adapter_type=adapter_type,
        adapter_config=adapter_config,
        enable_streaming=enable_streaming
    )
    
    # 自动启动
    if config.get('auto_start', True):
        framework.start()
    
    return framework


# 使用示例
if __name__ == "__main__":
    # 创建LLM框架实例
    framework = LLMFramework(adapter_type='mock')
    framework.start()
    
    # 测试指标分析
    test_metrics = {
        "epoch": 15,
        "train_loss": 0.234,
        "val_loss": 0.287,
        "train_accuracy": 0.894,
        "val_accuracy": 0.856,
        "learning_rate": 0.001,
        "gpu_memory_used": 6.2,
        "gpu_memory_total": 8.0
    }
    
    print("=== 测试指标分析 ===")
    analysis_result = framework.analyze_training_metrics(test_metrics)
    print(f"分析状态: {analysis_result.get('rule_analysis', {}).get('training_state', '未知')}")
    
    print("\n=== 测试超参数建议 ===")
    suggestion_result = framework.get_hyperparameter_suggestions(
        test_metrics, 
        {"batch_size": 32, "learning_rate": 0.001}
    )
    print(f"建议数量: {len(suggestion_result.get('rule_suggestions', []))}")
    
    print("\n=== 测试对话功能 ===")
    chat_result = framework.chat_with_training_context("当前训练效果如何？")
    print(f"对话响应长度: {len(chat_result.get('response', ''))}")
    
    print("\n=== 框架统计信息 ===")
    stats = framework.get_framework_stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    
    print("\n=== 系统健康状态 ===")
    health = framework.get_system_health()
    print(f"整体状态: {health['overall_status']}")
    
    framework.stop() 