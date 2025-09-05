"""
智能训练控制器 - 主动训练模型的核心组件

主要功能：
- 基于LLM分析实时监控训练状态
- 自动停止训练并生成优化参数
- 自动重启训练使用新参数
- 多轮训练参数迭代优化
- 智能训练策略管理
"""

import os
import json
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from PyQt5.QtCore import QObject, pyqtSignal, QTimer, QThread
from PyQt5.QtWidgets import QMessageBox

from .real_time_metrics_collector import get_global_metrics_collector
from ..llm.llm_framework import LLMFramework
from ..llm.analysis_engine import TrainingAnalysisEngine


@dataclass
class TrainingIntervention:
    """训练干预记录"""
    intervention_id: str
    timestamp: float
    trigger_reason: str
    original_metrics: Dict[str, Any]
    analysis_result: Dict[str, Any]
    suggested_params: Dict[str, Any]
    intervention_type: str  # 'auto_stop', 'parameter_tuning', 'restart'
    status: str  # 'pending', 'executing', 'completed', 'failed'


@dataclass
class TrainingSession:
    """训练会话记录"""
    session_id: str
    start_time: float
    end_time: Optional[float]
    total_epochs: int
    completed_epochs: int
    interventions: List[TrainingIntervention]
    final_metrics: Optional[Dict[str, Any]]
    status: str  # 'running', 'stopped', 'completed', 'failed'


class IntelligentTrainingController(QObject):
    """智能训练控制器"""
    
    # 信号定义
    intervention_triggered = pyqtSignal(dict)  # 干预触发信号
    training_restarted = pyqtSignal(dict)  # 训练重启信号
    analysis_completed = pyqtSignal(dict)  # 分析完成信号
    status_updated = pyqtSignal(str)  # 状态更新信号
    error_occurred = pyqtSignal(str)  # 错误信号
    
    def __init__(self, training_system=None):
        super().__init__()
        self.training_system = training_system
        self.llm_framework = None
        self.analysis_engine = None
        self.metrics_collector = None
        
        # 训练会话管理
        self.current_session: Optional[TrainingSession] = None
        self.intervention_history: List[TrainingIntervention] = []
        
        # 配置参数
        self.config = {
            'auto_intervention_enabled': True,
            'intervention_thresholds': {
                'overfitting_risk': 0.8,  # 过拟合风险阈值
                'underfitting_risk': 0.7,  # 欠拟合风险阈值
                'stagnation_epochs': 5,    # 停滞轮数阈值
                'divergence_threshold': 2.0,  # 发散阈值
                'min_training_epochs': 3,   # 最小训练轮数
            },
            'analysis_interval': 10,  # 分析间隔（epoch）
            'max_interventions_per_session': 3,  # 每会话最大干预次数
            'parameter_tuning_strategy': 'conservative',  # 参数调优策略
        }
        
        # 状态管理
        self.is_monitoring = False
        self.monitoring_thread = None
        self.last_analysis_time = 0
        
        # 初始化组件
        self._initialize_components()
        
    def _initialize_components(self):
        """初始化相关组件"""
        try:
            # 从AI设置中读取适配器配置
            adapter_type, adapter_config = self._load_ai_adapter_config()
            
            # 初始化LLM框架
            self.llm_framework = LLMFramework(
                adapter_type=adapter_type,
                adapter_config=adapter_config
            )
            if not self.llm_framework.is_active:
                self.llm_framework.start()
            
            # 初始化分析引擎
            self.analysis_engine = TrainingAnalysisEngine(self.llm_framework.llm_adapter)
            
            # 获取指标采集器
            self.metrics_collector = get_global_metrics_collector()
            
            self.status_updated.emit(f"智能训练控制器初始化完成，使用适配器: {adapter_type}")
            
        except Exception as e:
            self.error_occurred.emit(f"初始化组件失败: {str(e)}")
    
    def _load_ai_adapter_config(self) -> tuple:
        """从AI设置中加载适配器配置"""
        try:
            ai_config_file = "setting/ai_config.json"
            if os.path.exists(ai_config_file):
                with open(ai_config_file, 'r', encoding='utf-8') as f:
                    ai_config = json.load(f)
                
                # 获取默认适配器类型
                general_config = ai_config.get('general', {})
                default_adapter = general_config.get('default_adapter', 'mock')
                
                # 根据适配器类型构建配置
                if default_adapter == 'openai':
                    openai_config = ai_config.get('openai', {})
                    adapter_config = {
                        'api_key': openai_config.get('api_key', ''),
                        'base_url': openai_config.get('base_url', ''),
                        'model': openai_config.get('model', 'gpt-3.5-turbo'),
                        'temperature': openai_config.get('temperature', 0.7),
                        'max_tokens': openai_config.get('max_tokens', 1000)
                    }
                    return 'openai', adapter_config
                    
                elif default_adapter == 'deepseek':
                    deepseek_config = ai_config.get('deepseek', {})
                    adapter_config = {
                        'api_key': deepseek_config.get('api_key', ''),
                        'base_url': deepseek_config.get('base_url', 'https://api.deepseek.com/v1'),
                        'model': deepseek_config.get('model', 'deepseek-chat'),
                        'temperature': deepseek_config.get('temperature', 0.7),
                        'max_tokens': deepseek_config.get('max_tokens', 1000)
                    }
                    return 'deepseek', adapter_config
                    
                elif default_adapter == 'local':
                    ollama_config = ai_config.get('ollama', {})
                    adapter_config = {
                        'base_url': ollama_config.get('base_url', 'http://localhost:11434'),
                        'model': ollama_config.get('model', 'llama2'),
                        'temperature': ollama_config.get('temperature', 0.7),
                        'num_predict': ollama_config.get('num_predict', 1000),
                        'timeout': ollama_config.get('timeout', 120)
                    }
                    return 'ollama', adapter_config
                    
                elif default_adapter == 'custom':
                    custom_config = ai_config.get('custom_api', {})
                    adapter_config = {
                        'api_key': custom_config.get('api_key', ''),
                        'base_url': custom_config.get('base_url', ''),
                        'model': custom_config.get('model', ''),
                        'temperature': custom_config.get('temperature', 0.7),
                        'max_tokens': custom_config.get('max_tokens', 1000)
                    }
                    return 'custom', adapter_config
            
            # 默认使用mock适配器
            return 'mock', {}
            
        except Exception as e:
            print(f"加载AI适配器配置失败: {str(e)}")
            return 'mock', {}
    
    def start_monitoring(self, training_config: Dict[str, Any]):
        """开始智能监控训练"""
        if self.is_monitoring:
            self.status_updated.emit("监控已在运行中")
            return
        
        try:
            # 创建新的训练会话
            self.current_session = TrainingSession(
                session_id=f"session_{int(time.time())}",
                start_time=time.time(),
                end_time=None,
                total_epochs=training_config.get('num_epochs', 20),
                completed_epochs=0,
                interventions=[],
                final_metrics=None,
                status='running'
            )
            
            # 启动监控线程
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            
            self.status_updated.emit("智能训练监控已启动")
            
        except Exception as e:
            self.error_occurred.emit(f"启动监控失败: {str(e)}")
    
    def stop_monitoring(self):
        """停止智能监控"""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        if self.current_session:
            self.current_session.end_time = time.time()
            self.current_session.status = 'stopped'
        
        self.status_updated.emit("智能训练监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 检查是否需要分析
                if self._should_analyze():
                    self._perform_intelligent_analysis()
                
                # 检查是否需要干预
                if self._should_intervene():
                    self._trigger_intervention()
                
                time.sleep(5)  # 每5秒检查一次
                
            except Exception as e:
                self.error_occurred.emit(f"监控循环出错: {str(e)}")
                time.sleep(10)  # 出错后等待更长时间
    
    def _should_analyze(self) -> bool:
        """判断是否应该进行分析"""
        if not self.current_session:
            return False
        
        # 检查分析间隔
        current_time = time.time()
        if current_time - self.last_analysis_time < self.config['analysis_interval'] * 60:  # 转换为秒
            return False
        
        # 检查是否有新的训练数据
        if self.metrics_collector:
            try:
                data = self.metrics_collector.get_current_training_data_for_ai()
                if 'error' not in data:
                    return True
            except:
                pass
        
        return False
    
    def _should_intervene(self) -> bool:
        """判断是否应该进行干预"""
        if not self.current_session:
            return False
        
        # 检查干预次数限制
        if len(self.current_session.interventions) >= self.config['max_interventions_per_session']:
            return False
        
        # 检查最小训练轮数
        if self.current_session.completed_epochs < self.config['intervention_thresholds']['min_training_epochs']:
            return False
        
        # 基于实时指标判断是否需要干预
        if self.metrics_collector:
            try:
                data = self.metrics_collector.get_current_training_data_for_ai()
                if 'error' not in data:
                    return self._evaluate_intervention_need(data)
            except:
                pass
        
        return False
    
    def _evaluate_intervention_need(self, training_data: Dict[str, Any]) -> bool:
        """评估是否需要干预"""
        try:
            metrics = training_data.get('current_metrics', {})
            trends = training_data.get('training_trends', {})
            
            # 检查过拟合风险
            if self._check_overfitting_risk(metrics, trends):
                return True
            
            # 检查欠拟合风险
            if self._check_underfitting_risk(metrics, trends):
                return True
            
            # 检查训练停滞
            if self._check_training_stagnation(trends):
                return True
            
            # 检查训练发散
            if self._check_training_divergence(metrics, trends):
                return True
            
        except Exception as e:
            self.error_occurred.emit(f"评估干预需求时出错: {str(e)}")
        
        return False
    
    def _check_overfitting_risk(self, metrics: Dict, trends: Dict) -> bool:
        """检查过拟合风险"""
        try:
            train_losses = trends.get('train_losses', [])
            val_losses = trends.get('val_losses', [])
            
            if len(train_losses) >= 3 and len(val_losses) >= 3:
                # 检查验证损失是否持续上升
                recent_val_losses = val_losses[-3:]
                if all(recent_val_losses[i] > recent_val_losses[i-1] for i in range(1, len(recent_val_losses))):
                    return True
                
                # 检查训练损失和验证损失的差距
                if len(train_losses) > 0 and len(val_losses) > 0:
                    gap = abs(train_losses[-1] - val_losses[-1])
                    if gap > self.config['intervention_thresholds']['overfitting_risk']:
                        return True
        
        except Exception:
            pass
        
        return False
    
    def _check_underfitting_risk(self, metrics: Dict, trends: Dict) -> bool:
        """检查欠拟合风险"""
        try:
            train_losses = trends.get('train_losses', [])
            val_losses = trends.get('val_losses', [])
            
            if len(train_losses) >= 3 and len(val_losses) >= 3:
                # 检查训练损失是否持续很高
                recent_train_losses = train_losses[-3:]
                if all(loss > self.config['intervention_thresholds']['underfitting_risk'] for loss in recent_train_losses):
                    return True
                
                # 检查训练和验证损失都很高
                if len(train_losses) > 0 and len(val_losses) > 0:
                    if (train_losses[-1] > self.config['intervention_thresholds']['underfitting_risk'] and 
                        val_losses[-1] > self.config['intervention_thresholds']['underfitting_risk']):
                        return True
        
        except Exception:
            pass
        
        return False
    
    def _check_training_stagnation(self, trends: Dict) -> bool:
        """检查训练停滞"""
        try:
            train_losses = trends.get('train_losses', [])
            val_losses = trends.get('val_losses', [])
            
            if len(train_losses) >= self.config['intervention_thresholds']['stagnation_epochs']:
                # 检查训练损失是否长时间没有改善
                recent_losses = train_losses[-self.config['intervention_thresholds']['stagnation_epochs']:]
                if all(abs(recent_losses[i] - recent_losses[i-1]) < 0.001 for i in range(1, len(recent_losses))):
                    return True
        
        except Exception:
            pass
        
        return False
    
    def _check_training_divergence(self, metrics: Dict, trends: Dict) -> bool:
        """检查训练发散"""
        try:
            train_losses = trends.get('train_losses', [])
            val_losses = trends.get('val_losses', [])
            
            if len(train_losses) >= 2 and len(val_losses) >= 2:
                # 检查损失是否急剧上升
                if len(train_losses) >= 2:
                    loss_change = train_losses[-1] - train_losses[-2]
                    if loss_change > self.config['intervention_thresholds']['divergence_threshold']:
                        return True
                
                if len(val_losses) >= 2:
                    val_loss_change = val_losses[-1] - val_losses[-2]
                    if val_loss_change > self.config['intervention_thresholds']['divergence_threshold']:
                        return True
        
        except Exception:
            pass
        
        return False
    
    def _perform_intelligent_analysis(self):
        """执行智能分析"""
        try:
            self.last_analysis_time = time.time()
            
            # 获取实时训练数据
            if not self.metrics_collector:
                return
            
            training_data = self.metrics_collector.get_current_training_data_for_ai()
            if 'error' in training_data:
                return
            
            # 使用LLM进行分析
            if self.llm_framework and self.llm_framework.is_active:
                analysis_result = self.llm_framework.analyze_real_training_metrics()
                
                if 'error' not in analysis_result:
                    # 发射分析完成信号
                    self.analysis_completed.emit({
                        'session_id': self.current_session.session_id if self.current_session else 'unknown',
                        'analysis_result': analysis_result,
                        'training_data': training_data,
                        'timestamp': self.last_analysis_time
                    })
                    
                    self.status_updated.emit("智能分析完成")
            
        except Exception as e:
            self.error_occurred.emit(f"执行智能分析时出错: {str(e)}")
    
    def _trigger_intervention(self):
        """触发训练干预"""
        try:
            if not self.current_session:
                return
            
            # 获取当前训练数据
            if not self.metrics_collector:
                return
            
            training_data = self.metrics_collector.get_current_training_data_for_ai()
            if 'error' in training_data:
                return
            
            # 创建干预记录
            intervention = TrainingIntervention(
                intervention_id=f"intervention_{int(time.time())}",
                timestamp=time.time(),
                trigger_reason="智能检测到训练问题",
                original_metrics=training_data.get('current_metrics', {}),
                analysis_result={},
                suggested_params={},
                intervention_type='auto_stop',
                status='pending'
            )
            
            # 添加到会话记录
            self.current_session.interventions.append(intervention)
            
            # 发射干预触发信号
            self.intervention_triggered.emit(asdict(intervention))
            
            # 执行干预
            self._execute_intervention(intervention)
            
        except Exception as e:
            self.error_occurred.emit(f"触发干预时出错: {str(e)}")
    
    def _execute_intervention(self, intervention: TrainingIntervention):
        """执行干预"""
        try:
            intervention.status = 'executing'
            
            # 1. 停止当前训练
            if self.training_system:
                self.training_system.stop()
                self.status_updated.emit("已自动停止训练")
            
            # 2. 使用LLM生成优化参数
            if self.llm_framework and self.llm_framework.is_active:
                # 获取超参数优化建议
                suggestions = self.llm_framework.get_real_hyperparameter_suggestions()
                
                if 'error' not in suggestions:
                    intervention.analysis_result = suggestions
                    
                    # 提取建议的参数
                    suggested_params = self._extract_suggested_params(suggestions)
                    intervention.suggested_params = suggested_params
                    
                    # 3. 自动重启训练
                    if suggested_params:
                        self._restart_training_with_params(suggested_params)
                        intervention.status = 'completed'
                        self.status_updated.emit("已使用优化参数重启训练")
                    else:
                        intervention.status = 'failed'
                        self.error_occurred.emit("无法提取有效的参数建议")
                else:
                    intervention.status = 'failed'
                    self.error_occurred.emit(f"获取参数建议失败: {suggestions.get('error', '未知错误')}")
            
        except Exception as e:
            intervention.status = 'failed'
            self.error_occurred.emit(f"执行干预时出错: {str(e)}")
    
    def _extract_suggested_params(self, suggestions: Dict[str, Any]) -> Dict[str, Any]:
        """从LLM建议中提取参数"""
        try:
            # 这里需要根据实际的LLM输出格式来解析
            # 暂时返回一个示例结构
            suggested_params = {}
            
            # 尝试从建议中提取具体数值
            if 'suggestions' in suggestions:
                for suggestion in suggestions['suggestions']:
                    if 'parameter' in suggestion and 'value' in suggestion:
                        suggested_params[suggestion['parameter']] = suggestion['value']
            
            # 如果没有找到具体建议，使用默认的保守调整
            if not suggested_params:
                suggested_params = self._generate_conservative_params()
            
            return suggested_params
            
        except Exception as e:
            self.error_occurred.emit(f"提取建议参数时出错: {str(e)}")
            return self._generate_conservative_params()
    
    def _generate_conservative_params(self) -> Dict[str, Any]:
        """生成保守的参数调整"""
        return {
            'learning_rate': 0.0005,  # 降低学习率
            'batch_size': 16,         # 减小批次大小
            'dropout_rate': 0.3,      # 增加dropout
            'weight_decay': 0.0005,   # 增加权重衰减
        }
    
    def _restart_training_with_params(self, new_params: Dict[str, Any]):
        """使用新参数重启训练"""
        try:
            if not self.training_system:
                self.error_occurred.emit("训练系统不可用")
                return
            
            # 发射训练重启信号
            self.training_restarted.emit({
                'session_id': self.current_session.session_id if self.current_session else 'unknown',
                'new_params': new_params,
                'timestamp': time.time(),
                'intervention_count': len(self.current_session.interventions) if self.current_session else 0
            })
            
            # 这里需要实际的训练重启逻辑
            # 由于训练系统的复杂性，建议通过信号让UI层处理
            self.status_updated.emit("训练重启信号已发送，请检查UI界面")
            
        except Exception as e:
            self.error_occurred.emit(f"重启训练时出错: {str(e)}")
    
    def update_training_progress(self, metrics: Dict[str, Any]):
        """更新训练进度"""
        if self.current_session:
            # 更新完成的轮数
            if 'epoch' in metrics:
                self.current_session.completed_epochs = metrics['epoch']
            
            # 更新最终指标
            self.current_session.final_metrics = metrics
    
    def get_intervention_history(self) -> List[Dict[str, Any]]:
        """获取干预历史"""
        return [asdict(intervention) for intervention in self.intervention_history]
    
    def get_current_session_info(self) -> Optional[Dict[str, Any]]:
        """获取当前会话信息"""
        if self.current_session:
            return asdict(self.current_session)
        return None
    
    def save_session_report(self, file_path: str):
        """保存会话报告"""
        try:
            if not self.current_session:
                return
            
            report = {
                'session_info': asdict(self.current_session),
                'intervention_history': self.get_intervention_history(),
                'config': self.config,
                'generated_at': time.time()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.status_updated.emit(f"会话报告已保存到: {file_path}")
            
        except Exception as e:
            self.error_occurred.emit(f"保存会话报告失败: {str(e)}")
    
    def load_config(self, config_file: str):
        """加载配置文件"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                
                # 合并配置
                self.config.update(loaded_config)
                self.status_updated.emit("配置已加载")
            
        except Exception as e:
            self.error_occurred.emit(f"加载配置失败: {str(e)}")
    
    def save_config(self, config_file: str):
        """保存配置文件"""
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            self.status_updated.emit(f"配置已保存到: {config_file}")
            
        except Exception as e:
            self.error_occurred.emit(f"保存配置失败: {str(e)}") 