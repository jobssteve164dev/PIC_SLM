"""
智能配置生成器

基于训练配置和实时数据，使用大模型生成优化的训练配置
主要功能：
- 接收当前训练配置和实时训练数据
- 使用LLM分析生成优化配置
- 调用现有配置应用机制
- 记录配置调整历史
"""

import os
import json
import time
import copy
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import QMessageBox

from .real_time_metrics_collector import get_global_metrics_collector
from ..llm.llm_framework import LLMFramework
from ..llm.analysis_engine import TrainingAnalysisEngine


@dataclass
class ConfigAdjustment:
    """配置调整记录"""
    adjustment_id: str
    timestamp: float
    original_config: Dict[str, Any]
    adjusted_config: Dict[str, Any]
    changes: Dict[str, Any]  # 具体变更的参数
    reason: str  # 调整原因
    training_metrics: Dict[str, Any]  # 触发调整的训练指标
    llm_analysis: Dict[str, Any]  # LLM分析结果
    status: str  # 'pending', 'applied', 'failed'


@dataclass
class TrainingSession:
    """训练会话记录"""
    session_id: str
    start_time: float
    original_config: Dict[str, Any]
    adjustments: List[ConfigAdjustment]
    final_config: Optional[Dict[str, Any]]
    status: str  # 'running', 'completed', 'failed'


class IntelligentConfigGenerator(QObject):
    """智能配置生成器"""
    
    # 信号定义
    config_generated = pyqtSignal(dict)  # 配置生成完成信号
    config_applied = pyqtSignal(dict)    # 配置应用完成信号
    adjustment_recorded = pyqtSignal(dict)  # 调整记录信号
    status_updated = pyqtSignal(str)     # 状态更新信号
    error_occurred = pyqtSignal(str)     # 错误信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.llm_framework = None
        self.analysis_engine = None
        self.metrics_collector = None
        
        # 会话管理
        self.current_session: Optional[TrainingSession] = None
        self.adjustment_history: List[ConfigAdjustment] = []
        
        # 配置约束
        self.parameter_constraints = self._load_parameter_constraints()
        
        # 初始化组件
        self._initialize_components()
        
    def _initialize_components(self):
        """初始化相关组件"""
        try:
            # 获取LLM配置
            llm_config = self._load_llm_config()
            adapter_type = llm_config.get('adapter_type', 'openai')
            
            # 生产环境检查
            if adapter_type == 'mock':
                self.error_occurred.emit("❌ 生产环境不允许使用mock LLM适配器！请在智能训练设置中配置真实的LLM服务。")
                raise ValueError("生产环境不允许使用mock LLM适配器")
            
            # 初始化LLM框架
            self.llm_framework = LLMFramework(
                adapter_type=adapter_type,
                adapter_config=llm_config.get('adapter_config', {})
            )
            self.llm_framework.start()
            
            # 初始化分析引擎
            self.analysis_engine = TrainingAnalysisEngine(self.llm_framework.llm_adapter)
            
            # 获取指标采集器
            self.metrics_collector = get_global_metrics_collector()
            
            print(f"✅ 智能配置生成器初始化完成，使用LLM适配器: {adapter_type}")
            
        except Exception as e:
            self.error_occurred.emit(f"初始化组件失败: {str(e)}")
            # 生产环境中不允许使用模拟适配器作为后备
            raise
    
    def _load_llm_config(self) -> Dict[str, Any]:
        """加载LLM配置"""
        try:
            print("[DEBUG] 开始加载LLM配置...")
            
            # 首先尝试从智能训练设置中加载
            intelligent_config_file = "setting/intelligent_training_config.json"
            if os.path.exists(intelligent_config_file):
                print(f"[DEBUG] 检查智能训练配置文件: {intelligent_config_file}")
                with open(intelligent_config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    llm_config = config.get('llm_config', {})
                    if llm_config:
                        print(f"[DEBUG] 从智能训练配置中找到LLM配置: {llm_config}")
                        return llm_config
                    else:
                        print("[DEBUG] 智能训练配置中未找到llm_config")
            else:
                print(f"[DEBUG] 智能训练配置文件不存在: {intelligent_config_file}")
            
            # 然后尝试从AI设置中加载
            ai_config_file = "setting/ai_config.json"
            if os.path.exists(ai_config_file):
                print(f"[DEBUG] 检查AI配置文件: {ai_config_file}")
                with open(ai_config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                    # 获取默认适配器类型
                    default_adapter = config.get('general', {}).get('default_adapter', 'openai')
                    print(f"[DEBUG] 从AI配置中获取默认适配器: {default_adapter}")
                    
                    # 根据适配器类型获取对应配置
                    if default_adapter == 'deepseek':
                        adapter_config = config.get('deepseek', {})
                        print(f"[DEBUG] 加载DeepSeek配置: {adapter_config}")
                        result = {
                            'adapter_type': 'deepseek',
                            'adapter_config': {
                                'api_key': adapter_config.get('api_key', ''),
                                'base_url': adapter_config.get('base_url', ''),
                                'model': adapter_config.get('model', 'deepseek-chat'),
                                'temperature': adapter_config.get('temperature', 0.7),
                                'max_tokens': adapter_config.get('max_tokens', 3000)
                            }
                        }
                        print(f"[DEBUG] 返回DeepSeek配置: {result}")
                        return result
                    elif default_adapter == 'openai':
                        adapter_config = config.get('openai', {})
                        return {
                            'adapter_type': 'openai',
                            'adapter_config': {
                                'api_key': adapter_config.get('api_key', ''),
                                'base_url': adapter_config.get('base_url', ''),
                                'model': adapter_config.get('model', 'gpt-4'),
                                'temperature': adapter_config.get('temperature', 0.7),
                                'max_tokens': adapter_config.get('max_tokens', 1000)
                            }
                        }
                    elif default_adapter == 'ollama':
                        adapter_config = config.get('ollama', {})
                        return {
                            'adapter_type': 'ollama',
                            'adapter_config': {
                                'base_url': adapter_config.get('base_url', ''),
                                'model': adapter_config.get('model', 'llama2'),
                                'temperature': adapter_config.get('temperature', 0.7),
                                'num_predict': adapter_config.get('num_predict', 1000),
                                'timeout': adapter_config.get('timeout', 120)
                            }
                        }
                    elif default_adapter == 'custom':
                        adapter_config = config.get('custom_api', {})
                        return {
                            'adapter_type': 'custom',
                            'adapter_config': {
                                'api_key': adapter_config.get('api_key', ''),
                                'base_url': adapter_config.get('base_url', ''),
                                'model': adapter_config.get('model', ''),
                                'temperature': adapter_config.get('temperature', 0.7),
                                'max_tokens': adapter_config.get('max_tokens', 1000)
                            }
                        }
            
            # 如果都没有，返回默认配置
            return {
                'adapter_type': 'openai',
                'adapter_config': {
                    'api_key': '',
                    'model': 'gpt-4',
                    'temperature': 0.7,
                    'max_tokens': 1000
                }
            }
            
        except Exception as e:
            print(f"加载LLM配置失败: {str(e)}")
            # 返回默认配置
            return {
                'adapter_type': 'openai',
                'adapter_config': {
                    'api_key': '',
                    'model': 'gpt-4',
                    'temperature': 0.7,
                    'max_tokens': 1000
                }
            }
    
    def _load_parameter_constraints(self) -> Dict[str, Dict[str, Any]]:
        """加载参数约束配置"""
        return {
            'learning_rate': {
                'min': 1e-6,
                'max': 0.1,
                'step': 1e-6,
                'description': '学习率'
            },
            'batch_size': {
                'min': 1,
                'max': 128,
                'step': 1,
                'description': '批次大小'
            },
            'num_epochs': {
                'min': 1,
                'max': 1000,
                'step': 1,
                'description': '训练轮数'
            },
            'dropout_rate': {
                'min': 0.0,
                'max': 0.9,
                'step': 0.01,
                'description': 'Dropout率'
            },
            'weight_decay': {
                'min': 0.0,
                'max': 0.01,
                'step': 1e-6,
                'description': '权重衰减'
            },
            'early_stopping_patience': {
                'min': 1,
                'max': 50,
                'step': 1,
                'description': '早停耐心值'
            }
        }
    
    def start_training_session(self, initial_config: Dict[str, Any]) -> str:
        """开始新的训练会话"""
        try:
            session_id = f"session_{int(time.time())}"
            
            self.current_session = TrainingSession(
                session_id=session_id,
                start_time=time.time(),
                original_config=copy.deepcopy(initial_config),
                adjustments=[],
                final_config=None,
                status='running'
            )
            
            self.status_updated.emit(f"开始训练会话: {session_id}")
            return session_id
            
        except Exception as e:
            self.error_occurred.emit(f"开始训练会话失败: {str(e)}")
            return ""
    
    def generate_optimized_config(self, 
                                current_config: Dict[str, Any],
                                training_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成优化的训练配置"""
        try:
            self.status_updated.emit("正在生成优化配置...")
            
            # 获取实时训练数据
            if training_metrics is None:
                real_data = self.metrics_collector.get_current_training_data_for_ai()
                if 'error' in real_data:
                    self.error_occurred.emit(f"无法获取训练数据: {real_data['error']}")
                    return current_config
                training_metrics = real_data.get('current_metrics', {})
            
            # 使用LLM分析当前配置和训练数据
            analysis_result = self._analyze_config_and_metrics(current_config, training_metrics)
            
            # 生成优化建议
            optimization_suggestions = self._generate_optimization_suggestions(
                current_config, training_metrics, analysis_result
            )
            
            # 应用优化建议到配置
            optimized_config = self._apply_optimization_suggestions(
                current_config, optimization_suggestions
            )
            
            # 验证配置有效性
            validated_config = self._validate_config(optimized_config)
            
            # 记录配置调整
            # 确保analysis_result是字典格式
            if isinstance(analysis_result, str):
                # 如果分析结果是字符串，提取其中的关键信息
                analysis_dict = {
                    'reason': 'LLM智能分析建议',
                    'analysis': analysis_result[:500] + '...' if len(analysis_result) > 500 else analysis_result
                }
            else:
                analysis_dict = analysis_result
            
            self._record_config_adjustment(
                current_config, validated_config, training_metrics, analysis_dict
            )
            
            self.status_updated.emit("优化配置生成完成")
            return validated_config
            
        except Exception as e:
            self.error_occurred.emit(f"生成优化配置失败: {str(e)}")
            return current_config
    
    def _analyze_config_and_metrics(self, 
                                  config: Dict[str, Any], 
                                  metrics: Dict[str, Any]) -> Dict[str, Any]:
        """分析配置和训练指标"""
        try:
            # 添加调试信息
            adapter_type = type(self.llm_framework.llm_adapter).__name__
            self.status_updated.emit(f"🔍 开始LLM分析，使用适配器: {adapter_type}")
            print(f"[DEBUG] 使用LLM适配器: {adapter_type}")
            print(f"[DEBUG] 适配器类型: {getattr(self.llm_framework.llm_adapter, 'adapter_type', 'unknown')}")
            
            # 构建分析提示词
            prompt = self._build_config_analysis_prompt(config, metrics)
            print(f"[DEBUG] 分析提示词长度: {len(prompt)} 字符")
            
            # 使用LLM进行分析
            self.status_updated.emit("🤖 正在调用LLM进行训练分析...")
            print(f"[DEBUG] 开始调用LLM分析...")
            
            analysis_result = self.llm_framework.llm_adapter.analyze_metrics(metrics, prompt)
            
            print(f"[DEBUG] LLM分析结果类型: {type(analysis_result)}")
            print(f"[DEBUG] LLM分析结果长度: {len(str(analysis_result))} 字符")
            
            # 检查是否是模拟结果
            if isinstance(analysis_result, str) and "模拟分析结果" in analysis_result:
                self.status_updated.emit("⚠️ 检测到模拟分析结果！请配置真实的LLM服务")
                print("[WARNING] 检测到模拟分析结果，请配置真实的LLM服务")
            else:
                self.status_updated.emit("✅ LLM分析完成")
                print("[DEBUG] LLM分析完成，结果看起来是真实的")
            
            return analysis_result
            
        except Exception as e:
            self.error_occurred.emit(f"分析配置和指标失败: {str(e)}")
            print(f"[ERROR] LLM分析失败: {str(e)}")
            return {'error': str(e)}
    
    def _build_config_analysis_prompt(self, 
                                    config: Dict[str, Any], 
                                    metrics: Dict[str, Any]) -> str:
        """构建配置分析提示词"""
        return f"""
请基于以下训练配置和实时训练数据，提供专业的配置优化建议：

## 📋 当前训练配置
```json
{json.dumps(config, ensure_ascii=False, indent=2)}
```

## 📊 实时训练指标
```json
{json.dumps(metrics, ensure_ascii=False, indent=2)}
```

## 🎯 分析要求

请基于以上信息，提供以下分析：

### 1. 配置评估
- 当前配置是否适合当前数据集和任务？
- 是否存在明显的配置问题或冲突？
- 哪些参数可能需要调整？

### 2. 训练状态分析
- 当前训练状态（收敛、过拟合、欠拟合等）
- 训练指标反映的问题
- 与配置参数的关联性

### 3. 优化建议
请提供具体的参数调整建议，格式如下：
```json
{{
    "suggestions": [
        {{
            "parameter": "learning_rate",
            "current_value": 0.001,
            "suggested_value": 0.0005,
            "reason": "训练损失下降缓慢，建议降低学习率",
            "priority": "high"
        }},
        {{
            "parameter": "batch_size",
            "current_value": 32,
            "suggested_value": 16,
            "reason": "GPU内存使用率较高，建议减小批次大小",
            "priority": "medium"
        }}
    ]
}}
```

### 4. 注意事项
- 只建议调整现有配置中的参数
- 确保建议值在合理范围内
- 优先考虑高优先级建议
- 保持配置的完整性和一致性

请用中文回答，保持专业性和实用性。
"""
    
    def _generate_optimization_suggestions(self, 
                                         config: Dict[str, Any],
                                         metrics: Dict[str, Any],
                                         analysis_result: Any) -> List[Dict[str, Any]]:
        """生成优化建议"""
        suggestions = []
        
        try:
            print(f"[DEBUG] 分析结果类型: {type(analysis_result)}")
            print(f"[DEBUG] 分析结果内容: {str(analysis_result)[:200]}...")
            
            # 从LLM分析结果中提取建议
            if isinstance(analysis_result, dict) and 'suggestions' in analysis_result:
                suggestions.extend(analysis_result['suggestions'])
            elif isinstance(analysis_result, str):
                # 如果分析结果是字符串，尝试解析其中的建议
                parsed_suggestions = self._parse_suggestions_from_text(analysis_result)
                suggestions.extend(parsed_suggestions)
            
            # 基于规则生成额外建议
            rule_suggestions = self._generate_rule_based_suggestions(config, metrics)
            suggestions.extend(rule_suggestions)
            
            # 按优先级排序
            suggestions.sort(key=lambda x: x.get('priority', 'low'), reverse=True)
            
            return suggestions
            
        except Exception as e:
            self.error_occurred.emit(f"生成优化建议失败: {str(e)}")
            print(f"[ERROR] 生成优化建议失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _parse_suggestions_from_text(self, text: str) -> List[Dict[str, Any]]:
        """从文本中解析建议"""
        suggestions = []
        
        try:
            print(f"[DEBUG] 开始解析LLM返回的文本建议")
            
            # 首先尝试提取JSON格式的建议
            json_suggestions = self._extract_json_suggestions(text)
            if json_suggestions:
                print(f"[DEBUG] 成功提取到 {len(json_suggestions)} 个JSON格式建议")
                return json_suggestions
            
            # 如果没有JSON格式，则进行文本解析
            print(f"[DEBUG] 未找到JSON格式建议，进行文本解析")
            lines = text.split('\n')
            current_suggestion = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # 检测建议项（以数字开头或包含关键词）
                if (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) or
                    any(keyword in line.lower() for keyword in ['建议', '推荐', '调整', '优化', '增加', '减少', '降低', '提高'])):
                    
                    if current_suggestion:
                        suggestions.append(current_suggestion)
                    
                    current_suggestion = {
                        'type': 'parameter_adjustment',
                        'description': line,
                        'priority': 'medium',
                        'confidence': 0.7
                    }
                elif current_suggestion and line.startswith(('  ', '\t', '-', '*')):
                    # 这是建议的详细说明
                    current_suggestion['description'] += f" {line.strip()}"
            
            if current_suggestion:
                suggestions.append(current_suggestion)
                
        except Exception as e:
            print(f"[DEBUG] 解析文本建议失败: {str(e)}")
        
        return suggestions
    
    def _extract_json_suggestions(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取JSON格式的建议"""
        try:
            import re
            import json
            
            # 查找JSON代码块
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            matches = re.findall(json_pattern, text, re.DOTALL)
            
            if not matches:
                # 如果没有代码块标记，尝试直接查找JSON对象
                json_pattern = r'\{[^{}]*"suggestions"[^{}]*\[.*?\][^{}]*\}'
                matches = re.findall(json_pattern, text, re.DOTALL)
            
            for match in matches:
                try:
                    json_data = json.loads(match)
                    if 'suggestions' in json_data and isinstance(json_data['suggestions'], list):
                        print(f"[DEBUG] 成功解析JSON建议: {len(json_data['suggestions'])} 个建议")
                        return json_data['suggestions']
                except json.JSONDecodeError as e:
                    print(f"[DEBUG] JSON解析失败: {str(e)}")
                    continue
            
            return []
            
        except Exception as e:
            print(f"[DEBUG] 提取JSON建议失败: {str(e)}")
            return []
    
    def _generate_rule_based_suggestions(self, 
                                       config: Dict[str, Any],
                                       metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于规则生成建议"""
        suggestions = []
        
        try:
            train_loss = metrics.get('train_loss', 0)
            val_loss = metrics.get('val_loss', 0)
            train_acc = metrics.get('train_accuracy', 0)
            val_acc = metrics.get('val_accuracy', 0)
            epoch = metrics.get('epoch', 0)
            
            # 过拟合检测
            if val_loss > train_loss * 1.3 and epoch > 5:
                suggestions.append({
                    'parameter': 'dropout_rate',
                    'current_value': config.get('dropout_rate', 0.0),
                    'suggested_value': min(0.9, config.get('dropout_rate', 0.0) + 0.1),
                    'reason': '检测到过拟合，建议增加Dropout率',
                    'priority': 'high'
                })
                
                suggestions.append({
                    'parameter': 'weight_decay',
                    'current_value': config.get('weight_decay', 0.0001),
                    'suggested_value': config.get('weight_decay', 0.0001) * 1.5,
                    'reason': '检测到过拟合，建议增加权重衰减',
                    'priority': 'high'
                })
            
            # 欠拟合检测
            if train_acc < 0.6 and val_acc < 0.6 and epoch > 10:
                suggestions.append({
                    'parameter': 'learning_rate',
                    'current_value': config.get('learning_rate', 0.001),
                    'suggested_value': config.get('learning_rate', 0.001) * 1.5,
                    'reason': '检测到欠拟合，建议增加学习率',
                    'priority': 'high'
                })
            
            # 收敛缓慢检测
            if train_loss > 0.5 and epoch > 15:
                suggestions.append({
                    'parameter': 'learning_rate',
                    'current_value': config.get('learning_rate', 0.001),
                    'suggested_value': config.get('learning_rate', 0.001) * 0.5,
                    'reason': '训练收敛缓慢，建议降低学习率',
                    'priority': 'medium'
                })
            
            return suggestions
            
        except Exception as e:
            self.error_occurred.emit(f"生成规则建议失败: {str(e)}")
            return []
    
    def _apply_optimization_suggestions(self, 
                                      config: Dict[str, Any],
                                      suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """应用优化建议到配置"""
        try:
            optimized_config = copy.deepcopy(config)
            
            for suggestion in suggestions:
                param_name = suggestion.get('parameter')
                suggested_value = suggestion.get('suggested_value')
                
                if param_name in optimized_config and suggested_value is not None:
                    # 应用约束
                    constrained_value = self._apply_parameter_constraints(param_name, suggested_value)
                    optimized_config[param_name] = constrained_value
            
            return optimized_config
            
        except Exception as e:
            self.error_occurred.emit(f"应用优化建议失败: {str(e)}")
            return config
    
    def _apply_parameter_constraints(self, param_name: str, value: Any) -> Any:
        """应用参数约束"""
        if param_name not in self.parameter_constraints:
            return value
        
        constraints = self.parameter_constraints[param_name]
        
        # 应用范围约束
        if 'min' in constraints:
            value = max(value, constraints['min'])
        if 'max' in constraints:
            value = min(value, constraints['max'])
        
        # 应用步长约束
        if 'step' in constraints and isinstance(value, (int, float)):
            step = constraints['step']
            value = round(value / step) * step
        
        return value
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证配置有效性"""
        try:
            validated_config = copy.deepcopy(config)
            
            # 验证必需参数
            required_params = ['model_name', 'batch_size', 'learning_rate', 'num_epochs']
            for param in required_params:
                if param not in validated_config:
                    self.error_occurred.emit(f"缺少必需参数: {param}")
                    return config
            
            # 验证参数类型和范围
            for param_name, value in validated_config.items():
                if param_name in self.parameter_constraints:
                    validated_config[param_name] = self._apply_parameter_constraints(param_name, value)
            
            return validated_config
            
        except Exception as e:
            self.error_occurred.emit(f"验证配置失败: {str(e)}")
            return config
    
    def _record_config_adjustment(self, 
                                original_config: Dict[str, Any],
                                adjusted_config: Dict[str, Any],
                                training_metrics: Dict[str, Any],
                                analysis_result: Dict[str, Any]):
        """记录配置调整"""
        try:
            # 计算变更
            changes = {}
            for key, value in adjusted_config.items():
                if key in original_config and original_config[key] != value:
                    changes[key] = {
                        'from': original_config[key],
                        'to': value
                    }
            
            if not changes:
                return  # 没有变更，不记录
            
            # 创建调整记录
            adjustment = ConfigAdjustment(
                adjustment_id=f"adj_{int(time.time())}",
                timestamp=time.time(),
                original_config=copy.deepcopy(original_config),
                adjusted_config=copy.deepcopy(adjusted_config),
                changes=changes,
                reason=analysis_result.get('reason', '智能优化建议'),
                training_metrics=copy.deepcopy(training_metrics),
                llm_analysis=copy.deepcopy(analysis_result),
                status='pending'
            )
            
            # 添加到历史记录
            self.adjustment_history.append(adjustment)
            if self.current_session:
                self.current_session.adjustments.append(adjustment)
            
            # 发射信号
            self.adjustment_recorded.emit(asdict(adjustment))
            
        except Exception as e:
            self.error_occurred.emit(f"记录配置调整失败: {str(e)}")
    
    def apply_config_to_training_system(self, 
                                      config: Dict[str, Any],
                                      training_tab) -> bool:
        """将配置应用到训练系统"""
        try:
            self.status_updated.emit("正在应用配置到训练系统...")
            
            # 使用现有的ConfigApplier应用配置
            from ..ui.components.training.config_applier import ConfigApplier
            success = ConfigApplier.apply_to_training_tab(config, training_tab)
            
            if success:
                self.status_updated.emit("配置应用成功")
                self.config_applied.emit({
                    'config': config,
                    'timestamp': time.time(),
                    'success': True
                })
            else:
                self.error_occurred.emit("配置应用失败")
                self.config_applied.emit({
                    'config': config,
                    'timestamp': time.time(),
                    'success': False
                })
            
            return success
            
        except Exception as e:
            self.error_occurred.emit(f"应用配置到训练系统失败: {str(e)}")
            return False
    
    def get_adjustment_history(self) -> List[Dict[str, Any]]:
        """获取调整历史"""
        return [asdict(adj) for adj in self.adjustment_history]
    
    def get_current_session_info(self) -> Dict[str, Any]:
        """获取当前会话信息"""
        if not self.current_session:
            return {}
        
        return {
            'session_id': self.current_session.session_id,
            'start_time': self.current_session.start_time,
            'status': self.current_session.status,
            'adjustment_count': len(self.current_session.adjustments),
            'original_config': self.current_session.original_config,
            'final_config': self.current_session.final_config
        }
    
    def export_adjustment_report(self) -> Dict[str, Any]:
        """导出调整报告"""
        return {
            'export_time': time.time(),
            'current_session': self.get_current_session_info(),
            'adjustment_history': self.get_adjustment_history(),
            'parameter_constraints': self.parameter_constraints
        }
    
    def stop_training_session(self):
        """停止训练会话"""
        if self.current_session:
            self.current_session.status = 'completed'
            self.current_session.final_config = self.current_session.adjustments[-1].adjusted_config if self.current_session.adjustments else self.current_session.original_config
            self.status_updated.emit(f"训练会话 {self.current_session.session_id} 已停止")