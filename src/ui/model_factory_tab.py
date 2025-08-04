from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                           QSplitter, QGroupBox, QTabWidget, QTextEdit, QLineEdit,
                           QScrollArea, QFrame, QSizePolicy, QStackedWidget, QComboBox,
                           QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QTextCursor, QColor, QPalette
import os
import sys
import json
import time
from datetime import datetime
from .base_tab import BaseTab

# 导入Markdown渲染器
from .components.chat.markdown_renderer import render_markdown

# 导入LLM框架
try:
    from src.llm.llm_framework import LLMFramework
    from src.llm.model_adapters import create_llm_adapter
    from src.ui.components.model_analysis.model_selection_dialog import ModelSelectionDialog
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"LLM模块导入失败: {e}")
    LLM_AVAILABLE = False


class LLMChatThread(QThread):
    """LLM聊天处理线程"""
    
    # 定义信号
    chat_finished = pyqtSignal(str)  # 聊天完成，返回AI回复
    chat_error = pyqtSignal(str)     # 聊天出错
    analysis_finished = pyqtSignal(str)  # 分析完成
    analysis_error = pyqtSignal(str)     # 分析出错
    
    def __init__(self, llm_framework, parent=None):
        super().__init__(parent)
        self.llm_framework = llm_framework
        self.task_type = ""
        self.user_message = ""
        self.training_context = {}
        self.task_params = {}
        
    def set_chat_task(self, user_message, training_context=None):
        """设置聊天任务"""
        self.task_type = "chat"
        self.user_message = user_message
        self.training_context = training_context or {}
        
    def set_analysis_task(self, task_type, params=None):
        """设置分析任务"""
        self.task_type = task_type
        self.task_params = params or {}
        
    def run(self):
        """执行任务"""
        try:
            if not self.llm_framework:
                self.chat_error.emit("LLM框架未初始化")
                return
                
            if self.task_type == "chat":
                self._handle_chat()
            elif self.task_type == "analyze_training":
                self._handle_training_analysis()
            elif self.task_type == "get_suggestions":
                self._handle_suggestions()
            elif self.task_type == "diagnose_issues":
                self._handle_diagnosis()
            elif self.task_type == "compare_models":
                self._handle_model_comparison()
            else:
                self.chat_error.emit(f"未知的任务类型: {self.task_type}")
                
        except Exception as e:
            if self.task_type == "chat":
                self.chat_error.emit(f"处理聊天请求时出错: {str(e)}")
            else:
                self.analysis_error.emit(f"处理分析请求时出错: {str(e)}")
    
    def _handle_chat(self):
        """处理聊天请求"""
        # 获取训练配置上下文
        config_context = ""
        if hasattr(self.llm_framework, 'analysis_engine'):
            config_context = self.llm_framework.analysis_engine._get_training_config_context()
        
        # 构建增强的用户消息，包含训练配置信息
        enhanced_message = f"""
{config_context}

## 用户问题
{self.user_message}

请基于以上训练配置信息，针对用户的具体问题进行专业回答。如果用户的问题与训练相关，请结合训练配置参数进行分析和建议。
"""
        
        # 先更新训练上下文到LLM框架（如果有的话）
        if hasattr(self.llm_framework, 'analysis_engine') and self.training_context:
            self.llm_framework.analysis_engine.prompt_builder.add_context({
                'type': 'training_context',
                'data': self.training_context
            })
        
        # 调用LLM框架进行对话
        result = self.llm_framework.chat_with_training_context(enhanced_message)
        
        # 处理返回结果
        if isinstance(result, dict):
            if 'error' in result:
                self.chat_error.emit(result['error'])
            else:
                response_text = result.get('response', '抱歉，没有收到有效回复')
                self.chat_finished.emit(response_text)
        else:
            self.chat_finished.emit(str(result))
    
    def _handle_training_analysis(self):
        """处理训练分析请求"""
        # 使用真实训练数据进行分析
        try:
            # 尝试获取真实训练数据
            result = self.llm_framework.analyze_real_training_metrics()
            
            # 如果没有真实数据，回退到模拟数据
            if result.get('error') and '没有可用的训练数据' in str(result.get('error', '')):
                # 模拟当前训练指标（仅作为后备方案）
                current_metrics = {
                    'epoch': 15,
                    'train_loss': 0.234,
                    'val_loss': 0.287,
                    'train_accuracy': 0.894,
                    'val_accuracy': 0.856,
                    'learning_rate': 0.001,
                    'gpu_memory_used': 6.2,
                    'gpu_memory_total': 8.0
                }
                result = self.llm_framework.analyze_training_metrics(current_metrics)
                
                # 添加提示说明这是模拟数据
                if isinstance(result, dict) and 'combined_insights' in result:
                    result['combined_insights'] = "⚠️ **注意：当前使用模拟数据进行分析**\n\n" + result['combined_insights']
                    
        except Exception as e:
            # 如果出现异常，使用模拟数据作为后备
            current_metrics = {
                'epoch': 15,
                'train_loss': 0.234,
                'val_loss': 0.287,
                'train_accuracy': 0.894,
                'val_accuracy': 0.856,
                'learning_rate': 0.001,
                'gpu_memory_used': 6.2,
                'gpu_memory_total': 8.0
            }
            result = self.llm_framework.analyze_training_metrics(current_metrics)
        
        # 处理返回结果
        if isinstance(result, dict):
            if 'error' in result:
                self.analysis_error.emit(result['error'])
            else:
                # 尝试获取分析内容
                combined_insights = result.get('combined_insights', '')
                llm_analysis = result.get('llm_analysis', '')
                response_text = combined_insights or llm_analysis or '分析完成，但未获得有效结果'
                
                # 如果有建议，也显示出来
                recommendations = result.get('recommendations')
                if recommendations:
                    response_text += f"\n\n建议: {recommendations}"
                
                self.analysis_finished.emit(response_text)
        else:
            self.analysis_finished.emit(str(result))
    
    def _handle_suggestions(self):
        """处理建议请求"""
        # 使用真实训练数据进行建议生成
        try:
            # 尝试获取基于真实数据的建议
            result = self.llm_framework.get_real_hyperparameter_suggestions()
            
            # 如果没有真实数据，回退到模拟数据
            if result.get('error') and '无法获取真实训练数据' in str(result.get('error', '')):
                # 模拟训练历史（仅作为后备方案）
                current_metrics = {'train_loss': 0.234, 'val_loss': 0.287, 'accuracy': 0.856}
                current_params = {'batch_size': 32, 'learning_rate': 0.001}
                result = self.llm_framework.get_hyperparameter_suggestions(
                    current_metrics, current_params
                )
                
                # 添加提示说明这是模拟数据
                if isinstance(result, dict) and 'llm_suggestions' in result:
                    result['llm_suggestions'] = "⚠️ **注意：当前使用模拟数据进行建议生成**\n\n" + result['llm_suggestions']
                    
        except Exception as e:
            # 如果出现异常，使用模拟数据作为后备
            current_metrics = {'train_loss': 0.234, 'val_loss': 0.287, 'accuracy': 0.856}
            current_params = {'batch_size': 32, 'learning_rate': 0.001}
            result = self.llm_framework.get_hyperparameter_suggestions(
                current_metrics, current_params
            )
        
        # 处理返回结果
        if isinstance(result, dict):
            if 'error' in result:
                self.analysis_error.emit(result['error'])
            else:
                # 尝试获取建议内容
                llm_suggestions = result.get('llm_suggestions', '')
                rule_suggestions = result.get('rule_suggestions', [])
                response_text = llm_suggestions or '已生成优化建议'
                
                if rule_suggestions:
                    response_text += f"\n\n规则建议: {rule_suggestions}"
                
                self.analysis_finished.emit(response_text)
        else:
            self.analysis_finished.emit(str(result))
    
    def _handle_diagnosis(self):
        """处理诊断请求"""
        # 使用真实训练数据进行问题诊断
        try:
            # 尝试获取基于真实数据的诊断
            result = self.llm_framework.diagnose_real_training_problems()
            
            # 如果没有真实数据，回退到模拟数据
            if result.get('error') and '无法获取真实训练数据' in str(result.get('error', '')):
                # 模拟问题指标（仅作为后备方案）
                problem_metrics = {
                    'train_loss': 0.1,
                    'val_loss': 0.8,  # 明显的过拟合
                    'gradient_norm': 1e-8,  # 梯度消失
                    'epoch': 20
                }
                result = self.llm_framework.diagnose_training_problems(problem_metrics)
                
                # 添加提示说明这是模拟数据
                if isinstance(result, dict) and 'llm_diagnosis' in result:
                    result['llm_diagnosis'] = "⚠️ **注意：当前使用模拟数据进行问题诊断**\n\n" + result['llm_diagnosis']
                    
        except Exception as e:
            # 如果出现异常，使用模拟数据作为后备
            problem_metrics = {
                'train_loss': 0.1,
                'val_loss': 0.8,  # 明显的过拟合
                'gradient_norm': 1e-8,  # 梯度消失
                'epoch': 20
            }
            result = self.llm_framework.diagnose_training_problems(problem_metrics)
        
        # 处理返回结果
        if isinstance(result, dict):
            if 'error' in result:
                self.analysis_error.emit(result['error'])
            else:
                # 尝试获取诊断内容
                llm_diagnosis = result.get('llm_diagnosis', '')
                rule_diagnosis = result.get('rule_diagnosis', '')
                response_text = llm_diagnosis or rule_diagnosis or '诊断完成，但未发现明显问题'
                
                # 添加推荐行动
                recommended_actions = result.get('recommended_actions')
                if recommended_actions:
                    response_text += f"\n\n推荐行动: {recommended_actions}"
                
                self.analysis_finished.emit(response_text)
        else:
            self.analysis_finished.emit(str(result))
    
    def _handle_model_comparison(self):
        """处理模型对比请求"""
        # 使用传入的真实模型数据，如果没有则使用默认模拟数据
        if self.task_params and isinstance(self.task_params, list) and len(self.task_params) > 0:
            model_results = self.task_params
        else:
            # 默认模拟数据（保持兼容性）
            model_results = [
                {
                    'model_name': 'ResNet50',
                    'accuracy': 0.892,
                    'val_loss': 0.234,
                    'params': 25557032,
                    'inference_time': 0.05
                },
                {
                    'model_name': 'EfficientNet-B0',
                    'accuracy': 0.900,
                    'val_loss': 0.198,
                    'params': 5288548,
                    'inference_time': 0.03
                },
                {
                    'model_name': 'MobileNetV2',
                    'accuracy': 0.875,
                    'val_loss': 0.267,
                    'params': 3504872,
                    'inference_time': 0.02
                }
            ]
        
        result = self.llm_framework.analysis_engine.compare_models(model_results)
        
        # 处理返回结果
        if isinstance(result, dict):
            if 'error' in result:
                self.analysis_error.emit(result['error'])
            else:
                # 尝试获取对比分析内容
                llm_comparison = result.get('llm_comparison', '')
                rule_comparison = result.get('rule_comparison', '')
                response_text = llm_comparison or rule_comparison or '模型对比完成'
                
                # 添加最佳模型推荐
                best_model = result.get('best_model')
                if best_model:
                    response_text += f"\n\n推荐模型: {best_model}"
                
                self.analysis_finished.emit(response_text)
        else:
            self.analysis_finished.emit(str(result))


class LLMChatWidget(QWidget):
    """LLM聊天界面组件"""
    
    # 定义信号
    status_updated = pyqtSignal(str)
    analysis_requested = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.llm_framework = None
        self.chat_history = []
        self.training_context = {}
        self.chat_thread = None  # 聊天线程
        self.init_ui()
        self.init_llm_framework()
    
    def init_ui(self):
        """初始化聊天界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 标题
        title_label = QLabel("AI训练助手")
        title_label.setFont(QFont('微软雅黑', 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # LLM适配器选择和重新加载按钮
        adapter_layout = QHBoxLayout()
        adapter_layout.addWidget(QLabel("AI模型:"))
        self.adapter_combo = QComboBox()
        self.adapter_combo.addItems(["模拟适配器", "OpenAI GPT-4", "DeepSeek", "本地Ollama", "自定义API"])
        self.adapter_combo.currentTextChanged.connect(self.switch_adapter)
        adapter_layout.addWidget(self.adapter_combo)
        
        # 添加重新加载配置按钮
        self.reload_config_btn = QPushButton("🔄 重新加载配置")
        self.reload_config_btn.setToolTip("重新加载AI设置配置")
        self.reload_config_btn.clicked.connect(self.reload_ai_config)
        self.reload_config_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
        """)
        adapter_layout.addWidget(self.reload_config_btn)
        
        adapter_layout.addStretch()
        layout.addLayout(adapter_layout)
        
        # 聊天历史显示区域
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumHeight(300)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Microsoft YaHei', Arial, sans-serif;
                font-size: 10pt;
            }
        """)
        layout.addWidget(self.chat_display)
        
        # 快捷操作按钮
        quick_actions_group = QGroupBox("快捷操作")
        quick_layout = QHBoxLayout()
        
        self.analyze_btn = QPushButton("🔍 分析当前训练状态")
        self.analyze_btn.clicked.connect(self.analyze_training)
        quick_layout.addWidget(self.analyze_btn)
        
        self.suggest_btn = QPushButton("💡 获取优化建议")
        self.suggest_btn.clicked.connect(self.get_suggestions)
        quick_layout.addWidget(self.suggest_btn)
        
        self.diagnose_btn = QPushButton("🔧 诊断训练问题")
        self.diagnose_btn.clicked.connect(self.diagnose_issues)
        quick_layout.addWidget(self.diagnose_btn)
        
        self.compare_btn = QPushButton("📊 模型对比分析")
        self.compare_btn.clicked.connect(self.compare_models)
        quick_layout.addWidget(self.compare_btn)
        
        quick_actions_group.setLayout(quick_layout)
        layout.addWidget(quick_actions_group)
        
        # 消息输入区域
        input_layout = QHBoxLayout()
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("输入您的问题，例如：为什么验证损失在增加？")
        self.message_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.message_input)
        
        self.send_btn = QPushButton("发送")
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)
        
        layout.addLayout(input_layout)
        
        # 状态指示器
        self.status_label = QLabel("AI助手已就绪")
        self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
    
    def init_llm_framework(self):
        """初始化LLM框架"""
        if not LLM_AVAILABLE:
            self.add_system_message("❌ LLM模块不可用，请检查安装")
            self.set_ui_enabled(False)
            return
            
        try:
            # 加载AI设置配置
            ai_config = self.load_ai_config()
            
            # 根据配置初始化适配器
            default_adapter = ai_config.get('general', {}).get('default_adapter', 'mock')
            
            if default_adapter == 'openai':
                openai_config = ai_config.get('openai', {})
                api_key = openai_config.get('api_key', '')
                if api_key:
                    self.llm_framework = LLMFramework('openai', openai_config)
                    self.adapter_combo.setCurrentText("OpenAI GPT-4")
                    self.add_system_message("✅ AI助手已启动，使用OpenAI GPT-4")
                else:
                    # 没有API密钥，回退到模拟适配器
                    self.llm_framework = LLMFramework('mock')
                    self.adapter_combo.setCurrentText("模拟适配器")
                    self.add_system_message("⚠️ 未配置OpenAI API密钥，使用模拟适配器")
            elif default_adapter == 'deepseek':
                deepseek_config = ai_config.get('deepseek', {})
                api_key = deepseek_config.get('api_key', '')
                if api_key:
                    self.llm_framework = LLMFramework('deepseek', deepseek_config)
                    self.adapter_combo.setCurrentText("DeepSeek")
                    self.add_system_message("✅ AI助手已启动，使用DeepSeek")
                else:
                    # 没有API密钥，回退到模拟适配器
                    self.llm_framework = LLMFramework('mock')
                    self.adapter_combo.setCurrentText("模拟适配器")
                    self.add_system_message("⚠️ 未配置DeepSeek API密钥，使用模拟适配器")
            elif default_adapter == 'local':
                ollama_config = ai_config.get('ollama', {})
                self.llm_framework = LLMFramework('local', ollama_config)
                self.adapter_combo.setCurrentText("本地Ollama")
                self.add_system_message("✅ AI助手已启动，使用本地Ollama")
            elif default_adapter == 'custom':
                custom_config = ai_config.get('custom_api', {})
                api_key = custom_config.get('api_key', '')
                base_url = custom_config.get('base_url', '')
                if api_key and base_url:
                    self.llm_framework = LLMFramework('custom', custom_config)
                    self.adapter_combo.setCurrentText("自定义API")
                    api_name = custom_config.get('name', '自定义API')
                    self.add_system_message(f"✅ AI助手已启动，使用{api_name}")
                else:
                    # 没有配置，回退到模拟适配器
                    self.llm_framework = LLMFramework('mock')
                    self.adapter_combo.setCurrentText("模拟适配器")
                    self.add_system_message("⚠️ 未配置自定义API，使用模拟适配器")
            else:
                # 默认使用模拟适配器
                self.llm_framework = LLMFramework('mock')
                self.adapter_combo.setCurrentText("模拟适配器")
                self.add_system_message("✅ AI助手已启动，使用模拟适配器")
            
            # 启动框架
            self.llm_framework.start()
            self.status_label.setText("AI助手已就绪")
            self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
            
        except Exception as e:
            # 初始化失败，回退到模拟适配器
            try:
                self.llm_framework = LLMFramework('mock')
                self.llm_framework.start()
                self.adapter_combo.setCurrentText("模拟适配器")
                self.add_system_message(f"⚠️ 配置的适配器初始化失败: {str(e)}")
                self.add_system_message("✅ 已回退到模拟适配器")
                self.status_label.setText("AI助手已就绪")
                self.status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
            except Exception as fallback_error:
                self.add_system_message(f"❌ LLM框架初始化失败: {str(fallback_error)}")
                self.set_ui_enabled(False)
    
    def switch_adapter(self, adapter_name):
        """切换LLM适配器"""
        if not self.llm_framework:
            return
            
        try:
            self.status_label.setText("正在切换AI模型...")
            self.status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
            
            # 加载AI设置配置
            ai_config = self.load_ai_config()
            
            if adapter_name == "模拟适配器":
                adapter_type = 'mock'
                adapter_config = {}
            elif adapter_name == "OpenAI GPT-4":
                adapter_type = 'openai'
                openai_config = ai_config.get('openai', {})
                adapter_config = {
                    'api_key': openai_config.get('api_key', ''),
                    'model': openai_config.get('model', 'gpt-4'),
                    'base_url': openai_config.get('base_url', '') or None,
                    'temperature': openai_config.get('temperature', 0.7),
                    'max_tokens': openai_config.get('max_tokens', 1000)
                }
                
                # 检查API密钥
                if not adapter_config['api_key']:
                    self.add_system_message("❌ 未配置OpenAI API密钥，请在设置中配置")
                    self.status_label.setText("配置缺失")
                    self.status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
                    return
                    
            elif adapter_name == "DeepSeek":
                adapter_type = 'deepseek'
                deepseek_config = ai_config.get('deepseek', {})
                adapter_config = {
                    'api_key': deepseek_config.get('api_key', ''),
                    'model': deepseek_config.get('model', 'deepseek-coder'),
                    'base_url': deepseek_config.get('base_url', '') or None,
                    'temperature': deepseek_config.get('temperature', 0.7),
                    'max_tokens': deepseek_config.get('max_tokens', 1000)
                }
                if not adapter_config['api_key']:
                    self.add_system_message("❌ 未配置DeepSeek API密钥，请在设置中配置")
                    self.status_label.setText("配置缺失")
                    self.status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
                    return
                    
            elif adapter_name == "本地Ollama":
                adapter_type = 'local'
                ollama_config = ai_config.get('ollama', {})
                adapter_config = {
                    'model_name': ollama_config.get('model', 'llama2'),
                    'base_url': ollama_config.get('base_url', 'http://localhost:11434'),
                    'temperature': ollama_config.get('temperature', 0.7),
                    'num_predict': ollama_config.get('num_predict', 1000),
                    'timeout': ollama_config.get('timeout', 120)
                }
            elif adapter_name == "自定义API":
                adapter_type = 'custom'
                custom_config = ai_config.get('custom_api', {})
                adapter_config = {
                    'api_key': custom_config.get('api_key', ''),
                    'model': custom_config.get('model', 'custom-model'),
                    'base_url': custom_config.get('base_url', ''),
                    'provider_type': custom_config.get('provider_type', 'OpenAI兼容'),
                    'temperature': custom_config.get('temperature', 0.7),
                    'max_tokens': custom_config.get('max_tokens', 1000)
                }
                
                # 检查API密钥和基础URL
                if not adapter_config['api_key'] or not adapter_config['base_url']:
                    self.add_system_message("❌ 未配置自定义API密钥或基础URL，请在设置中配置")
                    self.status_label.setText("配置缺失")
                    self.status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
                    return
            else:
                return
                
            self.llm_framework.switch_adapter(adapter_type, adapter_config)
            success = True
            
            if success:
                self.add_system_message(f"✅ 已切换到: {adapter_name}")
                if adapter_name == "OpenAI GPT-4":
                    model_name = adapter_config.get('model', 'gpt-4')
                    self.add_system_message(f"   使用模型: {model_name}")
                elif adapter_name == "DeepSeek":
                    model_name = adapter_config.get('model', 'deepseek-coder')
                    self.add_system_message(f"   使用模型: {model_name}")
                elif adapter_name == "本地Ollama":
                    model_name = adapter_config.get('model_name', 'llama2')
                    base_url = adapter_config.get('base_url', 'localhost:11434')
                    self.add_system_message(f"   使用模型: {model_name}")
                    self.add_system_message(f"   服务器: {base_url}")
                elif adapter_name == "自定义API":
                    model_name = adapter_config.get('model', 'custom-model')
                    base_url = adapter_config.get('base_url', '')
                    api_name = custom_config.get('name', '自定义API')
                    self.add_system_message(f"   使用模型: {model_name}")
                    self.add_system_message(f"   API地址: {base_url}")
                    self.add_system_message(f"   API名称: {api_name}")
                    
                self.status_label.setText("AI助手已就绪")
                self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
            else:
                self.add_system_message(f"❌ 切换到 {adapter_name} 失败")
                self.status_label.setText("切换失败")
                self.status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
                
        except Exception as e:
            self.add_system_message(f"❌ 切换适配器时出错: {str(e)}")
            self.status_label.setText("切换出错")
            self.status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
    
    def set_ui_enabled(self, enabled):
        """设置UI组件启用状态"""
        self.analyze_btn.setEnabled(enabled)
        self.suggest_btn.setEnabled(enabled)
        self.diagnose_btn.setEnabled(enabled)
        self.compare_btn.setEnabled(enabled)
        self.send_btn.setEnabled(enabled)
        self.message_input.setEnabled(enabled)
    
    def add_system_message(self, message):
        """添加系统消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 使用Markdown渲染器处理消息内容
        try:
            rendered_content = render_markdown(message)
        except Exception as e:
            # 如果Markdown渲染失败，使用原始文本
            print(f"Markdown渲染失败: {e}")
            rendered_content = message
        
        formatted_message = f"<div style='color: #6c757d; font-size: 9pt; margin: 5px 0;'>[{timestamp}] {rendered_content}</div>"
        self.chat_display.append(formatted_message)
    
    def add_user_message(self, message):
        """添加用户消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"""
        <div style='margin: 10px 0; padding: 8px; background-color: #e3f2fd; color: #000000; border: 1px solid #2196f3; border-radius: 10px; text-align: right;'>
            <span style='color: #6c757d; font-size: 11px; font-weight: normal;'>您 [{timestamp}]:</span><br>
            <span style='color: #000000; font-style: italic;'>{message}</span>
        </div>
        """
        self.chat_display.append(formatted_message)
    
    def add_ai_message(self, message):
        """添加AI响应消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 使用Markdown渲染器处理消息内容
        try:
            rendered_content = render_markdown(message)
        except Exception as e:
            # 如果Markdown渲染失败，使用原始文本
            print(f"Markdown渲染失败: {e}")
            rendered_content = message
        
        formatted_message = f"""
        <div style='margin: 10px 0; padding: 12px; background-color: #f8f9fa; color: #000000; border: 1px solid #dee2e6; border-radius: 10px;'>
            <span style='color: #6c757d; font-size: 11px; font-weight: normal;'>AI助手 [{timestamp}]:</span><br>
            <div style='color: #000000; margin-top: 8px; line-height: 1.6;'>{rendered_content}</div>
        </div>
        """
        self.chat_display.append(formatted_message)
        
        # 自动滚动到底部
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_display.setTextCursor(cursor)
    
    def send_message(self):
        """发送用户消息"""
        message = self.message_input.text().strip()
        if not message or not self.llm_framework:
            return
            
        # 检查是否有线程正在运行
        if self.chat_thread and self.chat_thread.isRunning():
            QMessageBox.information(self, "提示", "AI正在处理中，请稍等...")
            return
            
        self.message_input.clear()
        self.add_user_message(message)
        
        # 保存用户消息用于聊天历史
        self._last_user_message = message
        
        # 显示进度
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.status_label.setText("AI正在思考...")
        self.status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        
        # 禁用UI控件
        self.set_ui_enabled(False)
        
        # 创建并启动聊天线程
        self.chat_thread = LLMChatThread(self.llm_framework)
        self.chat_thread.set_chat_task(message, self.training_context)
        
        # 连接信号
        self.chat_thread.chat_finished.connect(self.on_chat_finished)
        self.chat_thread.chat_error.connect(self.on_chat_error)
        
        # 启动线程
        self.chat_thread.start()
    
    def analyze_training(self):
        """分析当前训练状态"""
        if not self.llm_framework:
            return
            
        # 检查是否有线程正在运行
        if self.chat_thread and self.chat_thread.isRunning():
            QMessageBox.information(self, "提示", "AI正在处理中，请稍等...")
            return
        
        self.add_user_message("请分析当前训练状态")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("正在分析训练状态...")
        self.status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        
        # 禁用UI控件
        self.set_ui_enabled(False)
        
        # 创建并启动分析线程
        self.chat_thread = LLMChatThread(self.llm_framework)
        self.chat_thread.set_analysis_task("analyze_training")
        
        # 连接信号
        self.chat_thread.analysis_finished.connect(self.on_analysis_finished)
        self.chat_thread.analysis_error.connect(self.on_analysis_error)
        
        # 启动线程
        self.chat_thread.start()
    
    def get_suggestions(self):
        """获取优化建议"""
        if not self.llm_framework:
            return
            
        # 检查是否有线程正在运行
        if self.chat_thread and self.chat_thread.isRunning():
            QMessageBox.information(self, "提示", "AI正在处理中，请稍等...")
            return
        
        self.add_user_message("请给出超参数优化建议")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("正在生成优化建议...")
        self.status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        
        # 禁用UI控件
        self.set_ui_enabled(False)
        
        # 创建并启动分析线程
        self.chat_thread = LLMChatThread(self.llm_framework)
        self.chat_thread.set_analysis_task("get_suggestions")
        
        # 连接信号
        self.chat_thread.analysis_finished.connect(self.on_analysis_finished)
        self.chat_thread.analysis_error.connect(self.on_analysis_error)
        
        # 启动线程
        self.chat_thread.start()
    
    def diagnose_issues(self):
        """诊断训练问题"""
        if not self.llm_framework:
            return
            
        # 检查是否有线程正在运行
        if self.chat_thread and self.chat_thread.isRunning():
            QMessageBox.information(self, "提示", "AI正在处理中，请稍等...")
            return
        
        self.add_user_message("请诊断训练中的问题")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("正在诊断问题...")
        self.status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        
        # 禁用UI控件
        self.set_ui_enabled(False)
        
        # 创建并启动分析线程
        self.chat_thread = LLMChatThread(self.llm_framework)
        self.chat_thread.set_analysis_task("diagnose_issues")
        
        # 连接信号
        self.chat_thread.analysis_finished.connect(self.on_analysis_finished)
        self.chat_thread.analysis_error.connect(self.on_analysis_error)
        
        # 启动线程
        self.chat_thread.start()
    
    def compare_models(self):
        """模型对比分析 - 引导用户到模型评估Tab进行实际评估"""
        # 显示引导对话框
        reply = QMessageBox.question(
            self, "模型对比分析", 
            "模型对比分析需要使用真实的测试数据集对模型进行评估。\n\n"
            "系统将引导您切换到「模型评估与可视化」标签页进行以下操作：\n"
            "1. 选择要对比的多个模型\n"
            "2. 设置测试集数据目录\n" 
            "3. 执行模型评估获得真实性能数据\n"
            "4. 使用AI分析评估结果\n\n"
            "是否现在切换到模型评估页面？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            # 切换到模型评估Tab
            self.switch_to_evaluation_tab()
            
            # 在聊天界面显示引导信息
            self.add_ai_message(
                "已为您切换到模型评估页面。请按以下步骤进行模型对比分析：\n\n"
                "📋 操作步骤：\n"
                "1. 选择「模型目录」- 包含要对比的模型文件\n"
                "2. 选择「测试集目录」- 用于评估的测试数据\n"
                "3. 设置「模型类型」和「模型架构」\n"
                "4. 在模型列表中选择多个要对比的模型\n"
                "5. 点击「对比选中模型」进行评估\n"
                "6. 评估完成后，点击「AI结果分析」获得智能分析报告\n\n"
                                 "💡 提示：确保选择的模型类型和架构与训练时一致，以获得准确的评估结果。"
             )
    
    def switch_to_evaluation_tab(self):
        """切换到模型评估Tab"""
        # 通过parent链查找主窗口
        main_window = None
        widget = self.parent()
        while widget and not hasattr(widget, 'tabs'):
            widget = widget.parent()
        main_window = widget
        
        if main_window and hasattr(main_window, 'tabs'):
            # 获取主窗口的标签页控件
            tab_widget = main_window.tabs
            
            # 查找模型评估与可视化标签页的索引
            for i in range(tab_widget.count()):
                tab_text = tab_widget.tabText(i)
                if "模型评估与可视化" in tab_text:
                    tab_widget.setCurrentIndex(i)
                    
                    # 进一步切换到模型评估子标签页
                    evaluation_tab = tab_widget.widget(i)
                    if hasattr(evaluation_tab, 'switch_view'):
                        # 切换到模型评估子标签页（索引3）
                        evaluation_tab.switch_view(3)
                    break
        else:
            QMessageBox.warning(self, "错误", "无法找到主窗口，无法切换到模型评估页面")
        
    def on_models_selected_for_comparison(self, models_data):
        """处理用户选择的模型数据"""
        if not models_data or len(models_data) < 2:
            QMessageBox.warning(self, "警告", "需要至少选择2个模型进行对比")
            return
            
        # 显示用户选择的模型信息
        model_names = [model['model_name'] for model in models_data]
        user_message = f"请对比分析以下 {len(models_data)} 个模型：{', '.join(model_names)}"
        self.add_user_message(user_message)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("正在对比模型...")
        self.status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        
        # 禁用UI控件
        self.set_ui_enabled(False)
        
        # 创建并启动分析线程，传入真实的模型数据
        self.chat_thread = LLMChatThread(self.llm_framework)
        self.chat_thread.set_analysis_task("compare_models", models_data)
        
        # 连接信号
        self.chat_thread.analysis_finished.connect(self.on_analysis_finished)
        self.chat_thread.analysis_error.connect(self.on_analysis_error)
        
        # 启动线程
        self.chat_thread.start()
    
    def on_chat_finished(self, response_text):
        """聊天完成处理"""
        self.add_ai_message(response_text)
        
        # 更新聊天历史
        if hasattr(self, '_last_user_message'):
            self.chat_history.append({
                'user': self._last_user_message,
                'ai': response_text,
                'timestamp': datetime.now().isoformat()
            })
        
        self._reset_ui_state()
    
    def on_chat_error(self, error_message):
        """聊天错误处理"""
        self.add_ai_message(f"抱歉，处理您的问题时出现错误: {error_message}")
        self._reset_ui_state()
    
    def start_ai_analysis_with_data(self, analysis_data):
        """使用真实评估数据启动AI分析"""
        if not self.llm_framework:
            return
            
        # 检查是否有线程正在运行
        if self.chat_thread and self.chat_thread.isRunning():
            QMessageBox.information(self, "提示", "AI正在处理中，请稍等...")
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("正在分析模型评估结果...")
        self.status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        
        # 禁用UI控件
        self.set_ui_enabled(False)
        
        # 创建并启动分析线程，传入真实的评估数据
        self.chat_thread = LLMChatThread(self.llm_framework)
        self.chat_thread.set_analysis_task("compare_models", analysis_data)
        
        # 连接信号
        self.chat_thread.analysis_finished.connect(self.on_analysis_finished)
        self.chat_thread.analysis_error.connect(self.on_analysis_error)
        
        # 启动线程
        self.chat_thread.start()

    def on_analysis_finished(self, response_text):
        """分析完成处理"""
        self.add_ai_message(response_text)
        self._reset_ui_state()
    
    def on_analysis_error(self, error_message):
        """分析错误处理"""
        self.add_ai_message(f"分析时出现错误: {error_message}")
        self._reset_ui_state()
    
    def _reset_ui_state(self):
        """重置UI状态"""
        self.progress_bar.setVisible(False)
        self.status_label.setText("AI助手已就绪")
        self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        
        # 重新启用UI控件
        self.set_ui_enabled(True)
    
    def update_training_context(self, context):
        """更新训练上下文"""
        self.training_context.update(context)
    
    def load_ai_config(self):
        """加载AI设置配置"""
        import json
        import os
        
        config_file = "setting/ai_config.json"
        default_config = {
            'openai': {
                'api_key': '',
                'base_url': '',
                'model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 1000
            },
            'ollama': {
                'base_url': 'http://localhost:11434',
                'model': 'llama2',
                'temperature': 0.7,
                'num_predict': 1000
            },
            'deepseek': {
                'api_key': '',
                'base_url': '',
                'model': 'deepseek-coder',
                'temperature': 0.7,
                'max_tokens': 1000
            },
            'custom_api': {
                'name': '',
                'api_key': '',
                'base_url': '',
                'model': 'custom-model',
                'provider_type': 'OpenAI兼容',
                'temperature': 0.7,
                'max_tokens': 1000
            },
            'general': {
                'default_adapter': 'mock',
                'request_timeout': 60,
                'max_retries': 3,
                'enable_cache': True,
                'enable_streaming': False
            }
        }
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 合并默认配置，确保所有必需的键都存在
                    for key in default_config:
                        if key not in config:
                            config[key] = default_config[key]
                        elif isinstance(default_config[key], dict):
                            for subkey in default_config[key]:
                                if subkey not in config[key]:
                                    config[key][subkey] = default_config[key][subkey]
                    return config
            else:
                return default_config
        except Exception as e:
            print(f"加载AI配置失败: {str(e)}")
            return default_config
    
    def reload_ai_config(self):
        """重新加载AI配置"""
        try:
            self.add_system_message("🔄 正在重新加载AI配置...")
            self.status_label.setText("正在重新加载配置...")
            self.status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
            
            # 重新初始化LLM框架
            self.init_llm_framework()
            
            self.add_system_message("✅ AI配置重新加载完成")
            self.status_label.setText("AI助手已就绪")
            self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
            
        except Exception as e:
            self.add_system_message(f"❌ 重新加载AI配置失败: {str(e)}")
            self.status_label.setText("配置加载失败")
            self.status_label.setStyleSheet("color: #dc3545; font-weight: bold;")


class AnalysisPanelWidget(QWidget):
    """智能分析面板组件"""
    
    status_updated = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.analysis_results = {}
        self.init_ui()
    
    def init_ui(self):
        """初始化分析面板界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 标题
        title_label = QLabel("智能分析面板")
        title_label.setFont(QFont('微软雅黑', 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 创建分析结果标签页
        self.analysis_tabs = QTabWidget()
        
        # 训练状态分析
        self.training_status_widget = self.create_training_status_widget()
        self.analysis_tabs.addTab(self.training_status_widget, "训练状态")
        
        # 性能分析
        self.performance_widget = self.create_performance_widget()
        self.analysis_tabs.addTab(self.performance_widget, "性能分析")
        
        # 问题诊断
        self.diagnosis_widget = self.create_diagnosis_widget()
        self.analysis_tabs.addTab(self.diagnosis_widget, "问题诊断")
        
        # 优化建议
        self.suggestions_widget = self.create_suggestions_widget()
        self.analysis_tabs.addTab(self.suggestions_widget, "优化建议")
        
        layout.addWidget(self.analysis_tabs)
        
        # 刷新按钮
        refresh_btn = QPushButton("🔄 刷新分析")
        refresh_btn.clicked.connect(self.refresh_analysis)
        layout.addWidget(refresh_btn)
    
    def create_training_status_widget(self):
        """创建训练状态分析组件"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.status_display = QTextEdit()
        self.status_display.setReadOnly(True)
        self.status_display.setMaximumHeight(200)
        self.status_display.setPlaceholderText("训练状态分析将在这里显示...")
        layout.addWidget(self.status_display)
        
        return widget
    
    def create_performance_widget(self):
        """创建性能分析组件"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.performance_display = QTextEdit()
        self.performance_display.setReadOnly(True)
        self.performance_display.setMaximumHeight(200)
        self.performance_display.setPlaceholderText("性能分析结果将在这里显示...")
        layout.addWidget(self.performance_display)
        
        return widget
    
    def create_diagnosis_widget(self):
        """创建问题诊断组件"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.diagnosis_display = QTextEdit()
        self.diagnosis_display.setReadOnly(True)
        self.diagnosis_display.setMaximumHeight(200)
        self.diagnosis_display.setPlaceholderText("问题诊断结果将在这里显示...")
        layout.addWidget(self.diagnosis_display)
        
        return widget
    
    def create_suggestions_widget(self):
        """创建优化建议组件"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.suggestions_display = QTextEdit()
        self.suggestions_display.setReadOnly(True)
        self.suggestions_display.setMaximumHeight(200)
        self.suggestions_display.setPlaceholderText("优化建议将在这里显示...")
        layout.addWidget(self.suggestions_display)
        
        return widget
    
    def refresh_analysis(self):
        """刷新分析结果 - 使用真实训练数据"""
        self.status_updated.emit("正在获取真实训练数据...")
        
        try:
            # 获取实时指标采集器
            from src.training_components.real_time_metrics_collector import get_global_metrics_collector
            collector = get_global_metrics_collector()
            
            # 获取真实训练数据
            real_data = collector.get_current_training_data_for_ai()
            
            if "error" in real_data:
                # 如果没有真实数据，显示提示信息
                error_msg = real_data["error"]
                self.status_display.setText(f"⚠️ 无法获取训练数据\n📝 原因: {error_msg}\n💡 请确保训练正在进行中")
                self.performance_display.setText("📊 性能数据不可用\n🔄 请启动训练后再次刷新")
                self.diagnosis_display.setText("🔍 诊断功能需要训练数据\n⏳ 等待训练开始...")
                self.suggestions_display.setText("💭 优化建议需要基于真实数据\n🚀 开始训练后将提供个性化建议")
                self.status_updated.emit(f"数据获取失败: {error_msg}")
                return
            
            # 解析真实数据
            current_metrics = real_data.get("current_metrics", {})
            training_trends = real_data.get("training_trends", {})
            training_status = real_data.get("training_status", "unknown")
            session_id = real_data.get("session_id", "unknown")
            total_points = real_data.get("total_data_points", 0)
            duration = real_data.get("collection_duration", 0)
            
            # 1. 训练状态分析
            self._update_training_status_display(current_metrics, training_trends, training_status, session_id)
            
            # 2. 性能分析
            self._update_performance_display(current_metrics, training_trends, total_points, duration)
            
            # 3. 问题诊断
            self._update_diagnosis_display(current_metrics, training_trends)
            
            # 4. 优化建议
            self._update_suggestions_display(current_metrics, training_trends)
            
            self.status_updated.emit(f"分析结果已更新 (基于{total_points}个真实数据点)")
            
        except ImportError:
            # 如果无法导入采集器，显示模块不可用信息
            self.status_display.setText("❌ 实时数据采集模块不可用\n📦 请检查training_components模块")
            self.performance_display.setText("⚠️ 性能监控功能未启用")
            self.diagnosis_display.setText("🔧 诊断功能需要数据采集支持")
            self.suggestions_display.setText("💡 建议功能需要真实训练数据")
            self.status_updated.emit("数据采集模块不可用")
            
        except Exception as e:
            # 其他异常处理
            error_msg = str(e)
            self.status_display.setText(f"❌ 数据获取异常\n📝 错误: {error_msg}")
            self.performance_display.setText("⚠️ 性能数据获取失败")
            self.diagnosis_display.setText("🔧 诊断功能暂时不可用")
            self.suggestions_display.setText("💡 优化建议暂时不可用")
            self.status_updated.emit(f"分析异常: {error_msg}")
    
    def _update_training_status_display(self, current_metrics, training_trends, training_status, session_id):
        """更新训练状态显示"""
        try:
            status_text = f"📊 训练会话: {session_id}\n"
            status_text += f"🔄 状态: {training_status}\n"
            
            # 当前指标
            if current_metrics:
                epoch = current_metrics.get('epoch', 'N/A')
                phase = current_metrics.get('phase', 'N/A')
                loss = current_metrics.get('loss', 'N/A')
                accuracy = current_metrics.get('accuracy', 'N/A')
                
                status_text += f"📈 当前Epoch: {epoch}\n"
                status_text += f"🎯 训练阶段: {phase}\n"
                
                if isinstance(loss, (int, float)):
                    status_text += f"📉 损失值: {loss:.4f}\n"
                else:
                    status_text += f"📉 损失值: {loss}\n"
                    
                if isinstance(accuracy, (int, float)):
                    status_text += f"🎯 准确率: {accuracy:.1%}\n"
                else:
                    status_text += f"🎯 准确率: {accuracy}\n"
            
            # 趋势分析
            train_losses = training_trends.get('train_losses', [])
            val_losses = training_trends.get('val_losses', [])
            train_accs = training_trends.get('train_accuracies', [])
            val_accs = training_trends.get('val_accuracies', [])
            
            if train_losses and len(train_losses) >= 2:
                loss_trend = "📈 上升" if train_losses[-1] > train_losses[-2] else "📉 下降"
                status_text += f"📊 训练损失趋势: {loss_trend}\n"
                
            if val_losses and len(val_losses) >= 2:
                val_trend = "📈 上升" if val_losses[-1] > val_losses[-2] else "📉 下降"
                status_text += f"📊 验证损失趋势: {val_trend}\n"
                
            # 收敛状态判断
            if train_losses and val_losses and len(train_losses) >= 3:
                recent_train = train_losses[-3:]
                recent_val = val_losses[-3:]
                
                train_stable = max(recent_train) - min(recent_train) < 0.01
                val_stable = max(recent_val) - min(recent_val) < 0.01
                
                if train_stable and val_stable:
                    status_text += "✅ 收敛状态: 稳定\n"
                elif train_stable:
                    status_text += "⚠️ 收敛状态: 训练稳定，验证波动\n"
                else:
                    status_text += "🔄 收敛状态: 持续学习中\n"
            
            self.status_display.setText(status_text.strip())
            
        except Exception as e:
            self.status_display.setText(f"❌ 训练状态分析失败: {str(e)}")
    
    def _update_performance_display(self, current_metrics, training_trends, total_points, duration):
        """更新性能分析显示"""
        try:
            perf_text = f"📊 数据采集点数: {total_points}\n"
            perf_text += f"⏱️ 采集持续时间: {duration:.1f}秒\n"
            
            # 计算数据采集频率
            if duration > 0:
                frequency = total_points / duration
                perf_text += f"📈 数据采集频率: {frequency:.2f} 点/秒\n"
            
            # 训练速度分析
            train_losses = training_trends.get('train_losses', [])
            val_losses = training_trends.get('val_losses', [])
            epochs = training_trends.get('epochs', [])
            
            if epochs and len(epochs) >= 2:
                epoch_span = max(epochs) - min(epochs) + 1
                if duration > 0:
                    epoch_speed = epoch_span / (duration / 3600)  # epoch/小时
                    perf_text += f"🚀 训练速度: {epoch_speed:.2f} epoch/小时\n"
            
            # 损失变化率
            if train_losses and len(train_losses) >= 2:
                loss_change = abs(train_losses[-1] - train_losses[0])
                improvement_rate = loss_change / len(train_losses)
                perf_text += f"📉 训练损失改善率: {improvement_rate:.4f}/step\n"
            
            if val_losses and len(val_losses) >= 2:
                val_change = abs(val_losses[-1] - val_losses[0])
                val_improvement = val_change / len(val_losses)
                perf_text += f"📊 验证损失变化率: {val_improvement:.4f}/step\n"
            
            # 数据质量评估
            if train_losses and val_losses:
                data_quality = "🟢 优秀" if len(train_losses) > 10 and len(val_losses) > 5 else "🟡 一般"
                perf_text += f"🎯 数据质量: {data_quality}\n"
            
            self.performance_display.setText(perf_text.strip())
            
        except Exception as e:
            self.performance_display.setText(f"❌ 性能分析失败: {str(e)}")
    
    def _update_diagnosis_display(self, current_metrics, training_trends):
        """更新问题诊断显示"""
        try:
            diagnosis_text = ""
            issues_found = []
            
            train_losses = training_trends.get('train_losses', [])
            val_losses = training_trends.get('val_losses', [])
            train_accs = training_trends.get('train_accuracies', [])
            val_accs = training_trends.get('val_accuracies', [])
            
            # 过拟合检测
            if train_losses and val_losses and len(train_losses) >= 3 and len(val_losses) >= 3:
                avg_train_loss = sum(train_losses[-3:]) / 3
                avg_val_loss = sum(val_losses[-3:]) / 3
                
                if avg_val_loss > avg_train_loss * 1.5:
                    issues_found.append("⚠️ 检测到过拟合趋势")
                    issues_found.append("💡 建议: 增加正则化或减少模型复杂度")
                elif avg_val_loss < avg_train_loss * 0.8:
                    issues_found.append("⚠️ 可能存在欠拟合")
                    issues_found.append("💡 建议: 增加模型复杂度或减少正则化")
            
            # 学习停滞检测
            if train_losses and len(train_losses) >= 5:
                recent_losses = train_losses[-5:]
                loss_variance = max(recent_losses) - min(recent_losses)
                if loss_variance < 0.001:
                    issues_found.append("⚠️ 训练可能已停滞")
                    issues_found.append("💡 建议: 调整学习率或使用学习率调度器")
            
            # 梯度爆炸/消失检测（通过损失变化判断）
            if train_losses and len(train_losses) >= 2:
                loss_change = abs(train_losses[-1] - train_losses[-2])
                if loss_change > 1.0:
                    issues_found.append("⚠️ 可能存在梯度爆炸")
                    issues_found.append("💡 建议: 降低学习率或使用梯度裁剪")
                elif loss_change < 1e-6:
                    issues_found.append("⚠️ 可能存在梯度消失")
                    issues_found.append("💡 建议: 检查网络结构或使用残差连接")
            
            # 准确率异常检测
            if train_accs and val_accs and len(train_accs) >= 2 and len(val_accs) >= 2:
                train_acc_trend = train_accs[-1] - train_accs[-2]
                val_acc_trend = val_accs[-1] - val_accs[-2]
                
                if train_acc_trend > 0.1 and val_acc_trend < -0.05:
                    issues_found.append("⚠️ 训练准确率上升但验证准确率下降")
                    issues_found.append("💡 建议: 检查数据分布或增加数据增强")
            
            if not issues_found:
                diagnosis_text = "✅ 未发现明显训练问题\n📈 训练进展正常\n🎯 继续当前训练策略"
            else:
                diagnosis_text = "\n".join(issues_found)
            
            self.diagnosis_display.setText(diagnosis_text)
            
        except Exception as e:
            self.diagnosis_display.setText(f"❌ 问题诊断失败: {str(e)}")
    
    def _update_suggestions_display(self, current_metrics, training_trends):
        """更新优化建议显示"""
        try:
            suggestions = []
            
            train_losses = training_trends.get('train_losses', [])
            val_losses = training_trends.get('val_losses', [])
            train_accs = training_trends.get('train_accuracies', [])
            val_accs = training_trends.get('val_accuracies', [])
            
            # 基于损失趋势的建议
            if train_losses and len(train_losses) >= 3:
                recent_trend = sum(train_losses[-3:]) / 3 - sum(train_losses[-6:-3]) / 3 if len(train_losses) >= 6 else 0
                
                if recent_trend > 0.01:
                    suggestions.append("📉 训练损失上升，建议降低学习率")
                elif recent_trend < -0.05:
                    suggestions.append("🚀 训练进展良好，可考虑适当提高学习率")
                else:
                    suggestions.append("⚖️ 训练损失稳定，保持当前学习率")
            
            # 基于过拟合风险的建议
            if train_losses and val_losses and len(train_losses) >= 2 and len(val_losses) >= 2:
                train_val_gap = val_losses[-1] - train_losses[-1]
                
                if train_val_gap > 0.2:
                    suggestions.append("🛡️ 过拟合风险较高，建议增加Dropout或正则化")
                elif train_val_gap < 0.05:
                    suggestions.append("🎯 泛化能力良好，可考虑增加模型复杂度")
            
            # 基于准确率的建议
            if val_accs and len(val_accs) >= 1:
                current_val_acc = val_accs[-1]
                
                if current_val_acc < 0.7:
                    suggestions.append("📊 验证准确率较低，建议检查数据质量或模型架构")
                elif current_val_acc > 0.9:
                    suggestions.append("🎉 验证准确率优秀，可考虑进行模型压缩或部署")
            
            # 训练时间建议
            current_epoch = current_metrics.get('epoch', 0)
            if isinstance(current_epoch, (int, float)) and current_epoch > 0:
                if current_epoch < 10:
                    suggestions.append("⏰ 训练初期，建议密切观察损失变化")
                elif current_epoch > 50:
                    suggestions.append("⏳ 训练时间较长，建议评估是否需要早停")
            
            # 数据增强建议
            if train_losses and val_losses and len(train_losses) >= 5:
                train_stability = max(train_losses[-5:]) - min(train_losses[-5:])
                if train_stability < 0.01:
                    suggestions.append("🔄 训练稳定，可考虑增加数据增强多样性")
            
            if not suggestions:
                suggestions.append("📋 基于当前数据暂无特定建议")
                suggestions.append("💡 继续监控训练进展")
            
            suggestions_text = "\n".join(f"{i+1}. {suggestion}" for i, suggestion in enumerate(suggestions))
            self.suggestions_display.setText(suggestions_text)
            
        except Exception as e:
            self.suggestions_display.setText(f"❌ 优化建议生成失败: {str(e)}")


class ModelFactoryTab(BaseTab):
    """模型工厂标签页 - 集成LLM智能分析功能"""
    
    # 定义信号
    llm_analysis_requested = pyqtSignal(dict)
    training_context_updated = pyqtSignal(dict)
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent, main_window)
        self.main_window = main_window
        self.current_analysis = {}
        self.init_ui()
        
        # 使用新的智能配置系统
        config = self.get_config_from_manager()
        if config:
            self.apply_config(config)
    
    def init_ui(self):
        """初始化用户界面"""
        # 创建主布局
        main_layout = QVBoxLayout(self.scroll_content)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        

        
        # 创建垂直分割器
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.setChildrenCollapsible(False)
        
        # 上半部分：LLM聊天和分析面板 - 使用水平分割器
        upper_splitter = QSplitter(Qt.Horizontal)
        upper_splitter.setChildrenCollapsible(False)
        
        # 左侧：LLM聊天界面
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        self.chat_widget = LLMChatWidget()
        self.chat_widget.status_updated.connect(self.update_status)
        self.chat_widget.analysis_requested.connect(self.handle_analysis_request)
        left_layout.addWidget(self.chat_widget)
        
        upper_splitter.addWidget(left_widget)
        
        # 右侧：分析面板和系统状态
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # 分析面板
        self.analysis_panel = AnalysisPanelWidget()
        self.analysis_panel.status_updated.connect(self.update_status)
        right_layout.addWidget(self.analysis_panel)
        
        # 系统状态面板
        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout()
        
        self.system_status_display = QTextEdit()
        self.system_status_display.setReadOnly(True)
        self.system_status_display.setMaximumHeight(150)
        self.system_status_display.setPlaceholderText("系统状态信息将在这里显示...")
        status_layout.addWidget(self.system_status_display)
        
        # 系统控制按钮
        control_layout = QHBoxLayout()
        
        self.health_check_btn = QPushButton("🏥 健康检查")
        self.health_check_btn.clicked.connect(self.perform_health_check)
        control_layout.addWidget(self.health_check_btn)
        
        self.stats_btn = QPushButton("📊 系统统计")
        self.stats_btn.clicked.connect(self.show_system_stats)
        control_layout.addWidget(self.stats_btn)
        
        self.clear_btn = QPushButton("🗑️ 清空历史")
        self.clear_btn.clicked.connect(self.clear_chat_history)
        control_layout.addWidget(self.clear_btn)
        
        status_layout.addLayout(control_layout)
        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)
        
        upper_splitter.addWidget(right_widget)
        
        main_splitter.addWidget(upper_splitter)
        
        # 设置水平分割器的初始比例 (聊天界面60%, 分析面板40%)
        upper_splitter.setSizes([600, 400])
        
        # 下半部分：Batch分析触发控件和实时数据流监控
        lower_widget = QWidget()
        lower_layout = QVBoxLayout(lower_widget)
        lower_layout.setContentsMargins(5, 5, 5, 5)
        
        # 创建水平分割器用于下半部分
        lower_splitter = QSplitter(Qt.Horizontal)
        lower_splitter.setChildrenCollapsible(False)
        
        # 左侧：Batch分析触发控件
        left_lower_widget = QWidget()
        left_lower_layout = QVBoxLayout(left_lower_widget)
        left_lower_layout.setContentsMargins(5, 5, 5, 5)
        
        # 导入并创建Batch分析触发控件
        try:
            from src.ui.components.model_analysis.batch_analysis_trigger_widget import BatchAnalysisTriggerWidget
            self.batch_analysis_trigger = BatchAnalysisTriggerWidget()
            self.batch_analysis_trigger.status_updated.connect(self.update_status)
            self.batch_analysis_trigger.analysis_triggered.connect(self.handle_batch_analysis_triggered)
            left_lower_layout.addWidget(self.batch_analysis_trigger)
            
            # 从AI设置加载配置
            self.load_batch_analysis_config()
        except ImportError as e:
            # 如果导入失败，显示错误信息
            error_label = QLabel(f"⚠️ Batch分析触发控件加载失败: {str(e)}")
            error_label.setStyleSheet("color: #dc3545; padding: 20px; border: 1px solid #dc3545; border-radius: 5px;")
            error_label.setAlignment(Qt.AlignCenter)
            left_lower_layout.addWidget(error_label)
        
        lower_splitter.addWidget(left_lower_widget)
        
        # 右侧：实时数据流监控
        right_lower_widget = QWidget()
        right_lower_layout = QVBoxLayout(right_lower_widget)
        right_lower_layout.setContentsMargins(5, 5, 5, 5)
        
        # 导入并创建实时数据流监控控件
        try:
            from src.ui.components.model_analysis.real_time_stream_monitor import RealTimeStreamMonitor
            self.stream_monitor = RealTimeStreamMonitor()
            right_lower_layout.addWidget(self.stream_monitor)
        except ImportError as e:
            # 如果导入失败，显示错误信息
            error_label = QLabel(f"⚠️ 实时数据流监控控件加载失败: {str(e)}")
            error_label.setStyleSheet("color: #dc3545; padding: 20px; border: 1px solid #dc3545; border-radius: 5px;")
            error_label.setAlignment(Qt.AlignCenter)
            right_lower_layout.addWidget(error_label)
        
        lower_splitter.addWidget(right_lower_widget)
        
        # 设置下半部分分割器的比例 (Batch分析触发控件40%, 实时监控60%)
        lower_splitter.setSizes([400, 600])
        
        lower_layout.addWidget(lower_splitter)
        main_splitter.addWidget(lower_widget)
        
        # 设置分割器比例 (上半部分70%, 下半部分30%)
        main_splitter.setSizes([700, 300])
        
        main_layout.addWidget(main_splitter)
        
        # 初始化系统状态
        self.update_system_status()
    
    def handle_analysis_request(self, request_data):
        """处理分析请求"""
        self.llm_analysis_requested.emit(request_data)
        self.update_status("正在处理分析请求...")
    
    def handle_batch_analysis_triggered(self, analysis_data):
        """处理Batch分析触发事件"""
        try:
            # 更新状态
            trigger_type = analysis_data.get('trigger_type', 'unknown')
            batch_count = analysis_data.get('batch_count', 0)
            analysis_count = analysis_data.get('analysis_count', 0)
            
            self.update_status(f"Batch分析触发: {trigger_type} (Batch {batch_count}, 第{analysis_count}次)")
            
            # 如果有聊天组件，自动发送分析请求
            if hasattr(self, 'chat_widget') and self.chat_widget:
                # 构建分析提示
                metrics = analysis_data.get('metrics', {})
                analysis_prompt = self._build_batch_analysis_prompt(analysis_data)
                
                # 发送到聊天组件进行AI分析
                self.chat_widget.add_user_message(analysis_prompt)
                self.chat_widget.analyze_training()
                
        except Exception as e:
            self.update_status(f"处理Batch分析触发时出错: {str(e)}")
    
    def _build_batch_analysis_prompt(self, analysis_data):
        """构建Batch分析提示"""
        trigger_type = analysis_data.get('trigger_type', 'unknown')
        batch_count = analysis_data.get('batch_count', 0)
        epoch = analysis_data.get('epoch', 0)
        phase = analysis_data.get('phase', '')
        analysis_count = analysis_data.get('analysis_count', 0)
        metrics = analysis_data.get('metrics', {})
        
        prompt = f"""
## 🎯 Batch分析请求 (第{analysis_count}次)

**触发信息:**
- 触发类型: {trigger_type}
- 当前Epoch: {epoch}
- 当前Phase: {phase}
- 当前Batch: {batch_count}

**训练指标:**
{self._format_metrics_for_prompt(metrics)}

请基于以上信息，对当前训练状态进行专业分析，包括：
1. 训练进展评估
2. 潜在问题诊断
3. 优化建议
4. 下一步预测

请提供详细、专业的分析报告。
"""
        return prompt
    
    def _format_metrics_for_prompt(self, metrics):
        """格式化指标用于提示"""
        if not metrics:
            return "暂无详细指标数据"
            
        formatted = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                formatted.append(f"- {key}: {value}")
            else:
                formatted.append(f"- {key}: {value}")
                
        return "\n".join(formatted) if formatted else "暂无详细指标数据"
    
    def update_training_context(self, context):
        """更新训练上下文"""
        if hasattr(self, 'chat_widget'):
            self.chat_widget.update_training_context(context)
        self.training_context_updated.emit(context)
    
    def perform_health_check(self):
        """执行系统健康检查"""
        if not LLM_AVAILABLE:
            self.system_status_display.setText("❌ LLM模块不可用")
            return
            
        try:
            if hasattr(self.chat_widget, 'llm_framework') and self.chat_widget.llm_framework:
                health_status = self.chat_widget.llm_framework.get_system_health()
                self.system_status_display.setText(f"✅ 系统健康状态: {health_status}")
            else:
                self.system_status_display.setText("⚠️ LLM框架未初始化")
        except Exception as e:
            self.system_status_display.setText(f"❌ 健康检查失败: {str(e)}")
    
    def show_system_stats(self):
        """显示系统统计信息"""
        if not LLM_AVAILABLE:
            self.system_status_display.setText("❌ LLM模块不可用")
            return
            
        try:
            if hasattr(self.chat_widget, 'llm_framework') and self.chat_widget.llm_framework:
                stats = self.chat_widget.llm_framework.get_framework_stats()
                
                # 获取性能统计数据
                perf_stats = stats.get('performance_stats', {})
                adapter_info = stats.get('adapter_info', {})
                engine_stats = stats.get('engine_stats', {})
                
                # 计算成功率
                total_requests = perf_stats.get('total_requests', 0)
                successful_requests = perf_stats.get('successful_requests', 0)
                success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
                
                # 获取平均响应时间
                avg_response_time = perf_stats.get('average_response_time', 0)
                
                # 获取适配器信息
                adapter_type = adapter_info.get('type', 'Unknown')
                adapter_available = adapter_info.get('available', False)
                
                # 获取引擎统计
                analyses_performed = engine_stats.get('analyses_performed', 0)
                metrics_processed = engine_stats.get('metrics_processed', 0)
                
                # 获取请求类型统计
                request_types = perf_stats.get('request_types', {})
                
                # 计算运行时间
                start_time = perf_stats.get('start_time', time.time())
                uptime_seconds = time.time() - start_time
                uptime_hours = uptime_seconds / 3600
                
                stats_text = f"""
📊 框架统计信息:
• 框架状态: {stats.get('framework_status', 'Unknown')}
• 运行时间: {uptime_hours:.1f}小时
• 总请求数: {total_requests}
• 成功请求数: {successful_requests}
• 失败请求数: {perf_stats.get('failed_requests', 0)}
• 成功率: {success_rate:.1f}%
• 平均响应时间: {avg_response_time:.2f}秒

📈 请求类型分布:
• 指标分析: {request_types.get('analyze_metrics', 0)}次
• 获取建议: {request_types.get('get_suggestions', 0)}次
• 问题诊断: {request_types.get('diagnose_issues', 0)}次
• 对话交互: {request_types.get('chat', 0)}次
• 模型对比: {request_types.get('compare_models', 0)}次

🔧 适配器信息:
• 类型: {adapter_type}
• 状态: {'可用' if adapter_available else '不可用'}
• 模型: {adapter_info.get('model_name', 'Unknown')}

📊 引擎统计:
• 已执行分析: {analyses_performed}次
• 已处理指标: {metrics_processed}条
                """.strip()
                self.system_status_display.setText(stats_text)
            else:
                self.system_status_display.setText("⚠️ LLM框架未初始化")
        except Exception as e:
            self.system_status_display.setText(f"❌ 获取统计信息失败: {str(e)}")
    
    def clear_chat_history(self):
        """清空聊天历史"""
        if hasattr(self.chat_widget, 'chat_display'):
            self.chat_widget.chat_display.clear()
            self.chat_widget.chat_history.clear()
            self.chat_widget.add_system_message("聊天历史已清空")
    
    def update_system_status(self):
        """更新系统状态显示"""
        if LLM_AVAILABLE:
            status_text = """
🟢 AI模型工厂已启动
📡 数据流服务: 就绪
🤖 LLM框架: 已加载
💬 聊天服务: 正常
📊 分析引擎: 待命
            """.strip()
        else:
            status_text = """
🔴 AI模型工厂启动异常
❌ LLM框架: 未加载
⚠️ 请检查依赖安装
            """.strip()
        
        self.system_status_display.setText(status_text)
    
    def on_training_started(self, training_info):
        """训练开始时更新上下文"""
        context = {
            'training_active': True,
            'model_type': training_info.get('model_type', 'unknown'),
            'dataset': training_info.get('dataset', 'unknown'),
            'start_time': datetime.now().isoformat()
        }
        self.update_training_context(context)
        
        # 通知Batch分析触发控件训练已开始
        if hasattr(self, 'batch_analysis_trigger'):
            self.batch_analysis_trigger.on_training_started(training_info)
    
    def on_training_progress(self, metrics):
        """训练进度更新时更新上下文"""
        context = {
            'current_metrics': metrics,
            'last_update': datetime.now().isoformat()
        }
        self.update_training_context(context)
        
        # 更新Batch分析触发控件
        if hasattr(self, 'batch_analysis_trigger'):
            self.batch_analysis_trigger.update_training_progress(metrics)
    
    def on_training_completed(self, results):
        """训练完成时更新上下文"""
        context = {
            'training_active': False,
            'final_results': results,
            'completion_time': datetime.now().isoformat()
        }
        self.update_training_context(context)
        
        # 通知Batch分析触发控件训练已完成
        if hasattr(self, 'batch_analysis_trigger'):
            self.batch_analysis_trigger.on_training_completed(results)
    
    def on_training_stopped(self):
        """训练停止时更新上下文"""
        context = {
            'training_active': False,
            'stop_time': datetime.now().isoformat()
        }
        self.update_training_context(context)
        
        # 通知Batch分析触发控件训练已停止
        if hasattr(self, 'batch_analysis_trigger'):
            self.batch_analysis_trigger.on_training_stopped()
    
    def reload_ai_config(self):
        """重新加载AI配置"""
        try:
            self.add_system_message("🔄 正在重新加载AI配置...")
            self.status_label.setText("正在重新加载配置...")
            self.status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
            
            # 重新初始化LLM框架
            self.init_llm_framework()
            
            self.add_system_message("✅ AI配置重新加载完成")
            self.status_label.setText("AI助手已就绪")
            self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
            
        except Exception as e:
            self.add_system_message(f"❌ 重新加载AI配置失败: {str(e)}")
            self.status_label.setText("配置加载失败")
            self.status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
    
    def update_ai_adapter_from_settings(self, ai_config):
        """从设置更新AI适配器配置"""
        if hasattr(self, 'chat_widget') and self.chat_widget and hasattr(self.chat_widget, 'llm_framework'):
            try:
                default_adapter = ai_config.get('general', {}).get('default_adapter', 'mock')
                
                # 更新下拉框显示
                if default_adapter == 'openai':
                    self.chat_widget.adapter_combo.setCurrentText("OpenAI GPT-4")
                elif default_adapter == 'local':
                    self.chat_widget.adapter_combo.setCurrentText("本地Ollama")
                elif default_adapter == 'deepseek':
                    self.chat_widget.adapter_combo.setCurrentText("DeepSeek")
                elif default_adapter == 'custom':
                    self.chat_widget.adapter_combo.setCurrentText("自定义API")
                else:
                    self.chat_widget.adapter_combo.setCurrentText("模拟适配器")
                
                # 切换适配器
                self.chat_widget.switch_adapter(self.chat_widget.adapter_combo.currentText())
                
            except Exception as e:
                print(f"更新AI适配器配置时出错: {str(e)}")
        
        # 更新Batch分析触发控件配置
        self.load_batch_analysis_config()
    
    def load_batch_analysis_config(self):
        """从AI设置加载Batch分析配置"""
        try:
            # 从配置管理器获取AI配置
            from src.utils.config_manager import config_manager
            ai_config = config_manager.get_ai_config()
            
            if ai_config and hasattr(self, 'batch_analysis_trigger'):
                self.batch_analysis_trigger.update_config_from_ai_settings(ai_config)
                
        except Exception as e:
            print(f"加载Batch分析配置时出错: {str(e)}") 