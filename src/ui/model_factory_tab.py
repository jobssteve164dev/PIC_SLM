from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                           QSplitter, QGroupBox, QTabWidget, QTextEdit, QLineEdit,
                           QScrollArea, QFrame, QSizePolicy, QStackedWidget, QComboBox,
                           QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QTextCursor, QColor, QPalette
import os
import sys
import json
from datetime import datetime
from .base_tab import BaseTab

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
        # 先更新训练上下文到LLM框架（如果有的话）
        if hasattr(self.llm_framework, 'analysis_engine') and self.training_context:
            self.llm_framework.analysis_engine.prompt_builder.add_context({
                'type': 'training_context',
                'data': self.training_context
            })
        
        # 调用LLM框架进行对话
        result = self.llm_framework.chat_with_training_context(self.user_message)
        
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
        # 模拟当前训练指标
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
        # 模拟训练历史
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
        # 模拟问题指标
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
        
        # LLM适配器选择
        adapter_layout = QHBoxLayout()
        adapter_layout.addWidget(QLabel("AI模型:"))
        self.adapter_combo = QComboBox()
        self.adapter_combo.addItems(["模拟适配器", "OpenAI GPT-4", "DeepSeek", "本地Ollama"])
        self.adapter_combo.currentTextChanged.connect(self.switch_adapter)
        adapter_layout.addWidget(self.adapter_combo)
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
        formatted_message = f"<div style='color: #6c757d; font-size: 9pt; margin: 5px 0;'>[{timestamp}] {message}</div>"
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
        formatted_message = f"""
        <div style='margin: 10px 0; padding: 8px; background-color: #f8f9fa; color: #000000; border: 1px solid #dee2e6; border-radius: 10px;'>
            <span style='color: #6c757d; font-size: 11px; font-weight: normal;'>AI助手 [{timestamp}]:</span><br>
            <span style='color: #000000; font-weight: bold;'>{message}</span>
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
            
            # 查找模型评估标签页的索引
            for i in range(tab_widget.count()):
                tab_text = tab_widget.tabText(i)
                if "评估" in tab_text or "Evaluation" in tab_text:
                    tab_widget.setCurrentIndex(i)
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
        """刷新分析结果"""
        self.status_updated.emit("正在刷新分析结果...")
        
        # 模拟分析结果
        self.status_display.setText("✅ 训练状态正常\n📈 损失函数收敛良好\n🎯 准确率稳步提升")
        self.performance_display.setText("🚀 GPU利用率: 85%\n⚡ 训练速度: 1.2 samples/sec\n💾 内存使用: 6.2GB/8GB")
        self.diagnosis_display.setText("⚠️ 检测到轻微过拟合趋势\n💡 建议增加数据增强\n🔧 可考虑调整学习率")
        self.suggestions_display.setText("1. 降低学习率至0.0005\n2. 增加Dropout至0.3\n3. 使用余弦退火调度器")
        
        self.status_updated.emit("分析结果已更新")


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
        

        
        # 创建水平分割器
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        
        # 左侧：LLM聊天界面
        left_widget = QWidget()
        left_widget.setMinimumWidth(400)
        left_widget.setMaximumWidth(600)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        self.chat_widget = LLMChatWidget()
        self.chat_widget.status_updated.connect(self.update_status)
        self.chat_widget.analysis_requested.connect(self.handle_analysis_request)
        left_layout.addWidget(self.chat_widget)
        
        main_splitter.addWidget(left_widget)
        
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
        
        main_splitter.addWidget(right_widget)
        
        # 设置分割器比例 (左侧60%, 右侧40%)
        main_splitter.setSizes([600, 400])
        
        main_layout.addWidget(main_splitter)
        
        # 初始化系统状态
        self.update_system_status()
    
    def handle_analysis_request(self, request_data):
        """处理分析请求"""
        self.llm_analysis_requested.emit(request_data)
        self.update_status("正在处理分析请求...")
    
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
                stats_text = f"""
📊 框架统计信息:
• 总请求数: {stats.get('total_requests', 0)}
• 成功率: {stats.get('success_rate', 0):.1f}%
• 平均响应时间: {stats.get('avg_response_time', 0):.2f}秒
• 当前适配器: {stats.get('current_adapter', 'Unknown')}
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
    
    def on_training_progress(self, metrics):
        """训练进度更新时更新上下文"""
        context = {
            'current_metrics': metrics,
            'last_update': datetime.now().isoformat()
        }
        self.update_training_context(context)
    
    def on_training_completed(self, results):
        """训练完成时更新上下文"""
        context = {
            'training_active': False,
            'final_results': results,
            'completion_time': datetime.now().isoformat()
        }
        self.update_training_context(context)
    
    def reload_ai_config(self):
        """重新加载AI配置并更新适配器"""
        if hasattr(self, 'chat_widget') and self.chat_widget:
            try:
                # 重新初始化LLM框架
                self.chat_widget.init_llm_framework()
                self.update_status("AI配置已重新加载")
            except Exception as e:
                self.update_status(f"重新加载AI配置失败: {str(e)}")
    
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
                else:
                    self.chat_widget.adapter_combo.setCurrentText("模拟适配器")
                
                # 切换适配器
                self.chat_widget.switch_adapter(self.chat_widget.adapter_combo.currentText())
                
            except Exception as e:
                print(f"更新AI适配器配置时出错: {str(e)}") 