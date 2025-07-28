"""
AI设置组件 - 用于配置Ollama和OpenAI的API设置和模型选择
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                           QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
                           QCheckBox, QTextEdit, QTabWidget, QMessageBox,
                           QFormLayout, QDoubleSpinBox, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont
import json
import os
import requests
from typing import Dict, List

# 导入LLM相关模块
try:
    from src.llm.model_adapters import create_llm_adapter
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class OllamaTestThread(QThread):
    """Ollama连接测试线程"""
    
    test_completed = pyqtSignal(bool, str, list)  # 成功状态, 消息, 可用模型列表
    
    def __init__(self, base_url):
        super().__init__()
        self.base_url = base_url
    
    def run(self):
        try:
            # 测试连接
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                self.test_completed.emit(True, "连接成功", models)
            else:
                self.test_completed.emit(False, f"连接失败: HTTP {response.status_code}", [])
        except requests.exceptions.ConnectionError:
            self.test_completed.emit(False, "连接失败: 无法连接到Ollama服务", [])
        except requests.exceptions.Timeout:
            self.test_completed.emit(False, "连接失败: 请求超时", [])
        except Exception as e:
            self.test_completed.emit(False, f"连接失败: {str(e)}", [])


class OpenAITestThread(QThread):
    """OpenAI API测试线程"""
    
    test_completed = pyqtSignal(bool, str, list)  # 成功状态, 消息, 可用模型列表
    
    def __init__(self, api_key, base_url=None):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
    
    def run(self):
        try:
            if not LLM_AVAILABLE:
                self.test_completed.emit(False, "LLM模块不可用", [])
                return
            
            # 首先尝试获取可用模型列表
            models = self._fetch_available_models()
            
            if models:
                # 如果成功获取模型列表，再测试一个简单请求来验证API密钥
                adapter = create_llm_adapter('openai', 
                                           api_key=self.api_key, 
                                           base_url=self.base_url if self.base_url != "https://api.openai.com/v1" else None)
                
                # 测试简单请求
                response = adapter.generate_response("Hello", context={'type': 'test'})
                
                if response and not response.startswith("API调用失败"):
                    self.test_completed.emit(True, f"API密钥验证成功，发现 {len(models)} 个可用模型", models)
                else:
                    # API密钥无效，但可能是网络问题，仍返回获取到的模型列表
                    self.test_completed.emit(False, "API密钥验证失败，但已获取模型列表", models)
            else:
                # 无法获取模型列表，使用预定义列表
                fallback_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
                self.test_completed.emit(False, "无法获取模型列表，使用默认模型", fallback_models)
                
        except Exception as e:
            # 发生异常时使用预定义列表
            fallback_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
            self.test_completed.emit(False, f"测试失败: {str(e)}", fallback_models)
    
    def _fetch_available_models(self):
        """获取可用模型列表"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 调用OpenAI的models API端点
            response = requests.get(f"{self.base_url}/models", headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                models = []
                
                # 解析模型数据，只保留聊天模型
                for model in data.get('data', []):
                    model_id = model.get('id', '')
                    # 过滤出常用的聊天模型
                    if any(keyword in model_id.lower() for keyword in ['gpt-4', 'gpt-3.5', 'chatgpt']):
                        models.append(model_id)
                
                # 按模型名称排序
                models.sort(key=lambda x: (
                    0 if 'gpt-4' in x else 1 if 'gpt-3.5' in x else 2,  # 优先级排序
                    x  # 字母排序
                ))
                
                return models
            else:
                print(f"获取模型列表失败: HTTP {response.status_code}")
                return []
                
        except requests.exceptions.Timeout:
            print("获取模型列表超时")
            return []
        except Exception as e:
            print(f"获取模型列表异常: {str(e)}")
            return []


class DeepSeekTestThread(QThread):
    """DeepSeek API测试线程"""
    
    test_completed = pyqtSignal(bool, str, list)  # 成功状态, 消息, 可用模型列表
    
    def __init__(self, api_key, base_url=None):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url or "https://api.deepseek.com/v1"
    
    def run(self):
        try:
            if not LLM_AVAILABLE:
                self.test_completed.emit(False, "LLM模块不可用", [])
                return
            
            # 首先尝试获取可用模型列表
            models = self._fetch_available_models()
            
            if models:
                # 如果成功获取模型列表，再测试一个简单请求来验证API密钥
                adapter = create_llm_adapter('deepseek', 
                                           api_key=self.api_key, 
                                           base_url=self.base_url if self.base_url != "https://api.deepseek.com/v1" else None)
                
                # 测试简单请求
                response = adapter.generate_response("Hello", context={'type': 'test'})
                
                if response and not response.startswith("API调用失败"):
                    self.test_completed.emit(True, f"API密钥验证成功，发现 {len(models)} 个可用模型", models)
                else:
                    # API密钥无效，但可能是网络问题，仍返回获取到的模型列表
                    self.test_completed.emit(False, "API密钥验证失败，但已获取模型列表", models)
            else:
                # 无法获取模型列表，使用预定义列表
                fallback_models = ["deepseek-chat", "deepseek-coder"]
                self.test_completed.emit(False, "无法获取模型列表，使用默认模型", fallback_models)
                
        except Exception as e:
            self.test_completed.emit(False, f"测试失败: {str(e)}", [])
    
    def _fetch_available_models(self):
        """获取可用模型列表"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get('data', []):
                    model_id = model.get('id', '')
                    # 获取所有可用模型
                    models.append(model_id)
                return sorted(models)
            else:
                return []
                
        except Exception as e:
            print(f"获取DeepSeek模型列表失败: {str(e)}")
            return []


class CustomAPITestThread(QThread):
    """自定义API测试线程"""
    
    test_completed = pyqtSignal(bool, str, list)  # 成功状态, 消息, 可用模型列表
    model_selection_needed = pyqtSignal(list)  # 需要用户选择模型时发出信号
    
    def __init__(self, api_key, base_url, provider_type="openai", selected_model=None):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.provider_type = provider_type
        self.selected_model = selected_model
    
    def run(self):
        try:
            # 如果已经有选定的模型，直接测试API调用
            if self.selected_model:
                print(f"使用预选模型进行测试: {self.selected_model}")
                test_success = self._test_api_call(self.selected_model)
                if test_success:
                    self.test_completed.emit(True, f"API连接成功，使用模型: {self.selected_model}", self.models_list or [])
                else:
                    self.test_completed.emit(False, f"API密钥验证失败，模型 {self.selected_model} 不可用", self.models_list or [])
                return
            
            # 首先尝试获取可用模型列表
            models = self._fetch_available_models()
            
            if models:
                # 如果没有预选模型，发出信号让用户选择
                if not self.selected_model:
                    self.model_selection_needed.emit(models)
                    return
                
                # 使用选定的模型测试API调用
                test_success = self._test_api_call(self.selected_model)
                if test_success:
                    self.test_completed.emit(True, f"API连接成功，使用模型: {self.selected_model}", models)
                else:
                    self.test_completed.emit(False, f"API密钥验证失败，模型 {self.selected_model} 不可用", models)
            else:
                # 无法获取模型列表，尝试基本连接测试
                if self._test_basic_connection():
                    fallback_models = ["自定义模型"]
                    self.test_completed.emit(True, "基本连接成功，但无法获取模型列表", fallback_models)
                else:
                    self.test_completed.emit(False, "连接失败，请检查API地址和密钥", [])
                
        except Exception as e:
            print(f"自定义API测试异常: {str(e)}")
            self.test_completed.emit(False, f"测试失败: {str(e)}", [])
    
    def set_selected_model(self, model, models_list=None):
        """设置用户选择的模型"""
        self.selected_model = model
        self.models_list = models_list
        # 重新启动线程来测试API调用
        self.start()
    
    def _fetch_available_models(self):
        """获取可用模型列表"""
        try:
            # 基础认证头
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 对于OpenRouter，添加官方推荐的认证头
            if "openrouter.ai" in self.base_url:
                headers.update({
                    "HTTP-Referer": "https://github.com/ai-training-assistant",
                    "X-Title": "AI Training Assistant"
                })
            
            print(f"正在获取模型列表: {self.base_url}/models")
            print(f"认证头: {headers}")
            
            # 尝试标准的 /models 端点
            response = requests.get(f"{self.base_url}/models", headers=headers, timeout=10)
            
            print(f"模型列表响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                models = []
                
                # 处理不同的响应格式
                if isinstance(data, dict):
                    if 'data' in data:
                        # OpenAI格式
                        for model in data.get('data', []):
                            if isinstance(model, dict):
                                model_id = model.get('id', model.get('name', ''))
                                if model_id:
                                    models.append(model_id)
                            elif isinstance(model, str):
                                models.append(model)
                    elif 'models' in data:
                        # 其他格式
                        for model in data.get('models', []):
                            if isinstance(model, dict):
                                model_id = model.get('id', model.get('name', ''))
                                if model_id:
                                    models.append(model_id)
                            elif isinstance(model, str):
                                models.append(model)
                elif isinstance(data, list):
                    # 直接是模型列表
                    for model in data:
                        if isinstance(model, dict):
                            model_id = model.get('id', model.get('name', ''))
                            if model_id:
                                models.append(model_id)
                        elif isinstance(model, str):
                            models.append(model)
                
                print(f"获取到 {len(models)} 个模型")
                return sorted(models) if models else []
            else:
                print(f"获取模型列表失败，状态码: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"错误响应: {error_data}")
                except:
                    print(f"错误响应文本: {response.text}")
                return []
                
        except Exception as e:
            print(f"获取自定义API模型列表失败: {str(e)}")
            return []
    
    def _test_basic_connection(self):
        """测试基本连接"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 尝试GET请求到根路径
            response = requests.get(self.base_url, headers=headers, timeout=5)
            return response.status_code < 500  # 4xx错误也算连接成功
            
        except Exception:
            return False
    
    def _test_api_call(self, model_name):
        """测试API调用"""
        try:
            # 基础认证头
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 对于OpenRouter，添加官方推荐的认证头
            if "openrouter.ai" in self.base_url:
                headers.update({
                    "HTTP-Referer": "https://github.com/ai-training-assistant",
                    "X-Title": "AI Training Assistant"
                })
            
            # 尝试简单的聊天请求
            data = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            
            print(f"正在测试API调用: {self.base_url}/chat/completions")
            print(f"使用模型: {model_name}")
            print(f"认证头: {headers}")
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )
            
            print(f"API响应状态码: {response.status_code}")
            print(f"响应头: {dict(response.headers)}")
            
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    print(f"API错误响应: {error_data}")
                except:
                    print(f"API错误响应文本: {response.text}")
            else:
                print("API调用成功！")
            
            return response.status_code == 200
            
        except requests.exceptions.Timeout:
            print(f"API调用超时: {self.base_url}")
            return False
        except requests.exceptions.ConnectionError:
            print(f"API连接错误: {self.base_url}")
            return False
        except Exception as e:
            print(f"API调用异常: {str(e)}")
            return False


class AISettingsWidget(QWidget):
    """AI设置主组件"""
    
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config_file = "setting/ai_config.json"
        self.current_config = {}
        self.ollama_test_thread = None
        self.openai_test_thread = None
        self.deepseek_test_thread = None
        self.custom_test_thread = None
        
        self.init_ui()
        self.load_config()
        self._connect_signals()
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 创建标签页
        self.tabs = QTabWidget()
        
        # OpenAI设置标签页
        self.openai_tab = self.create_openai_tab()
        self.tabs.addTab(self.openai_tab, "OpenAI设置")
        
        # DeepSeek设置标签页
        self.deepseek_tab = self.create_deepseek_tab()
        self.tabs.addTab(self.deepseek_tab, "DeepSeek设置")
        
        # Ollama设置标签页
        self.ollama_tab = self.create_ollama_tab()
        self.tabs.addTab(self.ollama_tab, "Ollama设置")
        
        # 自定义API设置标签页
        self.custom_tab = self.create_custom_api_tab()
        self.tabs.addTab(self.custom_tab, "自定义API")
        
        # 通用设置标签页
        self.general_tab = self.create_general_tab()
        self.tabs.addTab(self.general_tab, "通用设置")
        
        layout.addWidget(self.tabs)
        
        # 添加重置按钮，并使其在左侧
        button_layout = QHBoxLayout()
        self.reset_btn = QPushButton("🔄 重置默认")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch() # 将按钮推到左侧
        layout.addLayout(button_layout)

        # 在主布局底部添加一个弹性空间，将所有内容向上推
        layout.addStretch()
    
    def create_openai_tab(self):
        """创建OpenAI设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # API配置组
        api_group = QGroupBox("API配置")
        api_layout = QFormLayout()
        
        # API密钥（带显示/隐藏按钮）
        key_layout = QHBoxLayout()
        self.openai_api_key = QLineEdit()
        self.openai_api_key.setEchoMode(QLineEdit.Password)
        self.openai_api_key.setPlaceholderText("输入您的OpenAI API密钥")
        key_layout.addWidget(self.openai_api_key)
        
        self.show_key_btn = QPushButton("👁")
        self.show_key_btn.setMaximumWidth(30)
        self.show_key_btn.clicked.connect(self.toggle_api_key_visibility)
        key_layout.addWidget(self.show_key_btn)
        api_layout.addRow("API密钥:", key_layout)
        
        # 自定义API基础URL
        self.openai_base_url = QLineEdit()
        self.openai_base_url.setPlaceholderText("https://api.openai.com/v1 (默认)")
        api_layout.addRow("基础URL:", self.openai_base_url)
        
        # 连接测试
        test_layout = QHBoxLayout()
        self.openai_test_btn = QPushButton("🔍 测试连接")
        self.openai_test_btn.clicked.connect(self.test_openai_connection)
        test_layout.addWidget(self.openai_test_btn)
        
        self.openai_test_progress = QProgressBar()
        self.openai_test_progress.setVisible(False)
        test_layout.addWidget(self.openai_test_progress)
        
        test_layout.addStretch()
        api_layout.addRow("连接测试:", test_layout)
        
        # 测试结果
        self.openai_test_result = QLabel("尚未测试")
        self.openai_test_result.setStyleSheet("color: #6c757d;")
        api_layout.addRow("测试结果:", self.openai_test_result)
        
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)
        
        # 模型配置组
        model_group = QGroupBox("模型配置")
        model_layout = QFormLayout()
        
        # 模型选择（可编辑下拉框）
        model_select_layout = QHBoxLayout()
        self.openai_model = QComboBox()
        self.openai_model.setEditable(True)  # 允许用户输入自定义模型名称
        self.openai_model.setPlaceholderText("请先测试连接以获取可用模型，或手动输入模型名称")
        # 初始为空，通过测试连接获取模型列表
        model_select_layout.addWidget(self.openai_model)
        
        # 刷新模型列表按钮
        self.refresh_models_btn = QPushButton("🔄")
        self.refresh_models_btn.setMaximumWidth(30)
        self.refresh_models_btn.setToolTip("刷新可用模型列表")
        self.refresh_models_btn.clicked.connect(self.refresh_model_list)
        model_select_layout.addWidget(self.refresh_models_btn)
        
        model_layout.addRow("模型名称:", model_select_layout)
        
        # 添加模型说明
        model_info = QLabel("💡 提示：测试连接成功后将自动获取可用模型列表，您也可以手动输入自定义模型名称")
        model_info.setStyleSheet("color: #6c757d; font-size: 12px;")
        model_info.setWordWrap(True)
        model_layout.addRow("", model_info)
        
        # 参数设置
        self.openai_temperature = QDoubleSpinBox()
        self.openai_temperature.setRange(0.0, 2.0)
        self.openai_temperature.setSingleStep(0.1)
        self.openai_temperature.setValue(0.7)
        model_layout.addRow("温度 (Temperature):", self.openai_temperature)
        
        self.openai_max_tokens = QSpinBox()
        self.openai_max_tokens.setRange(1, 8192)
        self.openai_max_tokens.setValue(1000)
        model_layout.addRow("最大令牌数:", self.openai_max_tokens)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # layout.addStretch() # 移除此行以消除空白
        return widget
    
    def create_deepseek_tab(self):
        """创建DeepSeek设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # API配置组
        api_group = QGroupBox("API配置")
        api_layout = QFormLayout()
        
        # API密钥（带显示/隐藏按钮）
        key_layout = QHBoxLayout()
        self.deepseek_api_key = QLineEdit()
        self.deepseek_api_key.setEchoMode(QLineEdit.Password)
        self.deepseek_api_key.setPlaceholderText("输入您的DeepSeek API密钥")
        key_layout.addWidget(self.deepseek_api_key)
        
        self.show_deepseek_key_btn = QPushButton("👁")
        self.show_deepseek_key_btn.setMaximumWidth(30)
        self.show_deepseek_key_btn.clicked.connect(self.toggle_deepseek_key_visibility)
        key_layout.addWidget(self.show_deepseek_key_btn)
        api_layout.addRow("API密钥:", key_layout)
        
        # 自定义API基础URL
        self.deepseek_base_url = QLineEdit()
        self.deepseek_base_url.setPlaceholderText("https://api.deepseek.com/v1 (默认)")
        api_layout.addRow("基础URL:", self.deepseek_base_url)
        
        # 连接测试
        test_layout = QHBoxLayout()
        self.deepseek_test_btn = QPushButton("🔍 测试连接")
        self.deepseek_test_btn.clicked.connect(self.test_deepseek_connection)
        test_layout.addWidget(self.deepseek_test_btn)
        
        self.deepseek_test_progress = QProgressBar()
        self.deepseek_test_progress.setVisible(False)
        test_layout.addWidget(self.deepseek_test_progress)
        
        test_layout.addStretch()
        api_layout.addRow("连接测试:", test_layout)
        
        # 测试结果
        self.deepseek_test_result = QLabel("尚未测试")
        self.deepseek_test_result.setStyleSheet("color: #6c757d;")
        api_layout.addRow("测试结果:", self.deepseek_test_result)
        
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)
        
        # 模型配置组
        model_group = QGroupBox("模型配置")
        model_layout = QFormLayout()
        
        # 模型选择（带刷新按钮）
        refresh_layout = QHBoxLayout()
        self.deepseek_models = QComboBox()
        self.deepseek_models.addItems(["deepseek-chat", "deepseek-coder"])
        self.deepseek_models.setEditable(True)
        refresh_layout.addWidget(self.deepseek_models)
        
        self.refresh_deepseek_models_btn = QPushButton("🔄")
        self.refresh_deepseek_models_btn.setMaximumWidth(30)
        self.refresh_deepseek_models_btn.clicked.connect(self.refresh_deepseek_models)
        refresh_layout.addWidget(self.refresh_deepseek_models_btn)
        model_layout.addRow("选择模型:", refresh_layout)
        
        # 参数设置
        self.deepseek_temperature = QDoubleSpinBox()
        self.deepseek_temperature.setRange(0.0, 2.0)
        self.deepseek_temperature.setSingleStep(0.1)
        self.deepseek_temperature.setValue(0.7)
        model_layout.addRow("温度 (Temperature):", self.deepseek_temperature)
        
        self.deepseek_max_tokens = QSpinBox()
        self.deepseek_max_tokens.setRange(1, 8192)
        self.deepseek_max_tokens.setValue(1000)
        model_layout.addRow("最大令牌数:", self.deepseek_max_tokens)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # layout.addStretch() # 移除此行以消除空白
        return widget
    
    def create_ollama_tab(self):
        """创建Ollama设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 服务器配置组
        server_group = QGroupBox("服务器配置")
        server_layout = QFormLayout()
        
        # 服务器地址
        self.ollama_base_url = QLineEdit()
        self.ollama_base_url.setText("http://localhost:11434")
        self.ollama_base_url.setPlaceholderText("http://localhost:11434")
        server_layout.addRow("服务器地址:", self.ollama_base_url)
        
        # 连接测试
        test_layout = QHBoxLayout()
        self.ollama_test_btn = QPushButton("🔍 测试连接")
        self.ollama_test_btn.clicked.connect(self.test_ollama_connection)
        test_layout.addWidget(self.ollama_test_btn)
        
        self.ollama_test_progress = QProgressBar()
        self.ollama_test_progress.setVisible(False)
        test_layout.addWidget(self.ollama_test_progress)
        
        test_layout.addStretch()
        server_layout.addRow("连接测试:", test_layout)
        
        # 测试结果
        self.ollama_test_result = QLabel("尚未测试")
        self.ollama_test_result.setStyleSheet("color: #6c757d;")
        server_layout.addRow("测试结果:", self.ollama_test_result)
        
        server_group.setLayout(server_layout)
        layout.addWidget(server_group)
        
        # 模型配置组
        model_group = QGroupBox("模型配置")
        model_layout = QFormLayout()
        
        # 模型选择（带刷新按钮）
        refresh_layout = QHBoxLayout()
        self.ollama_models = QComboBox()
        self.ollama_models.addItems(["llama2", "llama2:13b", "codellama", "mistral"])
        self.ollama_models.setEditable(True)
        refresh_layout.addWidget(self.ollama_models)
        
        self.refresh_models_btn = QPushButton("🔄")
        self.refresh_models_btn.setMaximumWidth(30)
        self.refresh_models_btn.clicked.connect(self.refresh_ollama_models)
        refresh_layout.addWidget(self.refresh_models_btn)
        model_layout.addRow("选择模型:", refresh_layout)
        
        # 参数设置
        self.ollama_temperature = QDoubleSpinBox()
        self.ollama_temperature.setRange(0.0, 2.0)
        self.ollama_temperature.setSingleStep(0.1)
        self.ollama_temperature.setValue(0.7)
        model_layout.addRow("温度 (Temperature):", self.ollama_temperature)
        
        self.ollama_num_predict = QSpinBox()
        self.ollama_num_predict.setRange(1, 4096)
        self.ollama_num_predict.setValue(1000)
        model_layout.addRow("预测令牌数:", self.ollama_num_predict)
        
        # 添加超时设置
        self.ollama_timeout = QSpinBox()
        self.ollama_timeout.setRange(30, 600)  # 30秒到10分钟
        self.ollama_timeout.setValue(120)  # 默认2分钟
        self.ollama_timeout.setSuffix(" 秒")
        model_layout.addRow("请求超时:", self.ollama_timeout)
        
        # 添加超时说明
        timeout_info = QLabel("💡 提示：大模型响应可能需要较长时间，建议设置2-5分钟超时")
        timeout_info.setStyleSheet("color: #6c757d; font-size: 12px;")
        timeout_info.setWordWrap(True)
        model_layout.addRow("", timeout_info)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # layout.addStretch() # 移除此行以消除空白
        return widget
    
    def create_custom_api_tab(self):
        """创建自定义API设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # API配置组
        api_group = QGroupBox("自定义API配置")
        api_layout = QFormLayout()
        
        # API名称
        self.custom_api_name = QLineEdit()
        self.custom_api_name.setPlaceholderText("为您的自定义API命名，如：Claude API、本地LLM等")
        api_layout.addRow("API名称:", self.custom_api_name)
        
        # API基础URL
        self.custom_base_url = QLineEdit()
        self.custom_base_url.setPlaceholderText("输入API基础URL，如：https://api.example.com/v1")
        api_layout.addRow("基础URL:", self.custom_base_url)
        
        # API密钥（带显示/隐藏按钮）
        key_layout = QHBoxLayout()
        self.custom_api_key = QLineEdit()
        self.custom_api_key.setEchoMode(QLineEdit.Password)
        self.custom_api_key.setPlaceholderText("输入您的API密钥")
        key_layout.addWidget(self.custom_api_key)
        
        self.show_custom_key_btn = QPushButton("👁")
        self.show_custom_key_btn.setMaximumWidth(30)
        self.show_custom_key_btn.clicked.connect(self.toggle_custom_key_visibility)
        key_layout.addWidget(self.show_custom_key_btn)
        api_layout.addRow("API密钥:", key_layout)
        
        # 提供商类型
        self.custom_provider_type = QComboBox()
        self.custom_provider_type.addItems(["OpenAI兼容", "自定义格式"])
        self.custom_provider_type.setCurrentText("OpenAI兼容")
        api_layout.addRow("API类型:", self.custom_provider_type)
        
        # 连接测试
        test_layout = QHBoxLayout()
        self.custom_test_btn = QPushButton("🔍 测试连接")
        self.custom_test_btn.clicked.connect(self.test_custom_connection)
        test_layout.addWidget(self.custom_test_btn)
        
        self.custom_test_progress = QProgressBar()
        self.custom_test_progress.setVisible(False)
        test_layout.addWidget(self.custom_test_progress)
        
        test_layout.addStretch()
        api_layout.addRow("连接测试:", test_layout)
        
        # 测试结果
        self.custom_test_result = QLabel("尚未测试")
        self.custom_test_result.setStyleSheet("color: #6c757d;")
        api_layout.addRow("测试结果:", self.custom_test_result)
        
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)
        
        # 模型配置组
        model_group = QGroupBox("模型配置")
        model_layout = QFormLayout()
        
        # 模型选择（可编辑下拉框）
        model_select_layout = QHBoxLayout()
        self.custom_model = QComboBox()
        self.custom_model.setEditable(True)  # 允许用户输入自定义模型名称
        self.custom_model.setPlaceholderText("请先测试连接以获取可用模型，或手动输入模型名称")
        model_select_layout.addWidget(self.custom_model)
        
        # 刷新模型列表按钮
        self.refresh_custom_models_btn = QPushButton("🔄")
        self.refresh_custom_models_btn.setMaximumWidth(30)
        self.refresh_custom_models_btn.setToolTip("刷新可用模型列表")
        self.refresh_custom_models_btn.clicked.connect(self.refresh_custom_model_list)
        model_select_layout.addWidget(self.refresh_custom_models_btn)
        
        model_layout.addRow("模型名称:", model_select_layout)
        
        # 添加模型说明
        model_info = QLabel("💡 提示：测试连接成功后将自动获取可用模型列表，您也可以手动输入自定义模型名称")
        model_info.setStyleSheet("color: #6c757d; font-size: 12px;")
        model_info.setWordWrap(True)
        model_layout.addRow("", model_info)
        
        # 参数设置
        self.custom_temperature = QDoubleSpinBox()
        self.custom_temperature.setRange(0.0, 2.0)
        self.custom_temperature.setSingleStep(0.1)
        self.custom_temperature.setValue(0.7)
        model_layout.addRow("温度 (Temperature):", self.custom_temperature)
        
        self.custom_max_tokens = QSpinBox()
        self.custom_max_tokens.setRange(1, 32768)
        self.custom_max_tokens.setValue(1000)
        model_layout.addRow("最大令牌数:", self.custom_max_tokens)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        return widget
    
    def create_general_tab(self):
        """创建通用设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 默认适配器组
        adapter_group = QGroupBox("默认适配器")
        adapter_layout = QFormLayout()
        
        self.default_adapter = QComboBox()
        self.default_adapter.addItems(["模拟适配器", "OpenAI", "DeepSeek", "Ollama", "自定义API"])
        adapter_layout.addRow("默认使用:", self.default_adapter)
        
        adapter_group.setLayout(adapter_layout)
        layout.addWidget(adapter_group)
        
        # 高级设置组
        advanced_group = QGroupBox("高级设置")
        advanced_layout = QFormLayout()
        
        # 请求超时
        self.request_timeout = QSpinBox()
        self.request_timeout.setRange(5, 300)
        self.request_timeout.setValue(60)
        self.request_timeout.setSuffix(" 秒")
        advanced_layout.addRow("请求超时:", self.request_timeout)
        
        # 重试次数
        self.max_retries = QSpinBox()
        self.max_retries.setRange(0, 10)
        self.max_retries.setValue(3)
        advanced_layout.addRow("最大重试次数:", self.max_retries)
        
        # 启用缓存
        self.enable_cache = QCheckBox("启用响应缓存")
        self.enable_cache.setChecked(True)
        advanced_layout.addRow("缓存设置:", self.enable_cache)
        
        # 启用流式响应
        self.enable_streaming = QCheckBox("启用流式响应")
        self.enable_streaming.setChecked(False)
        advanced_layout.addRow("流式处理:", self.enable_streaming)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        # layout.addStretch() # 移除此行以消除空白
        return widget
    
    def refresh_model_list(self):
        """刷新OpenAI模型列表"""
        api_key = self.openai_api_key.text().strip()
        if not api_key:
            QMessageBox.warning(self, "警告", "请先输入API密钥")
            return
        
        # 禁用刷新按钮，显示加载状态
        self.refresh_models_btn.setEnabled(False)
        self.refresh_models_btn.setText("⏳")
        
        base_url = self.openai_base_url.text().strip() or None
        self.model_refresh_thread = OpenAITestThread(api_key, base_url)
        self.model_refresh_thread.test_completed.connect(self.on_model_refresh_completed)
        self.model_refresh_thread.start()
    
    def on_model_refresh_completed(self, success, message, models):
        """模型列表刷新完成回调"""
        self.refresh_models_btn.setEnabled(True)
        self.refresh_models_btn.setText("🔄")
        
        if models:
            # 保存当前选中的模型
            current_model = self.openai_model.currentText()
            
            # 更新模型列表
            self.openai_model.clear()
            self.openai_model.addItems(models)
            
            # 恢复之前选中的模型（如果存在）
            if current_model and current_model in models:
                self.openai_model.setCurrentText(current_model)
            elif models:
                # 如果之前的模型不存在，选择第一个
                self.openai_model.setCurrentText(models[0])
            
            if success:
                QMessageBox.information(self, "成功", f"已获取 {len(models)} 个可用模型")
            else:
                QMessageBox.warning(self, "部分成功", f"{message}\n已更新模型列表")
        else:
            QMessageBox.warning(self, "失败", f"无法获取模型列表: {message}")

    def toggle_api_key_visibility(self):
        """切换API密钥显示/隐藏"""
        if self.openai_api_key.echoMode() == QLineEdit.Password:
            self.openai_api_key.setEchoMode(QLineEdit.Normal)
            self.show_key_btn.setText("🙈")
        else:
            self.openai_api_key.setEchoMode(QLineEdit.Password)
            self.show_key_btn.setText("👁")
    
    def toggle_deepseek_key_visibility(self):
        """切换DeepSeek API密钥显示/隐藏"""
        if self.deepseek_api_key.echoMode() == QLineEdit.Password:
            self.deepseek_api_key.setEchoMode(QLineEdit.Normal)
            self.show_deepseek_key_btn.setText("🙈")
        else:
            self.deepseek_api_key.setEchoMode(QLineEdit.Password)
            self.show_deepseek_key_btn.setText("👁")
    
    def toggle_custom_key_visibility(self):
        """切换自定义API密钥显示/隐藏"""
        if self.custom_api_key.echoMode() == QLineEdit.Password:
            self.custom_api_key.setEchoMode(QLineEdit.Normal)
            self.show_custom_key_btn.setText("🙈")
        else:
            self.custom_api_key.setEchoMode(QLineEdit.Password)
            self.show_custom_key_btn.setText("👁")
    
    def test_openai_connection(self):
        """测试OpenAI连接"""
        api_key = self.openai_api_key.text().strip()
        if not api_key:
            QMessageBox.warning(self, "警告", "请先输入API密钥")
            return
        
        self.openai_test_btn.setEnabled(False)
        self.openai_test_progress.setVisible(True)
        self.openai_test_progress.setRange(0, 0)
        self.openai_test_result.setText("正在测试...")
        self.openai_test_result.setStyleSheet("color: #ffc107;")
        
        base_url = self.openai_base_url.text().strip() or None
        self.openai_test_thread = OpenAITestThread(api_key, base_url)
        self.openai_test_thread.test_completed.connect(self.on_openai_test_completed)
        self.openai_test_thread.start()
    
    def on_openai_test_completed(self, success, message, models):
        """OpenAI测试完成回调"""
        self.openai_test_btn.setEnabled(True)
        self.openai_test_progress.setVisible(False)
        
        if success:
            self.openai_test_result.setText(f"✅ {message}")
            self.openai_test_result.setStyleSheet("color: #28a745;")
            
            # 更新模型列表
            if models:
                current_model = self.openai_model.currentText()
                self.openai_model.clear()
                self.openai_model.addItems(models)
                if current_model in models:
                    self.openai_model.setCurrentText(current_model)
                elif models:
                    self.openai_model.setCurrentText(models[0])
        else:
            self.openai_test_result.setText(f"❌ {message}")
            self.openai_test_result.setStyleSheet("color: #dc3545;")
    
    def refresh_model_list(self):
        """刷新OpenAI模型列表"""
        self.test_openai_connection()
    
    def test_ollama_connection(self):
        """测试Ollama连接"""
        base_url = self.ollama_base_url.text().strip()
        if not base_url:
            base_url = "http://localhost:11434"
        
        self.ollama_test_btn.setEnabled(False)
        self.ollama_test_progress.setVisible(True)
        self.ollama_test_progress.setRange(0, 0)
        self.ollama_test_result.setText("正在测试...")
        self.ollama_test_result.setStyleSheet("color: #ffc107;")
        
        self.ollama_test_thread = OllamaTestThread(base_url)
        self.ollama_test_thread.test_completed.connect(self.on_ollama_test_completed)
        self.ollama_test_thread.start()
    
    def on_ollama_test_completed(self, success, message, models):
        """Ollama测试完成回调"""
        self.ollama_test_btn.setEnabled(True)
        self.ollama_test_progress.setVisible(False)
        
        if success:
            self.ollama_test_result.setText(f"✅ {message}")
            self.ollama_test_result.setStyleSheet("color: #28a745;")
            
            # 更新模型列表
            if models:
                current_model = self.ollama_models.currentText()
                self.ollama_models.clear()
                self.ollama_models.addItems(models)
                if current_model in models:
                    self.ollama_models.setCurrentText(current_model)
        else:
            self.ollama_test_result.setText(f"❌ {message}")
            self.ollama_test_result.setStyleSheet("color: #dc3545;")
    
    def test_deepseek_connection(self):
        """测试DeepSeek连接"""
        api_key = self.deepseek_api_key.text().strip()
        if not api_key:
            QMessageBox.warning(self, "警告", "请先输入API密钥")
            return
        
        self.deepseek_test_btn.setEnabled(False)
        self.deepseek_test_progress.setVisible(True)
        self.deepseek_test_progress.setRange(0, 0)
        self.deepseek_test_result.setText("正在测试...")
        self.deepseek_test_result.setStyleSheet("color: #ffc107;")
        
        base_url = self.deepseek_base_url.text().strip() or None
        self.deepseek_test_thread = DeepSeekTestThread(api_key, base_url)
        self.deepseek_test_thread.test_completed.connect(self.on_deepseek_test_completed)
        self.deepseek_test_thread.start()
    
    def on_deepseek_test_completed(self, success, message, models):
        """DeepSeek测试完成回调"""
        self.deepseek_test_btn.setEnabled(True)
        self.deepseek_test_progress.setVisible(False)
        
        if success:
            self.deepseek_test_result.setText(f"✅ {message}")
            self.deepseek_test_result.setStyleSheet("color: #28a745;")
            
            # 更新模型列表
            if models:
                current_model = self.deepseek_models.currentText()
                self.deepseek_models.clear()
                self.deepseek_models.addItems(models)
                if current_model in models:
                    self.deepseek_models.setCurrentText(current_model)
        else:
            self.deepseek_test_result.setText(f"❌ {message}")
            self.deepseek_test_result.setStyleSheet("color: #dc3545;")
    
    def refresh_ollama_models(self):
        """刷新Ollama模型列表"""
        self.test_ollama_connection()
    
    def refresh_deepseek_models(self):
        """刷新DeepSeek模型列表"""
        self.test_deepseek_connection()
    
    def test_custom_connection(self):
        """测试自定义API连接"""
        api_key = self.custom_api_key.text().strip()
        if not api_key:
            QMessageBox.warning(self, "警告", "请先输入API密钥")
            return
        
        base_url = self.custom_base_url.text().strip()
        if not base_url:
            QMessageBox.warning(self, "警告", "请先输入API基础URL")
            return
        
        self.custom_test_btn.setEnabled(False)
        self.custom_test_progress.setVisible(True)
        self.custom_test_progress.setRange(0, 0)
        self.custom_test_result.setText("正在获取模型列表...")
        self.custom_test_result.setStyleSheet("color: #ffc107;")
        
        provider_type = self.custom_provider_type.currentText()
        self.custom_test_thread = CustomAPITestThread(api_key, base_url, provider_type)
        self.custom_test_thread.test_completed.connect(self.on_custom_test_completed)
        self.custom_test_thread.model_selection_needed.connect(self.on_custom_model_selection_needed)
        self.custom_test_thread.start()
    
    def refresh_custom_model_list(self):
        """刷新自定义API模型列表"""
        api_key = self.custom_api_key.text().strip()
        if not api_key:
            QMessageBox.warning(self, "警告", "请先输入API密钥")
            return
        
        base_url = self.custom_base_url.text().strip()
        if not base_url:
            QMessageBox.warning(self, "警告", "请先输入API基础URL")
            return
        
        self.refresh_custom_models_btn.setEnabled(False)
        self.refresh_custom_models_btn.setText("⏳")
        
        provider_type = self.custom_provider_type.currentText()
        self.custom_refresh_thread = CustomAPITestThread(api_key, base_url, provider_type)
        self.custom_refresh_thread.test_completed.connect(self.on_custom_refresh_completed)
        self.custom_refresh_thread.start()
    
    def on_custom_model_selection_needed(self, models):
        """自定义API需要用户选择模型"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
        
        # 创建模型选择对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("选择测试模型")
        dialog.setModal(True)
        dialog.setFixedSize(400, 150)
        
        layout = QVBoxLayout(dialog)
        
        # 说明文字
        info_label = QLabel(f"已获取到 {len(models)} 个可用模型，请选择一个进行连接测试：")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 模型选择下拉框
        model_combo = QComboBox()
        model_combo.addItems(models)
        layout.addWidget(model_combo)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        # 取消按钮
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        # 测试按钮
        test_btn = QPushButton("开始测试")
        test_btn.setDefault(True)
        button_layout.addWidget(test_btn)
        
        layout.addLayout(button_layout)
        
        # 连接测试按钮信号
        def start_test():
            selected_model = model_combo.currentText()
            dialog.accept()
            # 设置选定的模型并继续测试
            self.custom_test_thread.set_selected_model(selected_model, models)
            self.custom_test_result.setText("正在测试连接...")
            # 更新模型列表
            self.custom_model.clear()
            self.custom_model.addItems(models)
            self.custom_model.setCurrentText(selected_model)
        
        test_btn.clicked.connect(start_test)
        
        # 显示对话框
        if dialog.exec_() == QDialog.Accepted:
            # 对话框被接受，测试已经在start_test中开始
            pass
        else:
            # 用户取消，停止测试
            self.custom_test_btn.setEnabled(True)
            self.custom_test_progress.setVisible(False)
            self.custom_test_result.setText("测试已取消")
            self.custom_test_result.setStyleSheet("color: #6c757d;")
    
    def on_custom_test_completed(self, success, message, models):
        """自定义API测试完成回调"""
        self.custom_test_btn.setEnabled(True)
        self.custom_test_progress.setVisible(False)
        
        if success:
            self.custom_test_result.setText(f"✅ {message}")
            self.custom_test_result.setStyleSheet("color: #28a745;")
            
            # 更新模型列表
            if models:
                current_model = self.custom_model.currentText()
                self.custom_model.clear()
                self.custom_model.addItems(models)
                if current_model in models:
                    self.custom_model.setCurrentText(current_model)
        else:
            self.custom_test_result.setText(f"❌ {message}")
            self.custom_test_result.setStyleSheet("color: #dc3545;")
    
    def on_custom_refresh_completed(self, success, message, models):
        """自定义API模型列表刷新完成回调"""
        self.refresh_custom_models_btn.setEnabled(True)
        self.refresh_custom_models_btn.setText("🔄")
        
        if models:
            # 保存当前选中的模型
            current_model = self.custom_model.currentText()
            
            # 更新模型列表
            self.custom_model.clear()
            self.custom_model.addItems(models)
            
            # 恢复之前选中的模型（如果存在）
            if current_model and current_model in models:
                self.custom_model.setCurrentText(current_model)
            elif models:
                # 如果之前的模型不存在，选择第一个
                self.custom_model.setCurrentText(models[0])
            
            if success:
                QMessageBox.information(self, "成功", f"已获取 {len(models)} 个可用模型")
            else:
                QMessageBox.warning(self, "部分成功", f"{message}\n已更新模型列表")
        else:
            QMessageBox.warning(self, "失败", f"无法获取模型列表: {message}")
    
    def load_config(self):
        """加载配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.current_config = json.load(f)
                    self.apply_config_to_ui()
            else:
                self.reset_to_defaults()
        except Exception as e:
            print(f"加载AI配置失败: {str(e)}")
            self.reset_to_defaults()
    
    def apply_config_to_ui(self):
        """将配置应用到UI"""
        config = self.current_config
        
        # OpenAI设置
        openai_config = config.get('openai', {})
        self.openai_api_key.setText(openai_config.get('api_key', ''))
        self.openai_base_url.setText(openai_config.get('base_url', ''))
        
        # 处理模型配置 - 如果有配置的模型，设置到下拉框中
        configured_model = openai_config.get('model', '')
        if configured_model:
            # 如果下拉框中没有这个模型，先添加它
            if self.openai_model.findText(configured_model) == -1:
                self.openai_model.addItem(configured_model)
            self.openai_model.setCurrentText(configured_model)
        
        self.openai_temperature.setValue(openai_config.get('temperature', 0.7))
        self.openai_max_tokens.setValue(openai_config.get('max_tokens', 1000))
        
        # DeepSeek设置
        deepseek_config = config.get('deepseek', {})
        self.deepseek_api_key.setText(deepseek_config.get('api_key', ''))
        self.deepseek_base_url.setText(deepseek_config.get('base_url', 'https://api.deepseek.com/v1'))
        self.deepseek_models.setCurrentText(deepseek_config.get('model', 'deepseek-chat'))
        self.deepseek_temperature.setValue(deepseek_config.get('temperature', 0.7))
        self.deepseek_max_tokens.setValue(deepseek_config.get('max_tokens', 1000))
        
        # Ollama设置
        ollama_config = config.get('ollama', {})
        self.ollama_base_url.setText(ollama_config.get('base_url', 'http://localhost:11434'))
        self.ollama_models.setCurrentText(ollama_config.get('model', 'llama2'))
        self.ollama_temperature.setValue(ollama_config.get('temperature', 0.7))
        self.ollama_num_predict.setValue(ollama_config.get('num_predict', 1000))
        self.ollama_timeout.setValue(ollama_config.get('timeout', 120))
        
        # 自定义API设置
        custom_config = config.get('custom_api', {})
        self.custom_api_name.setText(custom_config.get('name', ''))
        self.custom_base_url.setText(custom_config.get('base_url', ''))
        self.custom_api_key.setText(custom_config.get('api_key', ''))
        self.custom_provider_type.setCurrentText(custom_config.get('provider_type', 'OpenAI兼容'))
        
        # 处理模型配置
        configured_model = custom_config.get('model', '')
        self.custom_model.clear()
        if configured_model:
            self.custom_model.addItem(configured_model)
            self.custom_model.setCurrentText(configured_model)
        
        self.custom_temperature.setValue(custom_config.get('temperature', 0.7))
        self.custom_max_tokens.setValue(custom_config.get('max_tokens', 1000))
        
        # 通用设置
        general_config = config.get('general', {})
        default_adapter = general_config.get('default_adapter', '模拟适配器')
        if default_adapter == 'openai':
            default_adapter = 'OpenAI'
        elif default_adapter == 'local':
            default_adapter = 'Ollama'
        elif default_adapter == 'deepseek':
            default_adapter = 'DeepSeek'
        elif default_adapter == 'custom':
            default_adapter = '自定义API'
        self.default_adapter.setCurrentText(default_adapter)
        self.request_timeout.setValue(general_config.get('request_timeout', 60))
        self.max_retries.setValue(general_config.get('max_retries', 3))
        self.enable_cache.setChecked(general_config.get('enable_cache', True))
        self.enable_streaming.setChecked(general_config.get('enable_streaming', False))
    
    def _save_config_to_file(self):
        """保存配置到文件（内部方法，由设置Tab调用）"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            # 使用当前配置（已通过update_settings_preview更新）
            config = self.current_config
            
            # 保存到文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"保存AI配置文件失败: {str(e)}")
            return False
    
    def reset_to_defaults(self):
        """重置为默认设置"""
        # 重置UI到默认值
        self.openai_api_key.clear()
        self.openai_base_url.clear()
        self.openai_model.clear()  # 清空模型选择，不设置默认值
        self.openai_temperature.setValue(0.7)
        self.openai_max_tokens.setValue(1000)
        
        self.deepseek_api_key.clear()
        self.deepseek_base_url.clear()
        self.deepseek_models.setCurrentText('deepseek-chat')
        self.deepseek_temperature.setValue(0.7)
        self.deepseek_max_tokens.setValue(1000)
        
        self.ollama_base_url.setText('http://localhost:11434')
        self.ollama_models.setCurrentText('llama2')
        self.ollama_temperature.setValue(0.7)
        self.ollama_num_predict.setValue(1000)
        self.ollama_timeout.setValue(120)
        
        self.custom_api_name.clear()
        self.custom_base_url.clear()
        self.custom_api_key.clear()
        self.custom_provider_type.setCurrentText("OpenAI兼容")
        self.custom_model.clear()
        self.custom_temperature.setValue(0.7)
        self.custom_max_tokens.setValue(1000)
        
        self.default_adapter.setCurrentText('模拟适配器')
        self.request_timeout.setValue(60)
        self.max_retries.setValue(3)
        self.enable_cache.setChecked(True)
        self.enable_streaming.setChecked(False)
        
        # 重置测试结果
        self.openai_test_result.setText("尚未测试")
        self.openai_test_result.setStyleSheet("color: #6c757d;")
        self.deepseek_test_result.setText("尚未测试")
        self.deepseek_test_result.setStyleSheet("color: #6c757d;")
        self.ollama_test_result.setText("尚未测试")
        self.ollama_test_result.setStyleSheet("color: #6c757d;")
        self.custom_test_result.setText("尚未测试")
        self.custom_test_result.setStyleSheet("color: #6c757d;")
    
    def get_config(self):
        """获取当前配置"""
        return self.current_config.copy()
    
    def _connect_signals(self):
        """连接所有控件的信号"""
        # OpenAI设置信号
        self.openai_api_key.textChanged.connect(self.update_settings_preview)
        self.openai_base_url.textChanged.connect(self.update_settings_preview)
        self.openai_model.currentTextChanged.connect(self.update_settings_preview)
        self.openai_temperature.valueChanged.connect(self.update_settings_preview)
        self.openai_max_tokens.valueChanged.connect(self.update_settings_preview)
        
        # DeepSeek设置信号
        self.deepseek_api_key.textChanged.connect(self.update_settings_preview)
        self.deepseek_base_url.textChanged.connect(self.update_settings_preview)
        self.deepseek_models.currentTextChanged.connect(self.update_settings_preview)
        self.deepseek_temperature.valueChanged.connect(self.update_settings_preview)
        self.deepseek_max_tokens.valueChanged.connect(self.update_settings_preview)
        
        # Ollama设置信号
        self.ollama_base_url.textChanged.connect(self.update_settings_preview)
        self.ollama_models.currentTextChanged.connect(self.update_settings_preview)
        self.ollama_temperature.valueChanged.connect(self.update_settings_preview)
        self.ollama_num_predict.valueChanged.connect(self.update_settings_preview)
        self.ollama_timeout.valueChanged.connect(self.update_settings_preview)
        
        # 自定义API设置信号
        self.custom_api_name.textChanged.connect(self.update_settings_preview)
        self.custom_base_url.textChanged.connect(self.update_settings_preview)
        self.custom_api_key.textChanged.connect(self.update_settings_preview)
        self.custom_provider_type.currentTextChanged.connect(self.update_settings_preview)
        self.custom_model.currentTextChanged.connect(self.update_settings_preview)
        self.custom_temperature.valueChanged.connect(self.update_settings_preview)
        self.custom_max_tokens.valueChanged.connect(self.update_settings_preview)
        
        # 通用设置信号
        self.default_adapter.currentTextChanged.connect(self.update_settings_preview)
        self.request_timeout.valueChanged.connect(self.update_settings_preview)
        self.max_retries.valueChanged.connect(self.update_settings_preview)
        self.enable_cache.toggled.connect(self.update_settings_preview)
        self.enable_streaming.toggled.connect(self.update_settings_preview)
    
    def update_settings_preview(self):
        """更新设置预览（当任何设置改变时调用）"""
        # 构建当前配置
        default_adapter_text = self.default_adapter.currentText()
        if default_adapter_text == 'OpenAI':
            default_adapter = 'openai'
        elif default_adapter_text == 'DeepSeek':
            default_adapter = 'deepseek'
        elif default_adapter_text == 'Ollama':
            default_adapter = 'local'
        elif default_adapter_text == '自定义API':
            default_adapter = 'custom'
        else:
            default_adapter = 'mock'
        
        config = {
            'openai': {
                'api_key': self.openai_api_key.text().strip(),
                'base_url': self.openai_base_url.text().strip(),
                'model': self.openai_model.currentText(),
                'temperature': self.openai_temperature.value(),
                'max_tokens': self.openai_max_tokens.value()
            },
            'deepseek': {
                'api_key': self.deepseek_api_key.text().strip(),
                'base_url': self.deepseek_base_url.text().strip(),
                'model': self.deepseek_models.currentText(),
                'temperature': self.deepseek_temperature.value(),
                'max_tokens': self.deepseek_max_tokens.value()
            },
            'ollama': {
                'base_url': self.ollama_base_url.text().strip() or 'http://localhost:11434',
                'model': self.ollama_models.currentText(),
                'temperature': self.ollama_temperature.value(),
                'num_predict': self.ollama_num_predict.value(),
                'timeout': self.ollama_timeout.value()
            },
            'custom_api': {
                'name': self.custom_api_name.text().strip(),
                'base_url': self.custom_base_url.text().strip(),
                'api_key': self.custom_api_key.text().strip(),
                'provider_type': self.custom_provider_type.currentText(),
                'model': self.custom_model.currentText(),
                'temperature': self.custom_temperature.value(),
                'max_tokens': self.custom_max_tokens.value()
            },
            'general': {
                'default_adapter': default_adapter,
                'request_timeout': self.request_timeout.value(),
                'max_retries': self.max_retries.value(),
                'enable_cache': self.enable_cache.isChecked(),
                'enable_streaming': self.enable_streaming.isChecked()
            }
        }
        
        # 更新当前配置
        self.current_config = config
        
        # 发出设置变更信号
        self.settings_changed.emit(config) 