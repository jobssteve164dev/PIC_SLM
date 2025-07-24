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


class AISettingsWidget(QWidget):
    """AI设置主组件"""
    
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config_file = "setting/ai_config.json"
        self.current_config = {}
        self.ollama_test_thread = None
        self.openai_test_thread = None
        
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
        
        # Ollama设置标签页
        self.ollama_tab = self.create_ollama_tab()
        self.tabs.addTab(self.ollama_tab, "Ollama设置")
        
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
    
    def create_general_tab(self):
        """创建通用设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 默认适配器组
        adapter_group = QGroupBox("默认适配器")
        adapter_layout = QFormLayout()
        
        self.default_adapter = QComboBox()
        self.default_adapter.addItems(["模拟适配器", "OpenAI", "Ollama"])
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
        else:
            self.openai_test_result.setText(f"❌ {message}")
            self.openai_test_result.setStyleSheet("color: #dc3545;")
        
        # 更新模型列表（无论测试成功与否，只要有模型列表就更新）
        if models:
            current_model = self.openai_model.currentText()
            self.openai_model.clear()
            self.openai_model.addItems(models)
            
            # 恢复之前选中的模型（如果存在）
            if current_model and current_model in models:
                self.openai_model.setCurrentText(current_model)
            elif models:
                # 如果之前的模型不存在，选择第一个
                self.openai_model.setCurrentText(models[0])
    
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
    
    def refresh_ollama_models(self):
        """刷新Ollama模型列表"""
        self.test_ollama_connection()
    
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
        
        # Ollama设置
        ollama_config = config.get('ollama', {})
        self.ollama_base_url.setText(ollama_config.get('base_url', 'http://localhost:11434'))
        self.ollama_models.setCurrentText(ollama_config.get('model', 'llama2'))
        self.ollama_temperature.setValue(ollama_config.get('temperature', 0.7))
        self.ollama_num_predict.setValue(ollama_config.get('num_predict', 1000))
        self.ollama_timeout.setValue(ollama_config.get('timeout', 120))
        
        # 通用设置
        general_config = config.get('general', {})
        default_adapter = general_config.get('default_adapter', '模拟适配器')
        if default_adapter == 'openai':
            default_adapter = 'OpenAI'
        elif default_adapter == 'local':
            default_adapter = 'Ollama'
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
        
        self.ollama_base_url.setText('http://localhost:11434')
        self.ollama_models.setCurrentText('llama2')
        self.ollama_temperature.setValue(0.7)
        self.ollama_num_predict.setValue(1000)
        self.ollama_timeout.setValue(120)
        
        self.default_adapter.setCurrentText('模拟适配器')
        self.request_timeout.setValue(60)
        self.max_retries.setValue(3)
        self.enable_cache.setChecked(True)
        self.enable_streaming.setChecked(False)
        
        # 重置测试结果
        self.openai_test_result.setText("尚未测试")
        self.openai_test_result.setStyleSheet("color: #6c757d;")
        self.ollama_test_result.setText("尚未测试")
        self.ollama_test_result.setStyleSheet("color: #6c757d;")
    
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
        
        # Ollama设置信号
        self.ollama_base_url.textChanged.connect(self.update_settings_preview)
        self.ollama_models.currentTextChanged.connect(self.update_settings_preview)
        self.ollama_temperature.valueChanged.connect(self.update_settings_preview)
        self.ollama_num_predict.valueChanged.connect(self.update_settings_preview)
        self.ollama_timeout.valueChanged.connect(self.update_settings_preview)
        
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
        elif default_adapter_text == 'Ollama':
            default_adapter = 'local'
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
            'ollama': {
                'base_url': self.ollama_base_url.text().strip() or 'http://localhost:11434',
                'model': self.ollama_models.currentText(),
                'temperature': self.ollama_temperature.value(),
                'num_predict': self.ollama_num_predict.value(),
                'timeout': self.ollama_timeout.value()
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