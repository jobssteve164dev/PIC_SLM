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
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"LLM模块导入失败: {e}")
    LLM_AVAILABLE = False


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
        self.adapter_combo.addItems(["模拟适配器", "OpenAI GPT-4", "本地Ollama"])
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
            # 创建默认的模拟适配器
            adapter = create_llm_adapter('mock')
            self.llm_framework = LLMFramework(adapter)
            self.add_system_message("✅ AI助手已启动，使用模拟适配器")
            self.status_label.setText("AI助手已就绪")
            self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        except Exception as e:
            self.add_system_message(f"❌ LLM框架初始化失败: {str(e)}")
            self.set_ui_enabled(False)
    
    def switch_adapter(self, adapter_name):
        """切换LLM适配器"""
        if not self.llm_framework:
            return
            
        try:
            self.status_label.setText("正在切换AI模型...")
            self.status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
            
            if adapter_name == "模拟适配器":
                adapter = create_llm_adapter('mock')
            elif adapter_name == "OpenAI GPT-4":
                # 这里需要配置API密钥
                adapter = create_llm_adapter('openai', api_key='your-api-key')
            elif adapter_name == "本地Ollama":
                adapter = create_llm_adapter('local', model_name='llama2')
            else:
                return
                
            success = self.llm_framework.switch_adapter(adapter)
            if success:
                self.add_system_message(f"✅ 已切换到: {adapter_name}")
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
        <div style='margin: 10px 0; padding: 8px; background-color: #007bff; color: white; border-radius: 10px; text-align: right;'>
            <strong>您 [{timestamp}]:</strong><br>{message}
        </div>
        """
        self.chat_display.append(formatted_message)
    
    def add_ai_message(self, message):
        """添加AI响应消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"""
        <div style='margin: 10px 0; padding: 8px; background-color: #28a745; color: white; border-radius: 10px;'>
            <strong>AI助手 [{timestamp}]:</strong><br>{message}
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
            
        self.message_input.clear()
        self.add_user_message(message)
        
        # 显示进度
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.status_label.setText("AI正在思考...")
        self.status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        
        try:
            # 调用LLM框架进行对话
            response = self.llm_framework.chat_with_training_context(
                message, self.training_context
            )
            self.add_ai_message(response)
            
            # 更新聊天历史
            self.chat_history.append({
                'user': message,
                'ai': response,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            self.add_ai_message(f"抱歉，处理您的问题时出现错误: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.status_label.setText("AI助手已就绪")
            self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
    
    def analyze_training(self):
        """分析当前训练状态"""
        if not self.llm_framework:
            return
            
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
        
        self.add_user_message("请分析当前训练状态")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        try:
            analysis = self.llm_framework.analyze_training_metrics(current_metrics)
            self.add_ai_message(analysis['combined_insights'])
            
            # 如果有建议，也显示出来
            if analysis.get('recommendations'):
                self.add_ai_message(f"建议: {analysis['recommendations']}")
                
        except Exception as e:
            self.add_ai_message(f"分析训练状态时出错: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
    
    def get_suggestions(self):
        """获取优化建议"""
        if not self.llm_framework:
            return
            
        # 模拟训练历史
        current_metrics = {'train_loss': 0.234, 'val_loss': 0.287, 'accuracy': 0.856}
        history = [
            {'epoch': i, 'train_loss': 0.5 - i*0.02, 'val_loss': 0.52 - i*0.018}
            for i in range(10)
        ]
        
        self.add_user_message("请给出超参数优化建议")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        try:
            suggestions = self.llm_framework.get_hyperparameter_suggestions(
                current_metrics, history
            )
            self.add_ai_message(suggestions)
        except Exception as e:
            self.add_ai_message(f"获取建议时出错: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
    
    def diagnose_issues(self):
        """诊断训练问题"""
        if not self.llm_framework:
            return
            
        # 模拟问题指标
        problem_metrics = {
            'train_loss': 0.1,
            'val_loss': 0.8,  # 明显的过拟合
            'gradient_norm': 1e-8,  # 梯度消失
            'epoch': 20
        }
        
        self.add_user_message("请诊断训练中的问题")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        try:
            diagnosis = self.llm_framework.diagnose_training_problems(problem_metrics)
            self.add_ai_message(diagnosis)
        except Exception as e:
            self.add_ai_message(f"诊断问题时出错: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
    
    def compare_models(self):
        """模型对比分析"""
        if not self.llm_framework:
            return
            
        # 模拟多个模型结果
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
        
        self.add_user_message("请对比分析这些模型")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        try:
            comparison = self.llm_framework.compare_model_results(model_results)
            self.add_ai_message(comparison['analysis'])
            if comparison.get('recommendation'):
                self.add_ai_message(f"推荐: {comparison['recommendation']}")
        except Exception as e:
            self.add_ai_message(f"模型对比时出错: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
    
    def update_training_context(self, context):
        """更新训练上下文"""
        self.training_context.update(context)


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
        
        # 添加标题
        title_label = QLabel("🏭 AI模型工厂")
        title_label.setFont(QFont('微软雅黑', 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                padding: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border-radius: 8px;
                margin-bottom: 10px;
            }
        """)
        main_layout.addWidget(title_label)
        
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