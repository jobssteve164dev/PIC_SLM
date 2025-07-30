"""
实时数据流监控控件

用于在AI模型工厂中展示训练时的实时数据流变化，包括SSE、WebSocket和REST API的数据流状态。
注意：此组件只作为客户端连接到已存在的数据流服务器，不会启动新的服务器实例。
"""

import json
import time
import threading
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QLabel, QTextEdit, QPushButton, QTabWidget,
                             QProgressBar, QTableWidget, QTableWidgetItem,
                             QHeaderView, QSplitter, QFrame)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt, QUrl
from PyQt5.QtGui import QFont, QColor, QPalette
import requests
import websockets
import asyncio
import logging


class StreamDataCollector(QThread):
    """数据流收集线程 - 仅作为客户端连接"""
    
    # 信号定义
    sse_data_received = pyqtSignal(dict)
    websocket_data_received = pyqtSignal(dict)
    rest_api_data_received = pyqtSignal(dict)
    connection_status_changed = pyqtSignal(str, str, bool)  # stream_type, status, connected
    error_occurred = pyqtSignal(str, str)  # stream_type, error_message
    
    def __init__(self):
        super().__init__()
        self.is_running = False
        self.sse_connected = False
        self.websocket_connected = False
        self.rest_api_connected = False
        
        # 服务器配置 - 连接到已存在的服务器
        self.sse_url = "http://127.0.0.1:8888/api/stream/metrics"
        self.websocket_url = "ws://127.0.0.1:8889"
        self.rest_api_url = "http://127.0.0.1:8890/api/metrics/current"
        
        # 数据缓存
        self.sse_data = []
        self.websocket_data = []
        self.rest_api_data = []
        
        # 连接状态
        self.connection_status = {
            'sse': False,
            'websocket': False,
            'rest_api': False
        }
        
        # 连接重试配置
        self.max_retries = 3
        self.retry_delay = 5  # 秒
        self.connection_timeout = 10  # 秒
    
    def run(self):
        """运行数据收集线程"""
        self.is_running = True
        
        # 创建子线程来处理各个数据流收集
        sse_thread = threading.Thread(target=self._start_sse_collection, daemon=True)
        websocket_thread = threading.Thread(target=self._start_websocket_collection, daemon=True)
        rest_api_thread = threading.Thread(target=self._start_rest_api_polling, daemon=True)
        
        # 启动子线程
        sse_thread.start()
        websocket_thread.start()
        rest_api_thread.start()
        
        # 保持主线程运行，监控子线程状态
        while self.is_running:
            time.sleep(1)
            
            # 检查子线程是否还在运行
            if not sse_thread.is_alive() and not websocket_thread.is_alive() and not rest_api_thread.is_alive():
                # 所有子线程都结束了，退出主线程
                break
    
    def stop(self):
        """停止数据收集"""
        self.is_running = False
        self.wait()
    
    def _start_sse_collection(self):
        """启动SSE数据收集 - 在主线程中运行"""
        retry_count = 0
        while self.is_running and retry_count < self.max_retries:
            try:
                # 使用requests的stream=True来正确处理SSE
                response = requests.get(self.sse_url, timeout=self.connection_timeout, stream=True)
                if response.status_code == 200:
                    self.connection_status['sse'] = True
                    self.connection_status_changed.emit('sse', 'connected', True)
                    retry_count = 0  # 重置重试计数
                    
                    # 开始SSE流处理
                    for line in response.iter_lines(decode_unicode=True):
                        if not self.is_running:
                            break
                            
                        if line and line.startswith('data: '):
                            try:
                                data_str = line[6:]
                                data = json.loads(data_str)
                                self.sse_data.append(data)
                                self.sse_data_received.emit(data)
                            except json.JSONDecodeError:
                                continue
                else:
                    self.connection_status['sse'] = False
                    self.connection_status_changed.emit('sse', f'HTTP {response.status_code}', False)
                    retry_count += 1
                    
            except requests.exceptions.ConnectionError as e:
                self.connection_status['sse'] = False
                self.connection_status_changed.emit('sse', 'connection_error', False)
                self.error_occurred.emit('sse', f'连接错误: {str(e)}')
                retry_count += 1
                
            except requests.exceptions.Timeout as e:
                self.connection_status['sse'] = False
                self.connection_status_changed.emit('sse', 'timeout', False)
                self.error_occurred.emit('sse', f'连接超时: {str(e)}')
                retry_count += 1
                
            except requests.exceptions.RequestException as e:
                self.connection_status['sse'] = False
                self.connection_status_changed.emit('sse', 'request_error', False)
                self.error_occurred.emit('sse', f'请求错误: {str(e)}')
                retry_count += 1
                
            except Exception as e:
                self.connection_status['sse'] = False
                self.connection_status_changed.emit('sse', 'unknown_error', False)
                self.error_occurred.emit('sse', f'未知错误: {str(e)}')
                retry_count += 1
                
            if retry_count < self.max_retries:
                time.sleep(self.retry_delay)
        
        # 如果达到最大重试次数，标记为永久失败
        if retry_count >= self.max_retries:
            self.connection_status['sse'] = False
            self.connection_status_changed.emit('sse', 'max_retries_exceeded', False)
    
    def _start_websocket_collection(self):
        """启动WebSocket数据收集 - 在主线程中运行"""
        retry_count = 0
        while self.is_running and retry_count < self.max_retries:
            try:
                # 创建新的事件循环
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def websocket_client():
                    try:
                        # 修复：websockets.connect()不支持timeout参数，使用asyncio.wait_for来控制超时
                        websocket = await asyncio.wait_for(
                            websockets.connect(self.websocket_url), 
                            timeout=self.connection_timeout
                        )
                        
                        async with websocket:
                            self.connection_status['websocket'] = True
                            self.connection_status_changed.emit('websocket', 'connected', True)
                            nonlocal retry_count
                            retry_count = 0  # 重置重试计数
                            
                            # 发送订阅消息
                            await websocket.send(json.dumps({
                                'type': 'subscribe',
                                'channel': 'metrics'
                            }))
                            
                            while self.is_running:
                                try:
                                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                                    data = json.loads(message)
                                    self.websocket_data.append(data)
                                    self.websocket_data_received.emit(data)
                                except asyncio.TimeoutError:
                                    continue
                                except Exception as e:
                                    break
                                    
                    except websockets.exceptions.ConnectionClosed:
                        self.connection_status['websocket'] = False
                        self.connection_status_changed.emit('websocket', 'connection_closed', False)
                        self.error_occurred.emit('websocket', 'WebSocket连接已关闭')
                        raise
                        
                    except websockets.exceptions.InvalidURI:
                        self.connection_status['websocket'] = False
                        self.connection_status_changed.emit('websocket', 'invalid_uri', False)
                        self.error_occurred.emit('websocket', '无效的WebSocket URI')
                        raise
                        
                    except Exception as e:
                        self.connection_status['websocket'] = False
                        self.connection_status_changed.emit('websocket', 'websocket_error', False)
                        self.error_occurred.emit('websocket', f'WebSocket错误: {str(e)}')
                        raise e
                
                # 运行WebSocket客户端
                loop.run_until_complete(websocket_client())
                loop.close()
                
            except websockets.exceptions.ConnectionClosed:
                retry_count += 1
                if retry_count < self.max_retries:
                    time.sleep(self.retry_delay)
                    
            except websockets.exceptions.InvalidURI:
                retry_count += 1
                if retry_count < self.max_retries:
                    time.sleep(self.retry_delay)
                    
            except Exception as e:
                self.connection_status['websocket'] = False
                self.connection_status_changed.emit('websocket', 'connection_error', False)
                self.error_occurred.emit('websocket', f'连接错误: {str(e)}')
                retry_count += 1
                
                if retry_count < self.max_retries:
                    time.sleep(self.retry_delay)
        
        # 如果达到最大重试次数，标记为永久失败
        if retry_count >= self.max_retries:
            self.connection_status['websocket'] = False
            self.connection_status_changed.emit('websocket', 'max_retries_exceeded', False)
    
    def _start_rest_api_polling(self):
        """启动REST API轮询 - 在主线程中运行"""
        retry_count = 0
        while self.is_running and retry_count < self.max_retries:
            try:
                response = requests.get(self.rest_api_url, timeout=self.connection_timeout)
                if response.status_code == 200:
                    data = response.json()
                    self.connection_status['rest_api'] = True
                    self.connection_status_changed.emit('rest_api', 'connected', True)
                    retry_count = 0  # 重置重试计数
                    self.rest_api_data.append(data)
                    self.rest_api_data_received.emit(data)
                else:
                    self.connection_status['rest_api'] = False
                    self.connection_status_changed.emit('rest_api', 'http_error', False)
                    retry_count += 1
                    
            except requests.exceptions.ConnectionError as e:
                self.connection_status['rest_api'] = False
                self.connection_status_changed.emit('rest_api', 'connection_error', False)
                self.error_occurred.emit('rest_api', f'连接错误: {str(e)}')
                retry_count += 1
                
            except requests.exceptions.Timeout as e:
                self.connection_status['rest_api'] = False
                self.connection_status_changed.emit('rest_api', 'timeout', False)
                self.error_occurred.emit('rest_api', f'连接超时: {str(e)}')
                retry_count += 1
                
            except requests.exceptions.RequestException as e:
                self.connection_status['rest_api'] = False
                self.connection_status_changed.emit('rest_api', 'request_error', False)
                self.error_occurred.emit('rest_api', f'请求错误: {str(e)}')
                retry_count += 1
                
            except Exception as e:
                self.connection_status['rest_api'] = False
                self.connection_status_changed.emit('rest_api', 'unknown_error', False)
                self.error_occurred.emit('rest_api', f'未知错误: {str(e)}')
                retry_count += 1
            
            # 如果达到最大重试次数，标记为永久失败
            if retry_count >= self.max_retries:
                self.connection_status['rest_api'] = False
                self.connection_status_changed.emit('rest_api', 'max_retries_exceeded', False)
                break
            
            time.sleep(2)  # 每2秒轮询一次


class RealTimeStreamMonitor(QWidget):
    """实时数据流监控控件 - 仅作为客户端"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_collector = None
        self.init_ui()
        self.init_data_collector()
        
        # 添加状态检查定时器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_server_status)
        self.status_timer.start(5000)  # 每5秒检查一次服务器状态
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 标题
        title_label = QLabel("📡 实时数据流监控 (客户端模式)")
        title_label.setFont(QFont('微软雅黑', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 说明文字
        info_label = QLabel("⚠️ 此组件仅作为客户端连接到已存在的数据流服务器，不会启动新的服务器实例")
        info_label.setStyleSheet("color: #856404; background-color: #fff3cd; padding: 8px; border: 1px solid #ffeaa7; border-radius: 4px;")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        # 连接状态面板
        self.create_connection_status_panel(layout)
        
        # 数据流标签页
        self.create_stream_tabs(layout)
        
        # 控制按钮
        self.create_control_buttons(layout)
    
    def create_connection_status_panel(self, parent_layout):
        """创建连接状态面板"""
        status_group = QGroupBox("🔗 数据流连接状态")
        status_layout = QHBoxLayout()
        
        # SSE状态
        self.sse_status_label = QLabel("SSE: 🔴 未连接")
        self.sse_status_label.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 3px;")
        status_layout.addWidget(self.sse_status_label)
        
        # WebSocket状态
        self.websocket_status_label = QLabel("WebSocket: 🔴 未连接")
        self.websocket_status_label.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 3px;")
        status_layout.addWidget(self.websocket_status_label)
        
        # REST API状态
        self.rest_api_status_label = QLabel("REST API: 🔴 未连接")
        self.rest_api_status_label.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 3px;")
        status_layout.addWidget(self.rest_api_status_label)
        
        status_layout.addStretch()
        status_group.setLayout(status_layout)
        parent_layout.addWidget(status_group)
    
    def create_stream_tabs(self, parent_layout):
        """创建数据流标签页"""
        self.tab_widget = QTabWidget()
        
        # SSE数据流标签页
        self.sse_tab = self.create_sse_tab()
        self.tab_widget.addTab(self.sse_tab, "SSE 数据流")
        
        # WebSocket数据流标签页
        self.websocket_tab = self.create_websocket_tab()
        self.tab_widget.addTab(self.websocket_tab, "WebSocket 数据流")
        
        # REST API数据流标签页
        self.rest_api_tab = self.create_rest_api_tab()
        self.tab_widget.addTab(self.rest_api_tab, "REST API 数据流")
        
        # 统计信息标签页
        self.stats_tab = self.create_stats_tab()
        self.tab_widget.addTab(self.stats_tab, "📊 统计信息")
        
        parent_layout.addWidget(self.tab_widget)
    
    def create_sse_tab(self):
        """创建SSE标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 数据展示区域
        self.sse_data_display = QTextEdit()
        self.sse_data_display.setReadOnly(True)
        self.sse_data_display.setFont(QFont('Consolas', 9))
        self.sse_data_display.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6;")
        layout.addWidget(self.sse_data_display)
        
        return tab
    
    def create_websocket_tab(self):
        """创建WebSocket标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 数据展示区域
        self.websocket_data_display = QTextEdit()
        self.websocket_data_display.setReadOnly(True)
        self.websocket_data_display.setFont(QFont('Consolas', 9))
        self.websocket_data_display.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6;")
        layout.addWidget(self.websocket_data_display)
        
        return tab
    
    def create_rest_api_tab(self):
        """创建REST API标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 数据展示区域
        self.rest_api_data_display = QTextEdit()
        self.rest_api_data_display.setReadOnly(True)
        self.rest_api_data_display.setFont(QFont('Consolas', 9))
        self.rest_api_data_display.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6;")
        layout.addWidget(self.rest_api_data_display)
        
        return tab
    
    def create_stats_tab(self):
        """创建统计信息标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 统计表格
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(4)
        self.stats_table.setHorizontalHeaderLabels(['数据流类型', '连接状态', '数据包数量', '最后更新时间'])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.stats_table)
        
        return tab
    
    def create_control_buttons(self, parent_layout):
        """创建控制按钮"""
        button_layout = QHBoxLayout()
        
        # 开始监控按钮
        self.start_btn = QPushButton("▶️ 开始监控")
        self.start_btn.clicked.connect(self.start_monitoring)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        button_layout.addWidget(self.start_btn)
        
        # 停止监控按钮
        self.stop_btn = QPushButton("⏹️ 停止监控")
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        button_layout.addWidget(self.stop_btn)
        
        # 刷新状态按钮
        self.refresh_btn = QPushButton("🔄 刷新状态")
        self.refresh_btn.clicked.connect(self.refresh_status)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        button_layout.addWidget(self.refresh_btn)
        
        # 测试连接按钮
        self.test_btn = QPushButton("🔍 测试连接")
        self.test_btn.clicked.connect(self.test_connections)
        self.test_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffc107;
                color: #212529;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0a800;
            }
        """)
        button_layout.addWidget(self.test_btn)
        
        # 清空数据按钮
        self.clear_btn = QPushButton("🗑️ 清空数据")
        self.clear_btn.clicked.connect(self.clear_all_data)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        button_layout.addWidget(self.clear_btn)
        
        button_layout.addStretch()
        parent_layout.addLayout(button_layout)
    
    def test_connections(self):
        """测试各个连接端点的可用性"""
        import threading
        
        def test_sse():
            try:
                response = requests.get(self.data_collector.sse_url, timeout=5)
                if response.status_code == 200:
                    self.data_collector.connection_status_changed.emit('sse', 'test_success', True)
                else:
                    self.data_collector.connection_status_changed.emit('sse', f'test_failed_http_{response.status_code}', False)
            except Exception as e:
                self.data_collector.connection_status_changed.emit('sse', f'test_failed_{type(e).__name__}', False)
        
        def test_websocket():
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def test_ws():
                    try:
                        # 修复：使用asyncio.wait_for来控制超时
                        websocket = await asyncio.wait_for(
                            websockets.connect(self.data_collector.websocket_url), 
                            timeout=5
                        )
                        async with websocket:
                            await websocket.ping()
                            return True
                    except Exception as e:
                        raise e
                
                result = loop.run_until_complete(test_ws())
                loop.close()
                if result:
                    self.data_collector.connection_status_changed.emit('websocket', 'test_success', True)
            except Exception as e:
                self.data_collector.connection_status_changed.emit('websocket', f'test_failed_{type(e).__name__}', False)
        
        def test_rest_api():
            try:
                response = requests.get(self.data_collector.rest_api_url, timeout=5)
                if response.status_code == 200:
                    self.data_collector.connection_status_changed.emit('rest_api', 'test_success', True)
                else:
                    self.data_collector.connection_status_changed.emit('rest_api', f'test_failed_http_{response.status_code}', False)
            except Exception as e:
                self.data_collector.connection_status_changed.emit('rest_api', f'test_failed_{type(e).__name__}', False)
        
        # 在后台线程中执行测试
        threading.Thread(target=test_sse, daemon=True).start()
        threading.Thread(target=test_websocket, daemon=True).start()
        threading.Thread(target=test_rest_api, daemon=True).start()
    
    def init_data_collector(self):
        """初始化数据收集器"""
        self.data_collector = StreamDataCollector()
        
        # 连接信号
        self.data_collector.sse_data_received.connect(self.on_sse_data_received)
        self.data_collector.websocket_data_received.connect(self.on_websocket_data_received)
        self.data_collector.rest_api_data_received.connect(self.on_rest_api_data_received)
        self.data_collector.connection_status_changed.connect(self.on_connection_status_changed)
        self.data_collector.error_occurred.connect(self.on_error_occurred)
    
    def start_monitoring(self):
        """开始监控"""
        if self.data_collector and not self.data_collector.isRunning():
            self.data_collector.start()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
    
    def stop_monitoring(self):
        """停止监控"""
        if self.data_collector and self.data_collector.isRunning():
            self.data_collector.stop()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def refresh_status(self):
        """刷新状态"""
        self.update_stats_table()
    
    def clear_all_data(self):
        """清空所有数据"""
        self.sse_data_display.clear()
        self.websocket_data_display.clear()
        self.rest_api_data_display.clear()
        if self.data_collector:
            self.data_collector.sse_data.clear()
            self.data_collector.websocket_data.clear()
            self.data_collector.rest_api_data.clear()
    
    def check_server_status(self):
        """检查服务器状态"""
        if not self.data_collector or not self.data_collector.isRunning():
            return
            
        # 检查各个端点的可用性
        try:
            # 检查SSE端点
            response = requests.get(self.data_collector.sse_url, timeout=3)
            if response.status_code != 200:
                self.data_collector.connection_status_changed.emit('sse', 'server_unavailable', False)
        except:
            pass
            
        try:
            # 检查REST API端点
            response = requests.get(self.data_collector.rest_api_url, timeout=3)
            if response.status_code != 200:
                self.data_collector.connection_status_changed.emit('rest_api', 'server_unavailable', False)
        except:
            pass
    
    def on_sse_data_received(self, data):
        """处理SSE数据"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_data = f"[{timestamp}] SSE: {json.dumps(data, indent=2, ensure_ascii=False)}\n"
        self.sse_data_display.append(formatted_data)
        
        # 自动滚动到底部
        cursor = self.sse_data_display.textCursor()
        cursor.movePosition(cursor.End)
        self.sse_data_display.setTextCursor(cursor)
    
    def on_websocket_data_received(self, data):
        """处理WebSocket数据"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_data = f"[{timestamp}] WebSocket: {json.dumps(data, indent=2, ensure_ascii=False)}\n"
        self.websocket_data_display.append(formatted_data)
        
        # 自动滚动到底部
        cursor = self.websocket_data_display.textCursor()
        cursor.movePosition(cursor.End)
        self.websocket_data_display.setTextCursor(cursor)
    
    def on_rest_api_data_received(self, data):
        """处理REST API数据"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_data = f"[{timestamp}] REST API: {json.dumps(data, indent=2, ensure_ascii=False)}\n"
        self.rest_api_data_display.append(formatted_data)
        
        # 自动滚动到底部
        cursor = self.rest_api_data_display.textCursor()
        cursor.movePosition(cursor.End)
        self.rest_api_data_display.setTextCursor(cursor)
    
    def on_connection_status_changed(self, stream_type, status, connected):
        """处理连接状态变化"""
        if stream_type == 'sse':
            if connected:
                if status == 'test_success':
                    self.sse_status_label.setText("SSE: 🟢 测试成功")
                    self.sse_status_label.setStyleSheet("padding: 5px; border: 1px solid #28a745; border-radius: 3px; background-color: #d4edda;")
                else:
                    self.sse_status_label.setText("SSE: 🟢 已连接")
                    self.sse_status_label.setStyleSheet("padding: 5px; border: 1px solid #28a745; border-radius: 3px; background-color: #d4edda;")
            else:
                if status == 'max_retries_exceeded':
                    self.sse_status_label.setText("SSE: 🔴 连接失败 (重试次数超限)")
                elif status == 'server_unavailable':
                    self.sse_status_label.setText("SSE: 🟡 服务器不可用")
                elif status == 'connection_error':
                    self.sse_status_label.setText("SSE: 🔴 连接错误")
                elif status == 'timeout':
                    self.sse_status_label.setText("SSE: 🔴 连接超时")
                elif status == 'request_error':
                    self.sse_status_label.setText("SSE: 🔴 请求错误")
                elif status == 'unknown_error':
                    self.sse_status_label.setText("SSE: 🔴 未知错误")
                elif status.startswith('HTTP'):
                    self.sse_status_label.setText(f"SSE: 🔴 {status}")
                elif status.startswith('test_failed'):
                    error_type = status.replace('test_failed_', '')
                    self.sse_status_label.setText(f"SSE: 🔴 测试失败 ({error_type})")
                else:
                    self.sse_status_label.setText("SSE: 🔴 未连接")
                self.sse_status_label.setStyleSheet("padding: 5px; border: 1px solid #dc3545; border-radius: 3px; background-color: #f8d7da;")
        
        elif stream_type == 'websocket':
            if connected:
                if status == 'test_success':
                    self.websocket_status_label.setText("WebSocket: 🟢 测试成功")
                    self.websocket_status_label.setStyleSheet("padding: 5px; border: 1px solid #28a745; border-radius: 3px; background-color: #d4edda;")
                else:
                    self.websocket_status_label.setText("WebSocket: 🟢 已连接")
                    self.websocket_status_label.setStyleSheet("padding: 5px; border: 1px solid #28a745; border-radius: 3px; background-color: #d4edda;")
            else:
                if status == 'max_retries_exceeded':
                    self.websocket_status_label.setText("WebSocket: 🔴 连接失败 (重试次数超限)")
                elif status == 'connection_closed':
                    self.websocket_status_label.setText("WebSocket: 🔴 连接已关闭")
                elif status == 'invalid_uri':
                    self.websocket_status_label.setText("WebSocket: 🔴 无效URI")
                elif status == 'websocket_error':
                    self.websocket_status_label.setText("WebSocket: 🔴 WebSocket错误")
                elif status == 'connection_error':
                    self.websocket_status_label.setText("WebSocket: 🔴 连接错误")
                elif status.startswith('test_failed'):
                    error_type = status.replace('test_failed_', '')
                    self.websocket_status_label.setText(f"WebSocket: 🔴 测试失败 ({error_type})")
                else:
                    self.websocket_status_label.setText("WebSocket: 🔴 未连接")
                self.websocket_status_label.setStyleSheet("padding: 5px; border: 1px solid #dc3545; border-radius: 3px; background-color: #f8d7da;")
        
        elif stream_type == 'rest_api':
            if connected:
                if status == 'test_success':
                    self.rest_api_status_label.setText("REST API: 🟢 测试成功")
                    self.rest_api_status_label.setStyleSheet("padding: 5px; border: 1px solid #28a745; border-radius: 3px; background-color: #d4edda;")
                else:
                    self.rest_api_status_label.setText("REST API: 🟢 已连接")
                    self.rest_api_status_label.setStyleSheet("padding: 5px; border: 1px solid #28a745; border-radius: 3px; background-color: #d4edda;")
            else:
                if status == 'max_retries_exceeded':
                    self.rest_api_status_label.setText("REST API: 🔴 连接失败 (重试次数超限)")
                elif status == 'server_unavailable':
                    self.rest_api_status_label.setText("REST API: 🟡 服务器不可用")
                elif status == 'connection_error':
                    self.rest_api_status_label.setText("REST API: 🔴 连接错误")
                elif status == 'timeout':
                    self.rest_api_status_label.setText("REST API: 🔴 连接超时")
                elif status == 'request_error':
                    self.rest_api_status_label.setText("REST API: 🔴 请求错误")
                elif status == 'http_error':
                    self.rest_api_status_label.setText("REST API: 🔴 HTTP错误")
                elif status == 'unknown_error':
                    self.rest_api_status_label.setText("REST API: 🔴 未知错误")
                elif status.startswith('test_failed'):
                    error_type = status.replace('test_failed_', '')
                    self.rest_api_status_label.setText(f"REST API: 🔴 测试失败 ({error_type})")
                else:
                    self.rest_api_status_label.setText("REST API: 🔴 未连接")
                self.rest_api_status_label.setStyleSheet("padding: 5px; border: 1px solid #dc3545; border-radius: 3px; background-color: #f8d7da;")
        
        # 更新统计表格
        self.update_stats_table()
    
    def on_error_occurred(self, stream_type, error_message):
        """处理错误"""
        timestamp = time.strftime("%H:%M:%S")
        error_text = f"[{timestamp}] {stream_type.upper()} 错误: {error_message}\n"
        
        if stream_type == 'sse':
            self.sse_data_display.append(error_text)
        elif stream_type == 'websocket':
            self.websocket_data_display.append(error_text)
        elif stream_type == 'rest_api':
            self.rest_api_data_display.append(error_text)
    
    def update_stats_table(self):
        """更新统计表格"""
        if not self.data_collector:
            return
            
        # 准备数据
        stats_data = [
            ('SSE', 
             '🟢 已连接' if self.data_collector.connection_status['sse'] else '🔴 未连接',
             len(self.data_collector.sse_data),
             time.strftime("%H:%M:%S") if self.data_collector.sse_data else 'N/A'),
            ('WebSocket',
             '🟢 已连接' if self.data_collector.connection_status['websocket'] else '🔴 未连接',
             len(self.data_collector.websocket_data),
             time.strftime("%H:%M:%S") if self.data_collector.websocket_data else 'N/A'),
            ('REST API',
             '🟢 已连接' if self.data_collector.connection_status['rest_api'] else '🔴 未连接',
             len(self.data_collector.rest_api_data),
             time.strftime("%H:%M:%S") if self.data_collector.rest_api_data else 'N/A')
        ]
        
        # 更新表格
        self.stats_table.setRowCount(len(stats_data))
        for row, (stream_type, status, count, last_update) in enumerate(stats_data):
            self.stats_table.setItem(row, 0, QTableWidgetItem(stream_type))
            self.stats_table.setItem(row, 1, QTableWidgetItem(status))
            self.stats_table.setItem(row, 2, QTableWidgetItem(str(count)))
            self.stats_table.setItem(row, 3, QTableWidgetItem(last_update))
    
    def export_data(self, stream_type):
        """导出数据"""
        if not self.data_collector:
            return
            
        data = []
        if stream_type == 'sse':
            data = self.data_collector.sse_data
        elif stream_type == 'websocket':
            data = self.data_collector.websocket_data
        elif stream_type == 'rest_api':
            data = self.data_collector.rest_api_data
        
        if data:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{stream_type}_data_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"数据已导出到: {filename}")
    
    def closeEvent(self, event):
        """关闭事件"""
        if self.data_collector and self.data_collector.isRunning():
            self.data_collector.stop()
        if self.status_timer:
            self.status_timer.stop()
        event.accept() 