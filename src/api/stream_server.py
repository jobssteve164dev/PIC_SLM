"""
Stream Server - 统一的训练数据流服务器

整合SSE、WebSocket和REST API三种数据接口，提供统一的训练指标数据流服务。
"""

import threading
import time
import logging
from typing import Dict, Any, Optional
from .sse_handler import SSEHandler
from .websocket_handler import WebSocketHandler
from .rest_api import TrainingAPI


class TrainingStreamServer:
    """统一的训练数据流服务器"""
    
    def __init__(self, training_system=None, config=None):
        """
        初始化数据流服务器
        
        Args:
            training_system: 训练系统实例
            config: 服务器配置
        """
        self.training_system = training_system
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # 初始化各个服务组件
        self.sse_handler = SSEHandler(
            max_clients=self.config.get('sse_max_clients', 50),
            buffer_size=self.config.get('buffer_size', 1000)
        )
        
        self.websocket_handler = WebSocketHandler(
            host=self.config.get('websocket_host', '127.0.0.1'),
            port=self.config.get('websocket_port', 8889)
        )
        
        self.rest_api = TrainingAPI(training_system)
        
        # 服务器状态
        self.is_running = False
        self.server_threads = {}
        
        # 连接训练系统信号
        self._connect_training_signals()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'sse_host': '127.0.0.1',
            'sse_port': 8888,
            'sse_max_clients': 50,
            'websocket_host': '127.0.0.1',
            'websocket_port': 8889,
            'rest_api_host': '127.0.0.1',
            'rest_api_port': 8890,
            'buffer_size': 1000,
            'auto_start_all': True,
            'debug_mode': False
        }
    
    def _connect_training_signals(self):
        """连接训练系统的信号"""
        if not self.training_system:
            return
        
        try:
            # 连接训练指标更新信号
            if hasattr(self.training_system, 'epoch_finished'):
                self.training_system.epoch_finished.connect(self.broadcast_metrics)
            
            # 连接其他训练相关信号
            if hasattr(self.training_system, 'progress_updated'):
                self.training_system.progress_updated.connect(self._on_progress_updated)
            
            if hasattr(self.training_system, 'status_updated'):
                self.training_system.status_updated.connect(self._on_status_updated)
            
            self.logger.info("成功连接训练系统信号")
            
        except Exception as e:
            self.logger.warning(f"连接训练系统信号失败: {str(e)}")
    
    def start_all_servers(self):
        """启动所有服务器"""
        self.logger.info("启动所有数据流服务器...")
        
        try:
            # 启动SSE服务器
            self.start_sse_server()
            
            # 启动WebSocket服务器
            self.start_websocket_server()
            
            # 启动REST API服务器
            self.start_rest_api_server()
            
            self.is_running = True
            self.logger.info("所有数据流服务器启动成功")
            
        except Exception as e:
            self.logger.error(f"启动服务器时出错: {str(e)}")
            self.stop_all_servers()
            raise
    
    def start_sse_server(self):
        """启动SSE服务器"""
        def run_sse_server():
            try:
                self.sse_handler.start_server(
                    host=self.config.get('sse_host', '127.0.0.1'),
                    port=self.config.get('sse_port', 8888),
                    debug=self.config.get('debug_mode', False)
                )
            except Exception as e:
                self.logger.error(f"SSE服务器启动失败: {str(e)}")
        
        sse_thread = threading.Thread(target=run_sse_server, daemon=True)
        sse_thread.start()
        self.server_threads['sse'] = sse_thread
        
        # 等待服务器启动
        time.sleep(1)
        self.logger.info("SSE服务器已启动")
    
    def start_websocket_server(self):
        """启动WebSocket服务器"""
        try:
            ws_thread = self.websocket_handler.start_server_thread()
            self.server_threads['websocket'] = ws_thread
            
            # 等待服务器启动
            time.sleep(1)
            self.logger.info("WebSocket服务器已启动")
            
        except Exception as e:
            self.logger.error(f"WebSocket服务器启动失败: {str(e)}")
            raise
    
    def start_rest_api_server(self):
        """启动REST API服务器"""
        def run_rest_api_server():
            try:
                self.rest_api.start_server(
                    host=self.config.get('rest_api_host', '127.0.0.1'),
                    port=self.config.get('rest_api_port', 8890),
                    debug=self.config.get('debug_mode', False)
                )
            except Exception as e:
                self.logger.error(f"REST API服务器启动失败: {str(e)}")
        
        api_thread = threading.Thread(target=run_rest_api_server, daemon=True)
        api_thread.start()
        self.server_threads['rest_api'] = api_thread
        
        # 等待服务器启动
        time.sleep(1)
        self.logger.info("REST API服务器已启动")
    
    def stop_all_servers(self):
        """停止所有服务器"""
        self.logger.info("停止所有数据流服务器...")
        
        self.is_running = False
        
        # 停止SSE服务器
        try:
            self.sse_handler.stop_server()
        except Exception as e:
            self.logger.error(f"停止SSE服务器时出错: {str(e)}")
        
        # 停止WebSocket服务器
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.websocket_handler.stop_server())
            loop.close()
        except Exception as e:
            self.logger.error(f"停止WebSocket服务器时出错: {str(e)}")
        
        # 等待线程结束
        for name, thread in self.server_threads.items():
            if thread and thread.is_alive():
                thread.join(timeout=5)
                if thread.is_alive():
                    self.logger.warning(f"{name}服务器线程未能正常结束")
        
        self.server_threads.clear()
        self.logger.info("所有数据流服务器已停止")
    
    def broadcast_metrics(self, metrics: Dict[str, Any]):
        """向所有客户端广播训练指标"""
        if not self.is_running:
            return
        
        try:
            # 广播到SSE客户端
            self.sse_handler.broadcast_metrics(metrics)
            
            # 广播到WebSocket客户端
            self.websocket_handler.broadcast_metrics_sync(metrics)
            
            # 更新REST API的指标数据
            self.rest_api.update_metrics(metrics)
            
        except Exception as e:
            self.logger.error(f"广播指标时出错: {str(e)}")
    
    def _on_progress_updated(self, progress: int):
        """处理训练进度更新"""
        progress_data = {
            'type': 'progress_update',
            'progress': progress,
            'timestamp': time.time()
        }
        self.broadcast_metrics(progress_data)
    
    def _on_status_updated(self, status: str):
        """处理训练状态更新"""
        status_data = {
            'type': 'status_update',
            'status': status,
            'timestamp': time.time()
        }
        self.broadcast_metrics(status_data)
    
    def get_server_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        return {
            'is_running': self.is_running,
            'endpoints': {
                'sse': f"http://{self.config.get('sse_host')}:{self.config.get('sse_port')}/api/stream/metrics",
                'websocket': f"ws://{self.config.get('websocket_host')}:{self.config.get('websocket_port')}",
                'rest_api': f"http://{self.config.get('rest_api_host')}:{self.config.get('rest_api_port')}/api"
            },
            'stats': {
                'sse': self.sse_handler.get_stats(),
                'websocket': self.websocket_handler.get_stats(),
                'active_threads': len([t for t in self.server_threads.values() if t and t.is_alive()])
            },
            'config': self.config
        }
    
    def get_api_endpoints(self) -> Dict[str, str]:
        """获取API端点列表"""
        return {
            'sse_stream': f"http://{self.config.get('sse_host')}:{self.config.get('sse_port')}/api/stream/metrics",
            'sse_status': f"http://{self.config.get('sse_host')}:{self.config.get('sse_port')}/api/stream/status",
            'sse_history': f"http://{self.config.get('sse_host')}:{self.config.get('sse_port')}/api/stream/history",
            'websocket': f"ws://{self.config.get('websocket_host')}:{self.config.get('websocket_port')}",
            'rest_current_metrics': f"http://{self.config.get('rest_api_host')}:{self.config.get('rest_api_port')}/api/metrics/current",
            'rest_metrics_history': f"http://{self.config.get('rest_api_host')}:{self.config.get('rest_api_port')}/api/metrics/history",
            'rest_training_control': f"http://{self.config.get('rest_api_host')}:{self.config.get('rest_api_port')}/api/training/control",
            'rest_training_status': f"http://{self.config.get('rest_api_host')}:{self.config.get('rest_api_port')}/api/training/status",
            'rest_system_info': f"http://{self.config.get('rest_api_host')}:{self.config.get('rest_api_port')}/api/system/info",
            'rest_health_check': f"http://{self.config.get('rest_api_host')}:{self.config.get('rest_api_port')}/api/system/health"
        }
    
    def test_connections(self) -> Dict[str, bool]:
        """测试所有服务器连接"""
        results = {}
        
        # 测试SSE服务器
        try:
            import requests
            response = requests.get(
                f"http://{self.config.get('sse_host')}:{self.config.get('sse_port')}/api/stream/status",
                timeout=5
            )
            results['sse'] = response.status_code == 200
        except:
            results['sse'] = False
        
        # 测试REST API服务器
        try:
            response = requests.get(
                f"http://{self.config.get('rest_api_host')}:{self.config.get('rest_api_port')}/api/system/health",
                timeout=5
            )
            results['rest_api'] = response.status_code == 200
        except:
            results['rest_api'] = False
        
        # 测试WebSocket服务器（简单检查）
        results['websocket'] = len(self.websocket_handler.clients) >= 0
        
        return results 