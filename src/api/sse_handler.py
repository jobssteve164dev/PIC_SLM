"""
SSE Handler - Server-Sent Events数据流处理器

提供训练指标的实时SSE数据流服务，支持多客户端连接和自动重连。
"""

import json
import time
import threading
from queue import Queue, Empty
from typing import Dict, Any, Set
from flask import Flask, Response, request
from flask_cors import CORS
import logging


class SSEHandler:
    """SSE数据流处理器"""
    
    def __init__(self, max_clients=50, buffer_size=1000):
        """
        初始化SSE处理器
        
        Args:
            max_clients: 最大客户端连接数
            buffer_size: 指标缓冲区大小
        """
        self.clients: Set[Queue] = set()
        self.max_clients = max_clients
        self.buffer_size = buffer_size
        self.metrics_history = []
        self.is_running = False
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # 创建Flask应用
        self.app = Flask(__name__)
        CORS(self.app)
        self._setup_routes()
        
    def _setup_routes(self):
        """设置Flask路由"""
        
        @self.app.route('/api/stream/metrics')
        def stream_metrics():
            """SSE数据流端点"""
            return self._create_event_stream()
        
        @self.app.route('/api/stream/status')
        def stream_status():
            """获取流状态"""
            return {
                'active_clients': len(self.clients),
                'max_clients': self.max_clients,
                'is_running': self.is_running,
                'metrics_count': len(self.metrics_history)
            }
        
        @self.app.route('/api/stream/history')
        def get_history():
            """获取历史指标"""
            limit = request.args.get('limit', 100, type=int)
            return {
                'metrics': self.metrics_history[-limit:] if self.metrics_history else [],
                'total_count': len(self.metrics_history)
            }
    
    def _create_event_stream(self):
        """创建SSE事件流"""
        def event_stream():
            # 为这个客户端创建专用队列
            client_queue = Queue(maxsize=self.buffer_size)
            
            with self.lock:
                if len(self.clients) >= self.max_clients:
                    yield f"data: {json.dumps({'error': 'Too many clients'})}\n\n"
                    return
                
                self.clients.add(client_queue)
                self.logger.info(f"新客户端连接，当前连接数: {len(self.clients)}")
            
            try:
                # 发送连接确认
                yield f"data: {json.dumps({'type': 'connection', 'status': 'connected'})}\n\n"
                
                # 发送最近的历史数据
                if self.metrics_history:
                    recent_metrics = self.metrics_history[-10:]  # 最近10条记录
                    for metrics in recent_metrics:
                        yield f"data: {json.dumps(metrics)}\n\n"
                
                # 持续发送新数据
                while True:
                    try:
                        # 等待新指标数据，超时检查连接状态
                        metrics = client_queue.get(timeout=30)
                        if metrics is None:  # 停止信号
                            break
                        yield f"data: {json.dumps(metrics)}\n\n"
                    except Empty:
                        # 发送心跳包保持连接
                        yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': time.time()})}\n\n"
                        
            except GeneratorExit:
                # 客户端断开连接
                pass
            finally:
                with self.lock:
                    self.clients.discard(client_queue)
                    self.logger.info(f"客户端断开连接，当前连接数: {len(self.clients)}")
        
        return Response(event_stream(), mimetype='text/plain')
    
    def broadcast_metrics(self, metrics: Dict[str, Any]):
        """广播训练指标到所有连接的客户端"""
        if not self.is_running:
            return
        
        # 添加时间戳和事件类型
        formatted_metrics = {
            'timestamp': time.time(),
            'event_type': 'metrics_update',
            'data': metrics
        }
        
        # 添加到历史记录
        with self.lock:
            self.metrics_history.append(formatted_metrics)
            # 保持历史记录在合理范围内
            if len(self.metrics_history) > self.buffer_size:
                self.metrics_history = self.metrics_history[-self.buffer_size//2:]
        
        # 广播到所有客户端
        dead_clients = []
        for client_queue in self.clients.copy():
            try:
                client_queue.put_nowait(formatted_metrics)
            except:
                # 客户端队列已满或已关闭
                dead_clients.append(client_queue)
        
        # 清理死连接
        if dead_clients:
            with self.lock:
                for dead_client in dead_clients:
                    self.clients.discard(dead_client)
    
    def start_server(self, host='127.0.0.1', port=8888, debug=False):
        """启动SSE服务器"""
        self.is_running = True
        self.logger.info(f"启动SSE服务器: http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)
    
    def stop_server(self):
        """停止SSE服务器"""
        self.is_running = False
        
        # 向所有客户端发送停止信号
        for client_queue in self.clients.copy():
            try:
                client_queue.put_nowait(None)
            except:
                pass
        
        self.clients.clear()
        self.logger.info("SSE服务器已停止")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务器统计信息"""
        return {
            'active_clients': len(self.clients),
            'max_clients': self.max_clients,
            'is_running': self.is_running,
            'metrics_count': len(self.metrics_history),
            'buffer_size': self.buffer_size
        } 