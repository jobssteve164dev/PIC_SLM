"""
WebSocket Handler - WebSocket实时通信处理器

提供训练指标的双向WebSocket通信服务，支持实时数据推送和命令控制。
"""

import json
import asyncio
import websockets
import threading
import time
from typing import Dict, Any, Set
from websockets.server import WebSocketServerProtocol
import logging


class WebSocketHandler:
    """WebSocket实时通信处理器"""
    
    def __init__(self, host='127.0.0.1', port=8889):
        """
        初始化WebSocket处理器
        
        Args:
            host: 服务器主机地址
            port: 服务器端口
        """
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.server = None
        self.is_running = False
        self.metrics_history = []
        self.logger = logging.getLogger(__name__)
        
        # 支持的消息类型
        self.message_handlers = {
            'subscribe': self._handle_subscribe,
            'unsubscribe': self._handle_unsubscribe,
            'get_history': self._handle_get_history,
            'ping': self._handle_ping
        }
    
    async def _handle_connection(self, websocket: WebSocketServerProtocol):
        """处理WebSocket连接（兼容websockets库）"""
        await self.register_client(websocket, "/")
    
    async def register_client(self, websocket: WebSocketServerProtocol, path: str):
        """注册新的WebSocket客户端"""
        self.clients.add(websocket)
        self.logger.info(f"新WebSocket客户端连接: {websocket.remote_address}, 当前连接数: {len(self.clients)}")
        
        try:
            # 发送连接确认
            await websocket.send(json.dumps({
                'type': 'connection',
                'status': 'connected',
                'client_id': id(websocket),
                'timestamp': time.time()
            }))
            
            # 发送最近的历史数据
            if self.metrics_history:
                recent_metrics = self.metrics_history[-5:]  # 最近5条记录
                for metrics in recent_metrics:
                    await websocket.send(json.dumps(metrics))
            
            # 监听客户端消息
            async for message in websocket:
                await self._handle_client_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"WebSocket客户端断开连接: {websocket.remote_address}")
        except Exception as e:
            self.logger.error(f"WebSocket客户端处理错误: {str(e)}")
        finally:
            self.clients.discard(websocket)
            self.logger.info(f"WebSocket客户端已移除，当前连接数: {len(self.clients)}")
    
    async def _handle_client_message(self, websocket: WebSocketServerProtocol, message: str):
        """处理客户端消息"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type in self.message_handlers:
                response = await self.message_handlers[message_type](websocket, data)
                if response:
                    await websocket.send(json.dumps(response))
            else:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': f'Unknown message type: {message_type}'
                }))
                
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Invalid JSON format'
            }))
        except Exception as e:
            self.logger.error(f"处理客户端消息时出错: {str(e)}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def _handle_subscribe(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """处理订阅请求"""
        subscription_type = data.get('subscription', 'all')
        return {
            'type': 'subscription_confirmed',
            'subscription': subscription_type,
            'client_id': id(websocket)
        }
    
    async def _handle_unsubscribe(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """处理取消订阅请求"""
        return {
            'type': 'unsubscription_confirmed',
            'client_id': id(websocket)
        }
    
    async def _handle_get_history(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """处理获取历史数据请求"""
        limit = data.get('limit', 10)
        history = self.metrics_history[-limit:] if self.metrics_history else []
        
        return {
            'type': 'history_data',
            'data': history,
            'count': len(history),
            'total': len(self.metrics_history)
        }
    
    async def _handle_ping(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """处理心跳请求"""
        return {
            'type': 'pong',
            'timestamp': time.time(),
            'client_id': id(websocket)
        }
    
    async def broadcast_metrics(self, metrics: Dict[str, Any]):
        """广播训练指标到所有连接的客户端"""
        if not self.clients:
            return
        
        # 格式化指标数据
        formatted_metrics = {
            'type': 'metrics_update',
            'timestamp': time.time(),
            'data': metrics
        }
        
        # 添加到历史记录
        self.metrics_history.append(formatted_metrics)
        # 保持历史记录在合理范围内
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-500:]
        
        # 广播到所有客户端
        if self.clients:
            # 创建所有发送任务
            tasks = []
            dead_clients = []
            
            for client in self.clients.copy():
                try:
                    task = client.send(json.dumps(formatted_metrics))
                    tasks.append(task)
                except websockets.exceptions.ConnectionClosed:
                    dead_clients.append(client)
                except Exception as e:
                    self.logger.warning(f"向客户端发送数据时出错: {str(e)}")
                    dead_clients.append(client)
            
            # 清理死连接
            for dead_client in dead_clients:
                self.clients.discard(dead_client)
            
            # 并发发送所有消息
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    def broadcast_metrics_sync(self, metrics: Dict[str, Any]):
        """同步方式广播指标（从其他线程调用）"""
        if not self.is_running:
            return
        
        # 在新的事件循环中运行异步广播
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.broadcast_metrics(metrics))
            loop.close()
        except Exception as e:
            self.logger.error(f"同步广播指标时出错: {str(e)}")
    
    async def start_server(self):
        """启动WebSocket服务器"""
        self.is_running = True
        self.logger.info(f"启动WebSocket服务器: ws://{self.host}:{self.port}")
        
        try:
            self.server = await websockets.serve(
                self._handle_connection,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=10
            )
            self.logger.info(f"WebSocket服务器启动成功: ws://{self.host}:{self.port}")
            return self.server
        except OSError as e:
            if e.errno == 10048:  # 端口已被占用
                self.logger.error(f"端口 {self.port} 已被占用，尝试停止现有服务器...")
                # 强制重置状态
                self.is_running = False
                self.server = None
                raise Exception(f"端口 {self.port} 已被占用，请确保没有其他WebSocket服务器在运行")
            else:
                raise
    
    def start_server_thread(self):
        """在独立线程中启动WebSocket服务器"""
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def server_main():
                await self.start_server()
                # 保持服务器运行，直到is_running变为False
                while self.is_running:
                    await asyncio.sleep(0.1)
            
            try:
                loop.run_until_complete(server_main())
            except KeyboardInterrupt:
                self.logger.info("WebSocket服务器收到停止信号")
            except Exception as e:
                self.logger.error(f"WebSocket服务器运行出错: {str(e)}")
            finally:
                try:
                    # 确保服务器完全关闭
                    if self.server:
                        try:
                            self.server.close()
                        except:
                            pass
                    loop.close()
                except:
                    pass
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        return server_thread
    
    async def stop_server(self):
        """停止WebSocket服务器"""
        self.is_running = False
        
        # 向所有客户端发送断开通知
        if self.clients:
            disconnect_message = json.dumps({
                'type': 'server_shutdown',
                'message': 'Server is shutting down',
                'timestamp': time.time()
            })
            
            tasks = []
            for client in self.clients.copy():
                try:
                    tasks.append(client.send(disconnect_message))
                except:
                    pass
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        # 关闭服务器
        if self.server:
            try:
                self.server.close()
                await self.server.wait_closed()
            except Exception as e:
                self.logger.error(f"关闭WebSocket服务器时出错: {str(e)}")
        
        self.clients.clear()
        self.logger.info("WebSocket服务器已停止")
    
    def stop_server_sync(self):
        """同步方式停止WebSocket服务器"""
        if not self.is_running:
            return
        
        try:
            # 强制重置状态
            self.is_running = False
            self.clients.clear()
            
            # 直接关闭服务器，不使用异步方法
            if self.server:
                try:
                    self.server.close()
                    self.logger.info("WebSocket服务器已关闭")
                except Exception as e:
                    self.logger.error(f"关闭WebSocket服务器时出错: {str(e)}")
                finally:
                    self.server = None
            else:
                self.server = None
                
            # 等待一段时间确保端口释放
            import time
            time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"同步停止WebSocket服务器时出错: {str(e)}")
            # 强制重置状态
            self.is_running = False
            self.clients.clear()
            self.server = None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务器统计信息"""
        return {
            'active_clients': len(self.clients),
            'is_running': self.is_running,
            'metrics_count': len(self.metrics_history),
            'server_address': f"ws://{self.host}:{self.port}"
        } 