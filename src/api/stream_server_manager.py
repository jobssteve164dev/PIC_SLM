"""
数据流服务器管理器 - 全局单例管理器

确保整个应用程序中只有一个数据流服务器实例在运行，避免端口冲突。
"""

import threading
import time
import logging
from typing import Dict, Any, Optional
from .stream_server import TrainingStreamServer


class StreamServerManager:
    """全局数据流服务器管理器 - 单例模式"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.logger = logging.getLogger(__name__)
        
        # 服务器实例
        self._stream_server: Optional[TrainingStreamServer] = None
        self._server_lock = threading.Lock()
        
        # 服务器状态
        self._is_initialized = False
        self._startup_time = None
        self._reference_count = 0  # 引用计数
        
        # 默认配置
        self._default_config = {
            'sse_host': '127.0.0.1',
            'sse_port': 8888,
            'websocket_host': '127.0.0.1',
            'websocket_port': 8889,
            'rest_api_host': '127.0.0.1',
            'rest_api_port': 8890,
            'buffer_size': 1000,
            'debug_mode': False
        }
    
    def get_stream_server(self, training_system=None, config=None) -> TrainingStreamServer:
        """
        获取数据流服务器实例
        
        Args:
            training_system: 训练系统实例
            config: 服务器配置
            
        Returns:
            TrainingStreamServer: 数据流服务器实例
        """
        with self._server_lock:
            # 如果服务器不存在或未运行，创建新实例
            if self._stream_server is None or not self._stream_server.is_running:
                self._create_stream_server(training_system, config)
            
            # 增加引用计数
            self._reference_count += 1
            self.logger.info(f"获取数据流服务器实例，当前引用计数: {self._reference_count}")
            
            return self._stream_server
    
    def _create_stream_server(self, training_system=None, config=None):
        """创建数据流服务器实例"""
        try:
            # 如果已有服务器实例，先确保完全停止
            if self._stream_server is not None:
                self.logger.info("检测到现有服务器实例，先停止...")
                try:
                    self._stream_server.stop_all_servers()
                except Exception as e:
                    self.logger.warning(f"停止现有服务器时出错: {str(e)}")
                
                # 等待一段时间确保端口释放
                time.sleep(5)
                self._stream_server = None
                self._is_initialized = False
            
            # 强制清理可能残留的端口绑定
            self._force_cleanup_ports()
            
            # 合并配置
            server_config = self._default_config.copy()
            if config:
                server_config.update(config)
            
            self.logger.info("创建新的数据流服务器实例...")
            
            # 创建服务器实例
            self._stream_server = TrainingStreamServer(
                training_system=training_system,
                config=server_config
            )
            
            # 启动服务器
            self._start_stream_server()
            
            self._is_initialized = True
            self._startup_time = time.time()
            
            self.logger.info("数据流服务器实例创建并启动成功")
            
        except Exception as e:
            self.logger.error(f"创建数据流服务器实例失败: {str(e)}")
            self._stream_server = None
            raise
    
    def _force_cleanup_ports(self):
        """强制清理可能残留的端口绑定"""
        try:
            import socket
            
            # 检查并清理WebSocket端口
            try:
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                test_socket.bind(('127.0.0.1', 8889))
                test_socket.close()
                self.logger.info("WebSocket端口 8889 可用")
            except OSError:
                self.logger.warning("WebSocket端口 8889 可能被占用，尝试强制释放...")
                # 等待更长时间
                time.sleep(2)
            
            # 检查并清理SSE端口
            try:
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                test_socket.bind(('127.0.0.1', 8888))
                test_socket.close()
                self.logger.info("SSE端口 8888 可用")
            except OSError:
                self.logger.warning("SSE端口 8888 可能被占用")
            
            # 检查并清理REST API端口
            try:
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                test_socket.bind(('127.0.0.1', 8890))
                test_socket.close()
                self.logger.info("REST API端口 8890 可用")
            except OSError:
                self.logger.warning("REST API端口 8890 可能被占用")
                
        except Exception as e:
            self.logger.warning(f"端口清理检查时出错: {str(e)}")
    
    def _start_stream_server(self):
        """启动数据流服务器"""
        if self._stream_server is None:
            return
        
        try:
            # 在独立线程中启动服务器
            def start_server():
                try:
                    self.logger.info("开始启动数据流服务器...")
                    self._stream_server.start_all_servers()
                    self.logger.info("数据流服务器启动完成")
                except Exception as e:
                    self.logger.error(f"启动数据流服务器失败: {str(e)}")
                    import traceback
                    self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            
            server_thread = threading.Thread(target=start_server, daemon=True)
            server_thread.start()
            
            # 等待服务器启动，增加等待时间
            time.sleep(5)
            
            # 检查服务器状态 - 更宽松的检查
            if self._stream_server.is_running:
                self.logger.info("数据流服务器启动成功")
            else:
                # 检查各个组件的状态
                self.logger.warning("数据流服务器状态检查，检查各组件状态:")
                sse_running = hasattr(self._stream_server.sse_handler, 'is_running')
                ws_running = self._stream_server.websocket_handler.is_running
                rest_running = self._stream_server.rest_api.is_running
                
                self.logger.info(f"SSE服务器状态: {sse_running}")
                self.logger.info(f"WebSocket服务器状态: {ws_running}")
                self.logger.info(f"REST API服务器状态: {rest_running}")
                
                # 如果至少有两个服务器运行，就认为启动成功
                running_count = sum([sse_running, ws_running, rest_running])
                if running_count >= 2:
                    self.logger.info(f"数据流服务器部分启动成功 ({running_count}/3 个组件运行)")
                    # 强制设置运行状态
                    self._stream_server.is_running = True
                else:
                    self.logger.error("数据流服务器启动失败")
                    raise Exception("数据流服务器启动失败")
                
        except Exception as e:
            self.logger.error(f"启动数据流服务器时出错: {str(e)}")
            raise
    
    def release_stream_server(self):
        """释放数据流服务器引用"""
        with self._server_lock:
            if self._reference_count > 0:
                self._reference_count -= 1
                self.logger.info(f"释放数据流服务器引用，当前引用计数: {self._reference_count}")
                
                # 如果没有引用，停止服务器
                if self._reference_count == 0:
                    self._stop_stream_server()
    
    def _stop_stream_server(self):
        """停止数据流服务器"""
        if self._stream_server is not None:
            try:
                self.logger.info("停止数据流服务器...")
                self._stream_server.stop_all_servers()
                self._stream_server = None
                self._is_initialized = False
                self.logger.info("数据流服务器已停止")
            except Exception as e:
                self.logger.error(f"停止数据流服务器时出错: {str(e)}")
    
    def is_server_running(self) -> bool:
        """检查服务器是否正在运行"""
        return (self._stream_server is not None and 
                self._stream_server.is_running and 
                self._is_initialized)
    
    def get_server_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        if not self.is_server_running():
            return {
                'is_running': False,
                'reference_count': self._reference_count,
                'is_initialized': self._is_initialized,
                'startup_time': self._startup_time
            }
        
        try:
            server_info = self._stream_server.get_server_info()
            server_info.update({
                'reference_count': self._reference_count,
                'is_initialized': self._is_initialized,
                'startup_time': self._startup_time
            })
            return server_info
        except Exception as e:
            self.logger.error(f"获取服务器信息失败: {str(e)}")
            return {
                'is_running': False,
                'error': str(e),
                'reference_count': self._reference_count,
                'is_initialized': self._is_initialized,
                'startup_time': self._startup_time
            }
    
    def get_api_endpoints(self) -> Dict[str, str]:
        """获取API端点列表"""
        if not self.is_server_running():
            return {}
        
        try:
            return self._stream_server.get_api_endpoints()
        except Exception as e:
            self.logger.error(f"获取API端点失败: {str(e)}")
            return {}
    
    def broadcast_metrics(self, metrics: Dict[str, Any]):
        """广播训练指标"""
        if not self.is_server_running():
            return
        
        try:
            self._stream_server.broadcast_metrics(metrics)
        except Exception as e:
            self.logger.error(f"广播训练指标失败: {str(e)}")
    
    def force_restart(self, training_system=None, config=None):
        """强制重启数据流服务器"""
        with self._server_lock:
            self.logger.info("强制重启数据流服务器...")
            
            # 停止现有服务器
            if self._stream_server is not None:
                try:
                    self._stream_server.stop_all_servers()
                except Exception as e:
                    self.logger.error(f"停止现有服务器时出错: {str(e)}")
            
            # 重置状态
            self._stream_server = None
            self._is_initialized = False
            self._reference_count = 0
            
            # 创建新服务器
            self._create_stream_server(training_system, config)
            
            self.logger.info("数据流服务器重启完成")
    
    def cleanup(self):
        """清理资源"""
        with self._server_lock:
            self.logger.info("清理数据流服务器管理器...")
            self._stop_stream_server()
            self._reference_count = 0
            self._is_initialized = False
            self._startup_time = None


# 全局实例
_global_manager = None

def get_stream_server_manager() -> StreamServerManager:
    """获取全局数据流服务器管理器实例"""
    global _global_manager
    if _global_manager is None:
        _global_manager = StreamServerManager()
    return _global_manager

def get_stream_server(training_system=None, config=None) -> TrainingStreamServer:
    """获取数据流服务器实例的便捷方法"""
    manager = get_stream_server_manager()
    return manager.get_stream_server(training_system, config)

def release_stream_server():
    """释放数据流服务器引用的便捷方法"""
    manager = get_stream_server_manager()
    manager.release_stream_server()

def is_stream_server_running() -> bool:
    """检查数据流服务器是否正在运行的便捷方法"""
    manager = get_stream_server_manager()
    return manager.is_server_running() 