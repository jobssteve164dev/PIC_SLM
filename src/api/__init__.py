"""
API模块 - 提供训练数据流和LLM集成的API接口

主要组件：
- SSE Handler: Server-Sent Events数据流服务
- WebSocket Handler: WebSocket实时通信服务  
- REST API: HTTP REST接口服务
- Stream Server: 统一的数据流服务器
"""

from .sse_handler import SSEHandler
from .websocket_handler import WebSocketHandler
from .rest_api import TrainingAPI
from .stream_server import TrainingStreamServer

__all__ = [
    'SSEHandler',
    'WebSocketHandler', 
    'TrainingAPI',
    'TrainingStreamServer'
] 