#!/usr/bin/env python3
"""
验证数据流服务器修复
"""

import sys
import os
import time
import json
import requests

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def test_stream_server():
    """测试数据流服务器"""
    print("🔧 验证数据流服务器修复...")
    print("=" * 60)
    
    try:
        # 导入数据流服务器管理器
        from api.stream_server_manager import get_stream_server
        
        # 创建配置
        config = {
            'sse_host': '127.0.0.1',
            'sse_port': 8888,
            'websocket_host': '127.0.0.1',
            'websocket_port': 8889,
            'rest_api_host': '127.0.0.1',
            'rest_api_port': 8890,
            'buffer_size': 1000,
            'debug_mode': False
        }
        
        print("📡 获取数据流服务器...")
        stream_server = get_stream_server(config=config)
        
        print("⏳ 等待服务器启动...")
        time.sleep(3)
        
        # 检查服务器状态
        if stream_server.is_running:
            print("✅ 数据流服务器启动成功!")
            
            # 测试REST API
            try:
                response = requests.get("http://127.0.0.1:8890/api/system/health", timeout=5)
                print(f"✅ REST API测试: {response.status_code}")
            except Exception as e:
                print(f"❌ REST API测试失败: {str(e)}")
            
            # 测试SSE
            try:
                response = requests.get("http://127.0.0.1:8888/api/stream/status", timeout=5)
                print(f"✅ SSE测试: {response.status_code}")
            except Exception as e:
                print(f"❌ SSE测试失败: {str(e)}")
            
            # 发送测试数据
            print("📊 发送测试数据...")
            test_metrics = {
                'epoch': 1,
                'train_loss': 2.1,
                'train_accuracy': 0.45,
                'timestamp': time.time()
            }
            stream_server.broadcast_metrics(test_metrics)
            
            print("✅ 测试数据发送成功!")
            
        else:
            print("❌ 数据流服务器启动失败!")
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_stream_server() 