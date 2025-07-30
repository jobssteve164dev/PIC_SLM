#!/usr/bin/env python3
"""
诊断数据流问题
"""

import sys
import os
import time
import json
import requests
import threading

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def test_rest_api():
    """测试REST API"""
    print("🔍 测试REST API...")
    try:
        # 测试健康检查
        response = requests.get("http://127.0.0.1:8890/api/system/health", timeout=5)
        print(f"✅ REST API健康检查: {response.status_code}")
        
        # 测试当前指标
        response = requests.get("http://127.0.0.1:8890/api/metrics/current", timeout=5)
        print(f"✅ REST API当前指标: {response.status_code}")
        data = response.json()
        print(f"   数据包数量: {len(data.get('data', {}))}")
        
        return True
    except Exception as e:
        print(f"❌ REST API测试失败: {str(e)}")
        return False

def test_sse_connection():
    """测试SSE连接"""
    print("🔍 测试SSE连接...")
    try:
        response = requests.get("http://127.0.0.1:8888/api/stream/status", timeout=5)
        print(f"✅ SSE状态检查: {response.status_code}")
        data = response.json()
        print(f"   活跃客户端: {data.get('active_clients', 0)}")
        print(f"   指标数量: {data.get('metrics_count', 0)}")
        
        # 尝试连接SSE流
        print("   尝试连接SSE流...")
        response = requests.get("http://127.0.0.1:8888/api/stream/metrics", stream=True, timeout=5)
        if response.status_code == 200:
            print("   ✅ SSE流连接成功")
            # 读取几行数据
            line_count = 0
            for line in response.iter_lines():
                if line_count >= 5:  # 只读取前5行
                    break
                print(f"   📨 数据: {line.decode('utf-8')[:100]}...")
                line_count += 1
        else:
            print(f"   ❌ SSE流连接失败: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ SSE测试失败: {str(e)}")
        return False

def test_websocket_connection():
    """测试WebSocket连接"""
    print("🔍 测试WebSocket连接...")
    try:
        import websockets
        import asyncio
        
        async def test_websocket():
            try:
                async with websockets.connect("ws://127.0.0.1:8889") as websocket:
                    print("   ✅ WebSocket连接成功")
                    
                    # 发送订阅消息
                    await websocket.send(json.dumps({
                        'type': 'subscribe',
                        'channel': 'metrics'
                    }))
                    
                    # 等待响应
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        print(f"   📨 订阅响应: {response[:100]}...")
                    except asyncio.TimeoutError:
                        print("   ⏰ 等待响应超时")
                    
                    # 发送ping
                    await websocket.send(json.dumps({
                        'type': 'ping'
                    }))
                    
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        print(f"   📨 Ping响应: {response[:100]}...")
                    except asyncio.TimeoutError:
                        print("   ⏰ Ping响应超时")
                        
            except Exception as e:
                print(f"   ❌ WebSocket连接失败: {str(e)}")
        
        # 运行异步测试
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(test_websocket())
        loop.close()
        
        return True
    except Exception as e:
        print(f"❌ WebSocket测试失败: {str(e)}")
        return False

def simulate_training_data():
    """模拟训练数据"""
    print("🔍 模拟训练数据...")
    try:
        # 导入数据流服务器管理器
        from api.stream_server_manager import get_stream_server
        
        # 获取数据流服务器
        stream_server = get_stream_server()
        
        # 模拟训练指标
        test_metrics = {
            'epoch': 1,
            'train_loss': 2.1,
            'train_accuracy': 0.45,
            'val_loss': 2.3,
            'val_accuracy': 0.42,
            'learning_rate': 0.001,
            'model_name': 'TestModel',
            'timestamp': time.time()
        }
        
        print(f"📊 发送测试指标: {test_metrics}")
        stream_server.broadcast_metrics(test_metrics)
        
        # 等待一下
        time.sleep(1)
        
        return True
    except Exception as e:
        print(f"❌ 模拟训练数据失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("🔧 数据流问题诊断")
    print("=" * 60)
    
    # 测试各个组件
    rest_ok = test_rest_api()
    sse_ok = test_sse_connection()
    websocket_ok = test_websocket_connection()
    
    print("\n📊 诊断结果:")
    print(f"REST API: {'✅ 正常' if rest_ok else '❌ 异常'}")
    print(f"SSE: {'✅ 正常' if sse_ok else '❌ 异常'}")
    print(f"WebSocket: {'✅ 正常' if websocket_ok else '❌ 异常'}")
    
    # 模拟训练数据
    print("\n🧪 模拟训练数据...")
    simulate_training_data()
    
    # 再次测试
    print("\n📊 模拟数据后的状态:")
    test_rest_api()
    test_sse_connection()
    test_websocket_connection()
    
    print("\n💡 问题分析:")
    print("1. REST API有数据是因为update_metrics被调用")
    print("2. SSE和WebSocket没有数据可能是因为没有客户端连接")
    print("3. 需要确保客户端正确连接到SSE和WebSocket服务器")

if __name__ == "__main__":
    main() 